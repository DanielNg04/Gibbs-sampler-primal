# primal_oracle_quantum_v1cube.py — SDP primal oracle (van Apeldoorn & Gilyén, arXiv:1804.05058 §2.2).

"""
Primal oracle with exact or MCMC Gibbs preparation (``gibbs_sampler_quantum_v2cube``).

MCMC mode samples the unpadded Gibbs state ρ_n and reconstructs the padded embedding
(ω, X′) from the partition function, because block-diagonal jumps do not mix the
corner coordinate. Jump operators are supplied by the driver (C and +F_i blocks).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as la
from tqdm import tqdm

from gibbs_sampler_quantum_v2cube import (
    QuantumGibbsSampler,
    jumps_from_hermitian_matrices,
    trace_distance,
)

# Real symmetric storage for all oracle matrices.
DTYPE = np.float64


def _to_symmetric_real(M: np.ndarray, *, name: str) -> np.ndarray:
    """Convert input to real symmetric float64; reject non-real data."""
    M = np.asarray(M)
    if np.iscomplexobj(M):
        im = la.norm(np.imag(M), ord="fro")
        if im > 1e-12 * max(1.0, la.norm(M, ord="fro")):
            raise ValueError(
                f"{name} must be real symmetric (imaginary Frobenius norm too large)."
            )
        M = np.real(M)
    M = np.asarray(M, dtype=DTYPE)
    return (M + M.T) / 2.0


def _is_symmetric(M: np.ndarray, tol: float) -> bool:
    M = np.asarray(M, dtype=DTYPE)
    return bool(la.norm(M - M.T, ord="fro") <= tol * max(1.0, la.norm(M, ord="fro")))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PrimalOracleProblem:
    """
    Input for the primal feasibility test at objective threshold g.

    Original SDP (paper Sec. 1.1): maximize Tr(CX) subject to Tr(A_j X) ≤ b_j,
    X ⪰ 0, with A_1 = I and b_1 = R so that Tr(X) ≤ R.

    The oracle augments A_0 = −C, b_0 = −g and works on the normalized variable
    X' = X/R with bounds b_j/R; see module docstring for the (n+1)-dimensional
    density-matrix embedding.

    **Real symmetric:** ``C`` and each ``A_j`` must be real symmetric (within tolerance).
    """

    A_matrices: list[np.ndarray]
    """Real symmetric matrices A_1, …, A_m (each n×n). Convention: A_1 = I, b_1 = R."""

    b: np.ndarray
    """Bounds b_1, …, b_m (e.g. b_1 = R for the trace constraint)."""

    C: np.ndarray
    """Objective matrix C (real symmetric, n×n)."""

    R: float
    """Trace bound: feasible X satisfy Tr(X) ≤ R (via A_1 = I, b_1 = R)."""

    g: float
    """Guessed dual-style objective threshold; b_0 = −g for the A_0 = −C constraint."""

    hermitian_tol: float = 1e-10
    trace_check_tol: float = 1e-9
    psd_check_tol: float = -1e-8
    offdiag_check_tol: float = 1e-7
    strict_first_constraint: bool = False
    """If True, raise when A_1, b_1 are far from I, R; else warn."""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.R <= 0:
            raise ValueError(f"R must be positive, got {self.R}")

        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        m = self.b.size
        if m < 1:
            raise ValueError("Need at least one constraint (m >= 1).")
        if len(self.A_matrices) != m:
            raise ValueError(
                f"len(A_matrices)={len(self.A_matrices)} must match len(b)={m}."
            )

        self.C = _to_symmetric_real(self.C, name="C")
        n = self.C.shape[0]
        if self.C.shape != (n, n):
            raise ValueError("C must be square.")

        for j, Aj in enumerate(self.A_matrices):
            Aj = _to_symmetric_real(Aj, name=f"A_{j+1}")
            if Aj.shape != (n, n):
                raise ValueError(f"A_{j+1} must be {n}×{n}, got {Aj.shape}.")
            self.A_matrices[j] = Aj
            if not _is_symmetric(Aj, self.hermitian_tol):
                raise ValueError(f"A_{j+1} is not symmetric within tolerance.")
        if not _is_symmetric(self.C, self.hermitian_tol):
            raise ValueError("C is not symmetric within tolerance.")

        id_n = np.eye(n, dtype=DTYPE)
        dev_A = la.norm(self.A_matrices[0] - id_n, ord="fro")
        dev_b = abs(float(self.b[0]) - self.R)
        if dev_A > 1e-6 * max(1.0, n) or dev_b > 1e-6 * max(1.0, abs(self.R)):
            msg = (
                f"Expected approximately A_1 = I and b_1 = R; "
                f"got ||A_1-I||_F = {dev_A:.3e}, |b_1-R| = {dev_b:.3e}."
            )
            if self.strict_first_constraint:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)


@dataclass
class ConstraintDiagnostics:
    """Tr(Ã_j ρ) versus normalized bounds b̃_j = b_j / R."""

    traces: np.ndarray
    bounds_tilde: np.ndarray
    slacks: np.ndarray  # b̃_j - Tr(Ã_j ρ); positive means satisfied
    max_violation_index: int | None
    """Index j in 0..m with largest violation Tr(Ã_j ρ) - b̃_j, or None if none > tol."""

    feasible_within_theta: bool


@dataclass
class GibbsDiagnostics:
    """Sanity check on the Gibbs state ρ."""

    trace_rho: float
    min_eig: float
    max_offdiag_block_norm: float
    omega_vs_one_minus_tr_Xp: float
    block_diagonal_ok: bool


@dataclass
class PrimalOracleResult:
    """Successful termination of :func:`run_primal_oracle`."""

    y: np.ndarray
    rho: np.ndarray
    X_prime: np.ndarray
    omega: float
    X: np.ndarray
    z: float
    constraint_diag: ConstraintDiagnostics
    gibbs_diag: GibbsDiagnostics
    iterations: int
    theta: float
    timing: dict[str, float] | None = None
    """Populated when ``run_primal_oracle(..., collect_timing=True)``."""

    gibbs_mode: str = "exact"
    """Which Gibbs preparation was used: ``"exact"`` or ``"mcmc"``."""

    gibbs_steps_per_iter: list[int] | None = None
    """MCMC mode only: channel applications to converge at each oracle iteration."""

    gibbs_converged_per_iter: list[bool] | None = None
    """MCMC mode only: whether Gibbs converged within the step cap at each iteration."""

    cutoff_trace_distance_per_iter: dict[int, list[float]] | None = None
    """MCMC + ``gibbs_step_cutoffs``: per cutoff c, the trace distance
    ``D(σ_cutoff, σ_full)`` between the early-terminated and fully-converged
    Gibbs states at each oracle iteration (same channel trajectory)."""


# ---------------------------------------------------------------------------
# Building blocks: feasibility form, scaling, embedding
# ---------------------------------------------------------------------------


def build_objective_and_full_constraints(problem: PrimalOracleProblem) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Return [A_0, A_1, …, A_m] with A_0 = −C and [b_0, …, b_m] with b_0 = −g.

    Indexing matches the paper: j = 0 is the objective constraint
    Tr(A_0 X) ≤ b_0  ⇔  Tr(−C X) ≤ −g  ⇔  Tr(C X) ≥ g for PSD X.
    """
    A0 = -problem.C
    b0 = -problem.g
    A_full = [A0] + list(problem.A_matrices)
    b_full = np.concatenate(([b0], np.asarray(problem.b, dtype=float)))
    return A_full, b_full


def normalized_bounds(b_full: np.ndarray, R: float) -> np.ndarray:
    """b̃_j = b_j / R for the X' = X/R formulation (matrices A_j are not scaled)."""
    if R <= 0:
        raise ValueError("R must be positive.")
    return np.asarray(b_full, dtype=float) / R


def embed_top_left(A: np.ndarray, extra: int = 1) -> np.ndarray:
    """
    Embed an n×n matrix into the top-left block of (n+extra)×(n+extra) zeros.

    For extra=1, Ã = diag(A, 0) as in the paper.
    """
    A = _to_symmetric_real(A, name="embed_top_left(A)")
    n = A.shape[0]
    out = np.zeros((n + extra, n + extra), dtype=DTYPE)
    out[:n, :n] = A
    return out


def extract_top_left(rho: np.ndarray, n: int) -> np.ndarray:
    """Top-left n×n block of ρ (the normalized primal part X')."""
    return np.asarray(rho, dtype=DTYPE)[:n, :n].copy()


def extract_omega(rho: np.ndarray) -> float:
    """Bottom-right scalar ω = ρ_{n,n} in the (n+1)×(n+1) embedding."""
    r = np.asarray(rho, dtype=DTYPE)
    return float(r[-1, -1])


def hamiltonian_padded(
    A_tilde: list[np.ndarray], y: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    """H = β ∑_j y_j Ã_j (real symmetric)."""
    if len(A_tilde) != len(y):
        raise ValueError("y length must match number of Ã_j.")
    d = A_tilde[0].shape[0]
    H = np.zeros((d, d), dtype=DTYPE)
    for coeff, At in zip(y, A_tilde, strict=True):
        H += float(coeff) * np.asarray(At, dtype=DTYPE)
    return beta * H


def gibbs_state_n_from_exponent(
    M: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Closed-form ``n``-dimensional Gibbs state and embedding corner.

    Given the exponent matrix ``M = β ∑_j y_j A_j`` (the already-scaled
    ``n×n`` Hamiltonian), return

    - ``ρ_n = exp(−M) / Z_n`` with ``Z_n = Tr exp(−M)`` (real symmetric, trace 1), and
    - ``ω = 1 / (Z_n + 1)`` — the padded corner of the paper's embedding.

    The primal block is then ``X' = (1 − ω) ρ_n`` (see the module embedding note),
    so this single routine yields the same ``X'`` and ``ω`` as the padded
    closed-form reference, and is the cheap path used for g-calibration.

    ``ω`` is computed from a log-sum-exp so it stays accurate even when ``Z_n``
    overflows ``float64``.
    """
    M = _to_symmetric_real(M, name="M")
    w, U = la.eigh(M)
    emin = float(np.min(w))
    p = np.exp(-(w - emin))
    Z_scaled = float(np.sum(p))
    if Z_scaled <= 0.0 or not np.isfinite(Z_scaled):
        raise RuntimeError("Scaled partition function invalid in gibbs_state_n_from_exponent.")
    probs = p / Z_scaled
    rho_n = (U * probs) @ U.T
    rho_n = (rho_n + rho_n.T) / 2.0
    # log Z_n = log(sum exp(-(w - emin))) - emin; omega = 1/(Z_n + 1).
    log_Z = float(np.log(Z_scaled) - emin)
    omega = float(np.exp(-np.logaddexp(log_Z, 0.0)))
    return rho_n, omega


def _padded_rho_from_n(rho_n: np.ndarray, omega: float, n: int) -> np.ndarray:
    """Assemble the padded ρ̃ = diag((1−ω) ρ_n, ω) for diagnostics / legacy callers."""
    rho = np.zeros((n + 1, n + 1), dtype=DTYPE)
    rho[:n, :n] = (1.0 - omega) * np.asarray(rho_n, dtype=DTYPE)
    rho[n, n] = float(omega)
    return rho


def _omega_from_sampler(sampler: QuantumGibbsSampler) -> float:
    """Recover ω = 1/(Z_n + 1) from a sampler's energies via log-sum-exp (stable)."""
    E = np.asarray(sampler.energies, dtype=float)
    emin = float(np.min(E))
    beta = float(getattr(sampler, "beta", 1.0))
    log_Z = float(np.log(np.sum(np.exp(-beta * (E - emin)))) - beta * emin)
    return float(np.exp(-np.logaddexp(log_Z, 0.0)))


def _build_constraint_jump_sampler(
    M: np.ndarray,
    jump_matrices: list[np.ndarray],
    *,
    scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.12, 0.06),
    normalize_jumps: bool = True,
) -> QuantumGibbsSampler:
    """
    Build a :class:`QuantumGibbsSampler` whose target is ``exp(−M)/Z`` and whose
    jump (proposal) operators are the supplied SDP matrices.

    ``M`` is the ``n×n`` exponent; we pass it as the sampler Hamiltonian with
    ``beta=1`` so the target is ``exp(−M)``. The proposal strength is reduced
    across ``scales`` until the channel's Kraus rescaling succeeds.
    """
    Mc = np.asarray(M, dtype=np.complex128)
    Mc = (Mc + Mc.conj().T) / 2.0
    last_err: Exception | None = None
    for scale in scales:
        jumps = jumps_from_hermitian_matrices(
            jump_matrices, scale=float(scale), normalize=normalize_jumps
        )
        try:
            return QuantumGibbsSampler(Mc, jumps, beta=1.0, verbose=False)
        except ValueError as e:
            last_err = e
    raise RuntimeError(
        "QuantumGibbsSampler (constraint jumps) failed to initialize across all "
        "proposal scales; reduce jump scale or check the jump set."
    ) from last_err


def gibbs_state_from_hamiltonian_quantum(
    H: np.ndarray,
    n: int,
    *,
    beta: float = 1.0,
    check: bool = True,
    psd_tol: float = -1e-8,
    trace_tol: float = 1e-9,
    offdiag_tol: float = 1e-7,
    compute_full_psd_check: bool = True,
) -> tuple[np.ndarray, GibbsDiagnostics]:
    """
    Closed-form padded Gibbs state for the **full** scaled Hamiltonian
    ``H = β ∑_j y_j Ã_j`` (padded ``(n+1)`` convention, e.g. from
    :func:`hamiltonian_padded`).

    Returns the padded ρ̃ = diag((1−ω) ρ_n, ω) and its diagnostics, matching the
    paper's embedding. (Backward-compatible signature; the channel-based path is
    inside :func:`run_primal_oracle` with ``gibbs_mode="mcmc"``.)
    """
    H = np.asarray(H, dtype=DTYPE)
    if not _is_symmetric(H, 1e-9):
        raise ValueError("Hamiltonian must be symmetric.")
    # Exponent of the n-dimensional block (β is already folded into H).
    M = H[:n, :n]
    rho_n, omega = gibbs_state_n_from_exponent(M)
    rho = _padded_rho_from_n(rho_n, omega, n)

    Xp = extract_top_left(rho, n)
    om = extract_omega(rho)
    gibbs = _collect_gibbs_diagnostics(
        rho,
        n,
        psd_tol,
        trace_tol,
        offdiag_tol,
        float(np.trace(Xp)),
        om,
        compute_full_psd_check=compute_full_psd_check,
    )
    if check:
        _assert_state(rho, gibbs, trace_tol, psd_tol)
    return rho, gibbs


def gibbs_state(
    A_tilde: list[np.ndarray],
    y: np.ndarray,
    *,
    beta: float = 1.0,
    check: bool = True,
    psd_tol: float = -1e-8,
    trace_tol: float = 1e-9,
    offdiag_tol: float = 1e-7,
) -> tuple[np.ndarray, GibbsDiagnostics]:
    """
    Padded closed-form Gibbs matrix for ``H_tot = β ∑_j y_j Ã_j``.
    """
    H = hamiltonian_padded(A_tilde, y, beta=beta)
    n = A_tilde[0].shape[0] - 1
    return gibbs_state_from_hamiltonian_quantum(
        H,
        n,
        beta=beta,
        check=check,
        psd_tol=psd_tol,
        trace_tol=trace_tol,
        offdiag_tol=offdiag_tol,
        compute_full_psd_check=check,
    )


def _collect_gibbs_diagnostics(
    rho: np.ndarray,
    n: int,
    psd_tol: float,
    trace_tol: float,
    offdiag_tol: float,
    tr_Xp: float,
    omega: float,
    *,
    compute_full_psd_check: bool = True,
) -> GibbsDiagnostics:
    r = np.asarray(rho, dtype=DTYPE)
    tr = float(np.trace(r))
    sym = (r + r.T) / 2.0
    if compute_full_psd_check:
        eig = la.eigvalsh(sym)
        min_eig = float(np.min(eig))
    else:
        min_eig = float("nan")

    off = r[:n, n:]
    max_off = float(la.norm(off, ord=2))

    block_ok = max_off <= offdiag_tol
    return GibbsDiagnostics(
        trace_rho=tr,
        min_eig=min_eig,
        max_offdiag_block_norm=max_off,
        omega_vs_one_minus_tr_Xp=abs(omega - (1.0 - tr_Xp)),
        block_diagonal_ok=block_ok,
    )


def _assert_state(
    rho: np.ndarray, gibbs: GibbsDiagnostics, trace_tol: float, psd_tol: float
) -> None:
    if abs(gibbs.trace_rho - 1.0) > trace_tol:
        raise RuntimeError(f"Tr(ρ) = {gibbs.trace_rho} not close to 1.")
    if not np.isnan(gibbs.min_eig) and gibbs.min_eig < psd_tol:
        raise RuntimeError(f"ρ not PSD within tolerance: λ_min = {gibbs.min_eig}.")


def constraint_traces_from_stack(Xp: np.ndarray, A_stack: np.ndarray) -> np.ndarray:
    """
    All ``Tr(A_j X')`` at once via Frobenius inner products (**O(m n²)**).

    ``A_stack`` has shape ``(m+1, n, n)`` for unpadded ``[A_0, …, A_m]``.
    """
    Xp = np.asarray(Xp, dtype=DTYPE)
    A_stack = np.asarray(A_stack, dtype=DTYPE)
    if A_stack.ndim != 3:
        raise ValueError("A_stack must have shape (num_constraints, n, n).")
    return np.einsum("kij,ij->k", A_stack, Xp, optimize=True)


def constraint_traces(rho: np.ndarray, A_tilde: list[np.ndarray]) -> np.ndarray:
    """
    ``Tr(Ã_j ρ)`` for all j, using ``Tr(Ã_j ρ) = Tr(A_j X')`` with unpadded blocks.
    """
    n = A_tilde[0].shape[0] - 1
    Xp = extract_top_left(rho, n)
    A_stack = np.stack([np.asarray(At, dtype=DTYPE)[:n, :n] for At in A_tilde])
    return constraint_traces_from_stack(Xp, A_stack)


def check_normalized_violations(
    rho: np.ndarray,
    A_tilde: list[np.ndarray],
    b_tilde: np.ndarray,
    theta: float,
) -> ConstraintDiagnostics:
    """
    Normalized feasibility: Tr(Ã_j ρ) ≤ b̃_j + θ for all j.
    """
    t = constraint_traces(rho, A_tilde)
    return _constraint_diagnostics_from_traces(t, b_tilde, theta)


def _constraint_diagnostics_from_traces(
    traces: np.ndarray,
    b_tilde: np.ndarray,
    theta: float,
) -> ConstraintDiagnostics:
    b_tilde = np.asarray(b_tilde, dtype=float).reshape(-1)
    t = np.asarray(traces, dtype=float).reshape(-1)
    slacks = b_tilde - t
    violations = t - b_tilde
    j_max = int(np.argmax(violations)) if violations.size else 0
    max_v = float(violations[j_max]) if violations.size else 0.0
    if max_v <= theta + 1e-15:
        max_idx: int | None = None
        feasible = True
    else:
        max_idx = j_max
        feasible = False
    return ConstraintDiagnostics(
        traces=t,
        bounds_tilde=b_tilde,
        slacks=slacks,
        max_violation_index=max_idx,
        feasible_within_theta=feasible,
    )


def check_normalized_violations_fast(
    Xp: np.ndarray,
    A_stack: np.ndarray,
    b_tilde: np.ndarray,
    theta: float,
) -> ConstraintDiagnostics:
    """
    Same test as :func:`check_normalized_violations`, but takes ``X'`` and
    pre-stacked unpadded ``A_stack``.
    """
    t = constraint_traces_from_stack(Xp, A_stack)
    return _constraint_diagnostics_from_traces(t, b_tilde, theta)


# ---------------------------------------------------------------------------
# Jump-operator derivation for the MCMC Gibbs channel
# ---------------------------------------------------------------------------


def _default_jump_matrices(
    A_full: list[np.ndarray],
    *,
    drop_identity: bool = True,
    tol: float = 1e-9,
    dedupe_tol: float = 1e-7,
) -> list[np.ndarray]:
    """
    Distinct Hermitian generators to use as channel jumps, taken from the full
    oracle constraint list ``[A_0 = −C, A_1 = I, A_2, …]``.

    The ``±`` pair coming from an equality constraint (``F_i`` and ``−F_i``)
    shares one Hermitian generator, so we deduplicate up to sign and scale. A
    pure multiple of the identity is dropped by default because, after Bohr
    reweighing, it is proportional to ``I`` and provides **no mixing** (it would
    only waste a Kraus slot and proposal-strength budget).
    """
    kept: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    for A in A_full:
        A = np.asarray(A, dtype=DTYPE)
        nrm = float(la.norm(A))
        if nrm <= tol:
            continue
        sig = A / nrm
        # Canonical sign: make the largest-magnitude entry non-negative.
        idx = np.unravel_index(int(np.argmax(np.abs(sig))), sig.shape)
        if sig[idx] < 0:
            sig = -sig
        if drop_identity:
            eye_sig = np.eye(A.shape[0], dtype=DTYPE) / np.sqrt(A.shape[0])
            if float(la.norm(sig - eye_sig)) <= dedupe_tol:
                continue
        if any(float(la.norm(sig - s)) <= dedupe_tol for s in signatures):
            continue
        signatures.append(sig)
        kept.append(A)
    if not kept:
        # Degenerate fallback (e.g. only the trace constraint): keep the objective.
        kept = [np.asarray(A_full[0], dtype=DTYPE)]
    return kept


def _select_violation_index(
    violations: np.ndarray,
    theta: float,
    selection: str,
    rng: np.random.Generator | None,
) -> int | None:
    """
    Return the index of the constraint to update, or ``None`` if θ-feasible.

    ``selection="max"`` picks the largest violation (paper default). ``"random"``
    picks a uniformly random constraint among those violated by more than θ
    (used to study the effect of the selection rule).
    """
    max_v = float(np.max(violations)) if violations.size else 0.0
    if max_v <= theta + 1e-15:
        return None
    if selection == "max":
        return int(np.argmax(violations))
    if selection == "random":
        violated = np.flatnonzero(violations > theta + 1e-15)
        if violated.size == 0:
            return int(np.argmax(violations))
        gen = rng if rng is not None else np.random.default_rng()
        return int(violated[gen.integers(violated.size)])
    raise ValueError(f"Unknown violation selection '{selection}'.")


# ---------------------------------------------------------------------------
# Main iteration (Section 2.2, PDF pp. 11–12)
# ---------------------------------------------------------------------------


def run_primal_oracle(
    problem: PrimalOracleProblem,
    epsilon: float,
    *,
    beta: float = 1.0,
    initial_y: np.ndarray | None = None,
    max_iterations: int | None = None,
    skip_gibbs_asserts: bool = False,
    show_progress: bool = False,
    collect_timing: bool = False,
    gibbs_mode: str = "exact",
    gibbs_jump_matrices: list[np.ndarray] | None = None,
    gibbs_max_steps: int = 5000,
    gibbs_step_cutoffs: list[int] | None = None,
    gibbs_target_theta: float | None = None,
    gibbs_warm_start: bool = False,
    normalize_jumps: bool = True,
    violation_selection: str = "max",
    violation_rng_seed: int | None = None,
    return_on_exhaustion: bool = False,
) -> PrimalOracleResult | None:
    """
    Primal-oracle feasibility loop at threshold ``g`` (Section 2.2).

    The ``n``-dimensional Gibbs state ``ρ_n ∝ exp(−M)`` (``M = β ∑_j y_j A_j``)
    is prepared either in closed form (``gibbs_mode="exact"``) or by iterating the
    discrete-time quantum Gibbs channel (``gibbs_mode="mcmc"``). The primal block
    is recovered as ``X' = (1 − ω) ρ_n`` with ``ω = 1/(Z_n + 1)``.

    Parameters
    ----------
    gibbs_mode :
        ``"exact"`` (closed-form eigendecomposition; cheap, for calibration) or
        ``"mcmc"`` (channel iteration; records convergence step counts).
    gibbs_jump_matrices :
        MCMC jump (proposal) operators (``n×n`` Hermitian). If ``None``, derived
        from the SDP constraints + objective via :func:`_default_jump_matrices`.
    gibbs_max_steps :
        Cap on channel applications per oracle iteration (convergence guard).
    gibbs_step_cutoffs :
        If given (MCMC only), also record the early-termination trace distance
        ``D(σ_cutoff, σ_full)`` at each of these step counts and each oracle
        iteration (for the cutoff experiment).
    gibbs_target_theta :
        Trace-distance target for MCMC Gibbs convergence. If ``None`` (default),
        use the oracle feasibility tolerance ``θ = ε/(2R)``. Set explicitly
        (e.g. a fixed ``0.01``) to decouple channel mixing difficulty from ``R``.
        Feasibility updates still use ``θ = ε/(2R)`` regardless.
    gibbs_warm_start :
        If True (MCMC only), use the previous iteration's channel endpoint as
        ``σ_0`` for the next Gibbs preparation instead of restarting from ``I/n``.
    return_on_exhaustion :
        If True, when the iteration cap is hit without θ-feasibility, still return
        the last :class:`PrimalOracleResult` (with ``feasible_within_theta=False``)
        so the per-iteration logs are preserved. Default False returns ``None``
        (keeps the binary-search solver's infeasible contract).
    violation_selection :
        ``"max"`` (largest violation, default) or ``"random"`` (uniform over
        violated constraints; seeded by ``violation_rng_seed``).
    collect_timing :
        Attach a ``timing`` dict with a ``perf_counter`` breakdown.

    Notes
    -----
    Update rule on a violated constraint j: ``y ← y + θ e_j``, ``θ = ε/(2R)``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if gibbs_mode not in ("exact", "mcmc"):
        raise ValueError("gibbs_mode must be 'exact' or 'mcmc'.")
    if beta != 1.0:
        warnings.warn(
            "beta != 1: Section 2.2 uses inverse temperature 1 in exp(-H). "
            "Non-unit beta scales H and changes the Gibbs state; keep beta=1 "
            "unless intentionally experimenting.",
            UserWarning,
            stacklevel=2,
        )

    wall_t0 = time.perf_counter()
    timing: dict[str, float] | None = (
        {
            "total_wall_time": 0.0,
            # gibbs_time is the sum of the three sub-buckets (kept for the solver).
            "gibbs_time": 0.0,
            "gibbs_construction_time": 0.0,
            "gibbs_channel_iteration_time": 0.0,
            "gibbs_convergence_check_time": 0.0,
            "trace_check_time": 0.0,
            "violation_logic_time": 0.0,
            "hamiltonian_update_time": 0.0,
            "result_packaging_time": 0.0,
        }
        if collect_timing
        else None
    )

    A_full, b_full = build_objective_and_full_constraints(problem)
    b_tilde = normalized_bounds(b_full, problem.R)
    n = problem.C.shape[0]

    # Unpadded stack [A_0 = −C, A_1 = I, …] and the n-dimensional exponent M.
    A_stack = np.ascontiguousarray(np.stack(A_full, axis=0), dtype=DTYPE)
    m1 = len(A_full)

    theta = epsilon / (2.0 * problem.R)
    gibbs_theta = float(gibbs_target_theta) if gibbs_target_theta is not None else theta
    if gibbs_theta <= 0:
        raise ValueError("gibbs_target_theta must be positive.")
    ln_n = np.log(max(n, 1))
    default_iters = int(np.ceil(ln_n / (theta**2)))
    iters = default_iters if max_iterations is None else int(max_iterations)

    if max_iterations is not None and iters < default_iters:
        warnings.warn(
            f"max_iterations={iters} is below the paper schedule "
            f"ceil(ln(n)/theta^2)={default_iters} (n={n}, theta={theta:.3g}). "
            f"The loop may end without theta-feasibility even when g is feasible "
            f"(false 'infeasible'). Use max_iterations=None for the full schedule.",
            UserWarning,
            stacklevel=2,
        )

    if initial_y is None:
        y = np.zeros(m1, dtype=float)
    else:
        y = np.asarray(initial_y, dtype=float).reshape(-1)
        if y.shape[0] != m1:
            raise ValueError(f"initial_y must have length {m1}.")
        if np.any(y < -1e-12):
            warnings.warn("initial_y has negative entries; clipping to 0.", UserWarning)
            y = np.clip(y, 0.0, None)

    check = not skip_gibbs_asserts

    # n-dimensional exponent, maintained by rank-one updates M += (β θ) A_j.
    M = beta * np.einsum("k,kij->ij", y, A_stack, optimize=True)
    M = np.ascontiguousarray((M + M.T) / 2.0, dtype=DTYPE)

    # MCMC setup: jump operators and per-iteration logs.
    use_mcmc = gibbs_mode == "mcmc"
    jump_matrices: list[np.ndarray] | None = None
    if use_mcmc:
        jump_matrices = (
            list(gibbs_jump_matrices)
            if gibbs_jump_matrices is not None
            else _default_jump_matrices(A_full)
        )
    cutoffs = sorted({int(c) for c in gibbs_step_cutoffs}) if (use_mcmc and gibbs_step_cutoffs) else []
    sigma0 = np.eye(n, dtype=np.complex128) / float(n)  # maximally mixed (or warm-started)
    v_rng = np.random.default_rng(violation_rng_seed)

    gibbs_steps_per_iter: list[int] = []
    gibbs_converged_per_iter: list[bool] = []
    cutoff_td_per_iter: dict[int, list[float]] = {c: [] for c in cutoffs}

    def _package(
        rho_n: np.ndarray,
        omega: float,
        Xp: np.ndarray,
        diag: ConstraintDiagnostics,
        iters_done: int,
    ) -> PrimalOracleResult:
        rho = _padded_rho_from_n(rho_n, omega, n)
        gibbs = _collect_gibbs_diagnostics(
            rho, n, problem.psd_check_tol, problem.trace_check_tol,
            problem.offdiag_check_tol, float(np.trace(Xp)), omega,
            compute_full_psd_check=check,
        )
        Xp_out = np.ascontiguousarray(Xp, dtype=DTYPE)
        if timing is not None:
            timing["gibbs_time"] = (
                timing["gibbs_construction_time"]
                + timing["gibbs_channel_iteration_time"]
                + timing["gibbs_convergence_check_time"]
            )
            timing["total_wall_time"] = time.perf_counter() - wall_t0
        return PrimalOracleResult(
            y=y.copy(),
            rho=rho,
            X_prime=Xp_out,
            omega=float(omega),
            X=problem.R * Xp_out,
            z=problem.R * float(np.trace(Xp_out)),
            constraint_diag=diag,
            gibbs_diag=gibbs,
            iterations=iters_done,
            theta=theta,
            timing=timing,
            gibbs_mode=gibbs_mode,
            gibbs_steps_per_iter=gibbs_steps_per_iter,
            gibbs_converged_per_iter=gibbs_converged_per_iter,
            cutoff_trace_distance_per_iter=(
                {c: cutoff_td_per_iter[c] for c in cutoffs} if cutoffs else None
            ),
        )

    last_state: tuple[np.ndarray, float, np.ndarray, ConstraintDiagnostics] | None = None

    it_range = (
        tqdm(
            range(iters),
            desc=f"Primal oracle ({gibbs_mode} Gibbs, cube)",
            unit="iter",
            leave=True,
        )
        if show_progress
        else range(iters)
    )

    for it in it_range:
        # --- Gibbs state preparation --------------------------------------
        if use_mcmc:
            tc0 = time.perf_counter()
            sampler = _build_constraint_jump_sampler(
                M, jump_matrices, normalize_jumps=normalize_jumps
            )
            if timing is not None:
                timing["gibbs_construction_time"] += time.perf_counter() - tc0

            conv = sampler.run_until_converged(
                sigma0,
                target_trace_distance=gibbs_theta,
                max_steps=gibbs_max_steps,
                step_cutoffs=cutoffs or None,
            )
            if timing is not None:
                timing["gibbs_channel_iteration_time"] += conv.timing["channel_apply_time"]
                timing["gibbs_convergence_check_time"] += conv.timing["convergence_check_time"]

            rho_n = np.asarray(np.real(conv.converged_state), dtype=DTYPE)
            rho_n = (rho_n + rho_n.T) / 2.0
            omega = _omega_from_sampler(sampler)

            if gibbs_warm_start:
                sigma0 = np.asarray(conv.converged_state, dtype=np.complex128).copy()
                tr0 = float(np.trace(sigma0).real)
                if tr0 > 0.0:
                    sigma0 = sigma0 / tr0

            gibbs_steps_per_iter.append(int(conv.steps_to_converge))
            gibbs_converged_per_iter.append(bool(conv.converged))
            for c in cutoffs:
                td = trace_distance(conv.cutoff_states[c], conv.converged_state)
                cutoff_td_per_iter[c].append(float(td))
        else:
            tc0 = time.perf_counter()
            rho_n, omega = gibbs_state_n_from_exponent(M)
            if timing is not None:
                timing["gibbs_construction_time"] += time.perf_counter() - tc0
            gibbs_steps_per_iter.append(0)
            gibbs_converged_per_iter.append(True)

        Xp = (1.0 - omega) * rho_n

        # --- Feasibility test ---------------------------------------------
        tt0 = time.perf_counter()
        traces = constraint_traces_from_stack(Xp, A_stack)
        if timing is not None:
            timing["trace_check_time"] += time.perf_counter() - tt0

        tv0 = time.perf_counter()
        violations = traces - b_tilde
        diag = _constraint_diagnostics_from_traces(traces, b_tilde, theta)
        j = _select_violation_index(violations, theta, violation_selection, v_rng)
        if timing is not None:
            timing["violation_logic_time"] += time.perf_counter() - tv0

        last_state = (rho_n, omega, Xp, diag)

        if j is None:
            tp0 = time.perf_counter()
            result = _package(rho_n, omega, Xp, diag, it + 1)
            if timing is not None:
                timing["result_packaging_time"] += time.perf_counter() - tp0
                timing["total_wall_time"] = time.perf_counter() - wall_t0
            return result

        # --- Coordinate update y ← y + θ e_j and rank-one M update --------
        y[j] += theta
        th0 = time.perf_counter()
        M += (beta * theta) * A_stack[j]
        if timing is not None:
            timing["hamiltonian_update_time"] += time.perf_counter() - th0

    if timing is not None:
        timing["gibbs_time"] = (
            timing["gibbs_construction_time"]
            + timing["gibbs_channel_iteration_time"]
            + timing["gibbs_convergence_check_time"]
        )
        timing["total_wall_time"] = time.perf_counter() - wall_t0

    # Iteration cap hit without θ-feasibility.
    if return_on_exhaustion and last_state is not None:
        return _package(last_state[0], last_state[1], last_state[2], last_state[3], iters)
    return None


__all__ = [
    "DTYPE",
    "PrimalOracleProblem",
    "ConstraintDiagnostics",
    "GibbsDiagnostics",
    "PrimalOracleResult",
    "build_objective_and_full_constraints",
    "normalized_bounds",
    "embed_top_left",
    "extract_top_left",
    "extract_omega",
    "hamiltonian_padded",
    "gibbs_state",
    "gibbs_state_from_hamiltonian_quantum",
    "gibbs_state_n_from_exponent",
    "constraint_traces",
    "constraint_traces_from_stack",
    "check_normalized_violations",
    "check_normalized_violations_fast",
    "run_primal_oracle",
]
