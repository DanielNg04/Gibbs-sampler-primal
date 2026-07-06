"""
Classical dense-matrix simulator for the discrete-time quantum Metropolis channel
(Gilyén et al., arXiv:2405.20322): coherent accept + single Kraus reject (Thm. 1,
Eq. (5)) with coherent Bohr reweighing (Sec. 1.4). Target ρ = exp(−βH)/Tr exp(−βH).

v3 — real-arithmetic rewrite of v2, built on three design decisions:

1.  **Everything is real.** H and the jump operators are assumed real symmetric,
    so the eigenvectors of H are real orthogonal, the Bohr weights are real, and
    the Gibbs state, every Kraus operator and every iterated state are real.
    All storage is float64: a real GEMM costs 1/4 the floating-point operations
    and 1/2 the memory traffic of the complex128 arithmetic used in v1/v2.

2.  **One diagonalization for H, one for the reject square root — nothing else.**
    H is diagonalized once (LAPACK ``dsyevd``, divide & conquer: backward stable
    and the fastest full-spectrum symmetric driver for large matrices). Every
    other spectral object — exp(−βH), ρ, ρ^{1/2}, ρ^{−1/2}, the Bohr-frequency
    weights — is derived from that single (E, U) pair by O(d²) element-wise work
    in the energy eigenbasis, where ρ is diagonal. There is no Padé/expm matrix
    exponential and no re-diagonalization of ρ anywhere.

3.  **The channel is fully vectorized.** All Kraus operators (accept + reject)
    live in one contiguous (q+1, d, d) stack. One channel application is a
    batched GEMM plus a single large GEMM (``tensordot``), both executed by
    multithreaded BLAS. There is no Python-level per-Kraus loop in the hot path.

Parallelism comes from the BLAS library (OpenBLAS/MKL) that backs every GEMM and
``eigh`` call; set BLAS thread counts externally on shared hosts — this module
does not pin threads.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import scipy.linalg as la

FDTYPE = np.float64


# --- Small linear-algebra utilities ------------------------------------------------

def _symmetrize(M: np.ndarray) -> np.ndarray:
    """
    Project onto the symmetric matrices: (M + Mᵀ)/2.

    Stability: floating-point GEMMs leave O(machine-eps) asymmetry; projecting it
    out keeps every state/operator exactly symmetric so that all downstream
    ``eigh``/``eigvalsh`` calls (which read only one triangle) are valid.
    Speed: O(d²) — negligible next to any O(d³) GEMM it follows.
    """
    return (M + M.T) * 0.5


def _as_real_symmetric(M: np.ndarray, *, atol: float, name: str) -> np.ndarray:
    """
    Validate and coerce an input matrix to a contiguous float64 symmetric array.

    This is the single boundary where inputs are checked: complex inputs are
    accepted only if their imaginary part is ≤ atol (then discarded), and the
    matrix must be square and symmetric within atol. Returning the symmetrized
    copy means no later routine ever needs to re-validate or re-symmetrize.
    """
    A = np.asarray(M)
    if np.iscomplexobj(A):
        if A.size and float(np.max(np.abs(A.imag))) > atol:
            raise ValueError(f"{name} has a non-negligible imaginary part; v3 is real-only.")
        A = A.real
    A = np.ascontiguousarray(A, dtype=FDTYPE)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {A.shape}.")
    if not np.allclose(A, A.T, atol=atol, rtol=0.0):
        raise ValueError(f"{name} must be symmetric within atol={atol}.")
    return _symmetrize(A)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Trace distance D(ρ, σ) = ½‖ρ − σ‖₁ for symmetric matrices.

    Stability: the difference is symmetrized and passed to ``eigvalsh``.
    Eigenvalues of a symmetric matrix are perfectly conditioned (Weyl's
    inequality: a perturbation of norm ε moves every eigenvalue by ≤ ε), so the
    computed trace norm carries no amplified error.
    Speed: one O(d³) LAPACK eigenvalue call — the trace norm genuinely needs the
    spectrum, so this is minimal; ``eigvalsh`` skips the eigenvector work that a
    full ``eigh`` would do.
    """
    d = _symmetrize(np.asarray(rho, dtype=FDTYPE) - np.asarray(sigma, dtype=FDTYPE))
    return float(0.5 * np.sum(np.abs(la.eigvalsh(d, check_finite=False))))


def frobenius_norm(M: np.ndarray) -> float:
    """‖M‖_F via ``np.linalg.norm``: a single fused BLAS ``nrm2``-style pass, O(d²)."""
    return float(np.linalg.norm(M, ord="fro"))


def expectation_value(sigma: np.ndarray, O: np.ndarray) -> float:
    """
    ⟨O⟩ = Tr(σ O) as a Frobenius inner product.

    Speed: ``einsum("ij,ji->")`` contracts directly in O(d²) without forming the
    O(d³) product σ @ O.
    Stability: a plain sum of d² products; no cancellation issues beyond those
    inherent in the data.
    """
    return float(np.einsum("ij,ji->", sigma, O, optimize=True))


# --- Bohr / frequency weighting ( Eq. (20), Eq. (33) ) ----------------------------

# A weight function must map the (d, d) Bohr-frequency matrix element-wise as one
# vectorized array operation (no scalar callables — a Python loop over d² entries
# was a measured hot spot in v1).
WeightFn = Callable[[np.ndarray], np.ndarray]


def coherent_S_minus_weight(nu: np.ndarray) -> np.ndarray:
    """
    Default coherent-accept weight f(ν) = ½(1 − tanh(ν/4))  (Sec. 1.4, S₋ weighting).

    Stability: ``tanh`` saturates smoothly to ±1, so f maps every real ν into
    [0, 1] with no overflow — unlike algebraically equivalent forms written with
    ``exp``, which overflow for large |ν| (deep spectra or large β).
    Speed: one vectorized ufunc pass over the d×d frequency matrix, O(d²).
    """
    return 0.5 * (1.0 - np.tanh(0.25 * nu))


def bohr_weight_matrix(energies: np.ndarray, beta: float, weight_fn: WeightFn) -> np.ndarray:
    """
    W_ij = f(β(E_i − E_j)) for all energy pairs at once.

    Speed: the frequency matrix ν is built by a broadcasted subtraction (O(d²),
    no Python loop) and ``weight_fn`` is applied to the whole array in one call.
    Stability: ν is exact up to one rounding per entry; all conditioning issues
    are delegated to the weight function (see :func:`coherent_S_minus_weight`).
    """
    E = np.asarray(energies, dtype=FDTYPE).reshape(-1, 1)
    nu = beta * (E - E.T)
    W = np.asarray(weight_fn(nu), dtype=FDTYPE)
    if W.shape != nu.shape:
        raise ValueError("weight_fn must act element-wise on the (d, d) frequency matrix.")
    return W


# --- Proposal jump factories -------------------------------------------------------


def jumps_from_symmetric_matrices(
    mats: Sequence[np.ndarray],
    *,
    scale: float = 1.0,
    normalize: bool = True,
    drop_zero: bool = True,
    zero_tol: float = 1e-12,
) -> List[np.ndarray]:
    """
    Build channel jump operators directly from problem matrices (SDP oracle entry
    point: the constraint matrices Ã_j plus the objective −C̃ become the proposals
    that drive the Metropolis dynamics toward ρ ∝ exp(−βH)).

    ``normalize`` divides each matrix by its Frobenius norm so constraints of very
    different magnitudes contribute comparable proposal weight (better-conditioned
    channel); ``drop_zero`` skips matrices that would be no-op Kraus slots. The
    sampler later rescales the whole set so λ_max(Σ ÃᵀÃ) ≤ 1 − margin, so
    ``scale`` only sets relative strength before that safety rescaling.

    Speed/stability: per matrix this is one O(d²) norm and one scaling — there is
    nothing to optimize further; the Frobenius norm is computed by a stable fused
    BLAS reduction.
    """
    out: List[np.ndarray] = []
    for k, M in enumerate(mats):
        A = _as_real_symmetric(M, atol=1e-10, name=f"mats[{k}]")
        nrm = float(np.linalg.norm(A))
        if drop_zero and nrm <= zero_tol:
            continue
        if normalize and nrm > zero_tol:
            A = A / nrm
        out.append(np.ascontiguousarray(float(scale) * A))
    if not out:
        raise ValueError("No usable (non-zero) jump matrices were provided.")
    return out


# Drop-in alias for callers written against the v2 (complex) module name.
jumps_from_hermitian_matrices = jumps_from_symmetric_matrices


def random_symmetric_jumps(
    dim: int,
    num_jumps: int,
    *,
    rng: Optional[np.random.Generator] = None,
    scale: float = 0.3,
) -> List[np.ndarray]:
    """
    Random real symmetric (GOE-like) proposals, mainly for tests and demos.

    Speed: all ``num_jumps`` Gaussian matrices are drawn in one batched RNG call
    and symmetrized with one batched transpose-add — no per-jump Python loop.
    """
    rng = rng or np.random.default_rng()
    G = rng.standard_normal((num_jumps, dim, dim))
    return list(0.5 * scale * (G + np.swapaxes(G, 1, 2)))


# --- Result containers ---------------------------------------------------------------


@dataclass
class TrajectoryResult:
    trace_dist: List[float]
    frobenius_dist: List[float]
    expectations: List[Optional[List[float]]]
    states: Optional[List[np.ndarray]] = None


@dataclass
class ConvergenceResult:
    """Outcome of :meth:`QuantumGibbsSampler.run_until_converged`."""

    steps_to_converge: int
    """Channel applications until ``trace_distance(σ, ρ) ≤ target`` (or ``max_steps``)."""

    converged: bool
    """True if the target trace distance was reached within ``max_steps``."""

    final_trace_distance: float
    """Trace distance to ρ at the stopping step."""

    converged_state: np.ndarray
    """The state σ reached at the stopping step (≈ ρ when ``converged``)."""

    cutoff_states: dict[int, np.ndarray]
    """Snapshots σ_{min(c, steps_to_converge)} for each requested cutoff ``c``."""

    timing: dict[str, float]
    """``channel_apply_time``, ``convergence_check_time``, ``total_run_time``, metadata."""


# --- Main sampler ------------------------------------------------------------------


class QuantumGibbsSampler:
    
    """
    Discrete-time quantum detailed-balanced channel (Theorem 1 + coherent accept),
    specialized to real symmetric H and jumps.

    Construction cost: exactly two O(d³) eigendecompositions (H once; the reject
    square root once) plus a handful of GEMMs — versus five eigendecompositions
    in v2 (H, ρ for its condition number, ρ^{1/2}, ρ^{−1/2}, and the inner square
    root). Everything in between is O(q·d²) element-wise work in the energy basis.
    """

    def __init__(
        self,
        H: np.ndarray,
        jumps: Sequence[np.ndarray],
        *,
        beta: float = 1.0,
        atol_sym: float = 1e-10,
        psd_eps: float = 1e-12,
        trace_tol: float = 1e-8,
        rescale_margin: float = 1e-6,
        weight_fn: WeightFn = coherent_S_minus_weight,
        symmetrize_output: bool = True,
        verbose: bool = False,
    ) -> None:
        if beta <= 0.0:
            raise ValueError("beta must be positive.")
        if rescale_margin < 0.0:
            raise ValueError("rescale_margin must be non-negative.")

        self.beta = float(beta)
        self.atol_sym = float(atol_sym)
        self.psd_eps = float(psd_eps)
        self.trace_tol = float(trace_tol)
        self.symmetrize_output = bool(symmetrize_output)
        self.verbose = bool(verbose)

        # ---- Validate inputs once at the boundary (everything after is trusted).
        self.H = _as_real_symmetric(H, atol=self.atol_sym, name="H")
        d = self.H.shape[0]
        self.dim = d

        jump_list = [
            _as_real_symmetric(A, atol=self.atol_sym, name=f"jumps[{a}]")
            for a, A in enumerate(jumps)
        ]
        if not jump_list:
            raise ValueError("Provide at least one jump operator.")
        for a, A in enumerate(jump_list):
            if A.shape != (d, d):
                raise ValueError(f"jumps[{a}] has shape {A.shape}, expected ({d}, {d}).")
        q = len(jump_list)
        self.num_jumps = q

        # ---- 1. The single diagonalization of H.
        # driver="evd" (LAPACK dsyevd, divide & conquer) is backward stable and the
        # fastest full-spectrum symmetric driver for large dense matrices.
        # check_finite=False skips an O(d²) scan; H was already validated above.
        self.energies, self.eigenvectors = la.eigh(self.H, driver="evd", check_finite=False)
        E, U = self.energies, self.eigenvectors

        # ---- 2. Gibbs state from the spectrum (no expm, no second eigh).
        # Shifted Boltzmann weights w_i = exp(−β(E_i − E_min)) are the softmax
        # trick: the common factor exp(β E_min) cancels in ρ, the largest weight
        # is exactly 1, so there is no overflow and no catastrophic underflow of
        # the *relative* weights — only states that are physically negligible
        # (ratio < 1e−308) flush to zero.
        emin = float(E[0])  # eigh returns ascending eigenvalues
        w = np.exp(-self.beta * (E - emin))
        Z_scaled = float(np.sum(w))
        if not (Z_scaled > 0.0 and math.isfinite(Z_scaled)):
            raise ValueError("Scaled partition function invalid.")
        probs = w / Z_scaled
        self.gibbs_probs = probs
        self.partition_function_scaled = Z_scaled
        # log Z is always representable; Z itself may overflow to inf for very
        # negative E_min — keep the log as the stable quantity of record.
        self.log_partition_function = math.log(Z_scaled) - self.beta * emin
        self.partition_function = float(np.exp(self.log_partition_function))

        # ρ = U diag(p) Uᵀ via column scaling (O(d²)) + one GEMM. Its spectrum is
        # `probs` by construction, so eigenvalue bounds are free — v2 spent an
        # extra O(d³) eigvalsh(ρ) to recover what we already know exactly.
        self.rho = _symmetrize((U * probs) @ U.T)
        self.rho_min_eig = float(np.min(probs))
        self.rho_max_eig = float(np.max(probs))
        self.rho_condition_number = self.rho_max_eig / max(self.rho_min_eig, self.psd_eps)

        # ---- 3. Coherent Bohr reweighing, batched in the energy basis.
        # Rotate all q jumps with two batched GEMMs (Uᵀ A_a U for every a at
        # once), then apply the weight matrix element-wise to the whole stack —
        # O(q·d²) after the rotations, no per-jump Python work.
        stack_E = U.T @ np.stack(jump_list) @ U
        W = bohr_weight_matrix(E, self.beta, weight_fn)
        accept_E = W[None, :, :] * stack_E

        # ---- 4. Safety rescaling so that D = Σ ÃᵀÃ satisfies λ_max(D) ≤ 1 − margin.
        # tensordot contracts D_jl = Σ_{a,i} Ã[a,i,j] Ã[a,i,l] as ONE large GEMM
        # of shape (d, q·d)·(q·d, d); as a Gram matrix the result is symmetric
        # PSD by construction (stable — no cancellation can make it indefinite
        # beyond rounding). Rescaling by c multiplies D by c² exactly, so D and
        # its eigenvalues are updated in place instead of being recomputed.
        D_E = _symmetrize(np.tensordot(accept_E, accept_E, axes=([0, 1], [0, 1])))
        ev_D = la.eigvalsh(D_E, check_finite=False)
        lam_max = float(ev_D[-1])
        target = max(0.0, 1.0 - float(rescale_margin))
        if lam_max > target:
            c = math.sqrt(target / lam_max)
            accept_E *= c
            D_E *= c * c
            lam_max *= c * c
            self.jump_scale = c
        else:
            self.jump_scale = 1.0
        if lam_max > 1.0 + 1e-7:
            raise ValueError(
                "T'†[I] has eigenvalues > 1 after rescaling; "
                "decrease proposal strength or margin."
            )

        # ---- 5. Reject Kraus K = sqrt(ρ^{1/2}(I−D)ρ^{1/2}) ρ^{−1/2}  (Thm. 1, Eq. (5)),
        # assembled in the energy basis where ρ = diag(p):
        #   ρ^{1/2}(I−D)ρ^{1/2}  →  M_ij = √p_i √p_j (I−D)_ij   (O(d²) scaling),
        #   ρ^{−1/2}             →  column scaling by 1/√p_j     (O(d²)).
        # Only the outer square root needs an eigendecomposition — the second and
        # last O(d³) eigh of the whole construction. Stability details:
        #  * √p_i·√p_j (precomputed square roots) cannot underflow where p_i·p_j
        #    would, and is exact to one rounding each.
        #  * 1/√p is clamped at psd_eps so a flushed-to-zero Gibbs weight cannot
        #    produce inf; mathematically ‖K‖ ≤ 1 keeps the result bounded.
        #  * Eigenvalues of M below zero are pure rounding noise (M is PSD by
        #    Theorem 1 once λ_max(D) ≤ 1); they are clipped at 0 before the
        #    square root, and anything below −1e−8 means a genuinely broken
        #    channel, which is reported instead of silently repaired.
        sqrt_p = np.sqrt(probs)
        inv_sqrt_p = 1.0 / np.sqrt(np.maximum(probs, self.psd_eps))
        M = _symmetrize(sqrt_p[:, None] * (np.eye(d) - D_E) * sqrt_p[None, :])
        w_M, v_M = la.eigh(M, driver="evd", check_finite=False)
        if float(w_M[0]) < -1e-8:
            raise RuntimeError(
                f"ρ^(1/2)(I−D)ρ^(1/2) is not PSD within tolerance: min eig = {float(w_M[0])}"
            )
        np.maximum(w_M, 0.0, out=w_M)
        sqrt_M = (v_M * np.sqrt(w_M)) @ v_M.T
        K_E = sqrt_M * inv_sqrt_p[None, :]

        # ---- 6. One contiguous Kraus stack (accept ops + reject op), rotated back
        # to the input basis with two batched GEMMs. A single C-contiguous block
        # lets _apply_channel_fast run the whole channel as batched BLAS calls.
        kraus_E = np.concatenate([accept_E, K_E[None, :, :]], axis=0)
        self.kraus_stack = np.ascontiguousarray(U @ kraus_E @ U.T, dtype=FDTYPE)
        self.accept_kraus = self.kraus_stack[:q]
        self.reject_kraus = self.kraus_stack[q]
        self.D_accept = _symmetrize(U @ D_E @ U.T)

        if self.verbose:
            print(
                "[QuantumGibbsSampler v3] dim=", d,
                "beta=", self.beta,
                "num_jumps=", q,
                "max_eig(D)=", lam_max,
                "jump_scale=", self.jump_scale,
            )

    # --- Channel application ---------------------------------------------------------

    def _apply_channel_fast(
        self,
        sigma: np.ndarray,
        *,
        symmetrize_output: Optional[bool] = None,
    ) -> np.ndarray:
        """
        M[σ] = Σ_a A_a σ A_aᵀ + K σ Kᵀ — the hot loop, as two BLAS-bound steps:

        1. ``tmp = kraus_stack @ σ``: a batched GEMM, (q+1) independent d×d
           multiplies dispatched straight to multithreaded dgemm.
        2. ``tensordot(tmp, kraus_stack, axes=([0, 2], [0, 2]))``: contracts both
           the Kraus index and the inner matrix index in ONE large GEMM of shape
           (d, (q+1)d)·((q+1)d, d). out[i,k] = Σ_{a,j} (A_a σ)[i,j]·A_a[k,j],
           i.e. exactly Σ_a A_a σ A_aᵀ, with the accumulation over Kraus
           operators fused into the GEMM instead of q Python-level adds (v2).

        Stability: every product is a plain GEMM (componentwise backward stable);
        the final symmetrization removes the O(eps) asymmetry so iterated states
        stay in the symmetric cone and downstream ``eigvalsh`` calls stay exact.
        """
        if symmetrize_output is None:
            symmetrize_output = self.symmetrize_output
        sigma = np.ascontiguousarray(sigma, dtype=FDTYPE)
        tmp = self.kraus_stack @ sigma
        out = np.tensordot(tmp, self.kraus_stack, axes=([0, 2], [0, 2]))
        return _symmetrize(out) if symmetrize_output else out

    def apply_channel(self, sigma: np.ndarray, *, check: bool = False) -> np.ndarray:
        """
        Validated single application of the CPTP map M (use ``_apply_channel_fast``
        inside loops — the validation here costs an extra O(d²), and the optional
        ``check`` adds an O(d³) eigenvalue scan).
        """
        sigma = _as_real_symmetric(sigma, atol=self.atol_sym * 10, name="sigma")
        if sigma.shape != (self.dim, self.dim):
            raise ValueError("sigma has wrong shape.")
        out = self._apply_channel_fast(sigma)
        if check:
            tr_in, tr_out = float(np.trace(sigma)), float(np.trace(out))
            if abs(tr_out - tr_in) > self.trace_tol:
                raise RuntimeError(
                    f"apply_channel: trace(out)={tr_out} deviates from trace(in)={tr_in}"
                )
            min_eig = float(la.eigvalsh(out, check_finite=False)[0])
            if min_eig < -1e-7:
                raise RuntimeError(f"apply_channel: min eigenvalue {min_eig}")
        return out

    # --- Trajectory drivers ------------------------------------------------------------

    def _coerce_state(self, sigma0: np.ndarray) -> np.ndarray:
        """Real-coerce, symmetrize and trace-normalize an initial state (boundary only)."""
        sigma = np.asarray(sigma0)
        if np.iscomplexobj(sigma):
            sigma = sigma.real
        sigma = _symmetrize(np.ascontiguousarray(sigma, dtype=FDTYPE))
        tr = float(np.trace(sigma))
        if abs(tr - 1.0) > 1e-6 and tr > 0.0:
            sigma = sigma / tr
        return sigma

    def run(
        self,
        sigma0: np.ndarray,
        steps: int,
        *,
        observables: Optional[Sequence[np.ndarray]] = None,
        store_states: bool = False,
    ) -> TrajectoryResult:
        """
        Iterate M for ``steps`` steps from ``sigma0``, recording distances to ρ.

        .. warning::
            Both distances are evaluated **every step** and the trace distance
            costs an O(d³) eigenvalue call each time. For timing studies use
            :meth:`run_until_converged` or iterate ``_apply_channel_fast`` directly.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        sigma = self._coerce_state(sigma0)
        obs_list = list(observables) if observables is not None else None

        tdist: List[float] = []
        fdist: List[float] = []
        exps: List[Optional[List[float]]] = []
        states: Optional[List[np.ndarray]] = [] if store_states else None

        for _ in range(steps + 1):
            if states is not None:
                states.append(sigma.copy())
            tdist.append(trace_distance(sigma, self.rho))
            fdist.append(frobenius_norm(sigma - self.rho))
            exps.append(
                None if obs_list is None
                else [expectation_value(sigma, O) for O in obs_list]
            )
            if len(tdist) <= steps:
                sigma = self._apply_channel_fast(sigma)

        return TrajectoryResult(
            trace_dist=tdist, frobenius_dist=fdist, expectations=exps, states=states
        )

    def run_until_converged(
        self,
        sigma0: np.ndarray,
        *,
        target_trace_distance: float,
        max_steps: int,
        step_cutoffs: Optional[Sequence[int]] = None,
        symmetrize_each_step: Optional[bool] = None,
    ) -> ConvergenceResult:
        """
        Iterate M from ``sigma0`` until ``D(σ_k, ρ) ≤ target_trace_distance``.

        Per step: one vectorized channel application (see
        :meth:`_apply_channel_fast`), a trace renormalization that prevents slow
        multiplicative drift of Tr σ over thousands of steps (stability), and one
        trace-distance check (the unavoidable O(d³) convergence criterion, timed
        separately so callers can see its share of the runtime).
        """
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if target_trace_distance <= 0:
            raise ValueError("target_trace_distance must be positive.")

        sym = self.symmetrize_output if symmetrize_each_step is None else symmetrize_each_step
        cutoffs = sorted({int(c) for c in step_cutoffs}) if step_cutoffs else []
        if any(c < 0 for c in cutoffs):
            raise ValueError("step_cutoffs must be non-negative.")
        cutoff_states: dict[int, np.ndarray] = {}

        sigma = self._coerce_state(sigma0)

        t_apply = 0.0
        t_check = 0.0
        t_run0 = time.perf_counter()

        tc0 = time.perf_counter()
        dist = trace_distance(sigma, self.rho)
        t_check += time.perf_counter() - tc0

        steps_to_converge = 0
        converged = dist <= target_trace_distance
        final_distance = dist
        if 0 in cutoffs:
            cutoff_states[0] = sigma.copy()

        if not converged:
            for step in range(1, int(max_steps) + 1):
                ta0 = time.perf_counter()
                sigma = self._apply_channel_fast(sigma, symmetrize_output=sym)
                tr = float(np.trace(sigma))
                if not math.isfinite(tr) or tr <= 0.0:
                    raise RuntimeError(
                        "Channel iteration produced a non-positive/inf trace; "
                        "Hamiltonian likely too stiff for the explicit reject Kraus."
                    )
                if abs(tr - 1.0) > 1e-12:
                    sigma = sigma / tr
                t_apply += time.perf_counter() - ta0

                if step in cutoffs:
                    cutoff_states[step] = sigma.copy()

                tc0 = time.perf_counter()
                dist = trace_distance(sigma, self.rho)
                t_check += time.perf_counter() - tc0

                steps_to_converge = step
                final_distance = dist
                if dist <= target_trace_distance:
                    converged = True
                    break

        converged_state = sigma.copy()
        for c in cutoffs:
            if c not in cutoff_states:
                cutoff_states[c] = converged_state

        timing_run = {
            "steps_to_converge": float(steps_to_converge),
            "converged": float(converged),
            "final_trace_distance": float(final_distance),
            "channel_apply_time": t_apply,
            "convergence_check_time": t_check,
            "total_run_time": time.perf_counter() - t_run0,
            "dim": float(self.dim),
            "num_jumps": float(self.num_jumps),
        }
        return ConvergenceResult(
            steps_to_converge=steps_to_converge,
            converged=converged,
            final_trace_distance=float(final_distance),
            converged_state=converged_state,
            cutoff_states=cutoff_states,
            timing=timing_run,
        )

    # --- Diagnostics -------------------------------------------------------------------

    def channel_diagnostics(self) -> dict[str, float]:
        """
        Validation metrics: trace preservation (D + KᵀK = I), fixed-point quality
        (M[ρ] ≈ ρ) and spectral bounds. Costs several O(d³) operations — for
        offline checking only, never inside iteration loops.
        """
        I = np.eye(self.dim, dtype=FDTYPE)
        K = self.reject_kraus
        dual_sum = self.D_accept + K.T @ K

        M_rho = self._apply_channel_fast(self.rho, symmetrize_output=True)
        ev_D = la.eigvalsh(self.D_accept, check_finite=False)

        return {
            "trace_preservation_fro": frobenius_norm(dual_sum - I),
            "fixed_point_fro": frobenius_norm(M_rho - self.rho),
            "fixed_point_trace_distance": trace_distance(M_rho, self.rho),
            "rho_min_eig": self.rho_min_eig,
            "rho_max_eig": self.rho_max_eig,
            "rho_condition_number": self.rho_condition_number,
            "D_min_eig": float(ev_D[0]),
            "D_max_eig": float(ev_D[-1]),
            "I_minus_D_min_eig": float(1.0 - ev_D[-1]),
            "jump_scale": self.jump_scale,
            "dim": float(self.dim),
            "num_jumps": float(self.num_jumps),
            "beta": self.beta,
        }

    @property
    def bohr_frequency_matrix(self) -> np.ndarray:
        """ν_ij = β(E_i − E_j); O(d²) broadcasted subtraction, built on demand."""
        E = self.energies.reshape(-1, 1)
        return self.beta * (E - E.T)


__all__ = [
    "DTYPE",
    "QuantumGibbsSampler",
    "TrajectoryResult",
    "ConvergenceResult",
    "coherent_S_minus_weight",
    "bohr_weight_matrix",
    "jumps_from_symmetric_matrices",
    "random_symmetric_jumps",
    "trace_distance",
    "frobenius_norm",
    "expectation_value",
]
