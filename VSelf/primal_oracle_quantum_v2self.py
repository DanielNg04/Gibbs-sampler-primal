# primal_oracle_quantum_v2self.py — SDP primal oracle (van Apeldoorn & Gilyén, arXiv:1804.05058 §2.2).

"""
Primal-oracle feasibility loop with exact or MCMC Gibbs preparation
(``gibbs_sampler_quantum_v4self``).

VSelf conventions:
- Every matrix is real symmetric float64. Inputs are validated once at the
  boundary (``validate_matrix``) and trusted everywhere after — no repeated
  shape/symmetry assertions, no complex handling.
- The oracle answers: is there X ⪰ 0 with Tr(A_j X) ≤ b_j (A_1 = I, b_1 = R)
  and Tr(CX) ≥ g? It maintains dual-style weights y ≥ 0, prepares the Gibbs
  state ρ_n ∝ exp(−M) with M = Σ_j y_j A_j, and either stops (θ-feasible)
  or bumps a violated constraint: y_j += θ with θ = ε/(2R).
- The paper's (n+1)-dimensional embedding is reconstructed from the partition
  function instead of being simulated: ω = 1/(Z_n + 1) and X' = (1 − ω) ρ_n.
  Block-diagonal jumps never mix the corner coordinate, so only the n×n block
  ever goes through the channel.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.linalg as la

from gibbs_sampler_quantum_v4self import (
    QuantumGibbsSampler,
    jumps_from_symmetric_matrices,
    trace_distance,
    validate_matrix,
)

DTYPE = np.float64


# --- Data containers ---

@dataclass
class PrimalOracleProblem:
    """
    Input for the primal feasibility test at objective threshold g.

    Convention: A_1 = I and b_1 = R encode the trace bound Tr(X) ≤ R.
    All matrices are symmetrized at construction and trusted afterwards.
    """

    A_matrices: List[np.ndarray]   # A_1..A_m, real symmetric n×n
    b: np.ndarray                  # b_1..b_m
    C: np.ndarray                  # objective matrix, real symmetric n×n
    R: float                       # trace bound (= b_1)
    g: float                       # objective threshold; b_0 = −g for A_0 = −C

    #Checks, corrections
    def __post_init__(self) -> None:
        self.C = validate_matrix(self.C)
        self.A_matrices = [validate_matrix(A) for A in self.A_matrices]
        self.b = np.asarray(self.b, dtype=DTYPE).reshape(-1)
        self.R = float(self.R)
        self.g = float(self.g)


@dataclass
class ConstraintDiagnostics:
    """Tr(A_j X') versus normalized bounds b̃_j = b_j / R."""

    traces: np.ndarray             # Tr(A_j X') for j = 0..m
    bounds_tilde: np.ndarray       # b̃_j = b_j / R
    slacks: np.ndarray             # b̃_j − Tr(A_j X'); positive means satisfied
    max_violation_index: Optional[int]   # worst violator, or None if θ-feasible
    feasible_within_theta: bool


@dataclass
class PrimalOracleResult:
    """Termination state of :func:`run_primal_oracle`."""

    y: np.ndarray                  # dual-style weights at the stopping iteration
    X_prime: np.ndarray            # normalized primal block X' = (1 − ω) ρ_n
    omega: float                   # embedding corner ω = 1/(Z_n + 1)
    X: np.ndarray                  # unnormalized candidate X = R · X'
    z: float                       # objective proxy z = R · Tr(X')
    constraint_diag: ConstraintDiagnostics
    iterations: int
    theta: float
    timing: Optional[dict[str, float]] = None
    gibbs_mode: str = "exact"

    gibbs_steps_per_iter: Optional[List[int]] = None
    """MCMC mode: channel applications to converge at each oracle iteration."""

    gibbs_converged_per_iter: Optional[List[bool]] = None
    """MCMC mode: whether Gibbs converged within the step cap at each iteration."""

    cutoff_trace_distance_per_iter: Optional[dict[int, List[float]]] = None
    """MCMC + cutoffs: per cutoff c, D(σ_cutoff, σ_full) at each iteration."""

    gibbs_gap_per_iter: Optional[List[tuple[int, float]]] = None
    """MCMC + gap stride: (iteration, spectral gap of the constructed channel)."""


# --- Gibbs state from the exponent matrix (exact mode + embedding corner) ---

def _omega_from_energies(E: np.ndarray) -> float:
    """
    ω = 1/(Z_n + 1) from the spectrum of M, via log-sum-exp.

    Stability: Z_n = Σ exp(−E_i) can overflow float64 long before ω loses
    meaning. Working with log Z_n = log Σ exp(−(E_i − E_min)) − E_min keeps
    every intermediate representable, and ω = exp(−logaddexp(log Z, 0))
    evaluates 1/(Z+1) without ever forming Z.
    """
    E = np.asarray(E, dtype=DTYPE)
    emin = float(np.min(E))
    log_Z = float(np.log(np.sum(np.exp(-(E - emin)))) - emin)
    return float(np.exp(-np.logaddexp(log_Z, 0.0)))


def gibbs_state_n_from_exponent(M: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Closed-form ρ_n = exp(−M)/Z_n and corner ω = 1/(Z_n + 1).

    Same construction as the sampler's step 2: one ``eigh`` (dsyevd — backward
    stable, fastest full-spectrum symmetric driver), shifted Boltzmann weights
    (softmax trick: the max weight is exactly 1, so no overflow and no loss of
    relative accuracy), then ρ_n = U diag(p) Uᵀ as a column scaling + one GEMM.
    This is the cheap path used for g-calibration; the channel path is
    ``run_primal_oracle(gibbs_mode="mcmc")``.
    """
    w, U = la.eigh(M, driver="evd", check_finite=False)
    p = np.exp(-(w - float(w[0])))          # eigh returns ascending eigenvalues
    probs = p / float(np.sum(p))
    rho_n = (U * probs) @ U.T
    return rho_n, _omega_from_energies(w)


# --- Channel diagnostics (MCMC mode) ---

def _channel_spectral_gap(kraus_stack: np.ndarray) -> float:
    """
    Spectral gap |λ₁| − |λ₂| of the channel superoperator S = Σ_a A_a ⊗ A_a.

    λ₁ = 1 belongs to the fixed point; the gap bounds the per-step contraction
    of everything orthogonal to it, so it is the channel's asymptotic mixing
    rate. Building S is O(q d⁴) and its eigenvalues O(d⁶) — this is a pure
    diagnostic for small instances, computed only every ``gibbs_gap_stride``
    iterations, never part of the solver hot path.
    """
    q1, d = kraus_stack.shape[0], kraus_stack.shape[1]
    S = np.zeros((d * d, d * d), dtype=DTYPE)
    for a in range(q1):
        S += np.kron(kraus_stack[a], kraus_stack[a])
    moduli = np.abs(la.eigvals(S, check_finite=False))
    moduli.sort()
    return float(moduli[-1] - moduli[-2])


# --- Feasibility test ---
def constraint_traces_from_stack(Xp: np.ndarray, A_stack: np.ndarray) -> np.ndarray:
    """
    All Tr(A_j X') at once as Frobenius inner products — O(m n²).

    Tr(A_j X') = Σ_ik (A_j)_ik (X')_ik (both symmetric), so flattening each
    constraint matrix into a row turns the whole test into one matrix-vector
    product (m+1, n²)·(n²,) — a single multithreaded BLAS dgemv instead of an
    einsum loop. Both reshapes are free: the stack and X' are C-contiguous,
    so only array metadata changes.
    """
    m1 = A_stack.shape[0]
    return A_stack.reshape(m1, -1) @ Xp.ravel()


def _constraint_diagnostics_from_traces(
    traces: np.ndarray,
    b_tilde: np.ndarray,
    theta: float,
) -> ConstraintDiagnostics:
    violations = traces - b_tilde
    j_max = int(np.argmax(violations))
    feasible = float(violations[j_max]) <= theta 
    return ConstraintDiagnostics(
        traces=traces,
        bounds_tilde=b_tilde,
        slacks=b_tilde - traces,
        max_violation_index=None if feasible else j_max,
        feasible_within_theta=feasible,
    )


def _select_violation_index(
    diag: ConstraintDiagnostics,
    theta: float,
    selection: str,
    rng: np.random.Generator,
) -> Optional[int]:
    """
    Index of the constraint to update, or None if θ-feasible.

    Reuses the argmax already computed in the diagnostics instead of rescanning.
    "max" picks the largest violation (paper default); "random" picks uniformly
    among constraints violated by more than θ (selection-rule experiments;
    violations are the negated slacks).
    """
    if diag.feasible_within_theta:
        return None
    if selection == "max":
        return diag.max_violation_index
    violated = np.flatnonzero(-diag.slacks > theta)
    return int(violated[rng.integers(violated.size)])


# --- Main iteration (Section 2.2) ---

def run_primal_oracle(
    problem: PrimalOracleProblem,
    epsilon: float,
    *,
    max_iterations: Optional[int] = None,
    collect_timing: bool = False,
    gibbs_mode: str = "exact",
    gibbs_jump_matrices: Optional[List[np.ndarray]] = None,
    gibbs_max_steps: int = 5000,
    gibbs_step_cutoffs: Optional[Sequence[int]] = None,
    gibbs_target_theta: Optional[float] = None,
    gibbs_warm_start: bool = False,
    gibbs_gap_stride: Optional[int] = None,
    normalize_jumps: bool = True,
    violation_selection: str = "max",
    violation_rng_seed: Optional[int] = None,
    return_on_exhaustion: bool = False,
) -> Optional[PrimalOracleResult]:
    """
    Primal-oracle feasibility loop at threshold g.

    Per iteration: prepare ρ_n ∝ exp(−M) (closed form or Gibbs channel),
    recover X' = (1 − ω) ρ_n, test all constraints at tolerance θ = ε/(2R),
    and on the selected violation j update y_j += θ and M += θ A_j (rank-one,
    O(n²) — M is never rebuilt from scratch).

    gibbs_mode:
        "exact"  — closed-form eigendecomposition each iteration (cheap; used
                   for g-calibration).
        "mcmc"   — iterate the detailed-balanced channel until the trace
                   distance to ρ_n drops below ``gibbs_target_theta`` (default:
                   the oracle tolerance θ). Requires ``gibbs_jump_matrices``.
    gibbs_step_cutoffs:
        MCMC only: also record D(σ_cutoff, σ_full) at these step counts each
        iteration (early-termination experiments).
    gibbs_warm_start:
        MCMC only: start each channel run from the previous iteration's
        endpoint instead of I/n. Consecutive exponents differ by one rank-one
        update, so their Gibbs states are close and mixing restarts warm.
    gibbs_gap_stride:
        MCMC only: every this many iterations, record the spectral gap of the
        freshly constructed channel (O(n⁶) diagnostic — see
        :func:`_channel_spectral_gap`); results land in ``gibbs_gap_per_iter``.
    return_on_exhaustion:
        If the iteration cap is hit without θ-feasibility, return the last
        state (feasible_within_theta=False) instead of None, preserving the
        per-iteration logs.

    Returns None when the cap is exhausted and ``return_on_exhaustion`` is
    False — the binary-search solver's "infeasible at this g" contract.
    """
    wall_t0 = time.perf_counter()
    timing: Optional[dict[str, float]] = (
        {
            "total_wall_time": 0.0,
            "gibbs_time": 0.0,                      # sum of the three buckets below
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

    # Full constraint list with the objective folded in as constraint 0:
    # Tr(A_0 X) ≤ b_0  ⇔  Tr(−C X) ≤ −g  ⇔  Tr(CX) ≥ g.
    # One contiguous (m+1, n, n) stack serves both the GEMV feasibility test
    # (via a free reshape to (m+1, n²)) and the additive M updates.
    A_stack = np.ascontiguousarray(
        np.stack([-problem.C] + problem.A_matrices, axis=0), dtype=DTYPE
    )
    b_tilde = np.concatenate(([-problem.g], problem.b)) / problem.R
    n = problem.C.shape[0]
    m1 = A_stack.shape[0]

    # Paper schedule: ceil(ln n / θ²) iterations suffice when g is feasible.
    theta = epsilon / (2.0 * problem.R)
    gibbs_theta = float(gibbs_target_theta) if gibbs_target_theta is not None else theta
    default_iters = int(np.ceil(np.log(n) / theta**2))
    iters = default_iters if max_iterations is None else int(max_iterations)

    y = np.zeros(m1, dtype=DTYPE)

    # Exponent M = Σ_j y_j A_j, maintained incrementally (y starts at 0).
    M = np.zeros((n, n), dtype=DTYPE)

    # MCMC setup. The jump matrices never change between iterations (only M
    # does), so they are normalized once here — not once per iteration.
    use_mcmc = gibbs_mode == "mcmc"
    jumps: Optional[List[np.ndarray]] = None
    if use_mcmc:
        jumps = jumps_from_symmetric_matrices(
            gibbs_jump_matrices, normalize=normalize_jumps
        )
    cutoffs = sorted({int(c) for c in gibbs_step_cutoffs}) if (use_mcmc and gibbs_step_cutoffs) else []
    sigma0 = np.eye(n, dtype=DTYPE) / float(n)     # maximally mixed (or warm-started)
    v_rng = np.random.default_rng(violation_rng_seed)

    gibbs_steps_per_iter: List[int] = []
    gibbs_converged_per_iter: List[bool] = []
    cutoff_td_per_iter: dict[int, List[float]] = {c: [] for c in cutoffs}
    gap_stride = int(gibbs_gap_stride) if (use_mcmc and gibbs_gap_stride) else 0
    gibbs_gap_per_iter: List[tuple[int, float]] = []

    def _package(rho_n: np.ndarray, omega: float, diag: ConstraintDiagnostics, iters_done: int) -> PrimalOracleResult:
        # X' = (1 − ω) ρ_n is only materialized here, at termination; the loop
        # itself never needs the matrix (see the trace-scaling note below).
        if timing is not None:
            timing["gibbs_time"] = (
                timing["gibbs_construction_time"]
                + timing["gibbs_channel_iteration_time"]
                + timing["gibbs_convergence_check_time"]
            )
            timing["total_wall_time"] = time.perf_counter() - wall_t0
        Xp = (1.0 - omega) * rho_n
        return PrimalOracleResult(
            y=y.copy(),
            X_prime=Xp,
            omega=float(omega),
            X=problem.R * Xp,
            z=problem.R * float(np.trace(Xp)),
            constraint_diag=diag,
            iterations=iters_done,
            theta=theta,
            timing=timing,
            gibbs_mode=gibbs_mode,
            gibbs_steps_per_iter=gibbs_steps_per_iter,
            gibbs_converged_per_iter=gibbs_converged_per_iter,
            cutoff_trace_distance_per_iter=(
                {c: cutoff_td_per_iter[c] for c in cutoffs} if cutoffs else None
            ),
            gibbs_gap_per_iter=gibbs_gap_per_iter if gap_stride else None,
        )

    last_state: Optional[tuple[np.ndarray, float, ConstraintDiagnostics]] = None

    for it in range(iters):
        # --- Gibbs state preparation ---
        if use_mcmc:
            tc0 = time.perf_counter()
            sampler = QuantumGibbsSampler(M, jumps)
            if timing is not None:
                timing["gibbs_construction_time"] += time.perf_counter() - tc0

            # Diagnostic: gap of this iteration's channel, on a stride so the
            # O(n⁶) eigendecomposition never dominates the run.
            if gap_stride and it % gap_stride == 0:
                gibbs_gap_per_iter.append(
                    (it + 1, _channel_spectral_gap(sampler.kraus_stack))
                )

            conv = sampler.run_until_converged(
                sigma0,
                target_trace_distance=gibbs_theta,
                max_steps=gibbs_max_steps,
                step_cutoffs=cutoffs or None,
            )
            if timing is not None:
                timing["gibbs_channel_iteration_time"] += conv.timing["channel_apply_time"]
                timing["gibbs_convergence_check_time"] += conv.timing["convergence_check_time"]

            rho_n = conv.converged_state          
            omega = _omega_from_energies(sampler.energies)

            if gibbs_warm_start:
                sigma0 = conv.converged_state

            gibbs_steps_per_iter.append(int(conv.steps_to_converge))
            gibbs_converged_per_iter.append(bool(conv.converged))
            for c in cutoffs:
                cutoff_td_per_iter[c].append(
                    float(trace_distance(conv.cutoff_states[c], conv.converged_state))
                )
        else:
            tc0 = time.perf_counter()
            rho_n, omega = gibbs_state_n_from_exponent(M)
            if timing is not None:
                timing["gibbs_construction_time"] += time.perf_counter() - tc0
            gibbs_steps_per_iter.append(0)
            gibbs_converged_per_iter.append(True)

        # --- Feasibility test ---
        # Traces are linear in the state: Tr(A_j X') = (1 − ω)·Tr(A_j ρ_n).
        # Scaling the m-vector by the scalar (1 − ω) avoids forming the O(n²)
        # matrix X' on every iteration.
        tt0 = time.perf_counter()
        traces = (1.0 - omega) * constraint_traces_from_stack(rho_n, A_stack)
        if timing is not None:
            timing["trace_check_time"] += time.perf_counter() - tt0

        tv0 = time.perf_counter()
        diag = _constraint_diagnostics_from_traces(traces, b_tilde, theta)
        j = _select_violation_index(diag, theta, violation_selection, v_rng)
        if timing is not None:
            timing["violation_logic_time"] += time.perf_counter() - tv0

        last_state = (rho_n, omega, diag)

        if j is None:
            tp0 = time.perf_counter()
            result = _package(rho_n, omega, diag, it + 1)
            if timing is not None:
                timing["result_packaging_time"] += time.perf_counter() - tp0
                timing["total_wall_time"] = time.perf_counter() - wall_t0
            return result

        # --- Coordinate update y ← y + θ e_j; rank-one exponent update ---
        y[j] += theta
        th0 = time.perf_counter()
        M += theta * A_stack[j]
        if timing is not None:
            timing["hamiltonian_update_time"] += time.perf_counter() - th0

    # Iteration cap hit without θ-feasibility.
    if return_on_exhaustion and last_state is not None:
        return _package(last_state[0], last_state[1], last_state[2], iters)
    return None


__all__ = [
    "DTYPE",
    "PrimalOracleProblem",
    "ConstraintDiagnostics",
    "PrimalOracleResult",
    "gibbs_state_n_from_exponent",
    "constraint_traces_from_stack",
    "run_primal_oracle",
]
