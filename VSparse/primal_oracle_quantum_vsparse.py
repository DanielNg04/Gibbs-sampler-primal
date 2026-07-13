# primal_oracle_quantum_vsparse.py — SDP primal oracle (van Apeldoorn & Gilyén, arXiv:1804.05058 §2.2).

"""
Primal-oracle feasibility loop with exact or MCMC Gibbs preparation
(``gibbs_sampler_quantum_vsparse``).

VSparse conventions:
- Problem matrices (C, A_j, M, jumps) are real symmetric CSR float64, validated
  once at the boundary (``validate_matrix``).
- Quantum states (rho_n, sigma, X') remain dense ndarrays — same return API as VSelf.
- Constraint traces use O(nnz(A_j)) inner products instead of O(n^2) GEMV rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix, issparse

from gibbs_sampler_quantum_vsparse import (
    MatrixLike,
    QuantumGibbsSampler,
    jumps_from_symmetric_matrices,
    trace_distance,
    validate_matrix,
    _symmetric_to_dense,
)

DTYPE = np.float64


# --- Data containers ---

@dataclass
class PrimalOracleProblem:
    """
    Input for the primal feasibility test at objective threshold g.

    Accepts dense ``ndarray`` inputs at construction; stores CSR internally.
    """

    A_matrices: List[MatrixLike]
    b: np.ndarray
    C: MatrixLike
    R: float
    g: float

    def __post_init__(self) -> None:
        self.C = validate_matrix(self.C)
        self.A_matrices = [validate_matrix(A) for A in self.A_matrices]
        self.b = np.asarray(self.b, dtype=DTYPE).reshape(-1)
        self.R = float(self.R)
        self.g = float(self.g)

    @property
    def n(self) -> int:
        return int(self.C.shape[0])


@dataclass
class ConstraintDiagnostics:
    traces: np.ndarray
    bounds_tilde: np.ndarray
    slacks: np.ndarray
    max_violation_index: Optional[int]
    feasible_within_theta: bool


@dataclass
class PrimalOracleResult:
    y: np.ndarray
    X_prime: np.ndarray
    omega: float
    X: np.ndarray
    z: float
    constraint_diag: ConstraintDiagnostics
    iterations: int
    theta: float
    timing: Optional[dict[str, float]] = None
    gibbs_mode: str = "exact"
    gibbs_steps_per_iter: Optional[List[int]] = None
    gibbs_converged_per_iter: Optional[List[bool]] = None
    cutoff_trace_distance_per_iter: Optional[dict[int, List[float]]] = None
    gibbs_gap_per_iter: Optional[List[tuple[int, float]]] = None


# --- Gibbs state from the exponent matrix ---

def _omega_from_energies(E: np.ndarray) -> float:
    E = np.asarray(E, dtype=DTYPE)
    emin = float(np.min(E))
    log_Z = float(np.log(np.sum(np.exp(-(E - emin)))) - emin)
    return float(np.exp(-np.logaddexp(log_Z, 0.0)))


def gibbs_state_n_from_exponent(M: MatrixLike) -> tuple[np.ndarray, float]:
    """
    Closed-form ρ_n = exp(−M)/Z_n and corner ω = 1/(Z_n + 1).

    M may be CSR or dense; densified once for full-spectrum ``eigh``.
    """
    M_csr = validate_matrix(M)
    n = M_csr.shape[0]
    M_dense = _symmetric_to_dense(M_csr, n)
    w, U = la.eigh(M_dense, driver="evd", check_finite=False)
    p = np.exp(-(w - float(w[0])))
    probs = p / float(np.sum(p))
    rho_n = (U * probs) @ U.T
    return rho_n, _omega_from_energies(w)


# --- Channel diagnostics ---

def _channel_spectral_gap(kraus_stack: np.ndarray) -> float:
    q1, d = kraus_stack.shape[0], kraus_stack.shape[1]
    S = np.zeros((d * d, d * d), dtype=DTYPE)
    for a in range(q1):
        S += np.kron(kraus_stack[a], kraus_stack[a])
    moduli = np.abs(la.eigvals(S, check_finite=False))
    moduli.sort()
    return float(moduli[-1] - moduli[-2])


# --- Feasibility test ---

def _trace_product(A: csr_matrix, X: np.ndarray) -> float:
    """Tr(A X) for symmetric A (CSR) and dense X — O(nnz(A))."""
    return float(A.multiply(X).sum())


def constraint_traces_from_stack(
    Xp: np.ndarray,
    A_stack: Union[np.ndarray, Sequence[csr_matrix]],
) -> np.ndarray:
    """
    All Tr(A_j X') at once.

    Accepts the VSelf dense ``(m+1, n, n)`` stack **or** a sequence of CSR matrices.
    """
    if isinstance(A_stack, np.ndarray):
        m1 = A_stack.shape[0]
        return A_stack.reshape(m1, -1) @ Xp.ravel()

    Xp = np.asarray(Xp, dtype=DTYPE)
    return np.asarray([_trace_product(A, Xp) for A in A_stack], dtype=DTYPE)


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
    if diag.feasible_within_theta:
        return None
    if selection == "max":
        return diag.max_violation_index
    violated = np.flatnonzero(-diag.slacks > theta)
    return int(violated[rng.integers(violated.size)])


# --- Main iteration ---

def run_primal_oracle(
    problem: PrimalOracleProblem,
    epsilon: float,
    *,
    max_iterations: Optional[int] = None,
    collect_timing: bool = False,
    gibbs_mode: str = "exact",
    gibbs_jump_matrices: Optional[List[MatrixLike]] = None,
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
    wall_t0 = time.perf_counter()
    timing: Optional[dict[str, float]] = (
        {
            "total_wall_time": 0.0,
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

    A_stack: List[csr_matrix] = [
        validate_matrix(-problem.C),
        *problem.A_matrices,
    ]
    b_tilde = np.concatenate(([-problem.g], problem.b)) / problem.R
    n = problem.n
    m1 = len(A_stack)

    theta = epsilon / (2.0 * problem.R)
    gibbs_theta = float(gibbs_target_theta) if gibbs_target_theta is not None else theta
    default_iters = int(np.ceil(np.log(n) / theta**2))
    iters = default_iters if max_iterations is None else int(max_iterations)

    y = np.zeros(m1, dtype=DTYPE)
    M = csr_matrix((n, n), dtype=DTYPE)

    use_mcmc = gibbs_mode == "mcmc"
    jumps: Optional[List[csr_matrix]] = None
    if use_mcmc:
        jumps = jumps_from_symmetric_matrices(
            gibbs_jump_matrices, normalize=normalize_jumps
        )
    cutoffs = sorted({int(c) for c in gibbs_step_cutoffs}) if (use_mcmc and gibbs_step_cutoffs) else []
    sigma0 = np.eye(n, dtype=DTYPE) / float(n)
    v_rng = np.random.default_rng(violation_rng_seed)

    gibbs_steps_per_iter: List[int] = []
    gibbs_converged_per_iter: List[bool] = []
    cutoff_td_per_iter: dict[int, List[float]] = {c: [] for c in cutoffs}
    gap_stride = int(gibbs_gap_stride) if (use_mcmc and gibbs_gap_stride) else 0
    gibbs_gap_per_iter: List[tuple[int, float]] = []

    def _package(
        rho_n: np.ndarray,
        omega: float,
        diag: ConstraintDiagnostics,
        iters_done: int,
    ) -> PrimalOracleResult:
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
        if use_mcmc:
            tc0 = time.perf_counter()
            sampler = QuantumGibbsSampler(M, jumps)
            if timing is not None:
                timing["gibbs_construction_time"] += time.perf_counter() - tc0

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

        y[j] += theta
        th0 = time.perf_counter()
        M = M + (theta * A_stack[j])
        if issparse(M):
            M.eliminate_zeros()
        if timing is not None:
            timing["hamiltonian_update_time"] += time.perf_counter() - th0

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
