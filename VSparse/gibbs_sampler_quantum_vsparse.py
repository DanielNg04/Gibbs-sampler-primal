"""
CSR sparse-matrix simulator for the discrete-time quantum Metropolis channel
(Gilyén et al., arXiv:2405.20322) — mirror of ``gibbs_sampler_quantum_v4self``.

Same public API; problem matrices (H, jumps) stored as ``scipy.sparse.csr_matrix``.
Quantum states (sigma, rho) remain dense.

Primary speedups:
  - Kraus apply (A @ sigma) @ A.T     : O(nnz(A)*n) vs O(n^3)
  - Bohr rotation U.T @ A @ U         : O(nnz(A)*n) + O(n^3) vs O(n^3) with dense A
  - Jump normalize / storage          : O(nnz) vs O(n^2)
  - Kraus kept sparse when fill-in is modest (no mandatory densification)

No meaningful speedup:
  - trace_distance (dense sigma - rho)
  - Full eigh(H) when H is densified for the full spectrum
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import norm as sparse_norm

DTYPE = np.float64
MatrixLike = Union[np.ndarray, csr_matrix]

# Densify H for eigh when n is small (BLAS wins) or H is already fairly dense.
_DENSE_EIGH_N_MAX = 40
_DENSE_EIGH_DENSITY = 0.25
# Store a Kraus operator as CSR when nnz / n^2 is below this after rotation.
_KRAUS_CSR_MAX_DENSITY = 0.40
_SYM_TOL = 1e-10


# --- Sparse / dense boundary utilities ---

def validate_matrix(H: MatrixLike) -> csr_matrix:
    """
    Validate a real symmetric matrix and return CSR storage.

    Accepts dense ``ndarray`` or ``csr_matrix`` at the API boundary.
    """
    if issparse(H):
        A = H.tocsr(copy=True)
        A.sum_duplicates()
    else:
        dense = np.ascontiguousarray(H, dtype=DTYPE)
        asym = float(np.max(np.abs(dense - dense.T))) if dense.size else 0.0
        if asym > _SYM_TOL:
            raise ValueError(
                f"validate_matrix: input is not symmetric (max |A - A^T| = {asym:.3e})."
            )
        A = csr_matrix(0.5 * (dense + dense.T))

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"validate_matrix: expected square matrix, got {A.shape}.")
    _check_csr_symmetry(A)
    A.eliminate_zeros()
    return A


def validate_dense_matrix(H: np.ndarray) -> np.ndarray:
    """Validate a dense state or difference matrix (sigma, rho, sigma - rho)."""
    A = np.ascontiguousarray(H, dtype=DTYPE)
    asym = float(np.max(np.abs(A - A.T))) if A.size else 0.0
    if asym > _SYM_TOL:
        raise ValueError(
            f"validate_dense_matrix: input is not symmetric (max |A - A^T| = {asym:.3e})."
        )
    return A


def _check_csr_symmetry(A: csr_matrix) -> None:
    if A.nnz == 0:
        return
    diff = A - A.T
    diff.eliminate_zeros()
    if diff.nnz > 0:
        asym = float(np.max(np.abs(diff.data)))
        if asym > _SYM_TOL:
            raise ValueError(
                f"validate_matrix: input is not symmetric (max |A - A^T| = {asym:.3e})."
            )


def _matrix_density(A: MatrixLike, n: int) -> float:
    if issparse(A):
        return float(A.nnz) / float(n * n)
    return 1.0


def _symmetric_to_dense(M: MatrixLike, n: int) -> np.ndarray:
    """Materialize a symmetric matrix for LAPACK (full-spectrum eigh)."""
    if issparse(M):
        if n <= _DENSE_EIGH_N_MAX or _matrix_density(M, n) > _DENSE_EIGH_DENSITY:
            return M.toarray()
        # Full spectrum still needs dense LAPACK today; eigsh(all) is not faster.
        return M.toarray()
    return np.ascontiguousarray(M, dtype=DTYPE)


def _jump_in_energy_basis(A: csr_matrix, U: np.ndarray) -> np.ndarray:
    """U.T @ A @ U with A in CSR: O(nnz(A)*n) + O(n^3)."""
    AU = A @ U
    return U.T @ AU


def _rotate_from_energy_basis(M_E: np.ndarray, U: np.ndarray) -> np.ndarray:
    """U @ M_E @ U.T — dense n×n in the original basis."""
    return U @ M_E @ U.T


def _array_to_csr(M: np.ndarray, zero_tol: float = 1e-12) -> csr_matrix:
    """Convert a dense matrix to CSR, dropping entries below ``zero_tol``."""
    if zero_tol > 0.0:
        M = M.copy()
        M[np.abs(M) <= zero_tol] = 0.0
    out = csr_matrix(M)
    out.eliminate_zeros()
    return out


def _kraus_from_dense(M: np.ndarray, n: int) -> Union[csr_matrix, np.ndarray]:
    """Prefer CSR storage when the rotated Kraus operator is sparse enough."""
    csr = _array_to_csr(M)
    if float(csr.nnz) / float(n * n) <= _KRAUS_CSR_MAX_DENSITY:
        return csr
    return np.ascontiguousarray(M, dtype=DTYPE)


def _kraus_matmul(A: Union[csr_matrix, np.ndarray], sigma: np.ndarray) -> np.ndarray:
    """(A @ sigma) @ A.T for symmetric A."""
    tmp = A @ sigma
    if issparse(A):
        return tmp @ A.T
    return tmp @ A.T


# --- Small linear algebra utilities (unchanged logic) ---

def coherent_S_minus_weight(nu: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.tanh(0.25 * nu))


def bohr_weight_matrix(energies: np.ndarray) -> np.ndarray:
    E = np.asarray(energies, dtype=DTYPE).reshape(-1, 1)
    nu = E - E.T
    return np.asarray(coherent_S_minus_weight(nu), dtype=DTYPE)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    d = validate_dense_matrix(np.asarray(rho, dtype=DTYPE) - np.asarray(sigma, dtype=DTYPE))
    return float(0.5 * np.sum(np.abs(la.eigvalsh(d, check_finite=False))))


def frobenius_norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, ord="fro"))


# --- Result dataclasses ---

@dataclass
class ConvergenceResult:
    steps_to_converge: int
    converged: bool
    final_trace_distance: float
    converged_state: np.ndarray
    cutoff_states: dict[int, np.ndarray]
    timing: dict[str, float]


@dataclass
class TrajectoryResult:
    trace_dist: List[float]
    frobenius_dist: List[float]
    states: Optional[List[np.ndarray]] = None


# --- Proposal jumps ---

def jumps_from_symmetric_matrices(
    mats: Sequence[MatrixLike],
    normalize: bool = True,
    drop_zero: bool = True,
    zero_tol: float = 1e-12,
) -> List[csr_matrix]:
    """
    Build CSR jump operators from problem matrices — O(nnz) norm and scale.
    """
    out: List[csr_matrix] = []
    for M in mats:
        A = validate_matrix(M)
        nrm = float(sparse_norm(A, ord="fro"))
        if drop_zero and nrm <= zero_tol:
            continue
        if normalize and nrm > zero_tol:
            A = A.copy()
            A.data /= nrm
        out.append(A)
    return out


# --- Main sampler ---

class QuantumGibbsSampler:
    """
    Discrete-time quantum detailed-balanced channel with CSR jumps and Kraus ops.
    """

    def __init__(
        self,
        H: MatrixLike,
        jumps: Sequence[MatrixLike],
        psd_eps: float = 1e-12,
        trace_tol: float = 1e-8,
        rescale_margin: float = 1e-6,
        symmetrize_output: bool = True,
        verbose: bool = False,
    ) -> None:
        self.psd_eps = float(psd_eps)
        self.trace_tol = float(trace_tol)
        self.symmetrize_output = bool(symmetrize_output)
        self.verbose = bool(verbose)

        self.H = validate_matrix(H)
        d = self.H.shape[0]
        self.dim = d

        jump_list = [validate_matrix(A) for A in jumps]
        q = len(jump_list)
        self.num_jumps = q

        H_dense = _symmetric_to_dense(self.H, d)
        self.energies, self.eigenvectors = la.eigh(H_dense, driver="evd", check_finite=False)
        E, U = self.energies, self.eigenvectors

        emin = float(E[0])
        w = np.exp(-(E - emin))
        probs = w / float(np.sum(w))
        self.rho = validate_dense_matrix((U * probs) @ U.T)

        self.rho_min_eig = float(np.min(probs))
        self.rho_max_eig = float(np.max(probs))
        self.rho_condition_number = self.rho_max_eig / max(self.rho_min_eig, self.psd_eps)

        # Bohr reweighing: sparse jumps → energy basis (sparse-dense matmul per jump).
        jumps_E = [_jump_in_energy_basis(A, U) for A in jump_list]
        W = bohr_weight_matrix(E)
        accept_E = [W * JE for JE in jumps_E]

        D_E = np.zeros((d, d), dtype=DTYPE)
        for AE in accept_E:
            D_E += AE.T @ AE
        D_E = validate_dense_matrix(D_E)

        ev_D = la.eigvalsh(D_E, check_finite=False)
        lam_max = float(ev_D[-1])
        target = 1.0 - float(rescale_margin)
        if lam_max > target:
            c = math.sqrt(target / lam_max)
            accept_E = [c * AE for AE in accept_E]
            D_E *= c * c
            lam_max *= c * c
            self.jump_scale = c
        else:
            self.jump_scale = 1.0

        sqrt_p = np.sqrt(probs)
        inv_sqrt_p = 1.0 / np.sqrt(np.maximum(probs, self.psd_eps))
        M_reject = validate_dense_matrix(sqrt_p[:, None] * (np.eye(d) - D_E) * sqrt_p[None, :])
        w_M, v_M = la.eigh(M_reject, driver="evd", check_finite=False)

        if float(w_M[0]) < -1e-8:
            raise RuntimeError(
                f"ρ^(1/2)(I−D)ρ^(1/2) is not PSD within tolerance: min eig = {float(w_M[0])}"
            )
        np.maximum(w_M, 0.0, out=w_M)

        sqrt_M = (v_M * np.sqrt(w_M)) @ v_M.T
        K_E = sqrt_M * inv_sqrt_p[None, :]

        # Rotate accept + reject Kraus back; keep CSR when fill-in is modest.
        self.kraus_ops: List[Union[csr_matrix, np.ndarray]] = []
        for AE in accept_E:
            self.kraus_ops.append(_kraus_from_dense(_rotate_from_energy_basis(AE, U), d))
        self.kraus_ops.append(_kraus_from_dense(_rotate_from_energy_basis(K_E, U), d))

        self.D_accept = validate_dense_matrix(_rotate_from_energy_basis(D_E, U))

        if self.verbose:
            nnz_kraus = sum(
                (op.nnz if issparse(op) else op.size) for op in self.kraus_ops
            )
            print(
                "[QuantumGibbsSampler/sparse] dim=", d,
                "num_jumps=", q,
                "Condition number of rho=", self.rho_condition_number,
                "max_eig(D)=", lam_max,
                "jump_scale=", self.jump_scale,
                "kraus_total_nnz=", nnz_kraus,
            )

    @property
    def kraus_stack(self) -> np.ndarray:
        """Dense (q+1, n, n) stack — API mirror for diagnostics (e.g. spectral gap)."""
        mats = [op.toarray() if issparse(op) else op for op in self.kraus_ops]
        return np.ascontiguousarray(np.stack(mats, axis=0), dtype=DTYPE)

    @property
    def accept_kraus(self) -> np.ndarray:
        return self.kraus_stack[: self.num_jumps]

    @property
    def reject_kraus(self) -> np.ndarray:
        return self.kraus_stack[self.num_jumps]

    def _apply_channel_fast(
        self,
        sigma: np.ndarray,
        symmetrize_output: Optional[bool] = None,
    ) -> np.ndarray:
        """
        M[σ] = Σ_a A_a σ A_aᵀ — sparse Kraus when stored as CSR: O(nnz(A)*n) each.
        """
        if symmetrize_output is None:
            symmetrize_output = self.symmetrize_output
        sigma = np.ascontiguousarray(sigma, dtype=DTYPE)
        out = np.zeros_like(sigma)
        for A in self.kraus_ops:
            out += _kraus_matmul(A, sigma)
        return (out + out.T) * 0.5 if symmetrize_output else out

    def apply_channel(self, sigma: np.ndarray, check: bool = False) -> np.ndarray:
        sigma = validate_dense_matrix(sigma)
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

    def _coerce_state(self, sigma0: np.ndarray) -> np.ndarray:
        sigma = validate_dense_matrix(sigma0)
        tr = float(np.trace(sigma))
        if abs(tr - 1.0) > 1e-6 and tr > 0.0:
            sigma = sigma / tr
        return sigma

    def run(
        self,
        sigma0: np.ndarray,
        steps: int,
        store_states: bool = False,
    ) -> TrajectoryResult:
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        sigma = self._coerce_state(sigma0)
        tdist: List[float] = []
        fdist: List[float] = []
        states: Optional[List[np.ndarray]] = [] if store_states else None

        for _ in range(steps + 1):
            if states is not None:
                states.append(sigma.copy())
            tdist.append(trace_distance(sigma, self.rho))
            fdist.append(frobenius_norm(sigma - self.rho))
            if len(tdist) <= steps:
                sigma = self._apply_channel_fast(sigma)

        return TrajectoryResult(trace_dist=tdist, frobenius_dist=fdist, states=states)

    def run_until_converged(
        self,
        sigma0: np.ndarray,
        target_trace_distance: float,
        max_steps: int,
        step_cutoffs: Optional[Sequence[int]] = None,
        symmetrize_each_step: Optional[bool] = None,
    ) -> ConvergenceResult:
        sym = self.symmetrize_output if symmetrize_each_step is None else symmetrize_each_step
        cutoffs = sorted({int(c) for c in step_cutoffs}) if step_cutoffs else []
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
