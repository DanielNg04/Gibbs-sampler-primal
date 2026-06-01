"""
Classical dense-matrix simulator for the discrete-time quantum Metropolis channel
(Gilyén et al., arXiv:2405.20322): accept + single Kraus reject (Thm. 1, Eq. (5))
with coherent Bohr reweighing (Sec. 1.4). Target state ρ ∝ exp(−βH)/Tr(exp(−βH)).

Set BLAS thread counts externally on shared hosts; this module does not pin threads.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import scipy.linalg as la

# --- Defaults: double-precision complex / float ------------------------------------

CDTYPE = np.complex128
FDTYPE = np.float64


# --- Small linear-algebra utilities ------------------------------------------------


def _is_hermitian(M: np.ndarray, *, atol: float) -> bool:
    M = np.asarray(M)
    return M.shape[0] == M.shape[1] and np.allclose(M, M.conj().T, atol=atol, rtol=0.0)


def _hermitian_sqrt_pow(
    rho: np.ndarray,
    power: float,
    *,
    eps: float,
) -> np.ndarray:
    rho = (np.asarray(rho) + np.asarray(rho).conj().T) / 2.0
    w, v = la.eigh(rho)
    w = np.maximum(np.real(w), eps)
    return (v * (w**power)) @ v.conj().T


def matrix_sqrt_psd(M: np.ndarray, *, eps: float = 1e-14) -> np.ndarray:
    return _hermitian_sqrt_pow(M, 0.5, eps=eps)


def matrix_inv_sqrt_psd(M: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    return _hermitian_sqrt_pow(M, -0.5, eps=eps)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance D_T(rho, sigma) = (1/2) ||rho - sigma||_1 for Hermitian states."""
    d = rho - sigma
    s = la.eigvalsh(0.5 * (d + d.conj().T))
    return float(0.5 * np.sum(np.abs(s)))


def frobenius_norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, ord="fro"))


def expectation_value(sigma: np.ndarray, O: np.ndarray) -> float:
    """
    ⟨O⟩ = Tr(σ O) as a Frobenius inner product — **O(d²)** instead of forming σ @ O.

    Uses ``Tr(AB) = sum_ij A_ij B_ji`` via ``einsum`` so BLAS can parallelize.
    """
    return float(np.einsum("ij,ji->", sigma, O, optimize=True).real)


def print_threadpool_info() -> None:
    """Try ``threadpoolctl.threadpool_info()``; no-op message if unavailable."""
    try:
        import threadpoolctl

        info = threadpoolctl.threadpool_info()
        print("threadpool_info:", info)
    except Exception as exc:  # pragma: no cover
        print(f"(threadpoolctl not available or failed: {exc})")


# --- Bohr / frequency weighting ( Eq. (20), Eq. (33) ) ----------------------------


def bohr_weight_matrix(
    energies: np.ndarray,
    beta: float,
    weight_fn: "WeightFn",
) -> np.ndarray:
    E = np.asarray(energies, dtype=float).reshape(-1, 1)
    nu = beta * (E - E.T)  # real Bohr-frequency matrix ν_ij = β (E_i − E_j)
    # The built-in weight functions are pure NumPy expressions (tanh of the real
    # part), so they already act element-wise on the whole ν matrix. Calling the
    # function once on the array avoids np.vectorize's Python-level loop over all
    # n² entries, which was a J·n² hot-spot when building every jump operator.
    try:
        W = np.asarray(weight_fn(nu), dtype=complex)
        if W.shape == nu.shape:
            return W
    except (TypeError, ValueError, FloatingPointError):
        pass
    # Fallback for genuinely scalar-only callables that cannot take an array.
    return np.vectorize(weight_fn)(nu).astype(complex)


WeightFn = Union[str, Callable[[complex], complex]]


def S_operator_bohr_weight(nu: complex) -> complex:
    return 0.5 * np.tanh(np.real(nu) / 4.0)


def default_coherent_reweigh_weight(nu: complex) -> complex:
    return 0.5 * (1.0 - np.tanh(np.real(nu) / 4.0))


def operator_to_energy_basis(A: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U.conj().T @ A @ U


def operator_from_energy_basis(A_E: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ A_E @ U.conj().T


def apply_bohr_elementwise_weight(
    A: np.ndarray,
    U: np.ndarray,
    energies: np.ndarray,
    beta: float,
    weight_fn: WeightFn = "coherent_S_minus",
) -> np.ndarray:
    if isinstance(weight_fn, str):
        if weight_fn == "coherent_S_minus":
            fn = default_coherent_reweigh_weight
        else:
            raise ValueError(f"Unknown weight_fn key: {weight_fn!r}")
    else:
        fn = weight_fn

    A_E = operator_to_energy_basis(A, U)
    W = bohr_weight_matrix(energies, beta, fn)
    return operator_from_energy_basis(W * A_E, U)


def coherent_reweigh_jump(
    A: np.ndarray,
    U: np.ndarray,
    energies: np.ndarray,
    beta: float,
    weight_fn: WeightFn = "coherent_S_minus",
) -> np.ndarray:
    """Same coherent reweighing as v1 (Eq. (11), S_- weighting)."""
    return apply_bohr_elementwise_weight(A, U, energies, beta, weight_fn=weight_fn)


# --- Proposal jump factories -------------------------------------------------------


def hermitian_random_jumps(
    dim: int,
    num_jumps: int,
    *,
    rng: Optional[np.random.Generator] = None,
    scale: float = 0.3,
) -> List[np.ndarray]:
    rng = rng or np.random.default_rng()
    jumps: List[np.ndarray] = []
    for _ in range(num_jumps):
        G = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        A = 0.5 * scale * (G + G.conj().T)
        jumps.append(A)
    return jumps


def jumps_from_hermitian_matrices(
    mats: Sequence[np.ndarray],
    *,
    scale: float = 1.0,
    normalize: bool = True,
    drop_zero: bool = True,
    zero_tol: float = 1e-12,
) -> List[np.ndarray]:
    """
    Build Hermitian channel jump operators directly from problem matrices.

    This is the entry point used by the SDP primal oracle so that the
    discrete-time quantum Gibbs channel uses the **SDP constraint matrices plus
    the objective matrix** as its proposal (jump) operators, instead of random
    Hermitian generators (:func:`hermitian_random_jumps`). Physically these are
    the operators whose Bohr-frequency reweighing drives the Metropolis-like
    detailed-balance dynamics toward ρ ∝ exp(-βH).

    Parameters
    ----------
    mats :
        Iterable of square matrices (e.g. the padded ``Ã_j`` and ``Ã_0 = -C̃``).
        Each is Hermitised as ``(A + A†)/2`` before use.
    scale :
        Global multiplicative factor applied to every jump. The sampler further
        rescales the whole set so ``λ_max(Σ Ã† Ã) ≤ 1 - margin``; ``scale`` only
        sets the *relative* proposal strength before that safety rescaling.
    normalize :
        If True, divide each matrix by its Frobenius norm first, so that
        matrices with very different magnitudes (common across SDP constraints)
        contribute comparable proposal weight and the channel is better
        conditioned. Zero matrices are skipped when ``drop_zero``.
    drop_zero :
        Skip matrices whose Frobenius norm is ``≤ zero_tol`` (they would be
        no-op jumps and only waste a Kraus slot).

    Returns
    -------
    list of complex128 Hermitian operators ready for :class:`QuantumGibbsSampler`.
    """
    out: List[np.ndarray] = []
    for M in mats:
        A = np.asarray(M, dtype=CDTYPE)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Each jump matrix must be square.")
        A = (A + A.conj().T) / 2.0
        nrm = float(np.linalg.norm(A))
        if drop_zero and nrm <= zero_tol:
            continue
        if normalize and nrm > zero_tol:
            A = A / nrm
        A = float(scale) * A
        out.append(np.ascontiguousarray(A, dtype=CDTYPE))
    if not out:
        raise ValueError("No usable (non-zero) jump matrices were provided.")
    return out


def pauli_x_site(n_qubits: int, site: int) -> np.ndarray:
    if site < 0 or site >= n_qubits:
        raise ValueError(f"site must be in [0, {n_qubits - 1}]")
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    ops: List[np.ndarray] = []
    for q in range(n_qubits):
        ops.append(sx if q == site else np.eye(2, dtype=complex))
    M = ops[0]
    for k in range(1, n_qubits):
        M = np.kron(M, ops[k])
    return M


def local_pauli_x_set(n_qubits: int) -> List[np.ndarray]:
    return [pauli_x_site(n_qubits, a) for a in range(n_qubits)]


def normalized_pauli_x_proposal(
    n_qubits: int,
    scale: float,
) -> List[np.ndarray]:
    xs = local_pauli_x_set(n_qubits)
    c = scale / np.sqrt(n_qubits)
    return [c * x for x in xs]


# --- Main sampler ------------------------------------------------------------------


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


class QuantumGibbsSampler:
    """
    Discrete-time quantum detailed-balanced channel (Theorem 1 + coherent accept).

    Same construction as v1; v2 adds timing, stable Gibbs weights, and optimized helpers.
    """

    def __init__(
        self,
        H: np.ndarray,
        jumps: Sequence[np.ndarray],
        *,
        beta: float = 1.0,
        atol_herm: float = 1e-10,
        psd_eps: float = 1e-12,
        trace_tol: float = 1e-8,
        rescale_to_trace_nonincreasing: bool = True,
        rescale_margin: float = 1e-6,
        weight_fn: WeightFn = "coherent_S_minus",
        reject_mode: str = "explicit",
        verbose: bool = False,
        collect_timing: bool = False,
        strict_psd_checks: bool = True,
        symmetrize_output: bool = True,
    ) -> None:
        self.atol_herm = float(atol_herm)
        self.psd_eps = float(psd_eps)
        self.trace_tol = float(trace_tol)
        self.beta = float(beta)
        self.weight_fn: WeightFn = weight_fn
        self.reject_mode = reject_mode
        self.verbose = verbose
        self.strict_psd_checks = strict_psd_checks
        self.symmetrize_output = symmetrize_output
        self.collect_timing = bool(collect_timing)
        self.timing: dict[str, float] = {}

        if self.beta <= 0.0:
            raise ValueError("beta must be positive.")
        if rescale_margin < 0:
            raise ValueError("rescale_margin must be non-negative.")

        t_wall0 = time.perf_counter()

        H = np.asarray(H, dtype=CDTYPE)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("H must be a square matrix.")
        if not _is_hermitian(H, atol=self.atol_herm):
            raise ValueError("H must be Hermitian (within atol_herm).")
        d = H.shape[0]
        self.dim = d

        jumps_list = [np.asarray(A, dtype=CDTYPE) for A in jumps]
        if not jumps_list:
            raise ValueError("Provide at least one jump operator.")
        for a, A in enumerate(jumps_list):
            if A.shape != (d, d):
                raise ValueError(f"jumps[{a}] has shape {A.shape}, expected ({d},{d}).")
            if not _is_hermitian(A, atol=self.atol_herm):
                raise ValueError(f"jumps[{a}] must be Hermitian (within atol_herm).")

        self.H = (H + H.conj().T) / 2.0

        t_eigh0 = time.perf_counter()
        self.energies, self.eigenvectors = la.eigh(self.H)
        t_eigh1 = time.perf_counter()

        t_gibbs0 = time.perf_counter()
        self._build_gibbs_state()
        t_gibbs1 = time.perf_counter()

        t_coh0 = time.perf_counter()
        self.raw_accept_kraus = [
            coherent_reweigh_jump(A, self.eigenvectors, self.energies, self.beta, self.weight_fn)
            for A in jumps_list
        ]
        self.accept_kraus = [np.ascontiguousarray(A) for A in self.raw_accept_kraus]
        t_coh1 = time.perf_counter()

        t_rs0 = time.perf_counter()
        if rescale_to_trace_nonincreasing:
            self._rescale_accept_kraus(margin=rescale_margin)
        else:
            self._last_kraus_scale = 1.0
        self.accept_kraus = [np.ascontiguousarray(A) for A in self.accept_kraus]
        # Stacked Kraus for optional future vectorization; the apply loop uses BLAS
        # matmul per (dim×dim) operator — best for large d and modest q.
        self.accept_kraus_stack = np.ascontiguousarray(
            np.stack(self.accept_kraus, axis=0), dtype=CDTYPE
        )
        t_rs1 = time.perf_counter()

        self.jump_scale = float(
            1.0 if not rescale_to_trace_nonincreasing else getattr(self, "_last_kraus_scale", 1.0)
        )

        t_d0 = time.perf_counter()
        self.D_accept = self._dual_identity_from_kraus(self.accept_kraus)
        t_d1 = time.perf_counter()

        lam_max = np.max(np.real(la.eigvalsh(self.D_accept)))
        if lam_max > 1.0 + 1e-7:
            raise ValueError(
                "T'†[I] has eigenvalues > 1 after rescaling; decrease proposal strength or margin."
            )

        if self.reject_mode != "explicit":
            raise NotImplementedError("Only reject_mode='explicit' is implemented in v2.")

        t_k0 = time.perf_counter()
        self.reject_kraus = np.ascontiguousarray(
            self._compute_reject_kraus_explicit(self.D_accept)
        )
        t_k1 = time.perf_counter()

        if self.verbose:
            print(
                "[QuantumGibbsSampler v2] dim=", d,
                "beta=", self.beta,
                "num_jumps=", len(self.accept_kraus),
                "max_eig(D)=", float(lam_max),
                "||K||=", float(np.linalg.norm(self.reject_kraus, ord=2)),
            )

        t_wall1 = time.perf_counter()

        if self.collect_timing:
            self.timing.update(
                {
                    "total_init_time": t_wall1 - t_wall0,
                    "h_eigh_time": t_eigh1 - t_eigh0,
                    "gibbs_state_time": t_gibbs1 - t_gibbs0,
                    "coherent_reweigh_time": t_coh1 - t_coh0,
                    "accept_rescale_time": t_rs1 - t_rs0,
                    "D_accept_build_time": t_d1 - t_d0,
                    "reject_kraus_time": t_k1 - t_k0,
                }
            )

    def _build_gibbs_state(self) -> None:
        """
        ρ = exp(-β H) / Z with **shifted** Boltzmann weights for numerical stability.

        ``w_i = exp(-β(E_i - E_min))`` has the same normalized Gibbs state as
        ``exp(-β E_i)`` because rescaling all weights by ``exp(β E_min)`` cancels in ρ.

        ``partition_function_scaled`` = Σ_i w_i = Z · exp(β E_min). The physical
        partition function ``Z = Tr exp(-βH)`` is recovered as
        ``partition_function_scaled * exp(-β E_min)``.
        """
        E = np.asarray(self.energies, dtype=FDTYPE)
        emin = float(np.min(E))
        shifted = E - emin
        w = np.exp(-self.beta * shifted)
        Z_scaled = float(np.sum(w))
        if Z_scaled <= 0.0 or not math.isfinite(Z_scaled):
            raise ValueError("Scaled partition function invalid.")

        self.partition_function_scaled = Z_scaled
        self.partition_function = Z_scaled * math.exp(-self.beta * emin)

        probs = w / Z_scaled
        U = self.eigenvectors
        self.rho = (U * probs) @ U.conj().T
        self.rho = (self.rho + self.rho.conj().T) / 2.0

        ev = np.real(la.eigvalsh(self.rho))
        self.rho_min_eig = float(np.min(ev))
        self.rho_max_eig = float(np.max(ev))
        denom = max(self.rho_min_eig, self.psd_eps)
        self.rho_condition_number = float(self.rho_max_eig / denom)

    @staticmethod
    def _dual_identity_from_kraus(kraus: Sequence[np.ndarray]) -> np.ndarray:
        acc = sum(K.conj().T @ K for K in kraus)
        return (acc + acc.conj().T) / 2.0

    def _rescale_accept_kraus(self, margin: float) -> None:
        D = self._dual_identity_from_kraus(self.accept_kraus)
        lam = float(np.max(np.real(la.eigvalsh(D))))
        target = max(0.0, 1.0 - margin)
        if lam <= target:
            self._last_kraus_scale = 1.0
            return
        c = np.sqrt(target / lam)
        self.accept_kraus = [c * A for A in self.accept_kraus]
        self._last_kraus_scale = float(c)

    def _compute_reject_kraus_explicit(self, D: np.ndarray) -> np.ndarray:
        """K = sqrt( sqrt(ρ) (I - D) sqrt(ρ) ) ρ^{-1/2}  (Theorem 1, Eq. (5))."""
        rho = self.rho
        rho_half = matrix_sqrt_psd(rho, eps=self.psd_eps)
        rho_inv_half = matrix_inv_sqrt_psd(rho, eps=self.psd_eps)
        I = np.eye(self.dim, dtype=CDTYPE)
        M_mid = I - D
        M_mid = (M_mid + M_mid.conj().T) / 2.0
        w_mid, v_mid = la.eigh(M_mid)
        min_w = float(np.min(np.real(w_mid)))
        if min_w < -1e-8:
            if self.strict_psd_checks:
                raise RuntimeError(
                    f"I - D is not PSD within tolerance: min eigenvalue = {min_w}"
                )
            if self.verbose:
                warnings.warn(
                    f"Clamping (I-D) negative eigenvalues (min was {min_w})",
                    UserWarning,
                    stacklevel=2,
                )
            w_mid = np.maximum(np.real(w_mid), 0.0)
        M_mid = (v_mid * w_mid) @ v_mid.conj().T

        inner = rho_half @ M_mid @ rho_half
        inner = (inner + inner.conj().T) / 2.0
        rho_inner_sqrt = matrix_sqrt_psd(inner, eps=self.psd_eps)
        K = rho_inner_sqrt @ rho_inv_half
        return K

    @property
    def bohr_frequency_matrix(self) -> np.ndarray:
        E = self.energies.reshape(-1, 1)
        return self.beta * (E - E.T)

    def channel_diagnostics(self) -> dict[str, float]:
        """
        Validation metrics (expensive fixed-point / spectral checks). Not for hot loops.
        """
        D = self.D_accept
        K = self.reject_kraus
        I = np.eye(self.dim, dtype=CDTYPE)
        dual_sum = D + K.conj().T @ K
        trace_preservation_fro = frobenius_norm(dual_sum - I)

        Mrho = self._apply_channel_fast(self.rho, symmetrize_output=True)
        fixed_point_fro = frobenius_norm(Mrho - self.rho)
        fixed_point_trace_distance = trace_distance(Mrho, self.rho)

        evD = np.real(la.eigvalsh(D))
        evImD = np.real(la.eigvalsh(I - D))

        return {
            "trace_preservation_fro": trace_preservation_fro,
            "fixed_point_fro": fixed_point_fro,
            "fixed_point_trace_distance": fixed_point_trace_distance,
            "rho_min_eig": self.rho_min_eig,
            "rho_max_eig": self.rho_max_eig,
            "rho_condition_number": self.rho_condition_number,
            "D_min_eig": float(np.min(evD)),
            "D_max_eig": float(np.max(evD)),
            "I_minus_D_min_eig": float(np.min(evImD)),
            "jump_scale": self.jump_scale,
            "dim": float(self.dim),
            "num_jumps": float(len(self.accept_kraus)),
            "beta": self.beta,
        }

    def _apply_channel_fast(
        self,
        sigma: np.ndarray,
        *,
        symmetrize_output: Optional[bool] = None,
    ) -> np.ndarray:
        """M[σ] = Σ Ã σ Ã† + K σ K†. Loop over Kraus: each step is a large BLAS matmul."""
        if symmetrize_output is None:
            symmetrize_output = self.symmetrize_output
        sigma = np.ascontiguousarray(np.asarray(sigma, dtype=CDTYPE))
        out = np.zeros_like(sigma, dtype=CDTYPE)
        for A in self.accept_kraus:
            out += A @ sigma @ A.conj().T
        K = self.reject_kraus
        out += K @ sigma @ K.conj().T
        if symmetrize_output:
            out = (out + out.conj().T) / 2.0
        return out

    def apply_channel(
        self,
        sigma: np.ndarray,
        *,
        check: bool = False,
        symmetrize_output: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Apply the CPTP channel M to σ.

        ``symmetrize_output``: default follows :attr:`symmetrize_output`; set ``False``
        for raw-speed benchmarking (output may be slightly non-Hermitian from rounding).
        """
        sigma = np.asarray(sigma, dtype=CDTYPE)
        if sigma.shape != (self.dim, self.dim):
            raise ValueError("sigma has wrong shape.")
        if not _is_hermitian(sigma, atol=self.atol_herm * 10):
            raise ValueError("sigma should be Hermitian (density matrix).")

        out = self._apply_channel_fast(sigma, symmetrize_output=symmetrize_output)

        if check or self.verbose:
            tr = np.trace(out).real
            tr_in = np.trace(sigma).real
            if abs(tr - tr_in) > self.trace_tol and abs(tr_in - 1.0) < self.trace_tol:
                msg = f"apply_channel: trace(out)={tr} deviates from trace(in)={tr_in}"
                if check:
                    raise RuntimeError(msg)
                print("[warn]", msg)
            w = la.eigvalsh(out)
            if np.min(w) < -1e-7:
                msg = f"apply_channel: min eigenvalue {float(np.min(w))}"
                if check:
                    raise RuntimeError(msg)
                print("[warn]", msg)
            if self.verbose:
                rho_applied = self._apply_channel_fast(self.rho, symmetrize_output=symmetrize_output)
                fd = frobenius_norm(rho_applied - self.rho)
                print(f"||M[rho] - rho||_F = {fd}")
        return out

    def run(
        self,
        sigma0: np.ndarray,
        steps: int,
        *,
        observables: Optional[Sequence[np.ndarray]] = None,
        store_states: bool = False,
    ) -> TrajectoryResult:
        """
        Iterate M for ``steps`` steps from ``sigma0``.

        .. warning::

            This evaluates trace distance to ρ **every step** (each call runs an
            eigendecomposition — **O(d³)** per step). For performance benchmarking,
            prefer :meth:`run_timed` with ``diagnostic_every`` set or distances disabled.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        sigma = np.asarray(sigma0, dtype=CDTYPE)
        sigma = (sigma + sigma.conj().T) / 2.0
        tr0 = np.trace(sigma).real
        if abs(tr0 - 1.0) > 1e-6:
            sigma = sigma / tr0

        obs_list = list(observables) if observables is not None else None
        tdist: List[float] = []
        fdist: List[float] = []
        exps: List[Optional[List[float]]] = []
        states: Optional[List[np.ndarray]] = [] if store_states else None

        for _ in range(steps):
            if store_states and states is not None:
                states.append(sigma.copy())
            tdist.append(trace_distance(sigma, self.rho))
            fdist.append(frobenius_norm(sigma - self.rho))
            if obs_list is None:
                exps.append(None)
            else:
                exps.append([expectation_value(sigma, O) for O in obs_list])
            sigma = self._apply_channel_fast(sigma)

        if store_states and states is not None:
            states.append(sigma.copy())
        tdist.append(trace_distance(sigma, self.rho))
        fdist.append(frobenius_norm(sigma - self.rho))
        if obs_list is None:
            exps.append(None)
        else:
            exps.append([expectation_value(sigma, O) for O in obs_list])

        return TrajectoryResult(
            trace_dist=tdist,
            frobenius_dist=fdist,
            expectations=exps,
            states=states,
        )

    def run_timed(
        self,
        sigma0: np.ndarray,
        steps: int,
        *,
        observables: Optional[Sequence[np.ndarray]] = None,
        diagnostic_every: Optional[int] = None,
        compute_trace_distance: bool = False,
        compute_frobenius_distance: bool = False,
        store_states: bool = False,
        symmetrize_each_step: Optional[bool] = None,
    ) -> tuple[TrajectoryResult, dict[str, float]]:
        """
        Same snapshot ordering as :meth:`run` (``steps + 1`` rows): row ``k`` uses
        σ after ``k`` channel applications (row ``0`` = initial state).

        Expensive trace / Frobenius distances and observable traces run only when
        the schedule says so; other rows store ``nan`` for distances and ``None``
        for expectations.

        **Schedule**

        - ``diagnostic_every = K > 0``: compute requested distances at rows where
          ``k % K == 0`` or ``k == steps`` (always include the final state row).
        - ``diagnostic_every is None``: if distance flags are set, compute distances
          only at ``k ∈ {0, steps}``; otherwise all distance slots are ``nan``.
        Observables use the same schedule whenever ``observables`` is not ``None``.

        ``apply_channel_time`` sums only :meth:`_apply_channel_fast` calls.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        if diagnostic_every is not None and diagnostic_every <= 0:
            raise ValueError("diagnostic_every must be positive or None.")

        sym = self.symmetrize_output if symmetrize_each_step is None else symmetrize_each_step

        t_run0 = time.perf_counter()
        t_apply = 0.0
        t_diag = 0.0
        t_obs = 0.0

        sigma = np.asarray(sigma0, dtype=CDTYPE)
        sigma = (sigma + sigma.conj().T) / 2.0
        tr0 = np.trace(sigma).real
        if abs(tr0 - 1.0) > 1e-6:
            sigma = sigma / tr0

        obs_list = list(observables) if observables is not None else None
        want_dist = compute_trace_distance or compute_frobenius_distance

        def snapshot_active(k: int) -> bool:
            """Whether row k gets real distance / observable work (not only nan)."""
            if diagnostic_every is not None:
                return (k % diagnostic_every == 0) or (k == steps)
            if want_dist or obs_list is not None:
                return k == 0 or k == steps
            return False

        def observable_active(k: int) -> bool:
            if obs_list is None:
                return False
            return snapshot_active(k)

        #Trace distances
        tdist: List[float] = []
        #Frobenius distances
        fdist: List[float] = []
        exps: List[Optional[List[float]]] = []
        states: Optional[List[np.ndarray]] = [] if store_states else None

        for k in range(steps):
            if store_states and states is not None:
                states.append(sigma.copy())

            if snapshot_active(k) and want_dist:
                td0 = time.perf_counter()
                if compute_trace_distance:
                    tdist.append(trace_distance(sigma, self.rho))
                else:
                    tdist.append(float("nan"))
                if compute_frobenius_distance:
                    fdist.append(frobenius_norm(sigma - self.rho))
                else:
                    fdist.append(float("nan"))
                t_diag += time.perf_counter() - td0
            else:
                tdist.append(float("nan"))
                fdist.append(float("nan"))

            if observable_active(k):
                to0 = time.perf_counter()
                exps.append([expectation_value(sigma, O) for O in obs_list])
                t_obs += time.perf_counter() - to0
            else:
                exps.append(None)

            ta0 = time.perf_counter()
            sigma = self._apply_channel_fast(sigma, symmetrize_output=sym)
            t_apply += time.perf_counter() - ta0

        if store_states and states is not None:
            states.append(sigma.copy())

        k = steps
        if snapshot_active(k) and want_dist:
            td0 = time.perf_counter()
            if compute_trace_distance:
                tdist.append(trace_distance(sigma, self.rho))
            else:
                tdist.append(float("nan"))
            if compute_frobenius_distance:
                fdist.append(frobenius_norm(sigma - self.rho))
            else:
                fdist.append(float("nan"))
            t_diag += time.perf_counter() - td0
        else:
            tdist.append(float("nan"))
            fdist.append(float("nan"))

        if observable_active(k):
            to0 = time.perf_counter()
            exps.append([expectation_value(sigma, O) for O in obs_list])
            t_obs += time.perf_counter() - to0
        else:
            exps.append(None)

        t_run1 = time.perf_counter()

        timing_run = {
            "steps": float(steps),
            "apply_channel_time": t_apply,
            "time_per_step": t_apply / max(steps, 1),
            "diagnostics_time": t_diag,
            "observable_time": t_obs,
            "total_run_time": t_run1 - t_run0,
            "dim": float(self.dim),
            "num_jumps": float(len(self.accept_kraus)),
        }

        return TrajectoryResult(
            trace_dist=tdist,
            frobenius_dist=fdist,
            expectations=exps,
            states=states,
        ), timing_run

    def run_until_converged(
        self,
        sigma0: np.ndarray,
        *,
        target_trace_distance: float,
        max_steps: int,
        step_cutoffs: Optional[Sequence[int]] = None,
        symmetrize_each_step: Optional[bool] = None,
    ) -> "ConvergenceResult":
        """Iterate ``M`` from ``sigma0`` until ``D(σ_k, ρ) ≤ target_trace_distance``."""
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if target_trace_distance <= 0:
            raise ValueError("target_trace_distance must be positive.")

        sym = self.symmetrize_output if symmetrize_each_step is None else symmetrize_each_step

        cutoffs = sorted({int(c) for c in step_cutoffs}) if step_cutoffs else []
        if any(c < 0 for c in cutoffs):
            raise ValueError("step_cutoffs must be non-negative.")
        cutoff_states: dict[int, np.ndarray] = {}

        sigma = np.ascontiguousarray(np.asarray(sigma0, dtype=CDTYPE))
        sigma = (sigma + sigma.conj().T) / 2.0
        tr0 = np.trace(sigma).real
        if abs(tr0 - 1.0) > 1e-6 and tr0 > 0:
            sigma = sigma / tr0

        t_apply = 0.0
        t_check = 0.0
        t_run0 = time.perf_counter()

        tc0 = time.perf_counter()
        dist = trace_distance(sigma, self.rho)
        t_check += time.perf_counter() - tc0

        steps_to_converge = 0
        converged = dist <= target_trace_distance
        if 0 in cutoffs:
            cutoff_states[0] = sigma.copy()
        final_distance = dist

        if not converged:
            for step in range(1, int(max_steps) + 1):
                ta0 = time.perf_counter()
                sigma = self._apply_channel_fast(sigma, symmetrize_output=sym)
                tr = np.trace(sigma).real
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

        t_run1 = time.perf_counter()
        timing_run = {
            "steps_to_converge": float(steps_to_converge),
            "converged": float(converged),
            "final_trace_distance": float(final_distance),
            "channel_apply_time": t_apply,
            "convergence_check_time": t_check,
            "total_run_time": t_run1 - t_run0,
            "dim": float(self.dim),
            "num_jumps": float(len(self.accept_kraus)),
        }
        return ConvergenceResult(
            steps_to_converge=steps_to_converge,
            converged=converged,
            final_trace_distance=float(final_distance),
            converged_state=converged_state,
            cutoff_states=cutoff_states,
            timing=timing_run,
        )


def channel_superoperator_matrix(
    sampler: "QuantumGibbsSampler",
    *,
    symmetrize_output: bool = False,
) -> np.ndarray:
    """
    Matrix ``S`` such that ``vec(Φ[σ]) = S @ vec(σ)`` (column-major vectorization).

    Built by applying the channel to each matrix basis element ``|i⟩⟨j|``.
    """
    d = sampler.dim
    apply = sampler._apply_channel_fast
    cols: list[np.ndarray] = []
    for j in range(d):
        for i in range(d):
            E = np.zeros((d, d), dtype=CDTYPE)
            E[i, j] = 1.0
            out = apply(E, symmetrize_output=symmetrize_output)
            cols.append(out.reshape(d * d, order="F"))
    return np.column_stack(cols)


def channel_superoperator_spectral_gap(
    sampler: "QuantumGibbsSampler",
    *,
    unit_tol: float = 1e-5,
) -> float:
    """
    Discrete-time mixing gap ``γ = 1 − max{|λ| : λ ∈ spec(S), |λ − 1| > unit_tol}``.

    For a trace-preserving quantum channel with unique stationary state, ``γ > 0``
    implies geometric convergence at rate ``(1 − γ)^k`` in the usual operator norm
    on the traceless subspace (cf. classical ``1 − λ_*`` gap on a stochastic matrix).
    """
    S = channel_superoperator_matrix(sampler)
    mods = np.abs(np.linalg.eigvals(S))
    non_unit = mods[mods <= 1.0 - unit_tol]
    if non_unit.size == 0:
        return 0.0
    return float(1.0 - np.max(non_unit))


def mixing_curve_trace_distances(
    sampler: "QuantumGibbsSampler",
    sigma0: np.ndarray,
    max_steps: int,
) -> list[float]:
    """
    ``D(σ_k, ρ)`` for ``k = 0, …, max_steps`` using the same per-step trace
    renormalization as :meth:`run_until_converged`.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative.")
    sigma = np.ascontiguousarray(np.asarray(sigma0, dtype=CDTYPE))
    sigma = (sigma + sigma.conj().T) / 2.0
    tr0 = float(np.trace(sigma).real)
    if tr0 > 0.0:
        sigma = sigma / tr0

    dists: list[float] = [float(trace_distance(sigma, sampler.rho))]
    for _ in range(int(max_steps)):
        sigma = sampler._apply_channel_fast(sigma)
        tr = float(np.trace(sigma).real)
        if not math.isfinite(tr) or tr <= 0.0:
            break
        if abs(tr - 1.0) > 1e-12:
            sigma = sigma / tr
        dists.append(float(trace_distance(sigma, sampler.rho)))
    return dists


__all__ = [
    "CDTYPE",
    "FDTYPE",
    "QuantumGibbsSampler",
    "TrajectoryResult",
    "ConvergenceResult",
    "jumps_from_hermitian_matrices",
    "trace_distance",
    "frobenius_norm",
    "expectation_value",
    "print_threadpool_info",
    "bohr_weight_matrix",
    "S_operator_bohr_weight",
    "default_coherent_reweigh_weight",
    "operator_to_energy_basis",
    "operator_from_energy_basis",
    "apply_bohr_elementwise_weight",
    "coherent_reweigh_jump",
    "hermitian_random_jumps",
    "pauli_x_site",
    "local_pauli_x_set",
    "normalized_pauli_x_proposal",
    "matrix_sqrt_psd",
    "matrix_inv_sqrt_psd",
    "channel_superoperator_matrix",
    "channel_superoperator_spectral_gap",
    "mixing_curve_trace_distances",
]
