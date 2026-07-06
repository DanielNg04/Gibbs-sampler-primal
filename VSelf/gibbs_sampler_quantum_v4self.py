"""
Classical dense-matrix simulator for the discrete-time quantum Metropolis channel
(Gilyén et al., arXiv:2405.20322):
- coherent accept + single Kraus reject (Thm. 1, Eq. (5)) with coherent Bohr reweighing (Sec. 1.4).
- Target ρ = exp(−H)/Tr exp(−H).
"""

from typing import Callable, List, Optional, Sequence

import numpy as np
import scipy.linalg as la
import math
import time
from dataclasses import dataclass

DTYPE = np.float64

# --- Small Linear Algebra Utilities ---
def _symmetrize(M: np.ndarray) -> np.ndarray:
    return (M + M.T) * 0.5

def prepare_matrix(H: np.ndarray) -> np.ndarray:
    #Store in contiguous array
    A = np.ascontiguousarray(H, dtype=DTYPE)

    #If not symmetric, symmetrize
    if not np.allclose(A, A.T):
        A = _symmetrize(A)
    return A

def coherent_S_minus_weight(nu: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.tanh(0.25 * nu))

def bohr_weight_matrix(energies: np.ndarray) -> np.ndarray:
    """
    W_ij = f(β(E_i − E_j)) for all energy pairs at once
    """
    E = np.asarray(energies, dtype=DTYPE).reshape(-1, 1)    #Converts energies into column vector
    nu = (E - E.T)
    W = np.asarray(coherent_S_minus_weight(nu), dtype=DTYPE)
    return W

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
    d = _symmetrize(np.asarray(rho, dtype=DTYPE) - np.asarray(sigma, dtype=DTYPE))
    return float(0.5 * np.sum(np.abs(la.eigvalsh(d, check_finite=False))))


def frobenius_norm(M: np.ndarray) -> float:
    """‖M‖_F via ``np.linalg.norm``: a single fused BLAS ``nrm2``-style pass, O(d²)."""
    return float(np.linalg.norm(M, ord="fro"))

# --- Result dataclasses
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

@dataclass
class TrajectoryResult:
    trace_dist: List[float]
    frobenius_dist: List[float]
    states: Optional[List[np.ndarray]] = None

# --- Proposal Jumps ---
def jumps_from_symmetric_matrices(
    mats: Sequence[np.ndarray],
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
    channel);
    ``drop_zero`` skips matrices that would be no-op Kraus slots.

    Speed/stability: per matrix this is one O(d²) norm and one scaling — there is
    nothing to optimize further; the Frobenius norm is computed by a stable fused
    BLAS reduction.
    """
    out: List[np.ndarray] = []
    for k, M in enumerate(mats):
        A = prepare_matrix(M)
        nrm = float(np.linalg.norm(A))
        if drop_zero and nrm <= zero_tol:
            continue
        if normalize and nrm > zero_tol:
            A = A / nrm
        out.append(np.ascontiguousarray(A))
    if not out:
        raise ValueError("No usable (non-zero) jump matrices were provided.")
    return out


# --- Main Sampler ---
class QuantumGibbsSampler:
    """
    Discrete-time quantum detailed-balanced channel ,
    specialized to real symmetric H and jumps.
    """

    def __init__(
            self,
            H: np.ndarray,
            jumps: Sequence[np.ndarray],
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

        #Prepare H and A_j jump matrices.
        self.H = prepare_matrix(H)
        d = self.H.shape[0]
        self.dim = d

        jump_list = [prepare_matrix(A) for A in jumps]

        q = len(jump_list)
        self.num_jumps = q

        #1. The single diagonalization of H
            #driver="evd" (LAPACK dsyevd, divide & conquer) is backward stable and the
            #fastest full-spectrum symmetric driver for large dense matrices.
            #check_finite=False skips an O(d²) scan; H was already validated above.
        self.energies, self.eigenvectors = la.eigh(self.H, driver="evd", check_finite=False)
        E, U = self.energies, self.eigenvectors
        #E: ndarray(N), eigenvalues in ascending order
        #U: ndarray(M, N), the i-th column is the normalized eigenvector corresponding to E[i]

        # 2. Calculate the exact Gibbs state (With stabilization trick - no underflow)
            # Shifted Boltzmann weights w_i = exp(−β(E_i − E_min))
            # The common factor exp(β E_min) cancels in ρ, the largest weight is exactly 1
            # There is no overflow and no catastrophic underflow of the *relative* weights.

        emin = float(E[0])
        w = np.exp(-(E - emin))
        Z_scaled = float(np.sum(w))
        probs = w / Z_scaled

        # ρ = U diag(p) Uᵀ --> (O(d²)) + one GEMM
        self.rho = _symmetrize((U * probs) @ U.T)

        #Calculate condition number for diagnostics
        self.rho_min_eig = float(np.min(probs))
        self.rho_max_eig = float(np.max(probs))
        self.rho_condition_number = self.rho_max_eig / max(self.rho_min_eig, self.psd_eps)

        #3. Coherent Bohr reweighing, batched in the energy basis.
            # Ã = U @ ( W ⊙ ( U^T @ A @ U)) @ U^T
            # Rotate all q jumps with two batched GEMMs at once
            # O(q·d²) + 2 GEMM
        stack_E = U.T @ np.stack(jump_list) @ U
        W = bohr_weight_matrix(E)
        accept_E = W[None, :, :] * stack_E

        #4. Safety rescaling so that D = Σ Ã^T Ã satisfies λ_max(D) ≤ 1 − margin.
            # tensordot contracts D_jl = Σ_{a,i} Ã[a,i,j] Ã[a,i,l] as ONE large GEMM of shape (d, q·d)·(q·d, d)
            # As a Gram matrix the result is symmetric PSD by construction (stable — no cancellation can make it indefinite beyond rounding).
            # Rescaling by c multiplies D by c² exactly, so D and its eigenvalues are updated in place instead of being recomputed.
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


        # 5. Reject Kraus K = sqrt(ρ^{1/2}(I−D)ρ^{1/2}) ρ^{−1/2},
            # assembled in the energy basis where ρ = diag(p):
            #   ρ^{1/2}(I−D)ρ^{1/2}  →  M_ij = √p_i √p_j (I−D)_ij   (O(d²) scaling),
            #   ρ^{−1/2}             →  column scaling by 1/√p_j     (O(d²)).
            # Only the outer square root needs an eigendecomposition — (O(d³)).

            # Stability details:
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

        #6. One contiguous Kraus stack (accept ops + reject op), rotated back
        # to the input basis with two batched GEMMs. A single C-contiguous block
        # lets _apply_channel_fast run the whole channel as batched BLAS calls.
        kraus_E = np.concatenate([accept_E, K_E[None, :, :]], axis=0)
        self.kraus_stack = np.ascontiguousarray(U @ kraus_E @ U.T, dtype=DTYPE)
        self.accept_kraus = self.kraus_stack[:q]
        self.reject_kraus = self.kraus_stack[q]
        self.D_accept = _symmetrize(U @ D_E @ U.T)

        if self.verbose:
            print(
                "[QuantumGibbsSampler] dim=", d,
                "num_jumps=", q,
                "Condition number of rho=", self.rho_condition_number,
                "max_eig(D)=", lam_max,
                "jump_scale=", self.jump_scale,
            )

    def _apply_channel_fast(self, sigma: np.ndarray, symmetrize_output: Optional[bool] = None) -> np.ndarray:
        """
        M[σ] = Σ_a A_a σ A_aᵀ + K σ Kᵀ — the hot loop, as two BLAS-bound steps:

        1. ``tmp = kraus_stack @ σ``: a batched GEMM, (q+1) independent d×d
           multiplies dispatched straight to multithreaded dgemm.
        2. ``tensordot(tmp, kraus_stack, axes=([0, 2], [0, 2]))``: contracts both
           the Kraus index and the inner matrix index in ONE large GEMM of shape
           (d, (q+1)d)·((q+1)d, d). out[i,k] = Σ_{a,j} (A_a σ)[i,j]·A_a[k,j],
           i.e. exactly Σ_a A_a σ A_aᵀ, with the accumulation over Kraus
           operators fused into the GEMM.

        Stability: every product is a plain GEMM (componentwise backward stable);
        the final symmetrization removes the O(eps) asymmetry so iterated states
        stay in the symmetric cone and downstream ``eigvalsh`` calls stay exact.
        """
        if symmetrize_output is None:
            symmetrize_output = self.symmetrize_output
        sigma = np.ascontiguousarray(sigma, dtype=DTYPE)
        tmp = self.kraus_stack @ sigma
        out = np.tensordot(tmp, self.kraus_stack, axes=([0, 2], [0, 2]))
        return _symmetrize(out) if symmetrize_output else out

    def apply_channel(self, sigma: np.ndarray, check: bool = False) -> np.ndarray:
        """
        Validated single application of the CPTP map M (use ``_apply_channel_fast``
        inside loops — the validation here costs an extra O(d²), and the optional
        ``check`` adds an O(d³) eigenvalue scan).
        """
        sigma = prepare_matrix(sigma)

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

# --- Run
    def _coerce_state(self, sigma0: np.ndarray) -> np.ndarray:
        """Symmetrize and trace-normalize an initial state (boundary only)."""

        sigma = prepare_matrix(sigma0)
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
        """
        Iterate M for ``steps`` steps from ``sigma0``, recording distances to ρ.
        ... warning::
            Both distances are evaluated **every step** and the trace distance
            costs an O(d³) eigenvalue call each time.
        """
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

        return TrajectoryResult(
            trace_dist=tdist, frobenius_dist=fdist, states=states
        )

    def run_until_converged(
            self,
            sigma0: np.ndarray,
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




