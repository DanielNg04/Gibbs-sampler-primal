# run_test_v2self.py — minimal benchmark driver for the VSelf primal oracle.

"""
Small-instance benchmark: MCMC oracle run + Gibbs-steps plot per instance.

Pipeline (same shape as the full ``run_test.py``, reduced to its essentials):
1. Load each converted instance (``SDP_problems_converted/<name>.npz``).
2. Use the lowest bracket threshold g = g_lo (comfortably feasible; the
   bisection calibration :func:`calibrate_g` stays available but is not used —
   on these instances the iteration count is set by θ = ε/(2R), not by g).
3. Run the primal oracle in MCMC mode (channel-based Gibbs preparation).
4. Plot Gibbs channel steps per oracle iteration and save the raw run JSON.

The instance list holds every converted problem that finishes at g_lo within
the budget (measured by exact-mode probes): hinf12 ~6 s, hinf1 ~6 s,
truss1 ~52 s, truss4 ~105 s. Excluded because θ = ε/(2R) makes them run for
hours at this precision: hinf2/hinf3/hinf4 (θ ≤ 1e-4), truss2/truss3
(θ ≈ 1e-3), and control1/control2 (not θ-feasible at g_lo even after 200k
exact-mode iterations). A wall-clock budget additionally skips remaining
instances instead of overrunning.

Single process: BLAS keeps all its threads (no pinning — that is only needed
when fanning out worker processes).
"""

import json
import os
import time

import numpy as np
import scipy.linalg as la

from primal_oracle_quantum_v2self import PrimalOracleProblem, run_primal_oracle

# --- Configuration (constants for now; grow into CLI arguments later) ---

INSTANCES = ["hinf12", "hinf1", "truss1", "truss4"]
TIME_BUDGET_S = 12 * 60        # stop starting new instances past this wall time
EPSILON = 0.1                  # SDP precision ε; oracle tolerance θ = ε/(2R)
MAX_ORACLE_ITERS = 200_000     # cap on primal-oracle iterations (θ can be ~2e-3)
GIBBS_MAX_STEPS = 2_000        # cap on channel applications per Gibbs preparation
GIBBS_WARM_START = True        # start each channel run from the previous endpoint

# Channel-gap diagnostic: record the superoperator spectral gap every this many
# oracle iterations (the O(n⁶) eigendecomposition is too costly per iteration).
# Strides sized from the known iteration counts to give ~100-200 samples each.
GAP_STRIDE = {"hinf12": 1, "hinf1": 20, "truss1": 200, "truss4": 250}
GAP_STRIDE_DEFAULT = 100

CALIB_TARGET_ITERS = 500       # calibration: aim for <= this many oracle iterations
CALIB_SEARCH_STEPS = 18        # calibration: bisection depth (bracket shrinks 2^18×)

_HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTED_DIR = os.path.join(os.path.dirname(_HERE), "SDP_problems_converted")
RESULTS_DIR = os.path.join(_HERE, "results")


# --- Instance loading ---

def load_instance(name: str) -> dict:
    """Load a converted instance .npz into plain arrays/scalars."""
    d = np.load(os.path.join(CONVERTED_DIR, f"{name}.npz"), allow_pickle=True)
    return {
        "name": str(d["name"]),
        "C": np.ascontiguousarray(d["C"]),
        "A": np.ascontiguousarray(d["A"]),
        "b": np.ascontiguousarray(d["b"]),
        "R": float(d["R"]),
        "n": int(d["n"]),
        "opt": float(d["opt"]),
    }


def build_problem(arrays: dict, g: float) -> PrimalOracleProblem:
    A = arrays["A"]
    return PrimalOracleProblem(
        A_matrices=[A[k] for k in range(A.shape[0])],
        b=arrays["b"],
        C=arrays["C"],
        R=arrays["R"],
        g=g,
    )


def cycle_adjacency_from_permutation(pi: np.ndarray) -> np.ndarray:
    """
    Adjacency of the random cycle from Gilyén–Vazirani Appendix C, Definition 6
    (arXiv:2011.09495): sample π on [n], connect i–j iff π(i) − π(j) ≡ ±1 (mod n).
    """
    pi = np.asarray(pi, dtype=np.intp)
    n = pi.shape[0]
    inv = np.empty(n, dtype=np.intp)
    inv[pi] = np.arange(n)
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for step in (-1, 1):
            j = int(inv[(int(pi[i]) + step) % n])
            adj[i, j] = 1.0
    return adj


def random_cycle_adjacency(rng: np.random.Generator, n: int) -> np.ndarray:
    """One independent uniformly random cycle (Definition 6) on vertex set [n]."""
    pi = rng.permutation(n)
    return cycle_adjacency_from_permutation(pi)


def jump_matrices_from_arrays(arrays: dict) -> list[np.ndarray]:
    """
    Channel proposals = SDP objective + distinct constraint generators
    + two random-cycle connectors (Appendix C, Definition 6).

    The converted stack is [I, F_1, −F_1, F_2, −F_2, …]: the identity is
    skipped (after Bohr reweighing it is ∝ I and drives no transitions) and
    only the +F_i at odd indices are kept (−F_i shares its generator up to
    sign, which the channel's ÃᵀÃ structure makes irrelevant).

    Two independent random cycles (Gilyén–Vazirani arXiv:2011.09495, App. C,
    Def. 6–7) replace the earlier dense GOE connector: each cycle is a sparse
    2-regular graph on [n], and their union is the standard building block for
    random regular expanders. This restores ergodicity when problem matrices
    share a block-diagonal sparsity pattern (see BLOCK_DIAGONAL_ISSUE.md).
  """
    A = arrays["A"]
    n = arrays["n"]
    rng = np.random.default_rng(0)          # fixed seed → reproducible channel
    cycle1 = random_cycle_adjacency(rng, n)
    cycle2 = random_cycle_adjacency(rng, n)
    return [arrays["C"]] + [A[k] for k in range(1, A.shape[0], 2)] + [cycle1, cycle2]


# --- Objective threshold g ---

def g_bracket(arrays: dict) -> tuple[float, float]:
    """
    Bracket [g_lo, g_hi] for the threshold search.

    g_hi = R·λ_max(C) is provably unreachable: Tr(CX) ≤ λ_max(C)·Tr(X) ≤ R·λ_max(C)
    for every feasible X. g_lo sits below the known optimum by a full |OPT|+1
    margin, so the oracle is comfortably feasible there.
    """
    lam_max = float(la.eigvalsh(arrays["C"], check_finite=False)[-1])
    g_hi = arrays["R"] * lam_max
    base = arrays["opt"] if np.isfinite(arrays["opt"]) else 0.0
    g_lo = min(base, 0.0) - (abs(base) + 1.0)
    return g_lo, g_hi


def _exact_iterations_for_g(arrays: dict, g: float, eps: float, cap: int) -> int:
    """
    Oracle iterations to reach θ-feasibility at threshold g, using the cheap
    closed-form Gibbs state (identical y-trajectory to MCMC mode, ~100× faster).
    Returns cap + 1 when infeasible within cap, so harder-than-cap and
    infeasible thresholds sort above every acceptable one.
    """
    res = run_primal_oracle(
        build_problem(arrays, g), eps,
        gibbs_mode="exact", max_iterations=cap, return_on_exhaustion=True,
    )
    if res is None or not res.constraint_diag.feasible_within_theta:
        return cap + 1
    return int(res.iterations)


def calibrate_g(arrays: dict, eps: float, target_iters: int, cap: int) -> float:
    """
    Largest g whose exact-mode run is θ-feasible within ``target_iters``.

    Valid bisection because iterations-to-feasibility is monotone in g
    (raising the threshold only shrinks the feasible set). Implemented for the
    full benchmark; the current specialized run uses g_lo instead.
    """
    lo, hi = g_bracket(arrays)
    chosen = lo
    for _ in range(CALIB_SEARCH_STEPS):
        mid = 0.5 * (lo + hi)
        if _exact_iterations_for_g(arrays, mid, eps, cap) > target_iters:
            hi = mid
        else:
            chosen, lo = mid, mid
    return chosen


# --- Plotting ---

def plot_gibbs_steps(steps: list[int], instance: str, g: float, out_path: str) -> None:
    """Dot plot of channel steps per oracle iteration, with mean/max lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(steps, dtype=float)
    x = np.arange(1, y.size + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x, y, s=14, color="#2E86AB", alpha=0.78, edgecolors="none", zorder=2)
    ax.axhline(float(np.mean(y)), color="#3D405B", linestyle="--", linewidth=1.5,
               label=f"mean = {float(np.mean(y)):.1f}")
    ax.axhline(float(np.max(y)), color="#81B29A", linestyle="-.", linewidth=1.5,
               label=f"max = {float(np.max(y)):.0f}")
    ax.set_xlabel("Primal-oracle iteration")
    ax.set_ylabel("Gibbs-sampler steps to convergence")
    ax.set_title(f"{instance}: MCMC steps per oracle iteration (g = {g:.4g}, ε = {EPSILON:g})")
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_channel_gap(gap_pairs: list[tuple[int, float]], instance: str, g: float,
                     out_path: str) -> None:
    """Spectral gap of the constructed channel vs oracle iteration (strided)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([p[0] for p in gap_pairs], dtype=float)
    y = np.asarray([p[1] for p in gap_pairs], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, color="#E07A5F", linewidth=1.2, alpha=0.6, zorder=1)
    ax.scatter(x, y, s=14, color="#E07A5F", edgecolors="none", zorder=2)
    ax.axhline(float(np.mean(y)), color="#3D405B", linestyle="--", linewidth=1.5,
               label=f"mean = {float(np.mean(y)):.4f}")
    ax.set_xlabel("Primal-oracle iteration")
    ax.set_ylabel("Channel spectral gap  1 − |λ₂|")
    ax.set_title(f"{instance}: channel gap per oracle iteration (g = {g:.4g}, ε = {EPSILON:g})")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- Driver ---

def run_instance(name: str) -> dict:
    """One MCMC oracle run at g = g_lo; writes run.json + plot, returns summary."""
    arrays = load_instance(name)
    g_lo, g_hi = g_bracket(arrays)
    g = g_lo    # user choice: the lowest (comfortably feasible) threshold
    theta = EPSILON / (2.0 * arrays["R"])
    print(f"[{name}] n={arrays['n']} R={arrays['R']:.4g} opt={arrays['opt']:.6g} "
          f"theta={theta:.4g}; g bracket = [{g_lo:.6g}, {g_hi:.6g}], using g = {g:.6g}")

    t0 = time.perf_counter()
    result = run_primal_oracle(
        build_problem(arrays, g),
        EPSILON,
        max_iterations=MAX_ORACLE_ITERS,
        collect_timing=True,
        gibbs_mode="mcmc",
        gibbs_jump_matrices=jump_matrices_from_arrays(arrays),
        gibbs_max_steps=GIBBS_MAX_STEPS,
        gibbs_warm_start=GIBBS_WARM_START,
        gibbs_gap_stride=GAP_STRIDE.get(name, GAP_STRIDE_DEFAULT),
        return_on_exhaustion=True,
    )
    wall = time.perf_counter() - t0

    steps = result.gibbs_steps_per_iter
    gaps = result.gibbs_gap_per_iter or []
    gap_vals = [gp for _, gp in gaps]
    feasible = bool(result.constraint_diag.feasible_within_theta)
    print(f"[{name}] oracle iterations = {result.iterations}, feasible = {feasible}, "
          f"z = {result.z:.6g}")
    print(f"[{name}] Gibbs steps per iteration: mean = {np.mean(steps):.1f}, "
          f"max = {max(steps)}, all converged = {all(result.gibbs_converged_per_iter)}")
    if gap_vals:
        print(f"[{name}] channel gap ({len(gap_vals)} samples): "
              f"first = {gap_vals[0]:.4f}, last = {gap_vals[-1]:.4f}, "
              f"min = {min(gap_vals):.4f}")
    print(f"[{name}] wall time = {wall:.2f}s "
          f"(gibbs {result.timing['gibbs_time']:.1f}s of "
          f"{result.timing['total_wall_time']:.1f}s)")

    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    raw = {
        "instance": name,
        "epsilon": EPSILON,
        "g": g,
        "g_bracket": [g_lo, g_hi],
        "R": arrays["R"],
        "iterations": int(result.iterations),
        "feasible": feasible,
        "theta": float(result.theta),
        "z": float(result.z),
        "omega": float(result.omega),
        "gibbs_warm_start": GIBBS_WARM_START,
        "gibbs_steps_per_iter": [int(s) for s in steps],
        "gibbs_converged_per_iter": [bool(c) for c in result.gibbs_converged_per_iter],
        "gap_stride": GAP_STRIDE.get(name, GAP_STRIDE_DEFAULT),
        "channel_gap_per_iter": [[int(i), float(gp)] for i, gp in gaps],
        "timing": result.timing,
        "wall_time": wall,
    }
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)

    plot_path = os.path.join(out_dir, "gibbs_steps_per_iteration.png")
    plot_gibbs_steps(steps, name, g, plot_path)
    gap_plot_path = os.path.join(out_dir, "channel_gap_per_iteration.png")
    if gaps:
        plot_channel_gap(gaps, name, g, gap_plot_path)
    print(f"[{name}] wrote {out_dir}\\run.json, {plot_path} and {gap_plot_path}")

    return {
        "instance": name,
        "n": arrays["n"],
        "iterations": int(result.iterations),
        "feasible": feasible,
        "gibbs_steps_mean": float(np.mean(steps)),
        "gibbs_steps_max": int(max(steps)),
        "gap_min": float(min(gap_vals)) if gap_vals else None,
        "wall_time_s": wall,
    }


def main() -> None:
    t_suite = time.perf_counter()
    summaries: list[dict] = []
    for name in INSTANCES:
        elapsed = time.perf_counter() - t_suite
        if elapsed > TIME_BUDGET_S:
            print(f"[suite] time budget ({TIME_BUDGET_S}s) reached after {elapsed:.0f}s; "
                  f"skipping remaining instances: {INSTANCES[INSTANCES.index(name):]}")
            break
        summaries.append(run_instance(name))
        print()

    print("=== suite summary ===")
    print(f"{'instance':<12} {'n':>4} {'iters':>8} {'feasible':>8} "
          f"{'mean_st':>8} {'max_st':>7} {'gap_min':>8} {'time_s':>8}")
    for s in summaries:
        gap_txt = f"{s['gap_min']:.4f}" if s["gap_min"] is not None else "-"
        print(f"{s['instance']:<12} {s['n']:>4} {s['iterations']:>8} "
              f"{str(s['feasible']):>8} {s['gibbs_steps_mean']:>8.2f} "
              f"{s['gibbs_steps_max']:>7} {gap_txt:>8} {s['wall_time_s']:>8.1f}")
    print(f"total wall time = {time.perf_counter() - t_suite:.1f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "suite_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)


if __name__ == "__main__":
    main()
