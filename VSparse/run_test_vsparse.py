# run_test_vsparse.py — minimal benchmark driver for the VSparse primal oracle.

"""
Same benchmark shape as ``VSelf/run_test_v2self.py``, using the CSR backend.

Pipeline:
1. Load each converted instance (``SDP_problems_converted/<name>.npz``).
2. Convert C and A to CSR at load time.
3. Build cycle-connector jumps as CSR directly (O(n) nonzeros).
4. Run the primal oracle in MCMC mode; save results under ``VSparse/results/``.
"""

import json
import os
import time

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix

from gibbs_sampler_quantum_vsparse import validate_matrix
from primal_oracle_quantum_vsparse import PrimalOracleProblem, run_primal_oracle

# --- Configuration ---

INSTANCES = ["hinf12", "hinf1", "truss1", "truss4"]
TIME_BUDGET_S = 12 * 60
EPSILON = 0.1
MAX_ORACLE_ITERS = 200_000
GIBBS_MAX_STEPS = 2_000
GIBBS_WARM_START = True

GAP_STRIDE = {"hinf12": 1, "hinf1": 20, "truss1": 200, "truss4": 250}
GAP_STRIDE_DEFAULT = 100

CALIB_TARGET_ITERS = 500
CALIB_SEARCH_STEPS = 18

_HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTED_DIR = os.path.join(os.path.dirname(_HERE), "SDP_problems_converted")
RESULTS_DIR = os.path.join(_HERE, "results")


# --- Instance loading ---

def load_instance(name: str) -> dict:
    """Load a converted instance .npz; store C and A as CSR."""
    d = np.load(os.path.join(CONVERTED_DIR, f"{name}.npz"), allow_pickle=True)
    C = validate_matrix(np.ascontiguousarray(d["C"]))
    A_dense = np.ascontiguousarray(d["A"])
    A_csr = [validate_matrix(A_dense[k]) for k in range(A_dense.shape[0])]
    return {
        "name": str(d["name"]),
        "C": C,
        "A": A_csr,
        "A_dense": A_dense,
        "b": np.ascontiguousarray(d["b"]),
        "R": float(d["R"]),
        "n": int(d["n"]),
        "opt": float(d["opt"]),
    }


def build_problem(arrays: dict, g: float) -> PrimalOracleProblem:
    return PrimalOracleProblem(
        A_matrices=list(arrays["A"]),
        b=arrays["b"],
        C=arrays["C"],
        R=arrays["R"],
        g=g,
    )


def cycle_adjacency_csr_from_permutation(pi: np.ndarray) -> csr_matrix:
    """
    CSR adjacency of the random cycle (Gilyén–Vazirani App. C, Def. 6) — O(n) nnz.
    """
    pi = np.asarray(pi, dtype=np.intp)
    n = pi.shape[0]
    inv = np.empty(n, dtype=np.intp)
    inv[pi] = np.arange(n)
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        for step in (-1, 1):
            j = int(inv[(int(pi[i]) + step) % n])
            rows.extend((i, j))
            cols.extend((j, i))
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def random_cycle_adjacency_csr(rng: np.random.Generator, n: int) -> csr_matrix:
    pi = rng.permutation(n)
    return cycle_adjacency_csr_from_permutation(pi)


def jump_matrices_from_arrays(arrays: dict) -> list:
    """Channel proposals as CSR: objective + constraint generators + two cycles."""
    A = arrays["A"]
    n = arrays["n"]
    rng = np.random.default_rng(0)
    cycle1 = random_cycle_adjacency_csr(rng, n)
    cycle2 = random_cycle_adjacency_csr(rng, n)
    return [arrays["C"]] + [A[k] for k in range(1, len(A), 2)] + [cycle1, cycle2]


# --- Objective threshold g ---

def g_bracket(arrays: dict) -> tuple[float, float]:
    C_dense = arrays["C"].toarray() if hasattr(arrays["C"], "toarray") else arrays["C"]
    lam_max = float(la.eigvalsh(C_dense, check_finite=False)[-1])
    g_hi = arrays["R"] * lam_max
    base = arrays["opt"] if np.isfinite(arrays["opt"]) else 0.0
    g_lo = min(base, 0.0) - (abs(base) + 1.0)
    return g_lo, g_hi


def _exact_iterations_for_g(arrays: dict, g: float, eps: float, cap: int) -> int:
    res = run_primal_oracle(
        build_problem(arrays, g),
        eps,
        gibbs_mode="exact",
        max_iterations=cap,
        return_on_exhaustion=True,
    )
    if res is None or not res.constraint_diag.feasible_within_theta:
        return cap + 1
    return int(res.iterations)


def calibrate_g(arrays: dict, eps: float, target_iters: int, cap: int) -> float:
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
    ax.set_title(
        f"{instance} [CSR]: MCMC steps per oracle iteration (g = {g:.4g}, ε = {EPSILON:g})"
    )
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_channel_gap(
    gap_pairs: list[tuple[int, float]],
    instance: str,
    g: float,
    out_path: str,
) -> None:
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
    ax.set_title(
        f"{instance} [CSR]: channel gap per oracle iteration (g = {g:.4g}, ε = {EPSILON:g})"
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- Driver ---

def run_instance(name: str) -> dict:
    arrays = load_instance(name)
    g_lo, g_hi = g_bracket(arrays)
    g = g_lo
    theta = EPSILON / (2.0 * arrays["R"])
    print(
        f"[{name}/CSR] n={arrays['n']} R={arrays['R']:.4g} opt={arrays['opt']:.6g} "
        f"theta={theta:.4g}; g bracket = [{g_lo:.6g}, {g_hi:.6g}], using g = {g:.6g}"
    )

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
    print(
        f"[{name}/CSR] oracle iterations = {result.iterations}, feasible = {feasible}, "
        f"z = {result.z:.6g}"
    )
    print(
        f"[{name}/CSR] Gibbs steps per iteration: mean = {np.mean(steps):.1f}, "
        f"max = {max(steps)}, all converged = {all(result.gibbs_converged_per_iter)}"
    )
    if gap_vals:
        print(
            f"[{name}/CSR] channel gap ({len(gap_vals)} samples): "
            f"first = {gap_vals[0]:.4f}, last = {gap_vals[-1]:.4f}, "
            f"min = {min(gap_vals):.4f}"
        )
    print(
        f"[{name}/CSR] wall time = {wall:.2f}s "
        f"(gibbs {result.timing['gibbs_time']:.1f}s of "
        f"{result.timing['total_wall_time']:.1f}s)"
    )

    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    raw = {
        "instance": name,
        "backend": "sparse_csr",
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
    print(f"[{name}/CSR] wrote {out_dir}\\run.json, {plot_path} and {gap_plot_path}")

    return {
        "instance": name,
        "backend": "sparse_csr",
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
            print(
                f"[suite/CSR] time budget ({TIME_BUDGET_S}s) reached after {elapsed:.0f}s; "
                f"skipping remaining instances: {INSTANCES[INSTANCES.index(name):]}"
            )
            break
        summaries.append(run_instance(name))
        print()

    print("=== VSparse suite summary ===")
    print(
        f"{'instance':<12} {'n':>4} {'iters':>8} {'feasible':>8} "
        f"{'mean_st':>8} {'max_st':>7} {'gap_min':>8} {'time_s':>8}"
    )
    for s in summaries:
        gap_txt = f"{s['gap_min']:.4f}" if s["gap_min"] is not None else "-"
        print(
            f"{s['instance']:<12} {s['n']:>4} {s['iterations']:>8} "
            f"{str(s['feasible']):>8} {s['gibbs_steps_mean']:>8.2f} "
            f"{s['gibbs_steps_max']:>7} {gap_txt:>8} {s['wall_time_s']:>8.1f}"
        )
    print(f"total wall time = {time.perf_counter() - t_suite:.1f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "suite_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)


if __name__ == "__main__":
    main()
