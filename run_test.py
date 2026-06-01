# run_test.py — thesis benchmark driver for the quantum SDP primal oracle.

"""
Benchmark converted SDPLIB instances: g-calibration (exact Gibbs), MCMC oracle runs,
and plots.

Per instance (default): main ε/(2R), fixed θ = 0.01, and ``--random-runs`` replicates
for each Gibbs criterion. Outputs under ``<results_dir>/``; see README.md.
"""

from __future__ import annotations

# --- Thread pinning MUST happen before NumPy / BLAS import -------------------
# On Windows (spawn) every worker re-imports this module, so setting the thread
# counts here guarantees each process uses a single BLAS thread.
#
# We *force* one BLAS/OpenMP thread per process (override, not setdefault). This
# is the first half of the hard core cap: combined with the worker-count clamp
# (see MAX_WORKERS / parse_config), total core usage is bounded by the number of
# worker processes. On a shared machine this prevents BLAS from silently fanning
# each worker out across many cores (e.g. an inherited OMP_NUM_THREADS=8 would
# otherwise turn 16 workers into 128 busy threads).
import os

for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_var] = "1"

# Hard upper bound on worker processes, enforced regardless of the --workers
# value. With one BLAS thread each (forced above), this caps total core usage at
# MAX_WORKERS. The runs are small-matrix, single-threaded NumPy workloads with
# hundreds of independent tasks, so process-level parallelism (many workers, one
# BLAS thread each) is far more efficient than threading BLAS within a worker.
MAX_WORKERS = 64

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Make console output robust to non-ASCII on legacy Windows code pages.
try:  # pragma: no cover - platform dependent
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from primal_oracle_quantum_v1cube import PrimalOracleProblem, run_primal_oracle
from sdp_conversion import instance_class

try:  # Progress bars are nice-to-have; degrade gracefully if tqdm is missing.
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None


class _ProgressBar:
    """
    Minimal progress-bar wrapper around tqdm.

    Falls back to a no-op (plain ``print`` for messages) when tqdm is not
    installed, so the driver keeps working in either case. ``write`` routes
    log lines through :func:`tqdm.write` so they appear *above* the live bar
    without corrupting it.
    """

    def __init__(self, total: int, desc: str) -> None:
        self._bar = (
            _tqdm(total=total, desc=desc, unit="run", dynamic_ncols=True, leave=True)
            if _tqdm is not None
            else None
        )

    def update(self, n: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(n)

    def set_postfix(self, text: str) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(text, refresh=False)

    def write(self, msg: str) -> None:
        if self._bar is not None:
            _tqdm.write(msg)
        else:
            print(msg)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

# Default benchmark instances and run parameters.
THESIS_INSTANCES = (
    "hinf1", "hinf4", "hinf12",
    "control1", "control2",
    "truss1", "truss3", "truss4",
)
DEFAULT_EPSILON = 1e-1
DEFAULT_FIXED_GIBBS_THETA = 0.01
DEFAULT_INSTANCES = THESIS_INSTANCES
DEFAULT_CUTOFFS = (50, 100, 200)
DEFAULT_TARGET_ITERS = 500
DEFAULT_MAX_ORACLE_ITERS = 550
DEFAULT_GIBBS_MAX_STEPS = 2000
DEFAULT_RANDOM_RUNS = 5
DEFAULT_WORKERS = 32
DEFAULT_GIBBS_WARM_START = True

# Natural / thesis plot palette (colorblind-friendly).
_PALETTE = {
    "rel": "#2E86AB",
    "fixed": "#E07A5F",
    "mean_line": "#3D405B",
    "max_line": "#81B29A",
    "cutoff_50": "#8E7DBE",
    "cutoff_100": "#2E86AB",
    "cutoff_200": "#E9C46A",
    "bar_rel_max": "#2E86AB",
    "bar_rel_mean": "#5BA4CF",
    "bar_theta_max": "#E07A5F",
    "bar_theta_mean": "#F4A261",
    "bar_oracle": "#3D405B",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def eps_tag(eps: float) -> str:
    """Filesystem-friendly tag, e.g. ``1e-02``."""
    return f"{eps:.0e}"


def fixed_theta_tag(theta: float) -> str:
    """Tag for fixed-θ runs, e.g. ``1e-02``."""
    return eps_tag(theta)


def load_instance_arrays(npz_path: str) -> dict:
    """Load a converted instance ``.npz`` into plain arrays/scalars."""
    d = np.load(npz_path, allow_pickle=True)
    return {
        "name": str(d["name"]),
        "C": np.ascontiguousarray(d["C"]),
        "A": np.ascontiguousarray(d["A"]),
        "b": np.ascontiguousarray(d["b"]),
        "R": float(d["R"]),
        "n": int(d["n"]),
        "m_oracle": int(d["m_oracle"]),
        "opt": float(d["opt"]),
    }


def build_problem(arrays: dict, g: float) -> PrimalOracleProblem:
    """Construct a :class:`PrimalOracleProblem` from loaded arrays at threshold ``g``."""
    A = arrays["A"]
    A_matrices = [A[k] for k in range(A.shape[0])]
    return PrimalOracleProblem(
        A_matrices=A_matrices,
        b=arrays["b"],
        C=arrays["C"],
        R=arrays["R"],
        g=float(g),
    )


def jump_matrices_from_arrays(arrays: dict) -> list[np.ndarray]:
    """
    Channel jump (proposal) operators = SDP constraints + objective.

    The converted constraint stack is ``[I, F_1, -F_1, F_2, -F_2, ...]``, so the
    distinct generators are the objective ``C`` together with the ``+F_i`` at odd
    indices (the identity is dropped: after Bohr reweighing it provides no mixing,
    and ``-F_i`` shares the generator of ``+F_i``). This avoids the costly generic
    deduplication inside the oracle.
    """
    A = arrays["A"]
    jumps = [np.ascontiguousarray(arrays["C"])]
    for k in range(1, A.shape[0], 2):
        jumps.append(np.ascontiguousarray(A[k]))
    return jumps


# ---------------------------------------------------------------------------
# g-calibration (exact mode)
# ---------------------------------------------------------------------------


def _exact_iterations_for_g(
    prob: PrimalOracleProblem,
    eps: float,
    g: float,
    cap: int,
) -> int:
    """
    Number of oracle iterations to reach θ-feasibility at threshold ``g`` using the
    cheap closed-form Gibbs state. Returns ``cap + 1`` if not feasible within ``cap``
    (treated as "needs at least the target"). Monotonically increasing in ``g``.
    """
    prob.g = float(g)
    res = run_primal_oracle(
        prob,
        eps,
        gibbs_mode="exact",
        max_iterations=cap,
        skip_gibbs_asserts=True,
        collect_timing=False,
        return_on_exhaustion=True,
    )
    if res is None or not res.constraint_diag.feasible_within_theta:
        return cap + 1
    return int(res.iterations)


def calibrate_g(
    arrays: dict,
    eps: float,
    target_iters: int,
    cap: int,
    search_steps: int = 18,
) -> dict:
    """
    Bisection on ``g`` so an exact-Gibbs feasibility run finishes in at most
    ``target_iters`` oracle steps (aiming near that budget).

    Bracket: ``g_lo`` (easy) .. ``g_hi`` (hard). We keep the **largest** ``g`` such
    that the run is θ-feasible within ``cap`` and uses ≤ ``target_iters`` steps.
    """
    import scipy.linalg as la

    C = arrays["C"]
    R = arrays["R"]
    opt = arrays["opt"]
    lam_max = float(np.max(la.eigvalsh(C)))
    g_hi = R * lam_max
    base = 0.0 if not np.isfinite(opt) else opt
    g_lo = min(base, 0.0) - (abs(base) + 1.0)
    if g_lo >= g_hi:
        g_lo = g_hi - (abs(g_hi) + 1.0)

    prob = build_problem(arrays, g_lo)

    lo, hi = g_lo, g_hi
    chosen = g_lo
    chosen_iters = _exact_iterations_for_g(prob, eps, lo, cap)
    for _ in range(search_steps):
        mid = 0.5 * (lo + hi)
        c = _exact_iterations_for_g(prob, eps, mid, cap)
        if c > cap:
            hi = mid
        elif c > target_iters:
            hi = mid
        else:
            chosen, chosen_iters = mid, c
            lo = mid
    return {
        "name": arrays["name"],
        "g": float(chosen),
        "calibrated_iters": int(chosen_iters),
        "calib_eps": float(eps),
        "g_lo": float(g_lo),
        "g_hi": float(g_hi),
        "opt": None if not np.isfinite(opt) else float(opt),
        "R": float(R),
    }


def _calibrate_worker(task: dict) -> dict:
    """Process-pool entry point for calibration of one instance."""
    import warnings
    warnings.simplefilter("ignore")
    try:
        arrays = load_instance_arrays(task["npz_path"])
        return calibrate_g(
            arrays,
            eps=task["eps"],
            target_iters=task["target_iters"],
            cap=task["cap"],
        )
    except Exception as exc:  # pragma: no cover - surfaced to the parent
        return {"name": task.get("name", "?"), "error": f"{exc}", "trace": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Oracle run (MCMC mode) workers
# ---------------------------------------------------------------------------


def _result_to_dict(res, *, eps: float) -> dict:
    """Serialize a :class:`PrimalOracleResult` to JSON-friendly primitives."""
    cutoff = None
    if res.cutoff_trace_distance_per_iter:
        cutoff = {
            str(c): [float(x) for x in vals]
            for c, vals in res.cutoff_trace_distance_per_iter.items()
        }
    return {
        "eps": float(eps),
        "iterations": int(res.iterations),
        "feasible": bool(res.constraint_diag.feasible_within_theta),
        "theta": float(res.theta),
        "z": float(res.z),
        "omega": float(res.omega),
        "gibbs_steps_per_iter": [int(x) for x in (res.gibbs_steps_per_iter or [])],
        "gibbs_converged_per_iter": [bool(x) for x in (res.gibbs_converged_per_iter or [])],
        "cutoff_td": cutoff,
        "timing": {k: float(v) for k, v in (res.timing or {}).items()},
    }


def _run_oracle_worker(task: dict) -> dict:
    """
    Process-pool entry point for a single MCMC oracle run.

    ``task`` keys: ``npz_path, kind, eps, g, max_iters, gibbs_max_steps, cutoffs,
    selection, seed, tag``.
    """
    import warnings
    warnings.simplefilter("ignore")
    try:
        arrays = load_instance_arrays(task["npz_path"])
        prob = build_problem(arrays, task["g"])
        jumps = jump_matrices_from_arrays(arrays)
        oracle_kw: dict = {
            "gibbs_mode": "mcmc",
            "gibbs_jump_matrices": jumps,
            "gibbs_max_steps": int(task["gibbs_max_steps"]),
            "gibbs_step_cutoffs": list(task["cutoffs"]) if task.get("cutoffs") else None,
            "max_iterations": int(task["max_iters"]),
            "skip_gibbs_asserts": True,
            "collect_timing": True,
            "violation_selection": task["selection"],
            "violation_rng_seed": task.get("seed"),
            "return_on_exhaustion": True,
            "gibbs_warm_start": bool(task.get("gibbs_warm_start")),
        }
        if task.get("gibbs_target_theta") is not None:
            oracle_kw["gibbs_target_theta"] = float(task["gibbs_target_theta"])
        res = run_primal_oracle(prob, task["eps"], **oracle_kw)
        if res is None:
            return {"name": arrays["name"], "tag": task["tag"], "error": "oracle returned None"}
        out = _result_to_dict(res, eps=task["eps"])
        out["name"] = arrays["name"]
        out["tag"] = task["tag"]
        out["kind"] = task["kind"]
        out["gibbs_warm_start"] = bool(task.get("gibbs_warm_start"))
        return out
    except Exception as exc:  # pragma: no cover
        return {
            "name": task.get("name", "?"),
            "tag": task.get("tag", "?"),
            "error": f"{exc}",
            "trace": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Plotting (lazy matplotlib import; Agg backend, no display)
# ---------------------------------------------------------------------------


def _new_axes(figsize=(9, 5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax


def plot_gibbs_steps(
    steps: list[int],
    eps: float,
    instance: str,
    out_path: str,
    *,
    convergence_label: str,
    violation_label: str = "max-violation",
    dot_color: str | None = None,
    title_suffix: str = "",
) -> None:
    """Dot plot of Gibbs steps per oracle iteration, with mean/max reference lines."""
    plt, fig, ax = _new_axes()
    x = np.arange(1, len(steps) + 1)
    y = np.asarray(steps, dtype=float)
    color = dot_color or _PALETTE["rel"]
    if y.size:
        ax.scatter(x, y, s=14, color=color, alpha=0.78, edgecolors="none", zorder=2)
        mean_v = float(np.mean(y))
        max_v = float(np.max(y))
        ax.axhline(mean_v, color=_PALETTE["mean_line"], linestyle="--", linewidth=1.5,
                   label=f"mean = {mean_v:.1f}")
        ax.axhline(max_v, color=_PALETTE["max_line"], linestyle="-.", linewidth=1.5,
                   label=f"max = {max_v:.0f}")
        ax.legend(loc="upper right")

    ax.set_xlabel("Primal-oracle iteration")
    ax.set_ylabel("Gibbs-sampler steps to convergence")
    ax.set_title(
        f"{instance}: Gibbs steps (ε = {eps:g}, target {convergence_label}, "
        f"{violation_label}){title_suffix}"
    )
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_gibbs_comparison(
    series: dict[str, list[int]],
    eps: float,
    instance: str,
    out_path: str,
    *,
    title_suffix: str = "",
) -> None:
    """Overlay Gibbs step counts for multiple convergence criteria (line plot)."""
    plt, fig, ax = _new_axes()
    for label, steps in series.items():
        y = np.asarray(steps, dtype=float)
        if y.size == 0:
            continue
        color = _PALETTE["rel"] if "ε" in label or "2R" in label else _PALETTE["fixed"]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, linewidth=1.3, alpha=0.88, label=label, color=color)
    ax.set_xlabel("Primal-oracle iteration")
    ax.set_ylabel("Gibbs-sampler steps to convergence")
    ax.set_title(
        f"{instance}: relative vs fixed Gibbs target (ε = {eps:g}){title_suffix}"
    )
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# Human-readable labels for the timing buckets used in the pie chart.
_PIE_BUCKETS = [
    ("gibbs_construction_time", "Gibbs sampler construction"),
    ("gibbs_channel_iteration_time", "Channel iteration (MCMC)"),
    ("gibbs_convergence_check_time", "Convergence checks"),
    ("trace_check_time", "Constraint traces"),
    ("violation_logic_time", "Violation selection"),
    ("hamiltonian_update_time", "Hamiltonian update"),
    ("result_packaging_time", "Result packaging"),
]


def plot_time_pie(timing_sum: dict, instance: str, out_path: str) -> None:
    """Pie chart of where time was spent across the combined ε runs (2.1.2)."""
    labels, values = [], []
    for key, label in _PIE_BUCKETS:
        v = float(timing_sum.get(key, 0.0))
        if v > 0:
            labels.append(label)
            values.append(v)
    if not values:
        return
    plt, fig, ax = _new_axes(figsize=(9, 7))
    total = sum(values)
    wedges, _ = ax.pie(values, startangle=90, counterclock=False)
    # Use a legend (label + seconds + percent) instead of inline labels so that
    # tiny slices do not overlap each other's text.
    legend_labels = [
        f"{lab}: {val:.3g}s ({100 * val / total:.1f}%)"
        for lab, val in zip(labels, values)
    ]
    ax.legend(
        wedges, legend_labels, loc="center left",
        bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False,
    )
    ax.set_title(f"{instance}: time breakdown over the ε runs (total {total:.3g}s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_cutoff_tracedist(
    cutoff_td: dict[str, list[float]],
    eps: float,
    R: float,
    instance: str,
    out_path: str,
) -> None:
    """Line chart of normalized early-termination trace distance per iteration (2.1.3)."""
    plt, fig, ax = _new_axes()
    scale = eps / R  # normalize by R/eps  <=>  multiply by eps/R
    colors = {
        "50": _PALETTE["cutoff_50"],
        "100": _PALETTE["cutoff_100"],
        "200": _PALETTE["cutoff_200"],
        "500": _PALETTE["max_line"],
    }
    for c in sorted(cutoff_td, key=lambda s: int(s)):
        vals = np.asarray(cutoff_td[c], dtype=float) * scale
        x = np.arange(1, len(vals) + 1)
        ax.plot(x, vals, linewidth=1.2, label=f"cutoff = {c} steps",
                color=colors.get(c, _PALETTE["rel"]))
    ax.set_xlabel("Primal-oracle iteration")
    ax.set_ylabel(r"$D(\sigma_{\mathrm{cutoff}}, \sigma_{\mathrm{full}}) \cdot \varepsilon / R$")
    ax.set_title(f"{instance}: early-termination error (ε = {eps:g})")
    ax.legend(loc="upper right")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_class_grouped_bar(
    class_name: str,
    per_instance: list[dict],
    out_path: str,
    *,
    reference_eps: float,
    fixed_theta: float,
) -> None:
    """
    Grouped bar chart for one problem class (random runs, ε/(2R) and fixed θ).
    """
    if not per_instance:
        return
    plt, fig, ax = _new_axes(figsize=(max(10, 2.2 * len(per_instance)), 6))
    names = [d["name"] for d in per_instance]
    rel_max = [d.get("avg_max_steps_rel", 0.0) for d in per_instance]
    rel_mean = [d.get("avg_mean_steps_rel", 0.0) for d in per_instance]
    th_max = [d.get("avg_max_steps_theta", 0.0) for d in per_instance]
    th_mean = [d.get("avg_mean_steps_theta", 0.0) for d in per_instance]
    avg_iters = [
        d.get("avg_iterations_feasible")
        if d.get("avg_iterations_feasible") is not None
        else d.get("avg_iterations", 0.0)
        for d in per_instance
    ]

    x = np.arange(len(names))
    n_bars = 5
    w = 0.15
    offsets = np.linspace(-2, 2, n_bars) * w
    ax.bar(x + offsets[0], rel_max, w, label="avg max Gibbs (ε/(2R), random)",
           color=_PALETTE["bar_rel_max"])
    ax.bar(x + offsets[1], rel_mean, w, label="avg mean Gibbs (ε/(2R), random)",
           color=_PALETTE["bar_rel_mean"])
    ax.bar(x + offsets[2], th_max, w, label=f"avg max Gibbs (θ={fixed_theta:g}, random)",
           color=_PALETTE["bar_theta_max"])
    ax.bar(x + offsets[3], th_mean, w, label=f"avg mean Gibbs (θ={fixed_theta:g}, random)",
           color=_PALETTE["bar_theta_mean"])
    ax.bar(x + offsets[4], avg_iters, w, label="avg oracle iters (feasible)",
           color=_PALETTE["bar_oracle"])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count (averaged over random runs)")
    ax.set_title(
        f"Class '{class_name}': random runs (ε = {reference_eps:g}, warm start)"
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class DriverConfig:
    converted_dir: str
    results_dir: str
    epsilon: float
    fixed_gibbs_theta: float
    cutoffs: tuple[int, ...]
    target_iters: int
    max_oracle_iters: int
    gibbs_max_steps: int
    random_runs: int
    gibbs_warm_start: bool
    workers: int
    instances: list[str] | None
    skip_existing: bool


def discover_instances(converted_dir: str, instances: list[str] | None) -> list[str]:
    """Return the list of instance ``.npz`` paths (optionally filtered by substring)."""
    files = sorted(
        os.path.join(converted_dir, f)
        for f in os.listdir(converted_dir)
        if f.endswith(".npz")
    )
    if instances:
        wanted = [s.lower() for s in instances]
        files = [f for f in files if any(w in os.path.basename(f).lower() for w in wanted)]
    return files


def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_calibration(cfg: DriverConfig, npz_paths: list[str]) -> dict[str, dict]:
    """Calibrate ``g`` for every instance (parallel). Cached in ``calibration.json``."""
    calib_path = os.path.join(cfg.results_dir, "calibration.json")
    cache: dict[str, dict] = {}
    if cfg.skip_existing and os.path.exists(calib_path):
        cache = {d["name"]: d for d in _load_json(calib_path)}

    tasks = []
    for path in npz_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in cache and "g" in cache[name]:
            continue
        tasks.append({
            "npz_path": path,
            "name": name,
            "eps": cfg.epsilon,
            "target_iters": cfg.target_iters,
            "cap": max(cfg.max_oracle_iters, cfg.target_iters + 50),
        })

    if tasks:
        print(f"[calibrate] {len(tasks)} instance(s) on {cfg.workers} worker(s) ...")
        bar = _ProgressBar(len(tasks), "calibrate g")
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            for res in pool.map(_calibrate_worker, tasks):
                bar.update(1)
                if "error" in res:
                    bar.write(f"[calibrate] ERROR {res['name']}: {res['error']}")
                    continue
                cache[res["name"]] = res
                bar.set_postfix(f"{res['name']} g={res['g']:.4g}")
                bar.write(f"[calibrate] {res['name']:<24} g={res['g']:.5g} "
                          f"iters~{res['calibrated_iters']} (target {cfg.target_iters})")
        bar.close()
        _save_json(calib_path, list(cache.values()))
    else:
        print("[calibrate] all instances cached.")
    return cache


def build_run_tasks(cfg: DriverConfig, npz_paths: list[str], calib: dict[str, dict]) -> list[dict]:
    """Assemble MCMC oracle-run tasks for the process pool."""
    tasks: list[dict] = []
    for path in npz_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in calib or "g" not in calib[name]:
            print(f"[run] skipping {name}: no calibrated g.")
            continue
        g = calib[name]["g"]
        eps = cfg.epsilon
        ft = float(cfg.fixed_gibbs_theta)
        warm = bool(cfg.gibbs_warm_start)
        base = {
            "npz_path": path,
            "name": name,
            "eps": eps,
            "g": g,
            "max_iters": cfg.max_oracle_iters,
            "gibbs_max_steps": cfg.gibbs_max_steps,
            "gibbs_warm_start": warm,
        }
        tasks.append({
            **base,
            "kind": "eps_rel",
            "cutoffs": list(cfg.cutoffs),
            "gibbs_target_theta": None,
            "gibbs_convergence_mode": "relative",
            "selection": "max",
            "seed": None,
            "tag": f"eps_{eps_tag(eps)}",
        })
        tasks.append({
            **base,
            "kind": "eps_fixed_theta",
            "cutoffs": None,
            "gibbs_target_theta": ft,
            "gibbs_convergence_mode": "fixed",
            "selection": "max",
            "seed": None,
            "tag": f"fixed_theta_{fixed_theta_tag(ft)}",
        })
        for r in range(cfg.random_runs):
            tasks.append({
                **base,
                "kind": "random_rel",
                "cutoffs": None,
                "gibbs_target_theta": None,
                "gibbs_convergence_mode": "relative",
                "selection": "random",
                "seed": 1000 + r,
                "tag": f"random_{r:02d}",
            })
            tasks.append({
                **base,
                "kind": "random_fixed_theta",
                "cutoffs": None,
                "gibbs_target_theta": ft,
                "gibbs_convergence_mode": "fixed",
                "selection": "random",
                "seed": 2000 + r,
                "tag": f"random_{r:02d}_theta",
            })
    return tasks


def execute_runs(cfg: DriverConfig, tasks: list[dict]) -> None:
    """Run every task (parallel) and persist each result under ``<instance>/raw/``."""
    # Optionally skip tasks whose raw JSON already exists.
    pending = []
    for t in tasks:
        raw_path = os.path.join(cfg.results_dir, t["name"], "raw", f"{t['tag']}.json")
        if cfg.skip_existing and os.path.exists(raw_path):
            continue
        pending.append(t)

    if not pending:
        print("[run] all run results cached.")
        return

    print(f"[run] {len(pending)} oracle run(s) on {cfg.workers} worker(s) ...")
    done = 0
    bar = _ProgressBar(len(pending), "oracle runs")
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(_run_oracle_worker, t): t for t in pending}
        for fut in as_completed(futures):
            t = futures[fut]
            res = fut.result()
            done += 1
            bar.update(1)
            if "error" in res:
                bar.write(f"[run] ({done}/{len(pending)}) ERROR {t['name']} {t['tag']}: {res['error']}")
                continue
            raw_path = os.path.join(cfg.results_dir, res["name"], "raw", f"{res['tag']}.json")
            _save_json(raw_path, res)
            feas = res.get("feasible")
            steps = res.get("gibbs_steps_per_iter", [])
            mx = max(steps) if steps else 0
            bar.set_postfix(f"{res['name']} {res['tag']} maxGibbs={mx}")
            bar.write(f"[run] ({done}/{len(pending)}) {res['name']:<22} {res['tag']:<12} "
                      f"iters={res['iterations']:<5} feasible={feas} maxGibbs={mx}")
    bar.close()


def _load_raw(cfg: DriverConfig, name: str, tag: str) -> dict | None:
    path = os.path.join(cfg.results_dir, name, "raw", f"{tag}.json")
    if not os.path.exists(path):
        return None
    return _load_json(path)


def make_instance_plots(cfg: DriverConfig, name: str, calib_entry: dict) -> dict | None:
    """Produce plots for one instance; return random-run aggregate stats."""
    inst_dir = os.path.join(cfg.results_dir, name)
    R = float(calib_entry.get("R", 1.0))
    timing_sum: dict[str, float] = {}
    eps = cfg.epsilon
    ft = cfg.fixed_gibbs_theta
    rel_raw = _load_raw(cfg, name, f"eps_{eps_tag(eps)}")
    fix_raw = _load_raw(cfg, name, f"fixed_theta_{fixed_theta_tag(ft)}")
    comparison: dict[str, list[int]] = {}

    if rel_raw is not None:
        steps = rel_raw.get("gibbs_steps_per_iter", [])
        plot_gibbs_steps(
            steps, eps, name,
            os.path.join(inst_dir, f"gibbs_steps_{eps_tag(eps)}.png"),
            convergence_label="ε/(2R)",
            violation_label="max-violation",
            dot_color=_PALETTE["rel"],
        )
        comparison["ε/(2R)"] = steps
        for k, v in rel_raw.get("timing", {}).items():
            timing_sum[k] = timing_sum.get(k, 0.0) + float(v)
        if rel_raw.get("cutoff_td"):
            plot_cutoff_tracedist(
                rel_raw["cutoff_td"], eps, R, name,
                os.path.join(inst_dir, "early_termination_tracedist.png"),
            )

    if fix_raw is not None:
        steps = fix_raw.get("gibbs_steps_per_iter", [])
        plot_gibbs_steps(
            steps, eps, name,
            os.path.join(inst_dir, f"gibbs_steps_fixed_theta_{fixed_theta_tag(ft)}.png"),
            convergence_label=f"θ={ft:g}",
            violation_label="max-violation",
            dot_color=_PALETTE["fixed"],
        )
        comparison[f"θ={ft:g}"] = steps
        for k, v in fix_raw.get("timing", {}).items():
            timing_sum[k] = timing_sum.get(k, 0.0) + float(v)

    if len(comparison) >= 2:
        plot_gibbs_comparison(
            comparison, eps, name,
            os.path.join(inst_dir, "gibbs_steps_comparison.png"),
        )
    if timing_sum:
        plot_time_pie(timing_sum, name, os.path.join(inst_dir, "time_breakdown_pie.png"))

    def _random_aggregate(tag_for_run) -> dict | None:
        maxima, means, iters, iters_feasible = [], [], [], []
        for r in range(cfg.random_runs):
            raw = _load_raw(cfg, name, tag_for_run(r))
            if raw is None:
                continue
            steps = np.asarray(raw.get("gibbs_steps_per_iter", []), dtype=float)
            if steps.size:
                maxima.append(float(np.max(steps)))
                means.append(float(np.mean(steps)))
            n_it = float(raw.get("iterations", 0))
            iters.append(n_it)
            if raw.get("feasible"):
                iters_feasible.append(n_it)
        if not maxima:
            return None
        return {
            "avg_max": float(np.mean(maxima)),
            "avg_mean": float(np.mean(means)),
            "avg_iterations": float(np.mean(iters)),
            "avg_iterations_feasible": (
                float(np.mean(iters_feasible)) if iters_feasible else None
            ),
            "feasible_fraction": float(len(iters_feasible) / len(iters)) if iters else 0.0,
            "n": len(maxima),
        }

    rel_agg = _random_aggregate(lambda r: f"random_{r:02d}")
    th_agg = _random_aggregate(lambda r: f"random_{r:02d}_theta")

    stats = None
    if rel_agg or th_agg:
        stats = {
            "name": name,
            "class": instance_class(name),
            "avg_max_steps_rel": rel_agg["avg_max"] if rel_agg else None,
            "avg_mean_steps_rel": rel_agg["avg_mean"] if rel_agg else None,
            "avg_max_steps_theta": th_agg["avg_max"] if th_agg else None,
            "avg_mean_steps_theta": th_agg["avg_mean"] if th_agg else None,
            "avg_iterations": (
                rel_agg["avg_iterations"] if rel_agg else (th_agg["avg_iterations"] if th_agg else 0.0)
            ),
            "avg_iterations_feasible": (
                rel_agg["avg_iterations_feasible"]
                if rel_agg and rel_agg["avg_iterations_feasible"] is not None
                else (th_agg["avg_iterations_feasible"] if th_agg else None)
            ),
            "feasible_fraction_rel": rel_agg["feasible_fraction"] if rel_agg else 0.0,
            "feasible_fraction_theta": th_agg["feasible_fraction"] if th_agg else 0.0,
            "num_random_runs": cfg.random_runs,
        }

    summary = {
        "name": name,
        "class": instance_class(name),
        "g": calib_entry.get("g"),
        "R": R,
        "opt": calib_entry.get("opt"),
        "epsilon": cfg.epsilon,
        "fixed_gibbs_theta": cfg.fixed_gibbs_theta,
        "gibbs_warm_start": cfg.gibbs_warm_start,
        "timing_sum": timing_sum,
        "random_stats": stats,
    }
    _save_json(os.path.join(inst_dir, "summary.json"), summary)
    return stats


def make_class_plots(cfg: DriverConfig, all_stats: list[dict]) -> None:
    """Group the per-instance 2.1.4 stats by class and emit one chart per class."""
    by_class: dict[str, list[dict]] = {}
    for s in all_stats:
        by_class.setdefault(s["class"], []).append(s)

    out_dir = os.path.join(cfg.results_dir, "_classes")
    os.makedirs(out_dir, exist_ok=True)
    for cls, items in sorted(by_class.items()):
        items = sorted(items, key=lambda d: d["name"])
        plot_class_grouped_bar(
            cls, items, os.path.join(out_dir, f"{cls}_grouped_bar.png"),
            reference_eps=cfg.epsilon,
            fixed_theta=cfg.fixed_gibbs_theta,
        )
        print(f"[class] {cls}: {len(items)} instance(s) -> {cls}_grouped_bar.png")


def _print_thesis_run_plan(cfg: DriverConfig, n_instances: int, n_tasks: int) -> None:
    """Summarize the planned benchmark."""
    per_inst = 2 + 2 * cfg.random_runs
    print("[driver] planned benchmark:")
    print(f"  instances ({n_instances}): {', '.join(cfg.instances or [])}")
    print(f"  ε = {cfg.epsilon:g}; fixed Gibbs θ = {cfg.fixed_gibbs_theta:g}")
    print(f"  warm start: {cfg.gibbs_warm_start}")
    print(f"  runs per instance: {per_inst} (main ε/(2R) + main θ + "
          f"{cfg.random_runs}× random each)")
    print(f"  early-termination L: {cfg.cutoffs} (ε/(2R) main run only)")
    print(f"  total oracle tasks: {n_tasks}")
    print(f"  g-calibration target iters: {cfg.target_iters}; "
          f"max_oracle_iters: {cfg.max_oracle_iters}; "
          f"gibbs_max_steps: {cfg.gibbs_max_steps}")
    print(f"  workers: {cfg.workers}")


def main(argv: list[str] | None = None) -> None:
    cfg = parse_config(argv)
    os.makedirs(cfg.results_dir, exist_ok=True)

    npz_paths = discover_instances(cfg.converted_dir, cfg.instances)
    if not npz_paths:
        raise FileNotFoundError(
            f"No converted .npz instances in {cfg.converted_dir!r}. "
            f"Run sdp_conversion.py first."
        )
    print(f"[driver] {len(npz_paths)} instance(s); workers={cfg.workers}; "
          f"ε={cfg.epsilon:g}; fixed Gibbs θ={cfg.fixed_gibbs_theta:g}; "
          f"target_iters={cfg.target_iters}")

    t0 = time.perf_counter()
    calib = run_calibration(cfg, npz_paths)
    tasks = build_run_tasks(cfg, npz_paths, calib)
    _print_thesis_run_plan(cfg, len(npz_paths), len(tasks))
    execute_runs(cfg, tasks)

    print("[driver] generating plots ...")
    all_stats: list[dict] = []
    for path in npz_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in calib:
            continue
        stats = make_instance_plots(cfg, name, calib[name])
        if stats is not None:
            all_stats.append(stats)
    if cfg.random_runs > 0:
        make_class_plots(cfg, all_stats)

    print(f"[driver] done in {time.perf_counter() - t0:.1f}s. Results in {cfg.results_dir}")


def parse_config(argv: list[str] | None) -> DriverConfig:
    here = _THIS_DIR
    p = argparse.ArgumentParser(description="Benchmark the quantum SDP primal oracle.")
    p.add_argument("--converted-dir", default=os.path.join(here, "SDP_problems_converted"))
    p.add_argument("--results-dir", default=os.path.join(here, "results"))
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help="SDP precision ε; Gibbs target θ = ε/(2R) for relative runs.")
    p.add_argument("--fixed-theta", type=float, default=DEFAULT_FIXED_GIBBS_THETA,
                   dest="fixed_gibbs_theta",
                   help="Fixed Gibbs trace-distance target.")
    p.add_argument("--instances", nargs="*", default=list(DEFAULT_INSTANCES),
                   help="Instance name substrings (default: thesis 8-instance set).")
    p.add_argument("--cutoffs", type=int, nargs="*", default=list(DEFAULT_CUTOFFS))
    p.add_argument("--target-iters", type=int, default=DEFAULT_TARGET_ITERS,
                   help="Target oracle iterations for g-calibration (exact Gibbs).")
    p.add_argument("--max-oracle-iters", type=int, default=DEFAULT_MAX_ORACLE_ITERS,
                   help="Hard cap on oracle iterations per MCMC run.")
    p.add_argument("--gibbs-max-steps", type=int, default=DEFAULT_GIBBS_MAX_STEPS,
                   help="Cap on channel applications per Gibbs preparation.")
    p.add_argument("--random-runs", type=int, default=DEFAULT_RANDOM_RUNS,
                   help="Random-violation runs per criterion (ε/(2R) and fixed θ).")
    p.add_argument(
        "--gibbs-warm-start",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GIBBS_WARM_START,
        help="Warm-start each Gibbs channel from the previous oracle iter (default: on).",
    )
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Worker processes (hard-capped at {MAX_WORKERS}).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse cached calibration.json and raw/*.json results.")
    args = p.parse_args(argv)

    max_iters = int(args.max_oracle_iters)

    # Hard cap on cores: clamp worker count to MAX_WORKERS no matter what was
    # requested (each worker already pinned to a single BLAS thread above).
    requested_workers = max(1, int(args.workers))
    workers = min(requested_workers, MAX_WORKERS)
    if requested_workers > MAX_WORKERS:
        print(f"[driver] --workers {requested_workers} exceeds the hard cap; "
              f"using {MAX_WORKERS} (shared-host limit).")

    return DriverConfig(
        converted_dir=args.converted_dir,
        results_dir=args.results_dir,
        epsilon=float(args.epsilon),
        fixed_gibbs_theta=float(args.fixed_gibbs_theta),
        cutoffs=tuple(args.cutoffs),
        target_iters=int(args.target_iters),
        max_oracle_iters=int(max_iters),
        gibbs_max_steps=int(args.gibbs_max_steps),
        random_runs=int(args.random_runs),
        gibbs_warm_start=bool(args.gibbs_warm_start),
        workers=workers,
        instances=args.instances if args.instances else None,
        skip_existing=bool(args.skip_existing),
    )


if __name__ == "__main__":
    main()
