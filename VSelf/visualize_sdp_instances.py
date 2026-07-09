# visualize_sdp_instances.py — sparsity / block-structure plots for converted SDP instances.
#
# Loads the same .npz instances as run_test_v2self.py and reads block metadata from
# the original SDPA .dat-s files (block sizes are not stored in the converted .npz).

"""
Visualise converted SDP instances: sparsity patterns, block coupling, and summary stats.

Each per-instance figure has **7 panels** (sparsity heatmaps, block adjacency, SDPA nnz profile).

Outputs (under ``VSelf/sdp_visualizations/`` by default):
  - ``<instance>_structure.png``  per-instance dashboard
  - ``suite_overview.png``        cross-instance comparison
  - ``suite_stats.json``          numeric summary for all instances

Usage::

    python visualize_sdp_instances.py
    python visualize_sdp_instances.py --instances hinf1 truss1
    python visualize_sdp_instances.py --out-dir path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sdp_conversion import parse_sdpa_sparse  # noqa: E402

# Same benchmark set as run_test_v2self.py
DEFAULT_INSTANCES = ["hinf12", "hinf1", "truss1", "truss4"]

CONVERTED_DIR = os.path.join(_ROOT, "SDP_problems_converted")
SDPA_DIR = os.path.join(_ROOT, "SDP_problems")
DEFAULT_OUT_DIR = os.path.join(_HERE, "sdp_visualizations")

_NNZ_EPS = 1e-12


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_instance(name: str) -> dict:
    """Load a converted instance .npz (same layout as run_test_v2self.load_instance)."""
    d = np.load(os.path.join(CONVERTED_DIR, f"{name}.npz"), allow_pickle=True)
    return {
        "name": str(d["name"]),
        "C": np.ascontiguousarray(d["C"]),
        "A": np.ascontiguousarray(d["A"]),
        "b": np.ascontiguousarray(d["b"]),
        "R": float(d["R"]),
        "n": int(d["n"]),
        "opt": float(d["opt"]),
        "m_oracle": int(d["m_oracle"]),
    }


def load_sdpa_metadata(name: str) -> dict:
    """Block sizes and raw SDPA constraint count from the source .dat-s file."""
    path = os.path.join(SDPA_DIR, f"{name}.dat-s")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SDPA source not found: {path}")
    problem = parse_sdpa_sparse(path)
    return {
        "m_sdpa": problem.m,
        "block_sizes": problem.block_sizes,
        "n": problem.n,
        "F": problem.F,
        "c": problem.c,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


@dataclass
class MatrixStats:
    label: str
    shape: tuple[int, int]
    nnz: int
    density: float
    max_abs: float
    cross_block_nnz: int
    cross_block_fraction: float


def block_offsets(block_sizes: list[int]) -> list[int]:
    abs_sizes = [abs(b) for b in block_sizes]
    offsets: list[int] = []
    running = 0
    for s in abs_sizes:
        offsets.append(running)
        running += s
    return offsets


def support_mask(M: np.ndarray, eps: float = _NNZ_EPS) -> np.ndarray:
    return np.abs(M) > eps


def nnz(M: np.ndarray, eps: float = _NNZ_EPS) -> int:
    return int(np.count_nonzero(support_mask(M, eps)))


def cross_block_nnz(M: np.ndarray, block_sizes: list[int], eps: float = _NNZ_EPS) -> int:
    """Count nonzeros that lie outside the block-diagonal support."""
    n = M.shape[0]
    block_mask = np.zeros((n, n), dtype=bool)
    off = 0
    for s in [abs(b) for b in block_sizes]:
        block_mask[off : off + s, off : off + s] = True
        off += s
    sup = support_mask(M, eps)
    return int(np.count_nonzero(sup & ~block_mask))


def matrix_stats(
    M: np.ndarray,
    label: str,
    block_sizes: list[int],
    eps: float = _NNZ_EPS,
) -> MatrixStats:
    nz = nnz(M, eps)
    total = M.size
    cb = cross_block_nnz(M, block_sizes, eps)
    return MatrixStats(
        label=label,
        shape=(int(M.shape[0]), int(M.shape[1])),
        nnz=nz,
        density=float(nz / total) if total else 0.0,
        max_abs=float(np.max(np.abs(M))) if total else 0.0,
        cross_block_nnz=cb,
        cross_block_fraction=float(cb / nz) if nz else 0.0,
    )


def union_support(matrices: list[np.ndarray], eps: float = _NNZ_EPS) -> np.ndarray:
    if not matrices:
        raise ValueError("union_support needs at least one matrix")
    acc = np.zeros(matrices[0].shape, dtype=bool)
    for M in matrices:
        acc |= support_mask(M, eps)
    return acc


def block_adjacency(support: np.ndarray, block_sizes: list[int]) -> np.ndarray:
    """
    B×B matrix: entry (i,j) is True when some nonzero lies in block pair (i,j).
  """
    abs_sizes = [abs(b) for b in block_sizes]
    B = len(abs_sizes)
    adj = np.zeros((B, B), dtype=bool)
    off_i = 0
    for bi, si in enumerate(abs_sizes):
        off_j = 0
        for bj, sj in enumerate(abs_sizes):
            sl = support[off_i : off_i + si, off_j : off_j + sj]
            adj[bi, bj] = bool(np.any(sl))
            off_j += sj
        off_i += si
    return adj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_block_grid(ax, block_sizes: list[int], n: int) -> None:
    """Light block-boundary grid on an n×n matrix axes."""
    off = 0
    for s in [abs(b) for b in block_sizes]:
        if off > 0:
            ax.axhline(off - 0.5, color="white", linewidth=0.8, alpha=0.85)
            ax.axvline(off - 0.5, color="white", linewidth=0.8, alpha=0.85)
        off += s
    if off < n:
        ax.axhline(off - 0.5, color="white", linewidth=0.8, alpha=0.85)
        ax.axvline(off - 0.5, color="white", linewidth=0.8, alpha=0.85)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)


def plot_binary_heatmap(ax, mask: np.ndarray, title: str, block_sizes: list[int]) -> None:
    n = mask.shape[0]
    ax.imshow(mask, cmap="Greys", interpolation="nearest", vmin=0, vmax=1, aspect="equal")
    draw_block_grid(ax, block_sizes, n)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("column")
    ax.set_ylabel("row")


def plot_signed_heatmap(ax, M: np.ndarray, title: str, block_sizes: list[int]) -> None:
    """Signed log-scale magnitudes for a single matrix (when values matter)."""
    n = M.shape[0]
    with np.errstate(divide="ignore"):
        display = np.sign(M) * np.log10(np.maximum(np.abs(M), _NNZ_EPS))
    display[~support_mask(M)] = np.nan
    vmax = np.nanmax(np.abs(display)) if np.any(np.isfinite(display)) else 1.0
    ax.imshow(display, cmap="RdBu_r", interpolation="nearest",
              vmin=-vmax, vmax=vmax, aspect="equal")
    draw_block_grid(ax, block_sizes, n)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("column")
    ax.set_ylabel("row")


def plot_nnz_bars(ax, labels: list[str], counts: list[int], title: str) -> None:
    x = np.arange(len(labels))
    ax.bar(x, counts, color="#2E86AB", edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("nnz")
    if len(labels) <= 20:
        ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
    else:
        ax.set_xlabel(f"matrix index (0 … {len(labels) - 1})")
        ax.set_xticks([])


def plot_block_adjacency(ax, adj: np.ndarray, title: str) -> None:
    B = adj.shape[0]
    ax.imshow(adj.astype(float), cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(B), [f"B{i + 1}" for i in range(B)], fontsize=7)
    ax.set_yticks(range(B), [f"B{i + 1}" for i in range(B)], fontsize=7)
    ax.set_title(title, fontsize=9)


def render_instance_figure(
    arrays: dict,
    sdpa: dict,
    out_path: str,
) -> dict:
    plt = _setup_matplotlib()

    name = arrays["name"]
    block_sizes: list[int] = sdpa["block_sizes"]
    n = arrays["n"]
    C = arrays["C"]
    A = arrays["A"]
    F_list: list[np.ndarray] = sdpa["F"]

    # Oracle stack: [I, F1, -F1, F2, -F2, …]
    generators = [C] + [A[k] for k in range(1, A.shape[0], 2)]

    union_oracle = union_support([C] + [A[k] for k in range(A.shape[0])])
    union_sdpa = union_support(F_list)
    union_gens = union_support(generators)

    stats = {
        "instance": name,
        "n": n,
        "m_sdpa": sdpa["m_sdpa"],
        "m_oracle": arrays["m_oracle"],
        "block_sizes": block_sizes,
        "n_blocks": len(block_sizes),
        "R": arrays["R"],
        "opt": arrays["opt"],
        "C": asdict(matrix_stats(C, "C (objective)", block_sizes)),
        "oracle_union": asdict(matrix_stats(union_oracle.astype(float), "oracle union", block_sizes)),
        "sdpa_union": asdict(matrix_stats(union_sdpa.astype(float), "SDPA union", block_sizes)),
        "generator_union": asdict(matrix_stats(union_gens.astype(float), "generator union", block_sizes)),
        "sdpa_matrices": [asdict(matrix_stats(F_list[i], f"F_{i}", block_sizes)) for i in range(len(F_list))],
        "oracle_matrices": [asdict(matrix_stats(A[k], f"A[{k}]", block_sizes)) for k in range(A.shape[0])],
    }

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        f"{name}  —  n={n},  SDPA m={sdpa['m_sdpa']},  oracle m={arrays['m_oracle']},  "
        f"blocks={block_sizes}",
        fontsize=11,
        y=0.98,
    )
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # Row 0: objective + union sparsity patterns
    ax1 = fig.add_subplot(gs[0, 0])
    plot_binary_heatmap(ax1, support_mask(C), "C — objective sparsity", block_sizes)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_binary_heatmap(ax2, union_sdpa, "Union support — all SDPA F_i", block_sizes)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_binary_heatmap(ax3, union_gens, "Union — C + distinct generators (+F_i)", block_sizes)

    # Row 1: magnitudes, example constraint, block adjacency
    ax4 = fig.add_subplot(gs[1, 0])
    plot_signed_heatmap(ax4, C, "C — signed log10|entry|", block_sizes)

    ax6 = fig.add_subplot(gs[1, 1])
    if len(F_list) > 1:
        plot_binary_heatmap(ax6, support_mask(F_list[1]), "F1 — first SDPA constraint", block_sizes)
    else:
        ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 2])
    adj = block_adjacency(union_gens, block_sizes)
    plot_block_adjacency(ax7, adj, "Block pairs touched by generators")

    # Row 2: per-matrix nnz (SDPA only)
    ax8 = fig.add_subplot(gs[2, :])
    sdpa_labels = [f"F{i}" for i in range(len(F_list))]
    plot_nnz_bars(ax8, sdpa_labels, [s["nnz"] for s in stats["sdpa_matrices"]], "NNZ per SDPA matrix")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return stats


def render_suite_overview(all_stats: list[dict], out_path: str) -> None:
    plt = _setup_matplotlib()

    names = [s["instance"] for s in all_stats]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("SDP benchmark instances — structural overview", fontsize=12)

    ax = axes[0, 0]
    ax.bar(x - width / 2, [s["n"] for s in all_stats], width, label="n", color="#3D405B")
    ax.bar(x + width / 2, [s["n_blocks"] for s in all_stats], width, label="# blocks", color="#81B29A")
    ax.set_xticks(x, names)
    ax.set_ylabel("count")
    ax.set_title("Dimension vs block count")
    ax.legend()

    ax = axes[0, 1]
    ax.bar(x - width / 2, [s["m_sdpa"] for s in all_stats], width, label="m (SDPA)", color="#2E86AB")
    ax.bar(x + width / 2, [s["m_oracle"] for s in all_stats], width, label="m (oracle)", color="#F2CC8F")
    ax.set_xticks(x, names)
    ax.set_ylabel("constraints")
    ax.set_title("Constraint counts")
    ax.legend()

    ax = axes[1, 0]
    densities = [100.0 * (1.0 - s["C"]["density"]) for s in all_stats]
    ax.bar(x, densities, color="#E07A5F")
    ax.set_xticks(x, names)
    ax.set_ylabel("% zeros")
    ax.set_title("Objective C sparsity (percent structural zeros)")
    ax.set_ylim(0, 105)

    ax = axes[1, 1]
    cross_frac = [100.0 * s["generator_union"]["cross_block_fraction"] for s in all_stats]
    ax.bar(x, cross_frac, color="#6D597A")
    ax.set_xticks(x, names)
    ax.set_ylabel("% of generator nnz")
    ax.set_title("Cross-block entries in generator union")
    ax.set_ylim(0, max(5.0, max(cross_frac) * 1.2 + 1.0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def visualize_instances(
    instances: list[str],
    out_dir: str,
) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    all_stats: list[dict] = []

    for name in instances:
        print(f"[{name}] loading …")
        arrays = load_instance(name)
        sdpa = load_sdpa_metadata(name)
        out_png = os.path.join(out_dir, f"{name}_structure.png")
        stats = render_instance_figure(arrays, sdpa, out_png)
        all_stats.append(stats)
        cb = stats["generator_union"]["cross_block_nnz"]
        print(
            f"[{name}] n={stats['n']} blocks={stats['block_sizes']} "
            f"m_sdpa={stats['m_sdpa']} m_oracle={stats['m_oracle']} "
            f"C sparsity={100*(1-stats['C']['density']):.1f}% "
            f"cross-block nnz={cb} -> {out_png}"
        )

    render_suite_overview(all_stats, os.path.join(out_dir, "suite_overview.png"))
    with open(os.path.join(out_dir, "suite_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2)

    print(f"\nWrote suite overview and stats to {out_dir}")
    return all_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise SDP instance structure and sparsity.")
    parser.add_argument(
        "--instances",
        nargs="+",
        default=DEFAULT_INSTANCES,
        help=f"Instance names (default: {' '.join(DEFAULT_INSTANCES)})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()
    visualize_instances(args.instances, args.out_dir)


if __name__ == "__main__":
    main()
