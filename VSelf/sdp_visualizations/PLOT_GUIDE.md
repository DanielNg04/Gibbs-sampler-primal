# SDP instance structure plots — guide

This document explains the **7 panels** on each per-instance figure produced by
`visualize_sdp_instances.py` (e.g. `hinf1_structure.png`). The script visualises
the converted oracle instances used in `run_test_v2self.py`, together with block
metadata from the original SDPA `.dat-s` files.

## Background: two matrix stacks

Each instance is stored in two related forms:

| Form | Matrices | Meaning |
|------|----------|---------|
| **SDPA (raw)** | `F₀, F₁, …, F_m` | Standard sparse SDPA format. `F₀` is the objective; `F₁…F_m` are equality constraints `tr(F_i Y) = c_i`. |
| **Oracle (converted)** | `A[0], A[1], …, A[M−1]` | Form used by the primal oracle. `A[0] = I` encodes the trace bound `Tr(Y) ≤ R`. Each SDPA equality becomes two inequalities: `tr(F_i Y) ≤ c_i` and `tr(−F_i Y) ≤ −c_i`, so `A = [I, F₁, −F₁, F₂, −F₂, …]`. |

The objective matrix in the converted file is `C = F₀`.

**Block structure** comes from the SDPA header (e.g. `hinf1` has blocks `[4, 4, 6]`, so
`n = 14`). Every matrix is `n × n` symmetric. White grid lines on the heatmaps mark
block boundaries; the block-size list also appears in the figure title.

**Sparsity convention:** an entry counts as nonzero if `|value| > 10⁻¹²`. Binary
heatmaps show only the *pattern* (zero vs nonzero), not magnitudes.

---

## Layout (3 rows × 3 columns)

```
 Row 0   [1 C sparsity]  [2 SDPA union]  [3 Generator union]
 Row 1   [4 C magnitudes] [5 F₁]          [6 Block adjacency]
 Row 2   [7 NNZ per SDPA — full width]
```

---

## Panel 1 — C — objective sparsity

**Type:** binary heatmap (`n × n`).

**What it shows:** which entries of the objective matrix `C = F₀` are nonzero
(black) vs exactly zero (white).

**How to read it:**

- Rows and columns index matrix entries `(i, j)`.
- White grid lines = block boundaries from the SDPA partition.
- Symmetry is visible: the pattern is symmetric about the main diagonal.

**Why it matters:** the objective often has a much sparser pattern than the
constraints (e.g. `truss1` has only one nonzero in `C`). This panel shows *where*
the objective lives inside the block layout.

---

## Panel 2 — Union support — all SDPA F_i

**Type:** binary heatmap.

**What it shows:** the union of nonzero positions across **all** SDPA matrices
`F₀, F₁, …, F_m`. If *any* `F_i` has a nonzero at `(i, j)`, that pixel is black.

**How to read it:**

- This is the combined sparsity *envelope* of the raw problem, before conversion to
  inequalities.
- Compare with panel 1: new black regions are entries that appear only in
  constraint matrices, not in the objective.

**Why it matters:** gives a single picture of which matrix coordinates ever appear
in the SDPA file, independent of how many constraints touch each entry.

---

## Panel 3 — Union — C + distinct generators (+F_i)

**Type:** binary heatmap.

**What it shows:** the union of nonzero positions for:

1. the objective `C`, and
2. the **distinct positive generators** `F₁, F₂, …, F_m` (one per SDPA constraint).

This matches the jump-operator set used in `run_test_v2self.py` (objective plus
`A[1], A[3], A[5], …`), **before** adding the random-cycle connectors.

**How to read it:**

- Differs from panel 2 only in that `F₀` and the `+F_i` generators are shown
  without double-counting negated copies (`−F_i` has the same support as `+F_i`).
- Excludes the identity `A[0] = I` (full diagonal), which is always present in
  the oracle stack but not among the problem-native jump operators.

**Why it matters:** for the quantum Gibbs channel, these are the problem-native
jump matrices. If this union is block-diagonal (no black pixels off the block
diagonal), the channel cannot move probability between blocks without extra
connectors — see `BLOCK_DIAGONAL_ISSUE.md`.

---

## Panel 4 — C — signed log₁₀|entry|

**Type:** colour heatmap (red–blue).

**What it shows:** the same matrix as panel 1, but now **magnitudes** matter:

- colour = `sign(entry) × log₁₀|entry|`
- red ≈ positive, blue ≈ negative
- white = exactly zero (masked out)

**How to read it:**

- Darker colour = larger magnitude on a log scale.
- Block boundaries are still drawn.
- Useful when panel 1 is mostly black: here you can see whether entries are
  `±1`, order-one, or very small/large.

**Why it matters:** sparsity alone hides coefficient scale. This panel shows
whether the objective is dominated by a few large entries or many moderate ones.

---

## Panel 5 — F₁ — first SDPA constraint

**Type:** binary heatmap.

**What it shows:** sparsity pattern of `F₁`, the **first equality constraint**
matrix from the SDPA file (not the objective).

**How to read it:**

- Representative example of a single constraint pattern.
- For `truss` instances, many constraints look similar; for `hinf`, constraints
  often differ substantially (compare panel 7).

**Why it matters:** a concrete example constraint, easier to interpret than the
full union in panel 2.

---

## Panel 6 — Block pairs touched by generators

**Type:** `B × B` binary heatmap (`B` = number of blocks).

**What it shows:** which **pairs of blocks** `(B_i, B_j)` have at least one
nonzero entry somewhere in the generator union (panel 3) lying in that block pair.

**How to read it:**

- Axes label blocks `B1 … B_B`.
- Blue cell `(i, j)` = some generator has a nonzero in block row `i`, block column `j`.
- The diagonal `(i, i)` = block `i` has some on-diagonal nonzero in a generator.
- Off-diagonal blue = **cross-block coupling** in the generator set.

**Why it matters:**

- If the matrix is **block-diagonal** (only diagonal cells blue), generators never
  link different blocks — the Gibbs channel preserves block weights.
- On the current benchmark set, off-diagonal cells are empty (no cross-block
  coupling), which is why random-cycle connectors were added in the sampler driver.

---

## Panel 7 — NNZ per SDPA matrix

**Type:** bar chart (full width).

**What it shows:** **nnz** (number of nonzeros) for each SDPA matrix `F₀, F₁, …, F_m`.

**How to read it:**

- `F0` = objective; `F1 … Fm` = constraints.
- Tall bars = relatively dense matrices; short bars = very sparse.
- For instances with many constraints (`hinf12`: `m = 43`), x-axis labels are
  omitted and the x-axis is indexed `0 … m`.

**Why it matters:** summarises per-matrix sparsity when plotting every `F_i` as a
heatmap would be unwieldy. Quickly shows whether a few constraints dominate the
total entry count.

---

## Figure title (banner)

Above the grid, the suptitle reports:

- **instance name**
- **`n`** — total matrix dimension
- **`SDPA m`** — number of equality constraints in the `.dat-s` file
- **`oracle m`** — number of oracle matrices `M = 2m + 1`
- **`blocks`** — SDPA block-size list (replaces the old block-partition bar chart)

---

## Related outputs (not part of the 7 panels)

| File | Contents |
|------|----------|
| `suite_overview.png` | Four comparison bar charts across all instances (dimension, constraints, C sparsity, cross-block fraction). |
| `suite_stats.json` | Machine-readable nnz, density, and cross-block counts for every matrix (including oracle matrices not plotted). |

---

## Regenerating the figures

```bash
python VSelf/visualize_sdp_instances.py
python VSelf/visualize_sdp_instances.py --instances hinf1 truss1
```

Default instance list matches `run_test_v2self.py`: `hinf12`, `hinf1`, `truss1`, `truss4`.
