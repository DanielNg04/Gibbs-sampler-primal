# Parameter choices for the primal-oracle benchmark

This note explains the parameters that govern a benchmark run — the trace bound
`R`, the precision `ε`, the derived convergence target `θ`, the objective
threshold `g`, and the inverse temperature `β` — how each one is currently
chosen, and **which problems those choices create**. It is meant to be read
alongside `sdp_conversion.py` (where `R` is set) and `run_test.py` /
`primal_oracle_quantum_v1cube.py` (where `θ`, `g`, and `β` are used).

The numbers in the tables come from the converted SDPLIB set
(`SDP_problems_converted/manifest.json`).

---

## 1. The trace bound `R`

**Definition (current).** In `sdp_conversion.trace_bound_from_opt`:

```
R = R_margin · max(|OPT|, 1),   R_margin = 2 (default)
```

`R` becomes the right-hand side of the trace constraint `A_1 = I`, `b_1 = R`,
i.e. it forces every candidate `X` to satisfy `Tr(X) ≤ R`. The whole oracle then
works on the normalized variable `X' = X/R`.

**Role.** `R` does two things at once:

1. it defines the **feasible region** (the trace ball `Tr(X) ≤ R` that must
  contain the optimizer `X*`), and
2. it sets the **scale of `θ`** through `θ = ε/(2R)` (Section 2).

**Problems with the current choice.**

- **It conflates the objective value with the trace.** `R` is supposed to
upper-bound `Tr(X*)` (the *size* of the optimizer). But `|OPT| = |Tr(C X*)|`
is a *different* quantity — it depends on `C`. There is no general inequality
`Tr(X*) ≤ 2|OPT|`. So `R = 2|OPT|` is a heuristic, not a guarantee:
  - If `‖C‖` is small but `Tr(X*)` is large, `R` can be **too small** and clip
  the true optimizer out of the trace ball → the feasibility test becomes
  ill-posed (the oracle can report infeasibility for a `g` that is actually
  achievable, or never reach `θ`-feasibility).
  - If `‖C‖` is large (objective scale ≫ trace scale), `R` is **too large**,
  which makes `θ` tiny and blows up the Gibbs step count (Section 2).
- **It is OPT-driven, so it inherits the OPT table's scale.** For the `hinf`
family the listed optima are large (`hinf4` OPT ≈ 274.8 ⇒ `R ≈ 549.5`), giving
the smallest `θ` and the worst Gibbs mixing cost in the whole set.

**Mitigation.** `R` is fully overridable per run (`--R` in `sdp_conversion.py`,
and the driver can override it too). For thesis runs it is often better to set
`R` from an actual bound on `Tr(X*)` (e.g. a known feasible point) than from
`|OPT|`, or to deliberately pick a *small* `R` to keep `θ` workable and accept
that the trace ball is then only a benchmark device rather than a certified
bound.

---

## 2. The precision `ε` and the convergence target `θ = ε/(2R)`

**Definition.** In `run_primal_oracle` (`primal_oracle_quantum_v1cube.py`):

```
θ      = ε / (2R)
#iters = ⌈ ln(n) / θ² ⌉          (paper schedule, Section 2.2)
```

`θ` is used in **three** places:

1. as the **per-iteration step size** of the mirror-descent update
  `y ← y + θ e_j` on the most-violated constraint,
2. as the **feasibility tolerance** (a state is "θ-feasible" when no constraint
  is violated by more than ≈ `θ`), and
3. **most importantly for cost**, as the **MCMC convergence target**: in
  `gibbs_mode="mcmc"` the channel `M[σ]` is iterated from the maximally mixed
   state `I/d` until `trace_distance(σ, ρ_exact) ≤ θ`.

**The epsilon sweep is `(0.1, 0.05, 0.01)`** (`DEFAULT_EPSILONS`), with `0.01`
as the reference used by experiments 2.1.3 / 2.1.4 and by `g`-calibration.

**Problems with the current choice.**

- `**θ` is extremely small, because `R` is large.** With `R` in the tens-to-
hundreds, even the *loosest* `ε = 0.1` gives `θ ∼ 10⁻⁴`. Concretely:

  | instance       | n   | R     | θ (ε=0.1) | θ (ε=0.05) | θ (ε=0.01) |
  | -------------- | --- | ----- | --------- | ---------- | ---------- |
  | hinf1          | 14  | 4.07  | 1.2e-2    | 6.1e-3     | 1.2e-3     |
  | truss1         | 13  | 18.0  | 2.8e-3    | 1.4e-3     | 2.8e-4     |
  | ThetaPrimeER23 | 116 | 192.5 | 2.6e-4    | 1.3e-4     | 2.6e-5     |
  | hinf4          | 16  | 549.5 | 9.1e-5    | 4.6e-5     | 9.1e-6     |

- **The theoretical step schedule `⌈ln n/θ²⌉` is astronomically large.** Because
it scales as `1/θ² = (2R/ε)²`, the paper's own step count is unusable as a
literal budget:

  | instance       | θ (ε=0.01) | ⌈ln n / θ²⌉ (steps) |
  | -------------- | ---------- | ------------------- |
  | hinf1          | 1.2e-3     | ~1.7 × 10⁶          |
  | ThetaPrimeER23 | 2.6e-5     | ~7 × 10⁹            |
  | hinf4          | 9.1e-6     | ~3 × 10¹⁰           |

  This is why the driver/oracle imposes `gibbs_max_steps` (a hard cap) instead of
  the schedule. **Consequence:** for the large-`R` instances the channel will
  almost never actually reach `trace_distance ≤ θ` within the cap. The recorded
  "steps to convergence" then *saturate at the cap*, and the convergence flag is
  `False`. Read the 2.1.1 spike plots accordingly: flat-topped spikes at the cap
  mean "did not converge to θ", not "θ reached at exactly that step".
- **The `ε` values mostly rescale cost, not behavior.** Since the mixing cost
grows like `1/θ² ∝ R²/ε²`, going from `ε = 0.1` to `ε = 0.01` multiplies the
*target* step count by `(0.1/0.01)² = 100`. The three-point sweep
`0.1 / 0.05 / 0.01` therefore spans ~`1 : 4 : 100`in target steps — useful for showing the`1/ε²`trend, but expect the smallest`ε`to be the one that pins against`gibbs_max_steps`.

**Mitigation.** Keep `R` small (Section 1) so `θ` is not microscopic; set
`gibbs_max_steps` consciously (it is the real runtime knob, not the paper
schedule); and interpret capped runs as lower bounds on the true mixing time.

---

## 3. The objective threshold `g`

**Definition.** `g` enters the oracle as the extra constraint `A_0 = −C`,
`b_0 = −g`, turning "is objective value `g` achievable?" into a feasibility
question. The benchmark does **one** feasibility check per run (no binary search
over `g`); instead `g` is *calibrated once* per instance.

`run_test.calibrate_g` runs a **bisection on `g`** between

```
g_lo = min(OPT, 0) − (|OPT| + 1)          (deeply feasible)
g_hi = R · λ_max(C)                         (spectral upper bound, infeasible)
```

using the cheap `gibbs_mode="exact"` Gibbs state, and keeps the **largest `g`
whose iteration count stays ≤ `target_iters`** (e.g. ~1000). The intent: pick `g`
just inside feasibility so the single run naturally takes ~`target_iters`
iterations rather than terminating immediately.

**Problems with the current choice.**

- `**g` controls oracle *iterations*, not Gibbs *steps*.** Calibration tunes how
many mirror-descent iterations the oracle performs, but each iteration's
channel-mixing cost is governed by `θ` (Section 2), independent of `g`. So
calibrating `g` makes the *x-axis* of the 2.1.1 plots ≈ `target_iters` long,
but does nothing to bound the *y-axis* (steps per iteration).
- **The bracket can degenerate.** `g_hi = R·λ_max(C)` assumes `λ_max(C) > 0`. For
the minimization-derived duals here, `C = F_0` can be negative (semi)definite,
making `g_hi ≤ 0` and possibly `g_lo ≥ g_hi`. The code repairs this case
(`g_lo = g_hi − (|g_hi| + 1)`), but the resulting `g` is then only a
"make-it-run-~N-iterations" device, not a meaningful objective certificate.
- **Calibration uses the exact Gibbs state, the real runs use MCMC.** The
iteration count that `g` is tuned against (exact mode) can differ slightly from
the MCMC run's iteration count, because the MCMC state is only `θ`-close. With
capped (non-converged) Gibbs states this gap can grow.
- `**g` depends on the OPT table being correct and correctly signed.** The OPT
CSV mixes signs/encodings (some entries are quoted with tabs; `control`/`hinf`
optima are listed positive). The loader now parses them robustly, but if a sign
convention is off, the `[g_lo, g_hi]` bracket is placed wrongly and calibration
yields a `g` that under- or over-shoots the intended ~`target_iters`.

**Mitigation.** Treat the calibrated `g` as an experiment-design knob (sets the
iteration budget), not as a tight bound on OPT. If a particular instance's
calibration lands far from `target_iters`, override `g` directly or widen the
bisection bracket.

---

## 4. The inverse temperature `β`

`β = 1` is **fixed** (the target Gibbs state is `exp(−M)/Z` with the `n×n`
exponent `M` built up as `M += θ A_j` over iterations). The oracle warns if
`β ≠ 1`, because Section 2.2 of the paper assumes unit inverse temperature; a
different `β` rescales `M` and changes both the target state and the mixing time.
There is no reason to change it for these benchmarks.

---

## 5. Summary: what to watch


| parameter | set by                       | governs                                     | main failure mode                                                |
| --------- | ---------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| `R`       | `2·max(                      | OPT                                         | ,1)` (overridable)                                               |
| `ε`       | `{0.1, 0.05, 0.01}`          | precision / `θ`                             | smaller `ε` ⇒ `1/ε²` more steps ⇒ caps out                       |
| `θ=ε/2R`  | derived                      | step size, feasibility tol, **MCMC target** | so small it is unreachable ⇒ steps saturate `gibbs_max_steps`    |
| `g`       | bisection to ~`target_iters` | # oracle iterations                         | decoupled from step count; bracket degenerates if `λ_max(C) ≤ 0` |
| `β`       | `1` (fixed)                  | Gibbs temperature                           | leave at 1                                                       |


**The single most important coupling:** `θ = ε/(2R)` and the cost `∝ 1/θ² = (2R/ε)²`. Both knobs you actually choose — `R` (via the margin) and `ε` — feed
this quadratically. Keeping `R` modest is the highest-leverage way to make the
Gibbs sampler converge within a sane `gibbs_max_steps`; otherwise the benchmark
measures "steps until the cap" rather than "steps until `θ`".