# Timing metrics (cube modules)

This document describes **what is measured**, **when**, and **what is not included** for the CPU-oriented timing hooks in:

- `primal_oracle_quantum_v1cube.py` — Arora–Kale-style **primal oracle** loop (Gibbs state at each iteration)
- `gibbs_sampler_quantum_v2cube.py` — **coherent-reweighing quantum Gibbs channel** (construction + optional MCMC run)
- `sdp_solver_primal_v1cube.py` — **full primal SDP solve**: binary search on `g` + oracle probes + optional v2 MCMC refinement

All reported times use `time.perf_counter()` (wall time in seconds, monotonic on the host). They are **not** CPU-time or per-thread accounting unless your BLAS library parallelizes inside a timed block (then that work still counts toward the same interval).

---

## 1. `primal_oracle_quantum_v1cube.py`

### How to enable

Pass `collect_timing=True` to `run_primal_oracle(...)` or `primal_oracle(...)`.

The result is attached on `PrimalOracleResult.timing` (or `info["timing"]` in the legacy wrapper). If `collect_timing=False` (default), no per-phase timers run and `timing` is `None`.

### Where the dict lives

| Field | Type | Meaning |
|-------|------|---------|
| `PrimalOracleResult.timing` | `dict[str, float] \| None` | Cumulative seconds over the oracle loop |

### Metrics (keys)

| Key | Aggregation | What is inside the timer |
|-----|-------------|---------------------------|
| **`total_wall_time`** | Set once at end (success or exhaustion) | Entire `run_primal_oracle` call from first line after validation through return, including **setup not listed below** (building `A_stack`, `A_tilde_stack`, initial `H`, tqdm, etc.). |
| **`gibbs_time`** | **Sum** over oracle iterations | One call per iteration to `gibbs_state_from_hamiltonian_quantum(H, n, ...)`. This includes: Hamiltonian checks; conversion to complex; **retry loop** that constructs `QuantumGibbsSampler` from `gibbs_sampler_quantum_v1` (full sampler **construction** each time—eigendecomposition of H, coherent reweighing, Kraus build, reject Kraus); reading `sampler.rho`; real projection and normalization; `_collect_gibbs_diagnostics` (optional full `eigvalsh` on ρ when `check=True`). **Does not** use `gibbs_sampler_quantum_v2cube` or its `collect_timing`. |
| **`trace_check_time`** | **Sum** over iterations | `constraint_traces_from_stack(Xp, A_stack)` — batched `einsum('kij,ij->k', ...)` for all normalized constraints (**O(m n²)**). |
| **`violation_logic_time`** | **Sum** over iterations | `_constraint_diagnostics_from_traces(traces, b_tilde, theta)` — slacks, argmax violation, feasible flag (cheap NumPy on the trace vector). |
| **`hamiltonian_update_time`** | **Sum** over iterations that continue | `H += (beta * theta) * A_tilde_stack[j]` after a violated constraint (rank-one Hamiltonian update). **Not** charged on the final successful iteration. |
| **`result_packaging_time`** | **Once** on success | Copy `X'`, `extract_omega`, scale to `X` and `z` when θ-feasibility is achieved. |

### Per-iteration flow (what gets timed)

```text
for each oracle iteration:
  [gibbs_time]       ρ, gibbs_diag ← gibbs_state_from_hamiltonian_quantum(H)
  [trace_check_time]   traces ← Tr(A_j X') via einsum
  [violation_logic_time]  feasible? which j?
  if feasible:
      [result_packaging_time]  build PrimalOracleResult; set total_wall_time; return
  else:
      y[j] += θ
      [hamiltonian_update_time]  H += (β θ) Ã_j
```

### What is **not** timed separately

- Initial problem embedding (`A_stack`, `A_tilde_stack`, first `H = β einsum(...)`).
- Progress bar (`tqdm`) overhead.
- Any work inside `QuantumGibbsSampler` beyond the outer `gibbs_time` wrapper (v1 has no `collect_timing` in this call path).
- MCMC / `apply_channel` / `run_timed` — the primal oracle only needs the **Gibbs matrix** `ρ`, not channel trajectories.

### Interpreting totals

- **`gibbs_time`** usually dominates when `dim = n+1` is large, because each iteration **rebuilds** a full quantum Gibbs sampler to obtain `exp(-βH)/Z` (same math as classical Gibbs, expensive construction).
- **`total_wall_time`** ≥ sum of the named buckets; the gap is setup + untimed loop overhead.
- On failure (`return None`), `result_packaging_time` stays `0`; `total_wall_time` is still set.

---

## 2. `gibbs_sampler_quantum_v2cube.py`

Two independent timing surfaces: **sampler construction** (`collect_timing` on `__init__`) and **channel trajectory** (`run_timed`). The ordinary `run()` method records **no** timing (its docstring points to `run_timed` for benchmarks).

`channel_diagnostics()` performs extra eigendecompositions and a fixed-point channel apply; it is **never** included in `self.timing` or `run_timed` unless you call it yourself outside those APIs.

### 2a. Construction timing — `QuantumGibbsSampler(..., collect_timing=True)`

#### How to enable

`collect_timing=True` in `QuantumGibbsSampler.__init__`. Results are stored in **`sampler.timing`** (`dict[str, float]`). If `collect_timing=False`, `self.timing` stays `{}` and no construction intervals are recorded.

#### Metrics (keys)

| Key | What is inside the timer |
|-----|---------------------------|
| **`total_init_time`** | Full `__init__`: validation, symmetrization of `H`, all steps below, optional verbose print. |
| **`h_eigh_time`** | `scipy.linalg.eigh(self.H)` — energy eigenvalues and eigenvectors for Bohr / Gibbs weights. |
| **`gibbs_state_time`** | `_build_gibbs_state()` — shifted Boltzmann weights `exp(-β(E_i - E_min))`, build ρ in eigenbasis, symmetrize; stores `rho_min_eig`, `rho_max_eig`, `rho_condition_number`, scaled partition function. |
| **`coherent_reweigh_time`** | For each jump: `coherent_reweigh_jump(...)`; then `ascontiguousarray` on accept Kraus list. |
| **`accept_rescale_time`** | Optional `_rescale_accept_kraus` so `λ_max(Σ Ã† Ã) ≤ 1 - margin`; refresh contiguous accept Kraus and build **`accept_kraus_stack`** `(num_jumps, dim, dim)`. |
| **`D_accept_build_time`** | `D_accept = Σ Ã† Ã` (dual channel on identity); check `λ_max(D) ≤ 1`. |
| **`reject_kraus_time`** | `_compute_reject_kraus_explicit(D_accept)` — PSD square roots / `eigh` on `I - D`, build reject Kraus `K`. |

Construction order in code:

```text
total_init_time  ⊃  validate H, jumps
                 ⊃  h_eigh_time
                 ⊃  gibbs_state_time
                 ⊃  coherent_reweigh_time
                 ⊃  accept_rescale_time
                 ⊃  D_accept_build_time
                 ⊃  reject_kraus_time
```

Sub-keys are **non-overlapping** slices of `__init__`; their sum is approximately `total_init_time` (small untimed gaps: eigenvalue check on `D`, verbose I/O).

### 2b. Channel run timing — `run_timed(...)`

#### How to enable

Call `sampler.run_timed(sigma0, steps, ...)` instead of `run(...)`. Returns `(TrajectoryResult, timing_run)`.

#### Metrics (keys)

| Key | What is inside the timer |
|-----|---------------------------|
| **`total_run_time`** | Entire `run_timed` call: initial state prep, loop, final snapshot row, building `timing_run`. |
| **`apply_channel_time`** | **Sum** of `_apply_channel_fast` only: `M[σ] = Σ_a Ã_a σ Ã_a† + K σ K†` (BLAS matmul loop + optional Hermitian symmetrization). This is the **MCMC / channel propagation** cost you usually want for scaling studies. |
| **`time_per_step`** | `apply_channel_time / max(steps, 1)` — average seconds per channel application (not per diagnostic row). |
| **`diagnostics_time`** | **Sum** of intervals where **trace distance** and/or **Frobenius distance** to `sampler.rho` are computed. Each uses `eigvalsh` (trace distance) or Frobenius norm. Only runs on “snapshot” steps (see schedule below). |
| **`observable_time`** | **Sum** of `expectation_value(sigma, O)` via `einsum('ij,ji->', ...)` for each observable, on the same snapshot schedule as distances when `observables` is set. |
| **`steps`** | Count of channel applications requested (metadata, not a duration). |
| **`dim`** | Hilbert space dimension (metadata). |
| **`num_jumps`** | Number of accept Kraus operators (metadata). |

#### Snapshot schedule (what affects `diagnostics_time` / `observable_time`)

- `diagnostic_every = K > 0`: compute requested distances / observables at rows with `k % K == 0` or `k == steps`.
- `diagnostic_every is None` and distance flags set: distances only at `k ∈ {0, steps}`.
- Observables follow the same schedule when `observables` is not `None`.
- Steps with no snapshot still append `nan` / `None` to trajectory lists but **do not** add to `diagnostics_time` or `observable_time`.

#### What is **not** in `run_timed` buckets

- `store_states` copies (listed in trajectory, not timed separately).
- Initial trace normalization of `sigma0`.
- Building the returned `TrajectoryResult` lists.

So:

```text
total_run_time  ≈  apply_channel_time + diagnostics_time + observable_time + small overhead
```

### 2c. Methods without automatic timing

| API | Timing |
|-----|--------|
| `apply_channel` / `_apply_channel_fast` | No built-in timer (use `run_timed` or wrap externally). |
| `run` | No timing; may compute trace distance **every** step if enabled — prefer `run_timed` for benchmarks. |
| `channel_diagnostics()` | On-demand validation only (CPTP residual, fixed point, spectral stats). |

---

## 3. `sdp_solver_primal_v1cube.py`

End-to-end solver for

```text
max Tr(C X)  s.t.  Tr(A_j X) ≤ b_j,  X ⪰ 0,  A_1 = I,  b_1 = R
```

using **binary search on `g`** (AG paper, Section 2.2 style) and, optionally, **v2cube channel MCMC** to refine the primal after a feasible oracle probe.

### How to enable

Call :func:`solve_sdp` with ``SDPSolverConfig(collect_timing=True)`` (default). Timings are on ``SDPSolverResult.timing``; per–binary-search-step rows are in ``SDPSolverResult.iteration_log``.

Subroutine flags:

| Subroutine | When it runs | Sub-timing source |
|------------|----------------|-------------------|
| `run_primal_oracle` | Every bisection probe at `g_mid` | §1 — `PrimalOracleResult.timing` **aggregated** into solver keys below |
| `QuantumGibbsSampler` **v2** + `run_timed` | If `mcmc_steps > 0` | §2 — optional init/run keys aggregated into `mcmc_*` |

Set ``mcmc_steps=0`` to skip v2 entirely (oracle-only solve).

### Solver-level metrics (`SDPSolverResult.timing`)

| Key | Aggregation | What is inside the timer |
|-----|-------------|---------------------------|
| **`total_wall_time`** | Once at end | Entire :func:`solve_sdp` call (bracket setup, all probes, optional MCMC, result assembly). |
| **`binary_search_wall_time`** | Once | Wall time of the bisection loop body (from first probe through last probe / break). |
| **`binary_search_logic_time`** | Sum over probes | Bisection bookkeeping **excluding** the wall time inside `run_primal_oracle` (computed as step wall minus `oracle_runtime_sec`). |
| **`primal_oracle_wall_time`** | **Sum** over probes | Wall time wrapping each `run_primal_oracle(...)` call (≥ oracle's own `total_wall_time`). |
| **`primal_oracle_gibbs_time`** | **Sum** over successful probes' oracle timings | Roll-up of §1 `gibbs_time`. |
| **`primal_oracle_trace_check_time`** | Sum | Roll-up of §1 `trace_check_time`. |
| **`primal_oracle_violation_logic_time`** | Sum | Roll-up of §1 `violation_logic_time`. |
| **`primal_oracle_hamiltonian_update_time`** | Sum | Roll-up of §1 `hamiltonian_update_time`. |
| **`primal_oracle_result_packaging_time`** | Sum | Roll-up of §1 `result_packaging_time`. |
| **`mcmc_wall_time`** | Sum over MCMC runs | Wall time around each `_run_mcmc_refinement` (v2 init + `run_timed` + averaging). |
| **`mcmc_sampler_init_time`** | Sum | Roll-up of v2 §2a `total_init_time` (when `mcmc_collect_channel_timing=True`). |
| **`mcmc_apply_channel_time`** | Sum | Roll-up of v2 §2b `apply_channel_time`. |
| **`mcmc_run_total_time`** | Sum | Roll-up of v2 §2b `total_run_time`. |
| **`mcmc_diagnostics_time`** | Sum | Roll-up of v2 §2b `diagnostics_time` (usually 0 if distances disabled). |
| **`mcmc_init_*` / `mcmc_run_*`** | Sum | Optional detail keys mirroring v2 construction/run sub-timers (e.g. `mcmc_init_coherent_reweigh_time`). |
| **`binary_search_steps`** | Count | Number of bisection iterations executed (metadata). |
| **`primal_oracle_calls`** | Count | Number of `run_primal_oracle` invocations. |
| **`primal_oracle_successes`** | Count | Probes that returned a feasible `PrimalOracleResult`. |
| **`mcmc_steps_executed`** | Count | Total channel steps requested (burn-in + steps) across MCMC runs. |

### Per–binary-search-step flow

```text
solve_sdp:
  for each bisection step at g_mid:
    [primal_oracle_wall_time]  run_primal_oracle(..., collect_timing=True)
         └─ internally §1 timers (gibbs_time, trace_check_time, …)
    if feasible and mcmc_steps > 0 and not mcmc_on_final_only:
         [mcmc_wall_time]  v2 sampler + run_timed from oracle (y, ρ)
    update g_lo / g_hi; append iteration_log row
  if mcmc_on_final_only and mcmc_steps > 0 and last feasible:
         [mcmc_wall_time]  one final v2 refinement
  set total_wall_time
```

### `iteration_log` rows (benchmark-friendly)

Each bisection probe appends one dict with fields such as:

- `phase` = `"binary_search"`
- `binary_search_step`, `candidate_value` (= `g_mid`), `g_lo`, `g_hi`, `feasible`
- `objective_estimate` (= best `Tr(C X)` so far)
- `primal_oracle_iterations`, `oracle_runtime_sec`, `gibbs_runtime_sec` (from that probe's oracle)
- `iteration_runtime_sec`, `cumulative_runtime_sec`

These align with SDPLIB benchmark CSV columns where applicable.

### What is **not** timed at solver level

- Default bracket heuristics (`eigvalsh` on `C` for `g_hi`) before the loop.
- Building `SDPInstance` / validation.
- Failed oracle calls still charge `primal_oracle_wall_time` but contribute **no** §1 sub-aggregates (oracle returned `None`).

### Interpreting totals

```text
total_wall_time  ≈  binary_search_wall_time  +  setup  +  (final MCMC if deferred)
binary_search_wall_time  ≈  primal_oracle_wall_time  +  mcmc_wall_time  +  binary_search_logic_time
primal_oracle_wall_time  ≥  primal_oracle_gibbs_time  +  other oracle buckets
```

For **benchmark CSV** mapping:

| Benchmark-style name | Solver key |
|----------------------|------------|
| `time_binary_search_sec` | `binary_search_wall_time` (or `total_wall_time` − MCMC) |
| `time_primal_oracle_sec` | `primal_oracle_wall_time` |
| `time_gibbs_state_sec` | `primal_oracle_gibbs_time` |
| `time_mcmc_sampling_sec` | `mcmc_apply_channel_time` |

---

## 4. Using the stack together

Typical call chain:

1. **`solve_sdp`** (`sdp_solver_primal_v1cube`) — bisection + aggregated timing.
2. Each probe → **`run_primal_oracle`** (`primal_oracle_quantum_v1cube`) — uses **v1** Gibbs for `ρ` (§1).
3. Optional refinement → **`run_timed`** (`gibbs_sampler_quantum_v2cube`) — discrete channel MCMC (§2).

---

## 5. Quick reference tables

### Primal oracle (`collect_timing=True`)

| Key | Per-iter sum? |
|-----|----------------|
| `gibbs_time` | Yes |
| `trace_check_time` | Yes |
| `violation_logic_time` | Yes |
| `hamiltonian_update_time` | Yes (except last success iter) |
| `result_packaging_time` | No (once) |
| `total_wall_time` | No (end-to-end) |

### Gibbs sampler v2cube

| Phase | Enable | Dict |
|-------|--------|------|
| Construction | `collect_timing=True` in `__init__` | `sampler.timing` |
| MCMC / channel | `run_timed(...)` | second return value `timing_run` |

| Construction key | Run key |
|------------------|---------|
| `total_init_time` | `total_run_time` |
| `h_eigh_time` | `apply_channel_time` |
| `gibbs_state_time` | `time_per_step` |
| `coherent_reweigh_time` | `diagnostics_time` |
| `accept_rescale_time` | `observable_time` |
| `D_accept_build_time` | `steps`, `dim`, `num_jumps` |
| `reject_kraus_time` | |

### SDP solver (`solve_sdp`, `collect_timing=True`)

| Key | Per-probe sum? |
|-----|----------------|
| `primal_oracle_wall_time` | Yes (all probes) |
| `primal_oracle_gibbs_time` | Sum over feasible probes only |
| `mcmc_wall_time` | Per MCMC invocation |
| `binary_search_wall_time` | Once (loop) |
| `total_wall_time` | Once (entire solve) |
