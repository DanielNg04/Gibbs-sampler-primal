# The Block-Diagonal (Ergodicity) Issue in the VSelf Gibbs Channel

**TL;DR.** The abnormally long "convergence" times on `hinf1` were not caused by
numerical instability. The jump operators handed to the quantum Metropolis
channel share a common block-diagonal sparsity structure, so the channel
conserves the probability weight of each block. The true Gibbs state $\rho$
distributes weight across the blocks differently than the initial state does,
so the trace distance $D(\sigma_k, \rho)$ plateaus at the block-weight mismatch
— **above** the convergence target $\theta$ — and every Gibbs preparation burns
its full step cap without ever being able to converge. Adding a single dense
(block-connecting) jump operator restores ergodicity and turns the multi-hour
run into a ~5-second one.

---

## 1. Background: what the channel needs to converge

The sampler (`gibbs_sampler_quantum_v4self.py`) implements the discrete-time
detailed-balanced channel of Gilyén et al. (arXiv:2405.20322):

$$
\mathcal{M}[\sigma] \;=\; \sum_{a=1}^{q} \tilde{A}_a\,\sigma\,\tilde{A}_a^{\mathsf T} \;+\; K\,\sigma\,K^{\mathsf T},
$$

where the $\tilde{A}_a$ are the Bohr-reweighed jump operators (the "accept"
Kraus operators) and $K$ is the single reject Kraus operator. Detailed balance
guarantees that the Gibbs state $\rho \propto e^{-H}$ is **a** fixed point:
$\mathcal{M}[\rho] = \rho$.

But being *a* fixed point is not enough. For `run_until_converged` to work,
$\rho$ must be the **unique** fixed point, and every initial state must be
driven toward it. That is the **ergodicity** (irreducibility) requirement:

> The jump operators must generate transitions that connect the whole space.
> If some subspace decomposition is invariant under **all** jumps, the channel
> decomposes into independent sectors, and the weight in each sector is a
> conserved quantity that no amount of iteration can change.

This is the quantum analogue of a classical Markov chain whose transition
graph has several disconnected components: the chain equilibrates *inside*
each component, but the total probability of each component stays frozen at
its initial value forever.

## 2. Why the mechanism survives every step of the construction

It is worth seeing why nothing in the pipeline breaks the block structure.
Suppose (after a permutation of coordinates) every jump matrix $A_a$ and the
Hamiltonian $M$ are block diagonal with the same partition
$\mathbb{R}^n = V_1 \oplus V_2 \oplus V_3$:

1. **The Hamiltonian is block diagonal by construction.** The oracle builds
   $M = \sum_j y_j A_j$ from the very same constraint matrices used as jumps,
   so $M$ inherits their sparsity. Its eigenvectors then live inside single
   blocks, and the eigenbasis rotation $U$ is itself block diagonal.
2. **Bohr reweighing is basis-local.** $\tilde{A} = U (W \odot U^{\mathsf T} A U) U^{\mathsf T}$
   only multiplies matrix elements by scalars in the energy basis; it cannot
   create matrix elements between blocks that were zero.
3. **The reject Kraus inherits the structure.** $D = \sum_a \tilde{A}_a^{\mathsf T}\tilde{A}_a$
   is block diagonal, hence so is $I - D$, hence so is
   $K = \sqrt{\rho^{1/2}(I-D)\rho^{1/2}}\;\rho^{-1/2}$ (square roots and inverses
   of block-diagonal PSD matrices stay block diagonal).

So **every** Kraus operator of the channel is block diagonal. Writing
$P_b$ for the orthogonal projector onto block $b$, each $P_b$ commutes with
every Kraus operator, and therefore

$$
\operatorname{Tr}\!\big(P_b\, \mathcal{M}[\sigma]\big) \;=\; \operatorname{Tr}\!\big(P_b\, \sigma\big)
\qquad \text{for every state } \sigma \text{ and every block } b .
$$

The block weights $w_b(\sigma) = \operatorname{Tr}(P_b \sigma)$ are exact
conserved quantities of the dynamics — in *exact arithmetic*, not as a
rounding artifact. This is why the behavior looked like "numerical
instability" but was insensitive to every stability-related parameter: it is
a structural property of the input, not an error of the implementation.

## 3. The concrete situation on `hinf1`

`hinf1` has $n = 14$, $R \approx 6.66$, and with $\varepsilon = 0.1$ the
convergence target is $\theta = \varepsilon / (2R) \approx 0.0075$.

### 3.1 The jump set is block diagonal

`jump_matrices_from_arrays` uses the objective $C$ plus the distinct
constraint generators $F_i$. Taking the union of their sparsity patterns and
computing connected components of the resulting graph on the 14 coordinates
gives

```
jump sparsity pattern: 3 connected components; sizes = [4, 4, 6]
```

No jump operator has a nonzero entry linking two different components, so the
channel decomposes into 3 independent sectors as described above.

### 3.2 The channel really has a 3-dimensional fixed-point space

Building the exact superoperator $S = \sum_a A_a \otimes A_a$ (a $196 \times 196$
matrix for $n = 14$) and counting eigenvalues of modulus $\ge 1 - 10^{-9}$:

| oracle depth $N$ | # eigenvalues with $|\lambda| \approx 1$ |
|---:|---:|
| 0 | 3 |
| 100 | 3 |
| 300 | 3 |
| 3131 | 3 |

Exactly one eigenvalue of modulus 1 per block — one conserved weight each.
An ergodic channel would show a single such eigenvalue. Note the count does
not change as the oracle stiffens $M$: the defect is present from iteration 0.

### 3.3 The trace distance plateaus above the target

The oracle starts each Gibbs preparation (cold start) from the maximally
mixed state $\sigma_0 = I/14$, whose block weights are $4/14$, $4/14$, $6/14$.
The true Gibbs state $\rho \propto e^{-M}$ has *different* block weights, and
the mismatch grows as $M$ stiffens (as the oracle keeps adding $\theta A_j$
to it). Since the block weights of $\sigma_k$ cannot move, the trace distance
converges to a strictly positive floor. Measured by iterating the channel
20,000 steps at oracle depth $N = 300$:

```
step      0:  D(sigma, rho) = 0.102935
step     10:  D(sigma, rho) = 0.049282
step    100:  D(sigma, rho) = 0.031907
step   1000:  D(sigma, rho) = 0.031237
step   5000:  D(sigma, rho) = 0.031237
step  10000:  D(sigma, rho) = 0.031237     <- flat forever; target is 0.0075
step  20000:  D(sigma, rho) = 0.031237
```

| oracle depth $N$ | plateau of $D(\sigma_\infty, \rho)$ | target $\theta$ | reachable? |
|---:|---:|---:|:---:|
| 300 | 0.0312 | 0.0075 | no |
| 3131 | 0.1822 | 0.0075 | no |

The channel does converge — quickly, in a few hundred steps — but to the
*wrong* state: the closest state to $\rho$ that has the frozen block weights.
Since that floor exceeds $\theta$, `run_until_converged` can never return
`converged=True` and exhausts the full `gibbs_max_steps = 10 000` cap on
**every** oracle iteration once $M$ is stiff enough (from roughly iteration
50 onward at $g_{\mathrm{lo}}$). That is the entire source of the observed
1–3 hour runtimes and the flat-at-cap Gibbs-steps plot.

Two knock-on effects are worth noting:

- **Warm start does not help.** The warm-started $\sigma_0$ carries whatever
  block weights the *previous* wrong fixed point had, which are again wrong
  for the new $\rho$. The conserved quantities make the error persistent.
- **The oracle trajectory itself is corrupted.** The state handed back to the
  feasibility test is not $\rho_n$, so the measured constraint traces, the
  violation selection, and hence the $y$-updates all deviate from the exact
  trajectory. The MCMC run at $N=300$ reported `feasible=False` where the
  exact-mode run behaves differently.

### 3.4 What was ruled out (the numerics are clean)

At every probed depth $N \in \{0, 25, 50, 100, 300, 1000, 3131\}$:

- the energy spread of $M$ stays modest (0 → 5.8), so **no Boltzmann-weight
  underflow**: the smallest Gibbs probability is $7\times 10^{-4}$ at worst,
  far above `psd_eps = 1e-12`, and the $1/\sqrt{p}$ clamp in the reject-Kraus
  construction never activates;
- the safety rescale never triggers (`jump_scale = 1.0` throughout), so the
  proposal strength is not being crushed;
- $\rho^{1/2}(I-D)\rho^{1/2}$ stays PSD well within tolerance.

The softmax-shifted weights, log-sum-exp $\omega$, symmetrizations, and the
Kraus assembly all behave exactly as designed. Nothing in
`gibbs_sampler_quantum_v4self.py` or `primal_oracle_quantum_v2self.py`
is numerically unstable here. The same defect also explains the earlier
`v1cube` runs: its `_default_jump_matrices` likewise built the jump set only
from the constraint matrices, so those channels were equally non-ergodic.

## 4. The fix: one block-connecting jump operator

Ergodicity only requires that the jump set, taken together, connects all
coordinates. The simplest robust choice is to append **one dense random
symmetric matrix** (a fixed-seed GOE sample) to the jump list:

```python
rng = np.random.default_rng(0)
B = rng.standard_normal((n, n))
G = (B + B.T) / np.sqrt(2.0)        # dense symmetric, connects everything
jump_matrices = jump_matrices_from_arrays(arrays) + [G]
```

A dense $G$ has (with probability 1) nonzero entries between every pair of
coordinates, so the joint sparsity graph becomes a single component, the
superoperator keeps exactly one eigenvalue at modulus 1, and $\rho$ becomes
the unique fixed point. Detailed balance is untouched — the Bohr reweighing
applies to $G$ exactly as to any other jump, so the channel still fixes
$\rho$; it now also *reaches* it.

Measured effect (cold start from $I/n$, target $\theta = 0.0075$):

| oracle depth $N$ | old jumps: steps to $\theta$ | +1 dense jump: steps to $\theta$ |
|---:|---:|---:|
| 300 | never (floor 0.0312) | 118 |
| 3131 | never (floor 0.1822) | 486 |

Full oracle run on `hinf1` at $g_{\mathrm{lo}}$, both caps at 10,000,
warm start on, with the augmented jump set:

```
iterations = 3203   feasible = True    wall time = 5.2 s
Gibbs steps per iteration: mean = 1.04, max = 26, capped runs = 0, all converged
```

With warm start the sampler needs about **one** channel step per oracle
iteration, because consecutive exponents differ by a single rank-one
$\theta$-update and their Gibbs states are extremely close — this is the
behavior the warm start was designed for, and it emerges as soon as the
channel is ergodic.

### Alternatives to a random dense jump

- **Targeted connectors.** Any set of sparse symmetric matrices whose pattern
  bridges the components works, e.g. $e_i e_j^{\mathsf T} + e_j e_i^{\mathsf T}$
  for one pair $(i, j)$ per pair of blocks. Cheaper per Kraus slot, but
  requires computing the components first and is easy to get wrong when new
  instances arrive.
- **The all-ones matrix** $J = \mathbf{1}\mathbf{1}^{\mathsf T}$. Connects
  everything, but is rank one and highly structured; a GOE sample mixes more
  isotropically for the same budget.

## 5. Practical recommendations

1. **Always augment the jump set** with one seeded dense symmetric matrix in
   `jump_matrices_from_arrays` (or in the oracle's jump preparation). This is
   the essential change; everything else is tuning.
2. **Keep `GIBBS_WARM_START = True`.** With an ergodic channel it reduces the
   per-iteration cost to ~1 step.
3. **`GIBBS_MAX_STEPS` can drop to ~2000.** The worst cold-start requirement
   observed was 486 steps; the cap becomes a pure safety net instead of a
   budget that non-ergodic runs silently exhaust.
4. **Fail loudly on non-ergodic jump sets.** A cheap guard at sampler
   construction: build the union sparsity graph of the jumps and check it has
   one connected component (an $O(qn^2)$ boolean pass plus a BFS). This turns
   a silent hours-long stall into an immediate, explainable error. (A block
   structure that only appears in a rotated basis would evade the sparsity
   check, but for SDPA-derived instances the coordinate-basis check catches
   the realistic failure mode.)
5. **Interpret `converged=False` as "wrong state", not "slow mixing".** When a
   Gibbs preparation hits the cap, the returned state may sit at a structural
   floor; downstream quantities (constraint traces, violation choice) are
   then untrustworthy. The per-iteration `gibbs_converged_per_iter` log
   already records this — a run where it is persistently `False` should be
   treated as invalid rather than merely expensive.
