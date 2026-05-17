# Binding Thermodynamics in the Lattice Protein Model

Trevor Bedford — 2026-05-11

## Physical picture vs lattice model

In real protein-ligand binding, binding is a two-step process:

1. **Diffusion**: the protein and ligand move through solution and occasionally encounter each other.
2. **Shape complementarity**: when they meet, the protein's folded shape either makes favorable contacts with the ligand or doesn't.

The binding constant depends on both steps. The lattice model **skips step 1 entirely**. The protein's first residue is always fixed at (0, 0) and the ligand occupies fixed lattice sites nearby. The protein never "floats away" — instead, the partition function sums over all possible conformations of the protein chain in the presence of the fixed ligand. What varies is the protein's conformation (how it folds), not its position.

This is the standard setup in lattice protein literature (e.g. Bloom's [latticeproteins](https://github.com/jbloomlab/latticeproteins)). It models **folding-mediated binding selectivity**: given that the protein is adjacent to the ligand, how strongly does its conformational ensemble favor ligand-complementary shapes?

## Two measures of binding

### Ensemble-averaged binding energy: ⟨E_bind⟩

The branch-and-bound folder (`fold.py`) accumulates, for every conformation $i$:

- $E_\text{intra}(i)$: intra-protein contact energy (sum of MJ contacts between protein residues)
- $E_\text{bind}(i)$: protein-ligand contact energy (sum of MJ contacts between protein and ligand residues)
- $E_\text{total}(i) = E_\text{intra}(i) + E_\text{bind}(i)$

The Boltzmann probability of conformation $i$ is:

$$p(i) = \frac{\exp(-E_\text{total}(i) / T)}{Z}$$

where $Z = \sum_i \exp(-E_\text{total}(i) / T)$ is the partition function summed over all valid conformations (those that don't overlap with ligand sites).

The ensemble-averaged binding energy is:

$$\langle E_\text{bind} \rangle = \sum_i E_\text{bind}(i) \cdot p(i)$$

This answers: **"when the protein is near the ligand, how favorable are the contacts on average across the conformational ensemble?"**

Conformations with many ligand contacts contribute strongly negative $E_\text{bind}(i)$. Conformations with no ligand contacts contribute $E_\text{bind}(i) = 0$. The ensemble average reflects how much of the Boltzmann weight falls on binding-competent conformations.

### Binding free energy: ΔG

A different quantity compares the partition function with and without the ligand:

$$\Delta G_\text{bind} = -T \ln \frac{Z_\text{ligand}}{Z_\text{free}}$$

where $Z_\text{free}$ is the partition function of the protein folding alone (no ligand present, with 8-fold symmetry correction) and $Z_\text{ligand}$ is the partition function of the protein folding in the presence of the fixed ligand (unreduced enumeration, avoiding ligand sites).

This answers the "bouncing around" question more directly: **"how much free energy does the system gain when the protein is adjacent to the ligand vs. free in solution?"** Negative $\Delta G$ means the ligand stabilizes the system.

$\Delta G$ captures both the energetic benefit of favorable contacts *and* the entropic effects:
- The ligand blocks lattice sites, reducing the number of accessible conformations (entropic cost).
- The ligand provides additional favorable contacts for conformations that reach it (energetic benefit).
- The balance determines whether binding is net favorable.

## How they differ

$\langle E_\text{bind} \rangle$ is a pure energy — it measures the average strength of protein-ligand contacts weighted by the full (ligand-present) Boltzmann distribution. It does not account for the conformational entropy cost of folding into a binding-competent shape.

$\Delta G$ is a free energy — it includes both the energetic benefit of ligand contacts and the entropic reorganization of the conformational ensemble. A protein that is already well-folded (high $Z_\text{free}$) gains less from the ligand's stabilizing effect than an intrinsically disordered protein (low $Z_\text{free}$), even if both make the same contacts in their native states.

### Numerical example

Using a 10-residue chain with FWYL ligand at positions (3, −2)...(6, −2), at T = 1.0:

| Sequence | $E_\text{native}^\text{free}$ | $E_\text{native}^\text{lig}$ | $Z_\text{free}$ | $Z_\text{lig}$ | $\Delta G$ | $\langle E_\text{bind} \rangle$ |
|----------|---:|---:|---:|---:|---:|---:|
| `ACDEFGHIKL` | −20.39 | −28.30 | 2.04e10 | 6.12e12 | −5.71 | −19.55 |
| `FFWWLLIIKK` | −21.87 | −29.54 | 4.92e11 | 5.30e13 | −4.68 | −11.81 |
| `KKDDEERRQQ` | −7.22 | −17.46 | 5.14e5 | 2.96e8 | −6.36 | −14.18 |

`KKDDEERRQQ` (charged/polar) has the most favorable $\Delta G$ despite having weaker MJ contacts. It is intrinsically disordered ($Z_\text{free}$ is five orders of magnitude smaller than the hydrophobic sequences), so the ligand provides a large relative stabilization. `FFWWLLIIKK` is already quite stable on its own (high $Z_\text{free}$), so the ligand's relative contribution to the free energy is smaller.

By contrast, $\langle E_\text{bind} \rangle$ ranks `ACDEFGHIKL` as the best binder (−19.55) because it has the strongest average contacts with the ligand, regardless of its intrinsic stability.

## Why we use ⟨E_bind⟩ for fitness

The design doc defines fitness as $f = -\langle E_\text{bind} \rangle$ (negated so that more negative binding = higher fitness). This choice is motivated by:

1. **Biological analogy.** We want fitness to reflect how well a protein binds a target — the selective pressure in the evolutionary model. $\langle E_\text{bind} \rangle$ directly measures the ensemble-averaged strength of protein-ligand interactions. Stability is handled implicitly: a protein that doesn't fold stably is spread across many conformations (most with poor ligand contact), yielding a weak $\langle E_\text{bind} \rangle$ and low fitness.

2. **No separate stability criterion needed.** The ensemble average naturally handles the coupling between stability and binding. A stably folded protein that contacts the ligand concentrates its Boltzmann weight on binding-competent conformations → strong $\langle E_\text{bind} \rangle$. An unstable protein spreads its weight across non-binding conformations → weak $\langle E_\text{bind} \rangle$. A protein with two low-energy states that both bind also scores well. The physics handles all these cases correctly without an ad hoc stability threshold.

3. **Sequence-level selectivity.** $\langle E_\text{bind} \rangle$ rewards the property we care about for generating evolutionary trajectories: how well does this sequence's conformational ensemble contact the ligand? $\Delta G$ mixes in intrinsic stability effects that would confound the fitness landscape — a disordered protein could have favorable $\Delta G$ simply because it has nowhere to go but toward the ligand, not because it has evolved good binding.

4. **Single-pass computation.** $\langle E_\text{bind} \rangle$ is accumulated during the branch-and-bound traversal at no extra cost — we already visit every conformation to compute $Z$. Computing $\Delta G$ requires two separate folds (with and without ligand), doubling the computational cost per fitness evaluation.

## Temperature dependence

At low temperature (T → 0), the ensemble is dominated by the native state, so $\langle E_\text{bind} \rangle$ approaches the binding energy of the native conformation. At high temperature, the protein is unfolded and $\langle E_\text{bind} \rangle$ approaches zero (the average over many random conformations, most of which don't contact the ligand).

Example for `ACDEFGHIKL` with FWYL ligand at (3, −2):

| T | P(native) | P(bound) | $\langle E_\text{bind} \rangle$ |
|---|-----------|----------|------|
| 0.1 | 100% | 100% | −20.29 |
| 0.5 | 82% | 100% | −20.25 |
| 1.0 | 32% | 99.7% | −19.55 |
| 2.0 | 6.8% | 89% | −15.03 |
| 5.0 | 0.4% | 41% | −4.75 |

Here P(native) is the Boltzmann probability of the single lowest-energy conformation, and P(bound) is the fraction of ensemble weight from conformations with at least one ligand contact. As temperature rises, the protein unfolds, fewer conformations contact the ligand, and $\langle E_\text{bind} \rangle$ weakens smoothly toward zero.
