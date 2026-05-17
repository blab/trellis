# Fitness Metric: Switching to Fraction-Folded × Native Binding Energy

Trevor Bedford — 2026-05-16

## Summary

We are switching the trellis fitness metric from ensemble-averaged binding energy (⟨E_bind⟩) to the standard lattice protein formulation used throughout the literature:

$$F(T) = f(T) \times BE(C_\text{native})$$

where f(T) is the fraction of the conformational ensemble in the native state and BE is the binding energy of the native conformation to the ligand. This change aligns with established practice, enables mean-field pruning for longer chain lengths, and produces effectively equivalent fitness landscapes for evolutionary trajectory generation.

## Previous approach: ensemble-averaged binding

Our original formulation defined fitness as −⟨E_bind⟩, the negated Boltzmann-weighted average of protein-ligand contact energy across all conformations:

$$\langle E_\text{bind} \rangle = \frac{\sum_i E_\text{bind}(i) \cdot \exp(-E_\text{total}(i) / T)}{\sum_i \exp(-E_\text{total}(i) / T)}$$

This is physically clean — it captures the full thermodynamic ensemble without requiring a separate stability criterion. A stably folded protein that contacts the ligand concentrates Boltzmann weight on binding-competent conformations; an unstable protein spreads weight across non-binding conformations and scores poorly. No ad hoc threshold needed.

However, this formulation requires accurate computation of both the numerator (binding-weighted sum) and denominator (partition function Z) over all conformations. This prevents the mean-field pruning of low-contact conformations that is essential for scaling to 18–20 residue chains (see `notes/mean-field-pruning-investigation.md`).

## New approach: fraction-folded × native binding energy

The standard formulation in the lattice protein literature defines fitness as:

$$F(T) = f(T) \times BE(C_\text{native})$$

where:

- $C_\text{native}$ is the lowest-energy conformation (including protein-ligand contacts)
- $BE(C_\text{native})$ is the binding energy of the native conformation to the ligand
- $f(T) = \frac{\exp(-E(C_\text{native}) / T)}{Q(T)}$ is the equilibrium fraction folded to the native state
- $Q(T) = \sum_i \exp(-E(C_i) / T)$ is the partition function over all conformations

This separates fitness into two components: stability (how reliably does the protein fold?) and function (how well does the folded protein bind?). The sigmoidal dependence on ΔG_f means unfolded proteins have near-zero fitness, while proteins above a stability threshold are evaluated primarily on binding.

Critically, mean-field pruning applies cleanly to this formulation. The native state is always a compact conformation with many contacts, so it is always explicitly scored. The mean-field approximation only adjusts Q(T) in the denominator of f(T), and Bloom et al. (2004) demonstrated that this introduces negligible error (RMSE < 0.002 in ΔG_f across 1000 random sequences at T=1.0).

## Literature precedent

The fraction-folded × native binding formulation is used consistently across the lattice protein ligand binding literature. We are not aware of any published lattice protein study that uses ensemble-averaged binding as the evolutionary fitness metric.

**Miller & Dill (1997).** Introduced lattice protein ligand binding. Their analysis explicitly examines when the assumption that "ligands mainly bind to native states" holds. For stable proteins in the lock-and-key and induced-fit regimes — the regime relevant to evolutionary trajectories of functional proteins — native-state binding dominates the ensemble. Edge cases where ensemble effects matter (ligand-induced denaturation, ANS-like non-specific binding) are not the regime we model.

> Miller DW, Dill KA. "Ligand binding to proteins: The binding landscape model." *Protein Science* 6:2166–2179 (1997).

**Williams, Pollock & Goldstein (2001).** Used 16-mer lattice proteins with ligand binding for evolutionary simulations. Fitness maps sequence to stability plus native-state binding propensity.

> Williams PD, Pollock DD, Goldstein RA. "Evolution of functionality in lattice proteins." *J Mol Graph Model* 19:150–156 (2001).

**Bloom, Wilke, Arnold & Adami (2004).** The most directly relevant precedent. Uses 18-mer lattice proteins with the MJ contact potential on a 2D lattice — the same setup as trellis. Fitness is defined as f(T) × BE(C_native). Ligand binding is computed for the native conformation only; the ensemble enters only through the fraction-folded term. Mean-field pruning of low-contact conformations reduces the scored set from 5.81M to 795K conformations (14%) with negligible error.

> Bloom JD, Wilke CO, Arnold FH, Adami C. "Stability and the evolvability of function in a model protein." *Biophysical Journal* 86:2758–2764 (2004). [arXiv:q-bio/0401038](https://arxiv.org/abs/q-bio/0401038)

**Bloom, Silberg, Wilke, Drummond, Adami & Arnold (2005).** Extended the same framework. Explicitly describes the mean-field approximation for low-contact conformations and validates it empirically.

> Bloom JD, Silberg JJ, Wilke CO, Drummond DA, Adami C, Arnold FH. "Thermodynamic prediction of protein neutrality." *PNAS* 102:606–611 (2005). [arXiv:q-bio/0409013](https://arxiv.org/abs/q-bio/0409013)

**Bloom, Labthavikul, Otey & Arnold (2006).** Combines lattice protein simulations with experimental P450 data using the same fitness formulation.

> Bloom JD, Labthavikul ST, Otey CR, Arnold FH. "Protein stability promotes evolvability." *PNAS* 103:5869–5874 (2006).

**Williams, Pollock & Goldstein (2006).** Uses the same native-state binding framework for studying marginal stability and functionality.

> Williams PD, Pollock DD, Goldstein RA. "Functionality and the evolution of marginal stability in proteins: Inferences from lattice simulations." *J Mol Evol* 62:698–706 (2006).

## Why the two metrics are equivalent for our use case

At T=1.0 with MJ energies, our binding thermodynamics analysis (`notes/binding-thermodynamics.md`) showed that for a typical sequence, P(native) ≈ 32% while P(bound) ≈ 99.7%. This means nearly all conformations with significant Boltzmann weight contact the ligand. The ensemble-averaged binding energy is therefore dominated by conformations near the native state, and should correlate strongly with native-state binding energy.

Before fully committing to the switch, we will validate empirically: compute both metrics for 1000 random sequences on 16-mer chains and confirm that the Spearman rank correlation is >0.95. If the fitness landscapes produce equivalent sequence rankings, SSWM trajectories through them will have the same statistical properties.

## Computational impact

Switching to f(T) × BE(C_native) enables Bloom-style mean-field pruning:

- Only conformations with >4 intra-protein contacts are explicitly scored (~14% of total for 18-mers)
- Low-contact conformations contribute to Z via a per-sequence mean-field correction
- The native state and its binding energy are always explicitly computed (compact conformations have many contacts)

This reduces the scored conformation set by ~7× and makes 18-mer and potentially 20-mer production runs feasible:

| Chain | Total conformations | Scored (>4 contacts) | Per-sequence time | Per SSWM step |
|-------|--------------------:|---------------------:|------------------:|--------------:|
| 16 | 3.3M | ~460K (14%) | ~1 ms | ~0.13s |
| 18 | 5.8M (reduced) | ~795K (14%) | ~1.7 ms | ~0.22s |
| 20 | ~60M (reduced) | ~8.4M (14%) | ~18 ms | ~2.3s |

Estimates assume the ~14% high-contact fraction observed for 18-mers scales similarly to other chain lengths. Actual fractions will be determined empirically during enumeration.
