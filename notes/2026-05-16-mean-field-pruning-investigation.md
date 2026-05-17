# Mean-Field Pruning Investigation: Negative Result

Trevor Bedford — 2026-05-16

## Motivation

For 20-mer lattice proteins, exhaustive scoring of all ~177M conformations takes ~49s per SSWM step, making production runs (~57 days per 32-ligand batch) infeasible. Bloom et al. (2005) [arXiv:q-bio/0401038](https://arxiv.org/abs/q-bio/0401038) demonstrated that for 18-mers, only 14% of conformations (those with >4 intra-protein contacts) needed explicit scoring. The remaining 86% were handled via a mean-field approximation of their collective partition function contribution. This reduced per-sequence scoring time by ~7×.

We investigated whether this approach could be adapted to our system.

## Bloom's Approach

For conformations with exactly `n` intra-protein contacts, their collective partition function contribution is approximated as:

```
Q_n(T) = N(n) × exp(-n × ⟨ε⟩ / T + n × σ²_ε / (2T²))
```

Where:
- `N(n)` = number of conformations with exactly n contacts (geometry-only, counted during enumeration)
- `⟨ε⟩` = mean MJ contact energy over all non-bonded residue pairs in the sequence
- `σ²_ε` = variance of MJ contact energies over those pairs
- `T` = temperature

This is a cumulant expansion: if each contact energy is drawn independently from a distribution with mean ⟨ε⟩ and variance σ², then the average Boltzmann weight for n contacts follows the formula above.

**Bloom's fitness metric:** fraction_folded × native_binding_energy, where:
- fraction_folded = exp(-E_native/T) / Z
- Z = partition function (sum of Boltzmann weights over all conformations)
- The mean-field correction only adjusts Z (the denominator)
- The native state (a single compact conformation) is always explicitly scored

## Our Fitness Metric: The Critical Difference

Our fitness uses **ensemble-averaged binding energy**:

```
fitness = -ensemble_binding_energy = -(binding_weighted_sum / Z)
```

Where:
- `Z = Σᵢ exp(-Eᵢ/T)` (partition function, summing over all conformations)
- `binding_weighted_sum = Σᵢ E_bind_i × exp(-Eᵢ/T)` (Boltzmann-weighted binding energies)
- `Eᵢ = E_intra_i + E_bind_i` (total energy includes both intra-protein and binding)

This requires accurate computation of BOTH the numerator (binding_weighted_sum) and denominator (Z). Pruning conformations removes their contributions from both.

## Why It Fails: Three Fundamental Problems

### Problem 1: All conformations have binding contacts

With the ligand anchored at (0, -1), residue 0 at the chain origin (0, 0) is always Manhattan-distance 1 from the ligand. Every self-avoiding walk has at least 1 binding contact.

Distribution for 16-mers with FWYL ligand at (0, -1):
- 0 binding contacts: 0 (0.0%)
- 1 binding contact: 1,986,460 (59.5%)
- 2 binding contacts: 816,485 (24.4%)
- 3 binding contacts: 278,398 (8.3%)
- 4+ binding contacts: 258,404 (7.7%)

This eliminates the strategy of "only prune conformations with no binding contacts" — there are none.

### Problem 2: Binding contacts are geometrically predictable

The mean-field assumption — that each contact energy is a random draw from the pool of all possible pair energies — breaks down for binding contacts. The first binding contact is always between residue 0 and a specific ligand position. The actual binding energy is `MJ[seq[0], lig[0]]`, not a random sample from the distribution over all 64 possible (chain, ligand) pairs.

This predictability means the variance term in the cumulant expansion systematically overcorrects, because the actual variance in binding energies across conformations with the same n_binding is much lower than the pooled variance would predict.

### Problem 3: Pruned conformations collectively dominate the partition function

For 16-mers with min_contacts=3:
- Stored (≥3 intra-contacts): 1,095,993 conformations (33%)
- Pruned (<3 intra-contacts): 2,243,754 conformations (67%)

The pruned conformations have Boltzmann weights of order exp(-E_bind/T) ≈ exp(2-5) ≈ 7-150 (since E_intra ≈ 0 and binding contributes favorable energies). Their collective Z contribution is enormous — comparable to or exceeding that of the stored compact conformations.

## Approaches Tested

All tested on 16-mers (3,339,747 conformations) with FWYL ligand, validated against exact full enumeration over 1000 random sequences.

### 1. Z-only correction (original Bloom approach)

Prune by intra-contacts. Apply mean-field Z correction to denominator only. Numerator uses stored conformations only.

**Result:** RMSE 5–15 depending on threshold. The inflated denominator without corresponding numerator correction pushes ensemble_binding toward zero.

### 2. No correction (stored conformations only)

Prune by intra-contacts. Compute ensemble_binding = binding_weighted_sum_stored / Z_stored without any correction.

**Result:** RMSE 5–12. Dropping extended conformations biases toward compact conformations with stronger binding, making ensemble_binding too negative.

### 3. Prune only non-binding conformations

Only prune conformations with BOTH (few intra-contacts AND zero binding contacts).

**Result:** Zero conformations pruned (Problem 1 above). The approach is inapplicable with a proximal ligand.

### 4. 2D mean-field with full cumulant expansion

Track 2D histogram N(n_intra, n_binding) for pruned conformations. Compute separate mean/variance for intra-protein and binding contact energies. Apply corrections to both Z and binding_weighted_sum:

```
Z_correction = Σ N(nᵢ,nᵦ) × exp(-(nᵢμᵢ + nᵦμᵦ)/T + (nᵢσ²ᵢ + nᵦσ²ᵦ)/(2T²))
binding_correction = Σ N(nᵢ,nᵦ) × nᵦ(μᵦ - σ²ᵦ/T) × exp(same)
```

**Result:** RMSE 15–20. The variance terms massively overcorrect due to non-Gaussian contact energy distributions and the geometric predictability issue (Problem 2).

### 5. 2D mean-field with means only (no variance)

Same as above but drop variance terms:

```
Z_correction = Σ N(nᵢ,nᵦ) × exp(-(nᵢμᵢ + nᵦμᵦ)/T)
binding_correction = Σ N(nᵢ,nᵦ) × nᵦμᵦ × exp(-(nᵢμᵢ + nᵦμᵦ)/T)
```

**Result:** RMSE 5–11. Better than full cumulant but still unacceptable. Underestimates the actual Z contribution (Jensen's inequality: E[exp(X)] > exp(E[X])).

## Summary Table

| Approach | min_contacts=3 RMSE | min_contacts=5 RMSE | Verdict |
|----------|--------------------:|--------------------:|---------|
| Z-only correction | 4.93 | 12.10 | Unusable |
| No correction | 4.93 | 12.10 | Unusable |
| 2D full cumulant | 20.29 | 16.03 | Worse than nothing |
| 2D means-only | 4.72 | 9.84 | Slightly better, still unusable |
| **Target** | **< 0.01** | **< 0.01** | — |

## Why Bloom's Approach Worked for Their Problem

1. **Their Boltzmann weights are intra-protein only.** In Bloom's formulation, binding doesn't affect thermodynamics — it's computed post-hoc for the native state. So Z depends only on intra-protein contacts, and the mean-field works because intra-protein contacts are more "random" (diverse pairs across many conformations).

2. **They only need Z, not a numerator sum.** The native state is a single conformation that's always explicitly scored. No approximation of a binding-weighted sum is needed.

3. **Their ligand placement allows non-binding conformations.** With different geometric setups, many extended conformations may not reach the ligand at all.

## Implications for Production Runs

The Bloom-style pruning approach is not viable for our fitness metric with a proximal ligand. The path to longer chains requires different strategies:

| Chain length | Strategy | Wall time (32 ligands, 32 workers) |
|---|---|---|
| 16 | Full scoring (current) | ~26 hours |
| 18 | Full scoring | ~7.4 days |
| 20 | Full scoring (memory-limited to 16 workers) | ~57 days |
| 20 | GPU-accelerated scoring | ~3-5 days (estimated) |

**Recommended approach for 20-mers:** GPU acceleration of the scoring loop (`_score_conformations`). The inner loop is embarrassingly parallel over conformations and involves only array lookups and reductions — ideal for GPU. A CUDA/Metal kernel could achieve 10-50× speedup over the numba CPU implementation.

## Code State

The `min_contacts` parameter and 2D `pruned_histogram` remain in `enumerate_conformations()` for analysis purposes. With `min_contacts=0` (default), all conformations are stored and behavior is unchanged from the pre-investigation baseline. The `fold()` function does not apply any mean-field correction — it scores only stored conformations.

The validation script `scripts/validate_pruning.py` documents and reproduces these findings.
