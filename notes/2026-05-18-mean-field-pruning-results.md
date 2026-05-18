# Mean-Field Pruning: Validation Results

Trevor Bedford — 2026-05-18

## Summary

Bloom-style mean-field pruning now works with the fraction-folded fitness metric. At `min_contacts=4` (the production default), only 17% of conformations are explicitly scored with RMSE 0.004 in fitness — a ~6× reduction in scored conformations with negligible accuracy loss.

## Background

Our previous investigation (`notes/2026-05-16-mean-field-pruning-investigation.md`) showed mean-field pruning fails catastrophically for the ensemble-averaged binding metric (RMSE 5–20). Switching to the literature-standard fitness metric (`fitness = f(T) × −BE_native`, commit cd42699) resolved this because:

1. The new metric only requires accurate Z (partition function denominator), not an accurate binding-weighted numerator sum
2. The native state is always a compact conformation with many contacts — always in the stored set
3. For well-folded proteins, the native state's Boltzmann weight dominates Z, making errors from pruned conformations negligible

This is exactly the formulation Bloom et al. (2004) used and validated (RMSE < 0.002 in ΔG_f for 18-mers).

## Method

The Bloom cumulant expansion approximates the partition function contribution of pruned conformations:

$$Q_n(T) = N(n) \times \exp\left(\frac{-n \langle\varepsilon\rangle}{T} + \frac{n \sigma^2_\varepsilon}{2T^2}\right)$$

Where N(n) is the count of conformations with exactly n intra-protein contacts, and ⟨ε⟩ and σ²_ε are the mean and variance of MJ contact energies over all non-bonded residue pairs in the sequence.

Implementation: `enumerate_conformations(chain_length, ligand, min_contacts=4)` stores only conformations with ≥4 intra-protein contacts. Pruned conformations are counted by contact number. During `fold()`, the mean-field correction is added to Z before computing `f(T) = exp(−E_native/T) / Z`.

## Validation results

Tested on 1000 random 16-mer sequences with a random 4-mer ligand (CSQK) at T=1.0. Exact results computed with `min_contacts=0` (all 3,339,747 conformations scored).

| min_contacts | Stored | Pruned | % stored | RMSE fitness | RMSE f(T) | Max error | Spearman |
|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 3 | 1,095,993 | 2,243,754 | 32.8% | 0.0014 | 0.0001 | 0.031 | 1.0000 |
| **4** | **554,999** | **2,784,748** | **16.6%** | **0.0043** | **0.0003** | **0.055** | **1.0000** |
| 5 | 244,144 | 3,095,603 | 7.3% | 0.034 | 0.002 | 0.618 | 0.999 |
| 6 | 90,708 | 3,249,039 | 2.7% | 0.356 | 0.016 | 5.66 | 0.964 |
| 7 | 30,400 | 3,309,347 | 0.9% | 0.851 | 0.047 | 8.10 | 0.796 |
| 8 | 6,515 | 3,333,232 | 0.2% | 1.029 | 0.069 | 7.66 | 0.699 |

There is a sharp accuracy cliff between min_contacts=5 and min_contacts=6. The production default of **min_contacts=4** provides an excellent tradeoff: 6× fewer conformations scored with fitness RMSE 0.004 and Spearman correlation effectively 1.0.

## Distribution of pruned conformations by intra-contact count

For 16-mers with min_contacts=4:

| Intra-contacts | Count | % of total |
|:---:|---:|:---:|
| 0 | 562,811 | 16.9% |
| 1 | 874,138 | 26.2% |
| 2 | 806,805 | 24.2% |
| 3 | 540,994 | 16.2% |

These are extended, unfolded conformations whose collective partition function contribution is well-approximated by the cumulant expansion.

## Projected impact on production runs

Assuming the ~17% storage fraction at min_contacts=4 scales to longer chains (Bloom et al. observed ~14% for 18-mers):

| Chain length | Total conformations | Scored (~17%) | Estimated per-sequence time |
|:---:|---:|---:|:---:|
| 16 | 3.3M | ~555K | ~0.6 ms |
| 18 | 5.8M | ~986K | ~1.1 ms |
| 20 | ~60M | ~10.2M | ~11 ms |

## References

Bloom JD, Wilke CO, Arnold FH, Adami C. "Stability and the evolvability of function in a model protein." *Biophysical Journal* 86:2758–2764 (2004). [arXiv:q-bio/0401038](https://arxiv.org/abs/q-bio/0401038)
