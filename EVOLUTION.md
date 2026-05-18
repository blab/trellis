# Evolution

Technical reference for the evolutionary model in trellis. For an overview
of the project and usage instructions, see [README.md](README.md). For the
folding model, see [FOLDING.md](FOLDING.md).

## Overview

Trellis generates evolutionary trajectories under strong-selection
weak-mutation (SSWM) dynamics at the DNA level. Fitness couples sequence to
function through folding thermodynamics: a DNA sequence is translated to a
protein, the protein is folded on the 2D lattice in the presence of a fixed
ligand, and fitness is computed from the protein's thermodynamic stability
and binding affinity. Mutations arise one at a time, fix or are lost
according to population genetics, and the trajectory records the sequence
of fixed substitutions.

Output is written as FASTA files compatible with the
[blab/trajectories](https://github.com/blab/trajectories) preprocessing
pipeline.

## Fitness function

**Module:** `trellis/fitness.py`

Fitness is defined as:

```
F(T) = f(T) × (−BE(C_native))
```

where:

- **C_native** is the lowest-energy conformation (including protein-ligand
  contacts).
- **BE(C_native)** is the binding energy of the native conformation to the
  ligand (negative = favorable). The negation makes higher fitness = better
  binding.
- **f(T) = exp(−E(C_native) / T) / Q(T)** is the equilibrium fraction of
  the conformational ensemble in the native state, where Q(T) is the
  partition function over all conformations.

This separates fitness into two components:

- **Stability**: f(T) measures how reliably the protein folds to its native
  state. An unstable protein spreads Boltzmann weight across many
  conformations, yielding f(T) near zero and near-zero fitness regardless
  of binding capacity.
- **Function**: −BE(C_native) measures how well the folded protein binds
  the ligand. Only matters if the protein is stable enough to fold.

Stop codons in the translated sequence yield fitness = −∞.

This is the standard formulation in the lattice protein literature,
following Bloom et al. (2004):

> Bloom JD, Wilke CO, Arnold FH, Adami C. "Stability and the evolvability
> of function in a model protein." *Biophysical Journal* 86:2758-2764 (2004).

## Binding thermodynamics

The lattice model places the protein and ligand on adjacent lattice sites.
The protein's first residue is always at (0, 0) and the ligand occupies
fixed positions nearby. The protein never diffuses away — the partition
function sums over all conformations of the protein chain in the presence
of the fixed ligand. This models **folding-mediated binding selectivity**:
given that the protein is near the ligand, how strongly does its
conformational ensemble favor ligand-complementary shapes?

An alternative fitness metric — ensemble-averaged binding energy ⟨E_bind⟩
— was considered and rejected. ⟨E_bind⟩ weights the binding energy of
every conformation by its Boltzmann probability, capturing the full
thermodynamic ensemble without a separate stability criterion. However:

1. The fraction-folded × native binding formulation is the standard in every
   published lattice protein ligand-binding study we are aware of (Miller &
   Dill 1997, Williams et al. 2001, Bloom et al. 2004-2006).
2. It enables mean-field pruning of low-contact conformations (see
   [FOLDING.md](FOLDING.md#mean-field-pruning)), which is essential for
   scaling to 18-20 residue chains.
3. Both metrics produce rugged, epistatic fitness landscapes suitable for
   generating training data. The choice is arbitrary for our use case.

The two metrics produce genuinely different landscapes (Spearman rho ≈ 0.58
across 1000 random 16-mers), but since the landscape is synthetic training
data, the literature-standard formulation provides a simple citation
justification.

See [`notes/2026-05-11-binding-thermodynamics.md`](notes/2026-05-11-binding-thermodynamics.md)
for the full thermodynamic discussion and
[`notes/2026-05-17-fitness-metric-switch-rationale.md`](notes/2026-05-17-fitness-metric-switch-rationale.md)
for the rationale behind the metric choice.

## Genetic code and mutations

**Module:** `trellis/genetic_code.py`

Evolution operates at the DNA level using the standard genetic code (64
codons mapping to 20 amino acids plus stop). Each protein-coding gene is a
sequence of 3N nucleotides encoding N amino acid residues.

**Mutation enumeration:** `single_nt_mutations()` generates all 3N
single-nucleotide mutations for an N-nucleotide DNA sequence (each position
can mutate to 3 alternative bases).

**Mutation classification:** Each mutation is classified as:

- **Synonymous** — same amino acid after translation (silent).
- **Nonsynonymous** — different amino acid (missense).
- **Nonsense** — introduces a stop codon.

**AA-level deduplication:** Due to codon degeneracy, multiple DNA mutations
can translate to the same amino acid sequence. `mutant_aa_sequences()`
groups all single-nucleotide mutants by their translated protein, returning
a dict of {aa_sequence: [mutant_dna_sequences]}. Each unique protein is
folded only once per SSWM step, then the fitness is applied to all DNA
variants that encode it.

## SSWM dynamics

**Module:** `trellis/sswm.py`

The strong-selection weak-mutation (SSWM) regime assumes mutations arise
rarely enough that each fixes or is lost before the next one appears. This
reduces the population genetics to a Markov chain on genotypes, where each
step is a single substitution event.

### Per-step algorithm

1. **Enumerate mutations.** Generate all 3N single-nucleotide mutations of
   the current DNA sequence.
2. **Deduplicate at the AA level.** Group mutant DNA sequences by their
   translated protein (via `mutant_aa_sequences()`).
3. **Fold unique proteins.** Compute fitness for each unique amino acid
   sequence. A fitness cache (`trellis/cache.py`) avoids redundant folding
   when the same protein was evaluated in a previous step.
4. **Compute selection coefficients.** For each mutation:
   s = fitness_mutant − fitness_current.
5. **Compute fixation probabilities.** Using the Kimura (1962) formula:

   ```
   P_fix(s) = (1 − exp(−2s)) / (1 − exp(−2 Ne s))
   ```

   with edge cases: s = 0 (neutral) → P_fix = 1/(2Ne);
   s = −∞ (lethal) → P_fix = 0; large |2 Ne s| handled for numerical
   stability.

6. **Sample the next substitution.** Choose one mutation with probability
   proportional to its fixation probability.
7. **Record.** Append the new DNA sequence, amino acid sequence, fitness,
   and mutation type (synonymous/nonsynonymous/nonsense) to the trajectory.

The mutation rate μ cancels from the sampling weights (all mutations share
the same per-site rate) and is stored in trajectory metadata only.

The trajectory terminates when either the requested number of steps is
reached or total fixation probability sums to zero (all mutations are
lethal).

### Effective population size

The parameter Ne (effective population size) controls the strength of
selection. Higher Ne means stronger selection: beneficial mutations fix with
higher probability, deleterious mutations are more effectively purged.
Lower Ne allows more genetic drift. Default: Ne = 50.

## Starting sequences

**Function:** `generate_start_sequence()` in `trellis/sswm.py`

Trajectories begin from a functional protein. The function samples random
sense codons (avoiding stop codons) until the translated protein meets a
minimum fitness threshold (default: 0.0). This ensures the evolutionary
trajectory starts from a sequence that folds and binds, rather than from a
random non-functional sequence.

## Trajectory output

**Module:** `trellis/trajectory_io.py`

Each trajectory is written as a FASTA file with one record per evolutionary
step:

```
>NODE_0000|0|0
GCTTGTGATGAATTTGGTCATATCAAACTTATGAATCCGCAAAGAACTTCTGTTTGGTAC
>NODE_0001|1|1
GCTTGTGATGAATTTGGTCATATCAAACTTATGAATCCGCAAAGAACTTCTGTTTGGTAC
...
>TIP_42|15|1
GCTTGTGATGAATTTGGTCATATCAAACTTATGAATCCGCAAAGAACTTCTGTTTGGTAC
```

Headers encode `name|cumulative_hamming|branch_hamming`. The final node is
named `TIP_{trajectory_id}`.

For bulk generation, trajectories are split into train/test sets by holding
out entire trajectories (not individual steps), then packaged into
compressed `forwards-{split}-{NNN}.tar.zst` shards. This format is
compatible with the [blab/trajectories](https://github.com/blab/trajectories)
preprocessing pipeline.

## Design history

- [`notes/2026-05-17-fitness-metric-switch-rationale.md`](notes/2026-05-17-fitness-metric-switch-rationale.md) — why fraction-folded × native binding was chosen over ensemble-averaged binding
- [`notes/2026-05-11-binding-thermodynamics.md`](notes/2026-05-11-binding-thermodynamics.md) — thermodynamic analysis of ⟨E_bind⟩ vs ΔG
