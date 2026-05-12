# trellis

Lattice protein synthetic data for evolutionary trajectory models.

The end goal is to generate evolutionary trajectories on a lattice-protein
fitness landscape — 20-residue chains on a 2D square lattice, folded by
exact branch-and-bound enumeration with a Miyazawa-Jernigan contact
potential, evolved under SSWM dynamics at the DNA level. Output is written
in a FASTA format compatible with the
[blab/trajectories](https://github.com/blab/trajectories) preprocessing
pipeline.

The full design is in
[`notes/lattice-protein-implementation-plan.md`](notes/lattice-protein-implementation-plan.md).

## Current state

| Step | Module | Status |
|------|--------|--------|
| 1 | `trellis/lattice.py` — 2D lattice, SAW enumeration, contacts | ✅ done |
| 2 | `trellis/energy.py` — MJ matrix, conformation energy | ✅ done |
| 3 | `trellis/fold.py` — branch-and-bound folding | ✅ done |
| 4 | `trellis/ligand.py` — ligand placement, binding energy | ✅ done |
| 5 | `trellis/fitness.py` — ensemble-averaged fitness | ✅ done |
| 6 | `trellis/genetic_code.py` — DNA ↔ AA, mutation enumeration | ✅ done |
| 7 | `trellis/sswm.py` — SSWM trajectory generation | ✅ done |
| 8 | `trellis/trajectory_io.py` — FASTA / tar.zst output | not started |
| 9 | `trellis/cache.py` — fitness cache | not started |
| — | `scripts/generate_trajectories.py` — main CLI | not started |

### Step 1: `lattice.py`

Provides:

- `Conformation = tuple[tuple[int, int], ...]` — sequence of (x, y) lattice positions, one per residue.
- `MOVES` — the four unit moves on the square lattice.
- `enumerate_saws(n, *, reduce_symmetry=True)` — depth-first SAW generator. With `reduce_symmetry=True` (default), residue 0 is fixed at (0, 0), residue 1 at (1, 0), and the first off-axis step is forced to +y, collapsing the 8-fold dihedral symmetry. With `reduce_symmetry=False`, all SAWs starting at the origin are emitted (counts match [OEIS A001411](https://oeis.org/A001411)).
- `get_contacts(conformation)` — non-bonded lattice-adjacent residue pairs.
- `is_self_avoiding(conformation)`, `occupied_sites(conformation)` — helpers used by later steps.

Symmetry-reduced SAW counts (against OEIS A001411 unreduced):

| n residues | unreduced (A001411[n−1]) | reduced |
|-----------:|-------------------------:|--------:|
|  2 |       4 |     1 |
|  3 |      12 |     2 |
|  4 |      36 |     5 |
|  5 |     100 |    13 |
|  6 |     284 |    36 |
|  7 |     780 |    98 |
|  8 |   2 172 |   272 |
|  9 |   5 916 |   740 |
| 10 |  16 268 | 2 034 |

The relation `unreduced = 8·reduced − 4` holds for n ≥ 2 (the unique
fully-on-axis straight walk is its own x-axis reflection, so it has 4
rotational images instead of 8).

### Step 2: `energy.py`

Provides:

- `AA_ALPHABET` / `AA_INDEX` — the 20 standard amino acids in alphabetical
  one-letter order, plus a letter→index lookup.
- `load_mj_matrix(path=DEFAULT_MJ_PATH)` — load and cache the 20×20
  Miyazawa-Jernigan contact-energy matrix as a read-only `float64`
  ndarray.
- `conformation_energy(sequence, conformation, mj_matrix)` — total
  contact energy by summing `mj[aa_i, aa_j]` over `lattice.get_contacts`.
- `max_contact_energy(n_remaining, mj_matrix)` — loose lower bound on
  additional energy from unplaced residues, used for branch-and-bound
  pruning in Step 3.
- `partition_function(energies, temperature=1.0)` — naive `Σ exp(-E/T)`
  for testing on small chains.

The MJ matrix lives at `data/mj_matrix.csv` (20×20, alphabetical AA
order, kT-like reduced units). Values come from the
`miyazawa_jernigan` dict in
[`jbloomlab/latticeproteins`](https://github.com/jbloomlab/latticeproteins/blob/master/src/interactions.py),
which sources Table V of:

> Miyazawa & Jernigan (1985). "Estimation of effective interresidue
> contact energies from protein crystal structures: Quasi-chemical
> approximation." *Macromolecules* 18:534–552.

Matrix range: most attractive `F-F = -6.85`, most repulsive `K-K = +0.13`.
A 1996 update of the MJ matrix exists; we may revisit and compare the
two landscapes once the pipeline is end-to-end working — see the design
notes for details.

### Step 3: `fold.py`

Provides:

- `FoldResult` — frozen dataclass with `native_conformation`,
  `native_energy`, `partition_function`, `ensemble_binding_energy`, and
  `n_conformations_enumerated`.
- `fold(sequence, mj_matrix, ligand=None, temperature=1.0)` —
  branch-and-bound folder that finds the native (lowest-energy)
  conformation while accumulating the Boltzmann partition function Z and
  ensemble-averaged binding energy in a single pass.

Without a ligand, the folder uses the same symmetry reduction as
`lattice._walk_reduced` (8-fold dihedral, corrected via `Z × 8 − 4`).
With a ligand the symmetry is broken, so the folder uses unreduced
enumeration and no correction.

Energy pruning skips subtrees whose lower bound (current energy +
`max_contact_energy` for remaining residues + binding bound) cannot
improve on the best complete conformation found so far. For a 20-mer
this prunes ~94% of the search tree, completing in ~14 seconds (pure
Python, no numba).

### Step 4: `ligand.py`

Provides:

- `Ligand` — frozen dataclass with `sequence` (amino acid string),
  `positions` (lattice coordinates), and `sites` (frozenset for O(1)
  collision lookup).
- `create_ligand(sequence, anchor=(0, -1), direction="horizontal")` —
  place a linear ligand on the lattice starting at an anchor position,
  extending horizontally (along x-axis) or vertically (along y-axis).
  Default placement puts a 4-residue ligand at `(0,−1)…(3,−1)`, one
  step below the protein origin.
- `binding_contacts(conformation, ligand)` — protein-ligand contact
  pairs `(protein_idx, ligand_idx)` at Manhattan distance 1.
- `binding_energy(protein_sequence, conformation, ligand, mj_matrix)` —
  sum of MJ contact energies over all binding contacts.

When a ligand is passed to `fold()`, the protein walk avoids ligand
sites (collision detection), and binding energy is computed at each
complete conformation and included in the Boltzmann weighting. The
ensemble-averaged binding energy `⟨E_bind⟩` is the basis for the
fitness function in Step 5 (fitness = `−⟨E_bind⟩`).

See `notes/binding-thermodynamics.md` for a discussion of `⟨E_bind⟩`
vs binding free energy `ΔG` and why `⟨E_bind⟩` was chosen.

### Step 5: `fitness.py`

Provides:

- `FitnessResult` — frozen dataclass with `fitness`, `fold_result`
  (`FoldResult | None`), `aa_sequence`, and `dna_sequence`.
- `compute_fitness(dna_sequence, ligand, mj_matrix, temperature=1.0)` —
  translates DNA to amino acids via `genetic_code.translate()`, folds
  with the ligand, and returns fitness = `−⟨E_bind⟩` (higher = fitter).
  Stop codons yield `fitness = −inf` and `fold_result = None`.
- `compute_fitness_aa(aa_sequence, ligand, mj_matrix, temperature=1.0)` —
  convenience function that skips translation, useful for direct
  amino-acid-level evaluation and the fitness cache (Step 9).

Fitness requires no separate stability criterion. The ensemble average
handles stability naturally: an unstable protein spreads its Boltzmann
weight across many non-binding conformations, yielding a weak `⟨E_bind⟩`
and low fitness. A stably folded protein that contacts the ligand
concentrates its weight on binding-competent conformations, yielding
strong `⟨E_bind⟩` and high fitness.

### Step 6: `genetic_code.py`

Provides:

- `CODON_TABLE` — standard genetic code mapping all 64 codons to amino
  acids (stop codons as `"*"`).
- `NUCLEOTIDES` — `"ACGT"`.
- `translate(dna_sequence)` — DNA to amino acid string. Length must be
  divisible by 3; stop codons included as `"*"` in output.
- `single_nt_mutations(dna_sequence)` — enumerate all single-nucleotide
  mutations as `(mutant_dna, position, ref_base, alt_base)` tuples.
  For a 60 nt sequence this yields 180 mutations.
- `classify_mutation(dna_ref, dna_alt)` — classify a single-nucleotide
  change as `"synonymous"`, `"nonsynonymous"`, or `"nonsense"`.
- `mutant_aa_sequences(dna_sequence)` — group all single-nt mutants by
  translated AA sequence. Returns `{aa_seq: [mutant_dna, ...]}`. This
  is the key deduplication for SSWM: many DNA mutations map to the same
  protein, so each unique AA sequence is folded only once.

### Step 7: `sswm.py`

Provides:

- `Trajectory` — dataclass with `dna_sequences`, `aa_sequences`,
  `fitness_values`, `mutation_types`, and `metadata`.
- `fixation_probability(s, Ne)` — Kimura (1962) fixation probability
  for a new mutation with selection coefficient `s` in a population of
  effective size `Ne`. Handles edge cases: `s = 0` (neutral),
  `s = −inf` (lethal), and numerically extreme `2·Ne·s`.
- `generate_trajectory(start_dna, ligand, mj_matrix, n_steps=100,
  Ne=1000, mu=1e-6, temperature=1.0, rng=None, fitness_cache=None)` —
  generate a single SSWM trajectory. Each step enumerates all
  single-nucleotide mutations, deduplicates by AA sequence (via
  `mutant_aa_sequences`), folds each unique protein once, and samples
  the next substitution proportional to Kimura fixation probability.
  `mu` cancels from the sampling weights and is stored in metadata only.
  An optional `dict[str, float]` cache avoids redundant folding across
  steps.
- `generate_start_sequence(n_codons, ligand, mj_matrix,
  min_fitness=0.0, max_attempts=10000, rng=None)` — sample random
  sense codons until the translated protein meets the fitness
  threshold.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11. Runtime dependency: `numpy`. Dev dependency: `pytest`.

## Run tests

```bash
pytest
```

Or a single module:

```bash
pytest tests/test_lattice.py -v
pytest tests/test_energy.py -v
```

## Repository layout

```
trellis/
├── trellis/
│   ├── __init__.py
│   ├── lattice.py           # Step 1 — 2D lattice, SAW enumeration
│   ├── energy.py            # Step 2 — MJ matrix, conformation energy
│   ├── fold.py              # Step 3 — branch-and-bound folding
│   ├── ligand.py            # Step 4 — ligand placement, binding energy
│   ├── fitness.py           # Step 5 — ensemble-averaged fitness
│   ├── genetic_code.py      # Step 6 — codon table, translation, mutations
│   └── sswm.py              # Step 7 — SSWM trajectory generation
├── tests/
│   ├── __init__.py
│   ├── test_lattice.py
│   ├── test_energy.py
│   ├── test_fold.py
│   ├── test_ligand.py
│   ├── test_fitness.py
│   ├── test_genetic_code.py
│   └── test_sswm.py
├── data/
│   ├── mj_matrix.csv        # MJ 1985 Table V, 20×20, alphabetical AA order
│   └── README.md            # citation, source URL, source commit SHA
├── notes/
│   ├── lattice-protein-implementation-plan.md
│   └── binding-thermodynamics.md
├── pyproject.toml
├── LICENSE.md
└── README.md
```
