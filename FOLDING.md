# Folding

Technical reference for how trellis folds lattice proteins. For an overview
of the project and usage instructions, see [README.md](README.md). For the
evolutionary model, see [EVOLUTION.md](EVOLUTION.md).

## Overview

Trellis models proteins as self-avoiding walks (SAWs) on a 2D square
lattice. Each residue occupies one lattice site; consecutive residues are
Manhattan-distance 1 apart. Non-bonded residues that land on adjacent
lattice sites form contacts, scored by the Miyazawa-Jernigan (MJ) contact
potential. A fixed ligand occupies nearby lattice sites and contributes
additional protein-ligand contact energies.

The Boltzmann partition function Z sums over all valid conformations
(self-avoiding walks that avoid ligand sites). The native state is the
lowest-energy conformation. The fraction folded f(T) = exp(-E_native/T) / Z
measures thermodynamic stability.

Two folding implementations exist:

- **`fold_bb.py`** — branch-and-bound folder. Pure Python, single-pass DFS
  with energy pruning. Reference implementation for correctness testing.
- **`fold_enum.py`** — pre-enumeration folder. Numba JIT-compiled. Enumerates
  all conformations once, stores contacts in compressed arrays, then scores
  any sequence by summing MJ lookups over precomputed contacts. 100-555x
  faster than branch-and-bound. Used in production.

## Lattice geometry

**Module:** `trellis/lattice.py`

Self-avoiding walks are enumerated by depth-first search. With symmetry
reduction enabled (default), the 8-fold dihedral symmetry of the square
lattice is collapsed:

- Residue 0 is fixed at (0, 0).
- Residue 1 is fixed at (1, 0).
- The first off-axis step is forced to +y, breaking the reflection
  symmetry across the x-axis.

This produces one representative per symmetry orbit. The exact relation
between reduced and unreduced counts is:

```
unreduced = 8 × reduced − 4
```

The −4 accounts for the single straight-line walk, which has only 4
rotational images (instead of 8) because it is its own x-axis reflection.

Symmetry-reduced SAW counts (verified against
[OEIS A001411](https://oeis.org/A001411)):

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

**Contacts** are non-bonded residue pairs (i, j) with j > i+1 that occupy
Manhattan-distance-1 lattice sites. These are the interactions scored by
the MJ potential.

## Contact potential

**Module:** `trellis/energy.py`

The 20×20 Miyazawa-Jernigan contact-energy matrix assigns an energy to each
pair of amino acid types in contact. Values are in kT-like reduced units
(most attractive: F-F = −6.85; most repulsive: K-K = +0.13).

Source: `miyazawa_jernigan` dictionary in
[jbloomlab/latticeproteins](https://github.com/jbloomlab/latticeproteins)
at commit `de1316a`, which cites Table V of:

> Miyazawa, S. & Jernigan, R. L. (1985). "Estimation of effective
> interresidue contact energies from protein crystal structures:
> Quasi-chemical approximation." *Macromolecules* 18:534-552.

The matrix is bundled in the Python package as `trellis/mj_matrix.csv`
(20 amino acids in alphabetical one-letter order). It is loaded once and
cached by `load_mj_matrix()`.

## Ligand model

**Module:** `trellis/ligand.py`

The ligand is a fixed linear chain of amino acid residues placed on the
lattice adjacent to the protein's origin. This models **folding-mediated
binding selectivity**: given that the protein is near the ligand, how
strongly does its conformational ensemble favor ligand-complementary shapes?
The lattice model does not model diffusion.

Default placement puts a 4-residue ligand at (0, −1)...(3, −1), one row
below the protein origin, extending horizontally. The `create_ligand()`
function supports arbitrary anchor positions and horizontal or vertical
orientation.

Protein-ligand **binding contacts** are (protein_idx, ligand_idx) pairs at
Manhattan distance 1. During SAW enumeration, the protein chain must avoid
ligand sites (collision detection). When a ligand is present, symmetry
reduction is disabled because the ligand breaks the lattice's dihedral
symmetry.

## Branch-and-bound folding

**Module:** `trellis/fold_bb.py`

Reference implementation, not used in production. A single-pass DFS that
simultaneously:

1. Finds the native (lowest-energy) conformation.
2. Accumulates the Boltzmann partition function Z.
3. Computes the ensemble-averaged binding energy (for the ligand case).

**Energy pruning** skips subtrees whose lower bound (current partial energy
+ `max_contact_energy` for remaining residues + binding bound) cannot
improve on the best complete conformation found so far. Each unplaced
interior residue can form at most 2 new contacts (4 lattice neighbors minus
2 chain bonds); the C-terminus can form 3. The bound uses `mj_matrix.min()`
per potential contact. For a 20-mer this prunes ~94% of the search tree.

**Symmetry handling:**

- Without a ligand: uses symmetry-reduced enumeration, then corrects
  Z via `Z_full = Z_sum × 8 − 4`.
- With a ligand: uses unreduced enumeration (the ligand breaks symmetry).

## Pre-enumeration folding

**Module:** `trellis/fold_enum.py`

The production folder. The key idea: enumerate all valid SAWs for a given
chain length and ligand **once**, store their contact lists, then score any
new sequence by summing MJ energies over precomputed contacts.

### Enumeration

`enumerate_conformations(chain_length, ligand)` runs a two-pass numba
JIT-compiled DFS:

1. **Pass 1 (counting):** Walk the full SAW tree to count conformations and
   total contacts, without storing anything.
2. **Pass 2 (storage):** Allocate exact-size numpy arrays and fill them.

The DFS uses an explicit stack on a 41×41 boolean grid (supporting
coordinates from −20 to +20, sufficient for chains up to ~30 residues).

### ConformationDatabase

A dataclass holding precomputed contacts in CSR-like (compressed sparse row)
layout:

- `contact_pairs` — (total_contacts, 2) int32 array of intra-protein
  contact pairs (i, j) with j > i+1.
- `contact_offsets` — (n_conformations + 1,) int64 array where
  `contact_offsets[k]` points to the first contact of conformation k.
- `binding_pairs` / `binding_offsets` — same layout for protein-ligand
  contacts.
- `coordinates` — optional (n_conformations, chain_length, 2) int32 array
  of (x, y) positions, stored when `store_coordinates=True`.

### Scoring

Scoring a single sequence iterates over conformations, accumulating
intra-protein and protein-ligand contact energies from the CSR arrays,
computing Boltzmann weights for Z and the binding-weighted ensemble average.

**Batch scoring** (used in SSWM) processes multiple sequences at once.
It precomputes a pair energy table — `mj[aa[s,i], aa[s,j]]` for all
residue pairs and sequences — before the conformation loop, eliminating
repeated double-indirect MJ matrix lookups.

### Performance

Benchmark: 10 random sequences, FWYL ligand, scoring only (excludes
one-time enumeration):

| Chain | B&B (10 seqs) | Numba scoring (10 seqs) | Speedup |
|-------|--------------|------------------------|---------|
| 8-mer | 0.041s | 0.0004s | 100x |
| 10-mer | 0.37s | 0.001s | 370x |
| 12-mer | 3.2s | 0.007s | 460x |
| 14-mer | 26.5s | 0.050s | 530x |
| 16-mer | 228s | 0.41s | 555x |

Enumeration is a one-time cost (~6s for 18-mers with a ligand). For SSWM
trajectories that fold ~60 sequences per step over 100 steps, the
enumeration cost is amortized over thousands of fold calls.

## Mean-field pruning

Following Bloom et al. (2004, 2005), conformations with fewer than
`min_contacts` (default 4) intra-protein contacts are not explicitly
scored. Instead, their aggregate contribution to the partition function Z
is approximated via a cumulant expansion:

```
Z_pruned(n) = count(n) × exp(−n × μ / T  +  n × σ² / (2T²))
```

where n is the contact count, μ is the mean MJ energy over all non-bonded
residue pairs in the sequence, and σ² is the variance. This is computed
per-sequence from the actual amino acid composition.

The approximation is valid because low-contact conformations contribute
individually small Boltzmann weights — their aggregate effect on Z matters,
but the distribution of their energies is well-captured by the first two
cumulants.

The native state is always a compact conformation with many contacts, so it
is always explicitly scored. The mean-field correction only adjusts the
denominator of the fraction-folded term f(T).

For 18-mers, this reduces the scored conformation set by ~7x (from ~5.8M
to ~795K conformations). Validated against exact enumeration with RMSE
< 0.002 in ΔG_f across 1000 random sequences at T=1.0 (Bloom et al. 2004).

> Bloom JD, Wilke CO, Arnold FH, Adami C. "Stability and the evolvability
> of function in a model protein." *Biophysical Journal* 86:2758-2764 (2004).

> Bloom JD, Silberg JJ, Wilke CO, Drummond DA, Adami C, Arnold FH.
> "Thermodynamic prediction of protein neutrality." *PNAS* 102:606-611 (2005).

## Conformation recovery

By default, `enumerate_conformations()` stores only contact lists (no
coordinates), which is all that's needed for fitness computation. For
visualization, pass `store_coordinates=True` to also store (x, y) positions
for every conformation during the enumeration pass. This enables O(1)
coordinate lookup when recovering the native conformation for display.

When coordinates are not stored, the native conformation can still be
recovered by re-enumerating SAWs and indexing into the sequence — slower but
memory-efficient.

## Design history

- [`notes/2026-05-07-exhaustive-enumeration-plan.md`](notes/2026-05-07-exhaustive-enumeration-plan.md) — original design for pre-enumeration folding
- [`notes/2026-05-16-numba-enumeration-plan.md`](notes/2026-05-16-numba-enumeration-plan.md) — numba JIT acceleration strategy
- [`notes/2026-05-16-mean-field-pruning-investigation.md`](notes/2026-05-16-mean-field-pruning-investigation.md) — mean-field pruning analysis
