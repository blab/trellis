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
| 5 | `trellis/fitness.py` — ensemble-averaged fitness | not started |
| 6 | `trellis/genetic_code.py` — DNA ↔ AA, mutation enumeration | not started |
| 7 | `trellis/sswm.py` — SSWM trajectory generation | not started |
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
│   ├── lattice.py           # Step 1 — implemented
│   └── energy.py            # Step 2 — implemented
├── tests/
│   ├── __init__.py
│   ├── test_lattice.py
│   └── test_energy.py
├── data/
│   ├── mj_matrix.csv        # MJ 1985 Table V, 20×20, alphabetical AA order
│   └── README.md            # citation, source URL, source commit SHA
├── notes/
│   └── lattice-protein-implementation-plan.md
├── pyproject.toml
├── LICENSE.md
└── README.md
```
