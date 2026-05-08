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
| 3 | `trellis/fold.py` — branch-and-bound folding (numba) | not started |
| 4 | `trellis/ligand.py` — ligand placement, binding energy | not started |
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
