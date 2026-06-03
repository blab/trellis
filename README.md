# trellis

Lattice protein synthetic data for evolutionary trajectory models.

Trellis generates evolutionary trajectories on a lattice-protein fitness
landscape. Proteins are 18-residue chains on a 2D square lattice, folded by
exhaustive enumeration with a Miyazawa-Jernigan contact potential and numba
JIT-compiled scoring. Fitness is fraction-folded times native binding energy
to a fixed lattice ligand, following Bloom et al. (2004). Evolution proceeds
under SSWM dynamics at the DNA level: single-nucleotide mutations arise and
fix according to Kimura fixation probabilities. Output is written in a FASTA
format compatible with the
[blab/trajectories](https://github.com/blab/trajectories) preprocessing
pipeline.

For technical details, see [FOLDING.md](FOLDING.md) (lattice geometry, MJ
potential, folding algorithms, mean-field pruning) and
[EVOLUTION.md](EVOLUTION.md) (fitness function, genetic code, SSWM dynamics).
The full design history is in [`notes/`](notes/).

## Install

```bash
pip install trellis-protein
```

Downstream code imports as `trellis` (the PyPI distribution is published as
`trellis-protein` because `trellis` is taken). For local development on
this repo:

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.11. Runtime dependencies: `numpy`, `numba`,
`zstandard`. Optional extras: `[analysis]` (matplotlib), `[dev]` (pytest).

## Fold a single sequence

`scripts/fold_sequence.py` folds a single amino acid or DNA sequence
and prints native energy, partition function, conformation coordinates,
and wall time. With a ligand, it also reports ensemble-averaged binding
energy and fitness.

```bash
# Fold an amino acid sequence
python scripts/fold_sequence.py --aa ACDEFGHIKLMNPQRSTVWY

# Fold with a ligand (reports binding energy and fitness)
python scripts/fold_sequence.py --aa ACDEFGHIKLMNPQRSTVWY --ligand-sequence FWYL

# Machine-readable JSON output
python scripts/fold_sequence.py --aa ACDEFG --json
```

## Generate and visualize a single trajectory

`scripts/generate_viz_trajectory.py` runs a short SSWM trajectory and
writes a JSON snapshot for the interactive D3 dashboard. A live example is
at [blab.github.io/trellis/viz](https://blab.github.io/trellis/viz/).

```bash
python scripts/generate_viz_trajectory.py
```

This runs with defaults (`--chain-length 10 --n-steps 100 --Ne 50
--temperature 1.0 --seed 42`) and writes `viz/viz_trajectory_data.json`.
Serve the repo root and open `viz/index.html`:

```bash
python -m http.server 8000
# open http://localhost:8000/viz/index.html
```

See [`viz/README.md`](viz/README.md) for details.

## Generate trajectories

`scripts/generate_trajectories.py` is the main CLI for bulk trajectory
generation. It runs SSWM trajectories in parallel, splits into train/test
sets, and packages them as compressed FASTA shards.

```bash
# Generate 1000 trajectories for a single ligand
python scripts/generate_trajectories.py \
    --n-trajectories 1000 \
    --n-steps 100 \
    --chain-length 18 \
    --ligand-sequence KEMN \
    --Ne 50 \
    --n-workers 8 \
    --seed 42

# Generate trajectories across multiple random ligands
python scripts/generate_trajectories.py \
    --n-random-ligands 8 \
    --n-trajectories 100 \
    --chain-length 18 \
    --Ne 50 \
    --n-workers 32 \
    --seed 42
```

Output:

```
results/trellis-18aa-KEMN/
├── forwards-train-000.tar.zst
├── forwards-test-000.tar.zst
└── metadata.json
```

Each worker process loads its own MJ matrix, ligand, and conformation
database. Per-trajectory RNG streams are created via `SeedSequence.spawn()`
for reproducibility regardless of worker count. See
[BENCHMARK.md](BENCHMARK.md) for timing data.

## Generate phylogenies

`scripts/generate_phylogeny.py` generates a single phylogenetic tree via
Yule-with-sampling on SSWM dynamics. At each time step, active lineages
may speciate (rate beta), get sampled as tips (rate psi), or continue as
single daughters. Mutations are drawn from SSWM fixation probabilities.
Output is Auspice v2 JSON viewable at [auspice.us](https://auspice.us).

```bash
# Generate a phylogeny with default parameters
# Default is 18-mer, which may take a while
python scripts/generate_phylogeny.py --output results/phylogeny.json

# Specify ligand and specify protein length
python scripts/generate_phylogeny.py \
    --chain-length 12 \
    --ligand-sequence KEMN \
    --output results/phylogeny.json
```

Key parameters:
- `--beta`: speciation probability per lineage per step (default 0.08)
- `--psi`: sampling probability per lineage per step (default 0.05)
- `--min-active`: minimum active lineages preserved against sampling (default 1)
- `--max-active`: cap on active lineages per step (default 100)
- `--max-nodes`: hard cap on total tree size (default 5000)

## Evaluation interface

`trellis.sswm.compute_sswm_probabilities` exposes the ground-truth SSWM
fixation-probability distribution over all single-nucleotide neighbors of
a reference DNA sequence. Downstream evaluation code uses this as the
oracle when scoring how well a learned model has recovered the fitness
landscape.

```python
from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fitness import compute_fitness_aa
from trellis.fold_enum import enumerate_conformations
from trellis.genetic_code import translate
from trellis.ligand import create_ligand
from trellis.sswm import compute_sswm_probabilities, pfix_for_target

# One-time setup per ligand
mj = load_mj_matrix()
ligand = create_ligand("KEMN", anchor=(0, -1))
db = enumerate_conformations(18, ligand)
cache = FitnessCache()

# Per-reference evaluation
ref_aa = translate(ref_dna)
if ref_aa not in cache:
    cache.put(ref_aa, compute_fitness_aa(ref_aa, ligand, mj, 1.0, db=db))
ref_fitness = cache.get(ref_aa).fitness

mutant_dnas, ground_truth = compute_sswm_probabilities(
    ref_dna, ref_fitness, ligand, mj, Ne=50.0, temperature=1.0,
    fitness_cache=cache, db=db,
)
# ground_truth sums to 1 (or is all zeros if every neighbor is lethal)

# Single mutation lookup
p = pfix_for_target(ref_dna, target_dna, ligand, mj,
                    Ne=50.0, temperature=1.0,
                    fitness_cache=cache, db=db)
```

Share the `FitnessCache` across all evaluations for a given ligand —
folding is the dominant cost, and the cache amortizes it.

The CLI `scripts/pfix_distribution.py` exercises the same API for
exploration:

```bash
# Full distribution for a reference sequence
python scripts/pfix_distribution.py \
    --reference CTCTGCAACACAACACGGCTCGATGTTCTCTATGGAGAATGTGACTGCGCGCTG \
    --ligand KEMN --Ne 50

# Single-mutation P_fix as JSON
python scripts/pfix_distribution.py \
    --reference <ref-dna> --target <neighbor-dna> \
    --ligand KEMN --Ne 50 --json
```

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
│   ├── lattice.py           # 2D lattice, SAW enumeration
│   ├── energy.py            # MJ matrix, conformation energy
│   ├── fold_bb.py           # branch-and-bound folding (reference)
│   ├── fold_enum.py         # pre-enumeration folding (production)
│   ├── ligand.py            # ligand placement, binding energy
│   ├── fitness.py           # fitness function
│   ├── genetic_code.py      # codon table, translation, mutations
│   ├── sswm.py              # SSWM trajectory generation
│   ├── phylogeny.py         # phylogenetic tree generation
│   ├── auspice_io.py        # Auspice v2 JSON serialization
│   ├── trajectory_io.py     # FASTA / tar.zst output
│   ├── cache.py             # fitness cache
│   └── mj_matrix.csv        # MJ 1985 contact potential
├── tests/
├── scripts/
│   ├── generate_trajectories.py     # bulk parallel trajectory generation
│   ├── generate_phylogeny.py        # single phylogenetic tree generation
│   ├── generate_viz_trajectory.py   # single trajectory for D3 dashboard
│   ├── pfix_distribution.py         # SSWM fixation-probability distribution
│   ├── fold_sequence.py             # fold a single sequence
│   ├── inspect_shard.py             # inspect / extract tar.zst shards
│   ├── benchmark.py                 # single-worker pipeline benchmark
│   └── benchmark_bb_vs_enum.py      # B&B vs pre-enumeration comparison
├── viz/
│   ├── index.html           # D3 dashboard
│   └── README.md
├── notes/                   # design history and rationale
├── FOLDING.md               # folding technical reference
├── EVOLUTION.md             # evolutionary model technical reference
├── BENCHMARK.md             # performance benchmarks
├── PRODUCTION.md            # production run instructions
├── pyproject.toml
├── LICENSE.md
└── README.md
```

