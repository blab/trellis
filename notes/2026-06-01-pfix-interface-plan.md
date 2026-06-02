# Plan: P_fix Evaluation Interface and PyPI Packaging

## Overview

Two changes to trellis:

1. **Evaluation interface.** Expose the SSWM probability computation as a clean public API that downstream repos can import. This is the ground-truth oracle for evaluating how well a diffusion model has learned the fitness landscape.
2. **PyPI packaging.** Publish trellis as a pip-installable package so downstream repos (pegasus-evals, diffusion-language-model) can declare it as a dependency.

## Part 1: Evaluation interface

### Refactor `compute_sswm_probabilities` into a shared helper

The SSWM probability computation — enumerate all single-nt mutations, fold unique AAs, compute fixation probabilities — is currently inlined in `generate_trajectory()` in `sswm.py`. The phylogeny generation plan also needs it. And now downstream evaluation repos need it. Refactor into a standalone function.

**File:** `trellis/sswm.py`

Add this function:

```python
def compute_sswm_probabilities(
    dna: str,
    current_fitness: float,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    db: ConformationDatabase,
    fitness_cache: FitnessCache,
    Ne: float,
    temperature: float = 1.0,
) -> tuple[list[str], np.ndarray]:
    """Compute normalized SSWM fixation probabilities for all single-nt neighbors.

    Returns
    -------
    mutant_dnas : list[str]
        All single-nucleotide mutant DNA sequences (~3*len(dna) entries).
    probs : np.ndarray
        Fixation probability for each mutant, normalized to sum to 1.
        If all mutations are lethal, returns all zeros.
    """
    mutations = single_nt_mutations(dna)
    mutation_aa = [translate(mutant_dna) for mutant_dna, _, _, _ in mutations]

    # Fold each unique AA sequence (cache-aware)
    unique_aas = set(mutation_aa)
    for aa_seq in unique_aas:
        if aa_seq not in fitness_cache:
            if "*" in aa_seq:
                fitness_cache.put(aa_seq, FitnessResult(
                    fitness=-inf, fold_result=None,
                    aa_sequence=aa_seq, dna_sequence="",
                ))
            else:
                r = compute_fitness_aa(aa_seq, ligand, mj_matrix, temperature, db=db)
                fitness_cache.put(aa_seq, r)

    # Compute fixation probabilities
    mutant_dnas = [m for m, _, _, _ in mutations]
    s_values = np.array([
        fitness_cache.get(aa).fitness - current_fitness for aa in mutation_aa
    ])
    probs = _vectorized_fixation_prob(s_values, Ne)

    total = probs.sum()
    if total > 0:
        probs = probs / total

    return mutant_dnas, probs
```

Then update `generate_trajectory()` to call this function instead of duplicating the logic. The inner loop body becomes:

```python
mutant_dnas, probs = compute_sswm_probabilities(
    current_dna, current_fitness, ligand, mj_matrix, db,
    fitness_cache, Ne, temperature,
)

if probs.sum() == 0:
    break

idx = rng.choice(len(mutant_dnas), p=probs)
chosen_dna = mutant_dnas[idx]
chosen_aa = translate(chosen_dna)
chosen_fitness = fitness_cache.get(chosen_aa).fitness
mut_type = classify_mutation(current_dna, chosen_dna)
```

Note: `probs` is already normalized inside `compute_sswm_probabilities`, so no need to normalize again. But `generate_trajectory` should check `probs.sum() == 0` (all lethal) to break, which means `compute_sswm_probabilities` should return unnormalized zeros in that case. Adjust the function to only normalize when total > 0, and return all-zeros otherwise. The caller checks `probs.sum() == 0` before sampling. (This is already reflected in the code above.)

### Add `pfix_for_target` convenience function

**File:** `trellis/sswm.py`

```python
def pfix_for_target(
    reference_dna: str,
    target_dna: str,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    db: ConformationDatabase,
    fitness_cache: FitnessCache,
    Ne: float,
    temperature: float = 1.0,
) -> float | None:
    """Return the normalized P_fix for mutating reference_dna to target_dna.

    Returns None if target_dna is not a single-nucleotide neighbor of
    reference_dna. Returns 0.0 if all neighbors are lethal.

    The returned value is the probability that *this specific mutation*
    would be the next substitution under SSWM dynamics, i.e. the
    fixation probability of target_dna normalized by the sum over all
    single-nt neighbors.
    """
    # Verify target is a single-nt neighbor
    diffs = sum(1 for a, b in zip(reference_dna, target_dna) if a != b)
    if diffs != 1 or len(reference_dna) != len(target_dna):
        return None

    # Compute current fitness
    ref_aa = translate(reference_dna)
    if ref_aa not in fitness_cache:
        r = compute_fitness_aa(ref_aa, ligand, mj_matrix, temperature, db=db)
        fitness_cache.put(ref_aa, r)
    current_fitness = fitness_cache.get(ref_aa).fitness

    mutant_dnas, probs = compute_sswm_probabilities(
        reference_dna, current_fitness, ligand, mj_matrix, db,
        fitness_cache, Ne, temperature,
    )

    try:
        idx = mutant_dnas.index(target_dna)
        return float(probs[idx])
    except ValueError:
        return None  # shouldn't happen if diffs == 1, but defensive
```

### Add CLI script

**File:** `scripts/pfix_distribution.py`

```python
#!/usr/bin/env python3
"""Print the SSWM fixation probability distribution for a reference sequence.

Examples:
    # Full distribution, sorted by probability
    python scripts/pfix_distribution.py \
        --reference GCTTGTGAT... --ligand KEMN --Ne 50

    # Probability for a specific target
    python scripts/pfix_distribution.py \
        --reference GCTTGTGAT... --target GCTTGTAAT... --ligand KEMN --Ne 50
"""

import argparse
import json

import numpy as np

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fitness import compute_fitness_aa
from trellis.fold_enum import enumerate_conformations
from trellis.genetic_code import classify_mutation, translate
from trellis.ligand import create_ligand
from trellis.sswm import compute_sswm_probabilities, pfix_for_target


def parse_anchor(value: str) -> tuple[int, int]:
    parts = value.split(",")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Reference DNA sequence")
    parser.add_argument("--target", default=None, help="Target DNA sequence (optional)")
    parser.add_argument("--ligand", required=True, help="Ligand AA sequence")
    parser.add_argument("--ligand-anchor", type=parse_anchor, default=(0, -1))
    parser.add_argument("--chain-length", type=int, default=18)
    parser.add_argument("--Ne", type=float, default=50.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N mutations (ignored with --target)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    mj = load_mj_matrix()
    ligand = create_ligand(args.ligand, anchor=args.ligand_anchor)
    db = enumerate_conformations(args.chain_length, ligand)
    cache = FitnessCache()

    # Compute reference fitness
    ref_aa = translate(args.reference)
    r = compute_fitness_aa(ref_aa, ligand, mj, args.temperature, db=db)
    cache.put(ref_aa, r)

    if args.target:
        # Single target mode
        p = pfix_for_target(
            args.reference, args.target, ligand, mj, db, cache,
            args.Ne, args.temperature,
        )
        if p is None:
            print("error: target is not a single-nucleotide neighbor of reference")
            return
        mut_type = classify_mutation(args.reference, args.target)
        if args.json:
            print(json.dumps({
                "reference": args.reference, "target": args.target,
                "pfix": p, "mutation_type": mut_type,
                "reference_fitness": r.fitness,
            }, indent=2))
        else:
            print(f"reference fitness: {r.fitness:.4f}")
            print(f"target pfix:       {p:.6f}")
            print(f"mutation type:     {mut_type}")
    else:
        # Full distribution mode
        mutant_dnas, probs = compute_sswm_probabilities(
            args.reference, r.fitness, ligand, mj, db, cache,
            args.Ne, args.temperature,
        )

        # Sort by probability descending
        order = np.argsort(-probs)

        n_nonzero = int((probs > 0).sum())
        n_above_1pct = int((probs > 0.01).sum())

        if args.json:
            entries = []
            for i in order:
                if probs[i] == 0:
                    continue
                entries.append({
                    "dna": mutant_dnas[i],
                    "aa": translate(mutant_dnas[i]),
                    "pfix": float(probs[i]),
                    "type": classify_mutation(args.reference, mutant_dnas[i]),
                })
            print(json.dumps({
                "reference": args.reference,
                "reference_fitness": r.fitness,
                "n_mutations": len(mutant_dnas),
                "n_nonzero": n_nonzero,
                "n_above_1pct": n_above_1pct,
                "distribution": entries,
            }, indent=2))
        else:
            print(f"reference fitness: {r.fitness:.4f}")
            print(f"mutations: {len(mutant_dnas)} total, "
                  f"{n_nonzero} nonzero, {n_above_1pct} above 1%")
            print()
            print(f"{'pfix':>10}  {'type':<14}  {'aa_change':<10}  mutation")
            print("-" * 60)
            for rank, i in enumerate(order[:args.top]):
                if probs[i] == 0:
                    break
                mut_type = classify_mutation(args.reference, mutant_dnas[i])
                ref_aa = translate(args.reference)
                mut_aa = translate(mutant_dnas[i])
                # Find the position that changed
                pos = next(j for j in range(len(args.reference))
                           if args.reference[j] != mutant_dnas[i][j])
                aa_pos = pos // 3
                aa_change = (f"{ref_aa[aa_pos]}{aa_pos+1}{mut_aa[aa_pos]}"
                             if ref_aa[aa_pos] != mut_aa[aa_pos] else "syn")
                nuc_change = f"{args.reference[pos]}{pos+1}{mutant_dnas[i][pos]}"
                print(f"{probs[i]:10.6f}  {mut_type:<14}  {aa_change:<10}  {nuc_change}")


if __name__ == "__main__":
    main()
```

## Part 2: PyPI packaging

### Add `pyproject.toml`

**File:** `pyproject.toml` (repo root)

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[project]
name = "trellis-lattice"
version = "0.1.0"
description = "2D lattice protein evolutionary trajectory simulation"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [
    {name = "Trevor Bedford"},
]
dependencies = [
    "numpy>=1.24",
    "numba>=0.58",
    "zstandard>=0.21",
]

[project.optional-dependencies]
aws = ["boto3>=1.28"]
viz = ["matplotlib>=3.7"]
dev = ["pytest>=7.0"]

[tool.setuptools.packages.find]
include = ["trellis*"]

[tool.setuptools.package-data]
trellis = ["../data/*.csv"]
```

The `data/mj_matrix.csv` path needs to be accessible after install. There are two approaches:

**Option A (simpler):** Move `data/mj_matrix.csv` inside the `trellis/` package directory as `trellis/data/mj_matrix.csv`. Update `energy.py` to reference it via `importlib.resources` or `Path(__file__).parent / "data" / "mj_matrix.csv"`. Then in `pyproject.toml`:

```toml
[tool.setuptools.package-data]
trellis = ["data/*.csv"]
```

**Option B:** Keep `data/` at repo root and use `importlib.resources` with a data package. More complex, no real benefit.

**Recommendation: Option A.** Move the file into the package so it installs cleanly:

```
trellis/
├── __init__.py
├── data/
│   └── mj_matrix.csv
├── energy.py      ← update DEFAULT_MJ_PATH
├── fitness.py
├── ...
```

Update `energy.py`:

```python
DEFAULT_MJ_PATH = Path(__file__).resolve().parent / "data" / "mj_matrix.csv"
```

This line is already correct if the data dir is a sibling of `energy.py`. Just needs the file physically moved.

### Verify local install works

Before publishing:

```bash
pip install -e ".[dev]"
pytest tests/
```

The editable install confirms that imports, data file loading, and all tests work from the packaged layout.

### Publish to PyPI

One-time setup:

```bash
pip install build twine
```

Build and publish:

```bash
python -m build
twine upload dist/*
```

You'll need a PyPI account and API token. Store the token in `~/.pypirc` or pass via environment variable.

After publishing, downstream repos install with:

```bash
pip install trellis-lattice
```

Or for development against a specific commit:

```bash
pip install git+https://github.com/blab/trellis.git@main
```

### Downstream usage in evaluation repos

In `pegasus-evals/requirements.txt` or `pyproject.toml`:

```
trellis-lattice>=0.1.0
```

Then in evaluation code:

```python
from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.ligand import create_ligand
from trellis.sswm import compute_sswm_probabilities

# One-time setup per ligand
mj = load_mj_matrix()
ligand = create_ligand("KEMN", anchor=(0, -1))
db = enumerate_conformations(18, ligand)
cache = FitnessCache()

# Per-branch evaluation
ref_aa = translate(ref_dna)
if ref_aa not in cache:
    r = compute_fitness_aa(ref_aa, ligand, mj, temperature=1.0, db=db)
    cache.put(ref_aa, r)
ref_fitness = cache.get(ref_aa).fitness

mutant_dnas, ground_truth_probs = compute_sswm_probabilities(
    ref_dna, ref_fitness, ligand, mj, db, cache, Ne=50.0, temperature=1.0,
)

# Compare with model predictions
pearson_r = np.corrcoef(ground_truth_probs, model_probs)[0, 1]
```

The cache should be shared across all evaluations for the same ligand to avoid redundant folding.

## Implementation order

1. **Refactor `compute_sswm_probabilities`** out of `generate_trajectory`. Update `generate_trajectory` to call it. Run tests to confirm identical behavior.
2. **Add `pfix_for_target`** to `sswm.py`.
3. **Add `scripts/pfix_distribution.py`** CLI.
4. **Move `data/mj_matrix.csv`** into `trellis/data/`. Update `energy.py` path. Run tests.
5. **Add `pyproject.toml`**. Verify `pip install -e ".[dev]"` and `pytest` pass.
6. **Publish to PyPI** as `trellis-lattice` version 0.1.0.
7. **Update downstream repos** to add `trellis-lattice` as a dependency.

## Validation

- `pytest tests/` passes after the refactor (step 1)
- `generate_trajectories.py` with a fixed seed produces bit-identical output before and after the refactor
- `scripts/pfix_distribution.py --reference ... --ligand KEMN` outputs a sensible distribution (top mutations are synonymous near the fitness peak, nonsynonymous early in a trajectory)
- `pip install trellis-lattice` in a fresh virtualenv, then `from trellis.sswm import compute_sswm_probabilities` works
- The `data/mj_matrix.csv` file loads correctly from the installed package (not just from the repo checkout)
