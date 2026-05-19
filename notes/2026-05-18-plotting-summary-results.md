# Plan: Batch Summary Analyses

## Overview

Two summary analyses for a completed batch of trellis trajectories. Both work directly from the FASTA shards in `results/trellis-18aa-{ligand}/` — no refolding needed, just reading DNA and AA sequences.

Output: a single script `analysis/summarize_batch.py` that reads a batch output directory, computes both analyses, and writes summary figures to `analysis/figures/`.

## Input

```bash
python analysis/summarize_batch.py
```

The script reads all `forwards-train-*.tar.zst` and `forwards-test-*.tar.zst` shards, extracts every trajectory FASTA, and parses the DNA sequences per step. AA sequences are derived by translating the DNA (using `trellis.genetic_code.translate`).

### Parsing

Each FASTA file is one trajectory with 101 entries (step 0 through step 100). Parse into a list of 1000 trajectories, each a list of 101 DNA strings. The header `>{name}|{cumulative_hamming}|{branch_hamming}` can be used for validation but the DNA sequences are the primary data.

## Analysis 1: Pairwise convergence

### Question

Are trajectories converging to the same sequences? If so, held-out test trajectories may overlap with training trajectories, compromising the evaluation.

### Method

At each step t (0, 10, 20, ..., 100), collect the DNA sequence (or AA sequence) at step t from all 1000 trajectories. Compute all-vs-all pairwise Hamming distances for that step. This is (1000 choose 2) = 499,500 pairs per step — fast for 54 nt strings.

Repeat separately for DNA and AA levels. DNA Hamming is on 54 characters; AA Hamming is on 18 characters.

### Output: `convergence.png`

Two-row figure (DNA top, AA bottom). Each row has:

**Panel A: Distribution of pairwise Hamming distances at selected steps.** Overlaid histograms (or violin plots) for steps 0, 25, 50, 75, 100. At step 0 (random starts), the distribution should be broad. If trajectories converge, the distribution shifts left over time. Use a different color per step.

**Panel B: Summary statistics over all steps.** Line plot with step on x-axis, showing:
- Mean pairwise Hamming distance (solid line)
- Minimum pairwise Hamming distance (dashed line, this is the critical one)
- 5th percentile pairwise Hamming distance (dotted line)

The minimum pairwise distance is the key diagnostic. If it reaches 0 at the DNA level, two trajectories produced identical sequences — a red flag for train/test contamination. If the minimum AA Hamming reaches 0 but DNA doesn't, trajectories found the same protein via different codons (less concerning but worth knowing).

**Panel C: Minimum pairwise Hamming at step 100 between train and test sets specifically.** The train/test split is recorded in which shard a trajectory came from. Compute the minimum Hamming between any test trajectory's final sequence and any training trajectory's final sequence. Report this number in the figure title. This is the direct measure of train/test bleed-through risk.

### Implementation notes

- Sample steps at intervals of 10 (0, 10, 20, ..., 100) rather than every step to keep computation and plot density reasonable.
- For the pairwise distance computation, use a vectorized approach: stack all sequences at step t into a 2D numpy array of character codes, then compute Hamming distances via broadcasting. For 1000 × 54, this is fast.
- Separate the train and test trajectories based on which shard file they came from (`forwards-train-*.tar.zst` vs `forwards-test-*.tar.zst`).

## Analysis 2: Cumulative unique AA sequences

### Question

How much of the reachable AA sequence space do 1000 trajectories cover? Is 1000 trajectories enough, or would more trajectories continue to discover new sequences?

### Method

Process trajectories in order (trajectory 0 through 999). Maintain a running set of all unique AA sequences seen. After each trajectory, record the set size. Also track:
- Total unique AA sequences after each trajectory
- New unique AA sequences contributed by each trajectory (marginal contribution)
- Total unique AA sequences at each step across all trajectories (how diversity grows over evolutionary time)

### Output: `sequence_diversity.png`

**Panel A: Cumulative unique AA sequences vs trajectory number.** X-axis: trajectory index (0–999). Y-axis: cumulative count of unique AA sequences seen. If the curve is still climbing steeply at 1000, more trajectories would sample new regions. If it flattens, the reachable space is saturated. Draw a horizontal dashed line at the theoretical maximum for the final panel to contextualize the saturation level.

**Panel B: Marginal contribution per trajectory.** X-axis: trajectory index. Y-axis: number of new unique AA sequences added by each trajectory. This should decrease over time as the cumulative set fills up. A bar chart or scatter plot.

**Panel C: Unique AA sequences at each step, averaged across trajectories.** X-axis: step (0–100). Y-axis: mean number of unique AA sequences seen up to step t within a single trajectory. This shows how quickly each individual trajectory explores — early steps (climbing) should add many new sequences, later steps (plateau) should add fewer. Plot mean ± 1 SD as a shaded envelope.

### Implementation notes

- The trajectory processing order matters for Panel A. Use the order they appear in the shards (which reflects the random train/test split). Could also shuffle and repeat to show the variance, but for a first pass the natural order is fine.
- For Panel C, "unique AA sequences within a trajectory up to step t" means the cumulative set {aa_0, aa_1, ..., aa_t}. Many consecutive steps will share the same AA (synonymous mutations), so this count grows in jumps.

## Script structure

```python
# scripts/summarize_batch.py

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from trellis.genetic_code import translate
# Shard reading utilities (parse tar.zst → list of trajectory DNA sequences)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("batch_dir", type=Path)
    p.add_argument("--step-interval", type=int, default=10)
    return p.parse_args()

def load_trajectories(batch_dir):
    """Read all shards, return (train_trajectories, test_trajectories).
    Each trajectory is a list of DNA strings (101 entries)."""
    ...

def analyze_convergence(train_trajs, test_trajs, step_interval):
    """Compute pairwise Hamming distances at sampled steps."""
    ...

def analyze_diversity(all_trajs):
    """Compute cumulative unique AA sequence counts."""
    ...

def plot_convergence(convergence_data, output_path):
    ...

def plot_diversity(diversity_data, output_path):
    ...

def main():
    args = parse_args()
    train, test = load_trajectories(args.batch_dir)
    fig_dir = args.batch_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    conv = analyze_convergence(train, test, args.step_interval)
    plot_convergence(conv, fig_dir / "convergence.png")

    div = analyze_diversity(train + test)
    plot_diversity(div, fig_dir / "sequence_diversity.png")
```

## Dependencies

- `matplotlib` for figures (already in the project for `visualize_conformation.py`)
- `zstandard` and `tarfile` for reading shards (already dependencies)
- `numpy` for vectorized Hamming computation

## Output

```
results/trellis-18aa-KEMN/
├── forwards-train-000.tar.zst
├── forwards-test-000.tar.zst
├── metadata.json
└── figures/
    ├── convergence.png
    └── sequence_diversity.png
```
