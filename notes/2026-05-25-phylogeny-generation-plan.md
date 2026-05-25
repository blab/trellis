# Plan: Phylogenetic Trajectory Generation

## Motivation

Current trellis output from `generate_trajectories.py` consists of independent SSWM walks: each trajectory is a single root-to-tip path through sequence space. For evaluation, holdout trajectories begin from random starts far from training sequences and never converge with training data. This is a worst-case generalization test — a model must predict evolutionary dynamics with zero local context.

Empirical training data from `blab/trajectories` is structurally different. It is extracted from phylogenetic reconstructions: a tree of related sequences where branches share recent common ancestors, and tips sampled at different times provide multiple windows on the local fitness landscape. Train/test splits use clade excision, which keeps training and test data within the same broad sequence neighborhood while ensuring strict ancestor-descendant separation.

We want a synthetic dataset with the same structural properties: phylogenetic trees with shared ancestry, sampled tips throughout evolutionary time, and tree topology that can be processed by the existing `blab/trajectories` pipeline. Each tree represents one population evolving under SSWM dynamics with periodic speciation and sampling events.

## The stochastic process

At each step, every currently-active lineage independently undergoes three events in sequence:

1. **Sampling** with probability ψ. The lineage becomes a tip at the current depth and stops evolving.
2. **Speciation** with probability β. The lineage forks into two daughter lineages.
3. **Substitution.** Each surviving lineage (or each daughter after speciation) samples one mutation from the SSWM fixation-probability distribution.

This is a forward-time **Yule-with-sampling** process layered on SSWM dynamics. It is structurally similar to birth-death-sampling models from phylodynamics, simplified by dropping the death (lineage extinction) term — under SSWM, lineages don't go extinct unless all single-nucleotide mutations are lethal (stop codons), which is rare and handled separately as a terminating condition.

### Why Yule-with-sampling rather than other tree-generation schemes

The Yule-with-sampling process matches the empirical situation we want to model. In real phylogenetic data:

- **Speciation events occur at roughly constant rate per lineage.** Two populations diverge into independent lineages through geographic or ecological isolation. The rate of these events doesn't depend on the fitness landscape — it depends on external factors like population structure. A constant per-lineage speciation rate (the Yule assumption) is the simplest model consistent with this.

- **Sampling events occur throughout the time range, not just at the present.** This is the key distinguishing feature from pure Yule trees. Influenza phylogenies sample sequences continuously from 2000 to 2026, producing the characteristic ladder-like topology with side clades falling off the trunk throughout the time range. Test data must include side clades from across this entire range, not just near the present. The per-step sampling probability ψ produces this naturally — early lineages have many chances to be sampled, so tips appear throughout the tree, not just at the maximum depth.

### Why sample two mutations independently at speciation

When a lineage speciates, both daughters independently sample from the same SSWM distribution. Two outcomes are possible:

1. **Different mutations.** The daughters diverge immediately. This is the standard textbook case and produces clean phylogenetic trees with branch length 1 between parent and each daughter.

2. **Same mutation.** Both daughters acquire the same substitution. They sit on the same genotype for one or more steps before drifting apart.

An alternative implementation could enforce that daughters sample *different* mutations (sample without replacement, weighted by residual probability). We reject this approach because:

- In real populations, two daughter lineages can sit on identical genotypes for many generations before stochastic drift separates them. The "same mutation" outcome is biologically realistic, not an artifact.
- More importantly, allowing the same mutation lets the model see that when one mutation strongly dominates the SSWM probability, multiple independent lineages converge to take it. This is information about the fitness landscape — specifically that mutation A is the obvious next step from sequence X. Forcing different mutations would hide this signal.

The Auspice JSON format and downstream tree processing handle zero-Hamming branches without issue. Both daughter nodes are recorded as distinct entities with their own subsequent evolutionary history.

### Parameter intuition

With per-lineage rates β (speciation) and ψ (sampling), the expected number of active lineages per step changes by factor (1 − ψ)(1 + β). For roughly stable populations, β ≈ ψ/(1 − ψ). For growing populations (more lineages later, like influenza fitness exploration after the trunk emerges), set β slightly higher than this stable rate.

Target parameters for 1000-2000 total nodes per tree over 100 steps:

| Parameter | Value | Rationale |
|---|---|---|
| ψ (per-lineage sampling rate) | 0.05 | ~5% of active lineages sampled per step |
| β (per-lineage speciation rate) | 0.08 | Slight growth: ~2.6% per step → ~13× total |
| n_steps | 100 | Same as independent trajectory dataset |
| max_active_lineages | 100 | Safety cap to prevent runaway growth |
| max_total_nodes | 5000 | Safety cap on tree size |

These parameters produce trees with roughly 500–1000 tips sampled across depths 1–100, plus a similar number of internal nodes. The cap parameters prevent edge cases (e.g., a tree where stochastic luck leads to explosive lineage growth) from breaking downstream processing or exhausting memory.

## Implementation

### Module: `trellis/phylogeny.py`

New module containing the tree generation logic. Imports from existing modules (`fitness`, `sswm`, `genetic_code`, `fold_enum`).

### Data structures

A tree is built as a list of nodes and a list of edges. Each node tracks its DNA sequence, amino acid sequence, fitness, depth from root, and tip status.

```python
@dataclass
class PhylogenyNode:
    """A node in the phylogenetic tree."""
    id: int
    dna: str
    aa: str
    fitness: float
    depth: int           # Hamming distance from root (= step number)
    is_tip: bool
    parent_id: int | None  # None for root


@dataclass
class Phylogeny:
    """A phylogenetic tree generated by Yule-with-sampling on SSWM dynamics."""
    nodes: list[PhylogenyNode]
    root_id: int
    metadata: dict
```

The metadata dict records all generation parameters (β, ψ, n_steps, Ne, temperature, ligand info, seed) for reproducibility and downstream pipeline use.

### Core algorithm

The algorithm processes lineages level-by-level (all active lineages at depth d → all at depth d+1). This keeps the bookkeeping simple and matches how the trees are conceptually structured.

```python
def generate_phylogeny(
    start_dna: str,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    db: ConformationDatabase,
    n_steps: int = 100,
    Ne: float = 50.0,
    beta: float = 0.08,
    psi: float = 0.05,
    temperature: float = 1.0,
    max_active_lineages: int = 100,
    max_total_nodes: int = 5000,
    rng: np.random.Generator | None = None,
    fitness_cache: FitnessCache | None = None,
) -> Phylogeny:
    """Generate a phylogenetic tree via Yule-with-sampling on SSWM dynamics."""

    if rng is None:
        rng = np.random.default_rng()
    if fitness_cache is None:
        fitness_cache = FitnessCache()

    # Initialize root
    root_aa = translate(start_dna)
    root_fitness_result = compute_fitness_aa(
        root_aa, ligand, mj_matrix, temperature, db=db
    )
    fitness_cache.put(root_aa, root_fitness_result)

    root = PhylogenyNode(
        id=0, dna=start_dna, aa=root_aa,
        fitness=root_fitness_result.fitness, depth=0,
        is_tip=False, parent_id=None,
    )
    nodes = [root]

    # Active lineages: list of node ids that may evolve further
    active = [0]

    for depth in range(1, n_steps + 1):
        if not active:
            break  # all lineages sampled or terminated
        if len(nodes) >= max_total_nodes:
            # Cap: convert all active lineages to tips and stop
            for node_id in active:
                nodes[node_id].is_tip = True
            break

        new_active = []
        for parent_id in active:
            parent = nodes[parent_id]

            # Sample? Lineage becomes a tip and terminates.
            if rng.random() < psi:
                parent.is_tip = True
                continue

            # Speciate? One or two daughter lineages.
            n_daughters = 2 if rng.random() < beta else 1

            # Compute SSWM probabilities (cached fitness lookup for neighbors)
            mutations, probs = compute_sswm_probabilities(
                parent.dna, parent.fitness, ligand, mj_matrix, db,
                fitness_cache, Ne, temperature,
            )

            if probs.sum() == 0:
                # All neighbors lethal — terminate as tip
                parent.is_tip = True
                continue

            probs = probs / probs.sum()

            # Sample n_daughters mutations independently (with replacement)
            for _ in range(n_daughters):
                idx = rng.choice(len(mutations), p=probs)
                child_dna = mutations[idx]
                child_aa = translate(child_dna)
                child_fitness = fitness_cache.get(child_aa).fitness

                child = PhylogenyNode(
                    id=len(nodes), dna=child_dna, aa=child_aa,
                    fitness=child_fitness, depth=depth,
                    is_tip=False, parent_id=parent_id,
                )
                nodes.append(child)
                new_active.append(child.id)

        # Enforce max active lineages: if exceeded, randomly sample some as tips
        if len(new_active) > max_active_lineages:
            n_to_sample = len(new_active) - max_active_lineages
            indices_to_sample = rng.choice(
                len(new_active), size=n_to_sample, replace=False
            )
            for idx in indices_to_sample:
                nodes[new_active[idx]].is_tip = True
            new_active = [
                new_active[i] for i in range(len(new_active))
                if i not in set(indices_to_sample.tolist())
            ]

        active = new_active

    # Anything still active at end of n_steps becomes a tip
    for node_id in active:
        nodes[node_id].is_tip = True

    metadata = {
        "n_steps": n_steps,
        "Ne": Ne,
        "beta": beta,
        "psi": psi,
        "temperature": temperature,
        "ligand_sequence": ligand.sequence,
        "ligand_positions": list(ligand.positions),
        "n_nodes": len(nodes),
        "n_tips": sum(1 for n in nodes if n.is_tip),
    }

    return Phylogeny(nodes=nodes, root_id=0, metadata=metadata)
```

### Helper function for SSWM probabilities

The SSWM mutation enumeration and fixation probability computation is the same logic used in `generate_trajectory()` in `sswm.py`. Refactor that into a shared helper to avoid duplication:

```python
def compute_sswm_probabilities(
    dna: str,
    current_fitness: float,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    db: ConformationDatabase,
    fitness_cache: FitnessCache,
    Ne: float,
    temperature: float,
) -> tuple[list[str], np.ndarray]:
    """Return (mutant_dna_list, fixation_probabilities) for all single-nt neighbors.

    Folds each unique AA neighbor (cache-aware), computes selection
    coefficients, applies the Kimura formula. Identical logic to one
    SSWM step in generate_trajectory().
    """
    mutations = single_nt_mutations(dna)
    mutation_aa = [translate(m) for m, _, _, _ in mutations]

    # Fold each unique AA
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
    probs = vectorized_fixation_prob(s_values, Ne)

    return mutant_dnas, probs
```

This function should be moved into `trellis/sswm.py` and `generate_trajectory()` updated to call it, eliminating duplicate logic between the two trajectory generation paths.

## Auspice JSON serialization

Output format follows the Nextstrain Auspice v2 specification, restricted to the fields needed by the `blab/trajectories` preprocessing pipeline.

### Module: `trellis/auspice_io.py`

New module. Converts a `Phylogeny` into a JSON document and writes it to disk.

### Output structure

```json
{
  "version": "v2",
  "meta": {
    "title": "Trellis phylogeny | ligand=KEMN | seed=42",
    "extensions": {
      "trellis": {
        "ligand_sequence": "KEMN",
        "ligand_positions": [[0,-1],[1,-1],[2,-1],[3,-1]],
        "chain_length": 18,
        "Ne": 50.0,
        "beta": 0.08,
        "psi": 0.05,
        "temperature": 1.0,
        "n_steps": 100,
        "seed": 42
      }
    }
  },
  "tree": {
    "name": "NODE_0000",
    "node_attrs": {
      "div": 0.0
    },
    "branch_attrs": {
      "mutations": {
        "nuc": []
      }
    },
    "children": [
      {
        "name": "NODE_0001",
        "node_attrs": {"div": 1.0},
        "branch_attrs": {"mutations": {"nuc": ["A23G"]}},
        "children": [...]
      }
    ]
  }
}
```

Field choices:

- **No `num_date`.** There is no clock to interpret in this synthetic system. Depth is captured implicitly through `div` (cumulative Hamming distance from root, which equals depth from root since each step is one mutation, except for zero-Hamming branches following speciation events with identical mutation sampling).
- **`div`** is the cumulative Hamming distance from the root. For trellis, this equals the number of mutations along the path from root to this node, since each step is a single-nucleotide substitution. Speciation events with identical daughter mutations produce parent/child pairs with `div` differing by zero — this is intentional and handled correctly by downstream processing.
- **`branch_attrs.mutations.nuc`** lists the nucleotide mutations on the branch leading to this node, in the format `{REF}{POS}{ALT}` with 1-indexed positions. For speciation branches with no mutation, this is an empty list.
- **`meta.extensions.trellis`** carries the simulation parameters for reproducibility. The `blab/trajectories` pipeline ignores unknown extensions, so this is non-intrusive.

### Conversion logic

```python
def phylogeny_to_auspice(phy: Phylogeny, title: str | None = None) -> dict:
    """Convert a Phylogeny to an Auspice v2 JSON dict."""

    # Build child-of mapping
    children_of: dict[int, list[int]] = {n.id: [] for n in phy.nodes}
    for n in phy.nodes:
        if n.parent_id is not None:
            children_of[n.parent_id].append(n.id)

    def build_subtree(node_id: int) -> dict:
        node = phy.nodes[node_id]
        parent = phy.nodes[node.parent_id] if node.parent_id is not None else None
        mutations = []
        if parent is not None:
            for i, (a, b) in enumerate(zip(parent.dna, node.dna)):
                if a != b:
                    mutations.append(f"{a}{i + 1}{b}")  # 1-indexed
        subtree = {
            "name": f"NODE_{node_id:04d}" if not node.is_tip else f"TIP_{node_id:04d}",
            "node_attrs": {"div": float(node.depth)},
            "branch_attrs": {"mutations": {"nuc": mutations}},
        }
        if children_of[node_id]:
            subtree["children"] = [build_subtree(c) for c in children_of[node_id]]
        return subtree

    auspice = {
        "version": "v2",
        "meta": {
            "title": title or f"Trellis phylogeny | ligand={phy.metadata['ligand_sequence']}",
            "extensions": {"trellis": phy.metadata},
        },
        "tree": build_subtree(phy.root_id),
    }
    return auspice


def write_auspice_json(phy: Phylogeny, path: str | Path, title: str | None = None) -> None:
    """Write a phylogeny to disk as an Auspice v2 JSON file."""
    auspice = phylogeny_to_auspice(phy, title=title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(auspice, f, indent=2)
        f.write("\n")
```

The recursion is fine for trees of this size (a few thousand nodes max). Python's default recursion limit (1000) might be hit for very deep tip-poor trees, but the max_total_nodes safety cap prevents this.

## Script: `scripts/generate_phylogeny.py`

CLI for generating a single phylogeny. Useful for testing and visualization. Bulk generation will come later in its own script.

```python
#!/usr/bin/env python3
"""Generate a single phylogenetic tree and write it to Auspice JSON."""

import argparse
from pathlib import Path

import numpy as np

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.ligand import create_ligand
from trellis.phylogeny import generate_phylogeny
from trellis.auspice_io import write_auspice_json
from trellis.sswm import generate_start_sequence


def parse_anchor(value: str) -> tuple[int, int]:
    parts = value.split(",")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand-sequence", type=str, default="KEMN")
    parser.add_argument("--ligand-anchor", type=parse_anchor, default=(0, -1))
    parser.add_argument("--chain-length", type=int, default=18)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--Ne", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--psi", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-fitness", type=float, default=0.0)
    parser.add_argument("--max-active", type=int, default=100)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mj = load_mj_matrix()
    ligand = create_ligand(args.ligand_sequence, anchor=args.ligand_anchor)
    print(f"enumerating conformations for ligand {args.ligand_sequence}...")
    db = enumerate_conformations(args.chain_length, ligand)
    print(f"  {db.n_conformations} conformations")

    rng = np.random.default_rng(args.seed)
    cache = FitnessCache()

    print(f"generating start sequence (min_fitness={args.min_fitness})...")
    start_dna = generate_start_sequence(
        args.chain_length, ligand, mj,
        min_fitness=args.min_fitness,
        temperature=args.temperature,
        rng=rng, db=db,
    )

    print(f"generating phylogeny (beta={args.beta}, psi={args.psi})...")
    phy = generate_phylogeny(
        start_dna=start_dna,
        ligand=ligand,
        mj_matrix=mj,
        db=db,
        n_steps=args.n_steps,
        Ne=args.Ne,
        beta=args.beta,
        psi=args.psi,
        temperature=args.temperature,
        max_active_lineages=args.max_active,
        max_total_nodes=args.max_nodes,
        rng=rng,
        fitness_cache=cache,
    )

    n_tips = sum(1 for n in phy.nodes if n.is_tip)
    print(f"tree: {len(phy.nodes)} nodes, {n_tips} tips")
    print(f"writing to {args.output}")
    write_auspice_json(phy, args.output)


if __name__ == "__main__":
    main()
```

## Validation

Before scaling to bulk generation, validate on a single tree:

1. **Visualize in Auspice.** Load the output JSON into [auspice.us](https://auspice.us/) and confirm the tree renders as expected — sampled tips distributed across the depth range, internal nodes connecting branches, branch labels showing mutations.

2. **Check tip count distribution.** Generate 10 trees with the default parameters. Verify total node counts fall in the 500–2000 range without hitting `max_total_nodes`. Verify tip count is reasonable (several hundred per tree).

3. **Check tip depth distribution.** Plot the histogram of tip depths across multiple trees. Should be a roughly geometric distribution with mean around 1/ψ = 20 steps, plus a spike at depth 100 from forced termination of any remaining active lineages.

4. **Check fitness trajectory along the trunk.** Identify the deepest tip in a tree and trace its path from root. Plot fitness vs depth. Should look qualitatively like a single SSWM trajectory: initial climb, then plateau near a fitness peak.

5. **Round-trip through the `blab/trajectories` preprocessing pipeline.** Generate one tree, write Auspice JSON, run the `trajectories` repo's preprocessing on it. Verify that train/test FASTAs come out and look reasonable. This is the critical compatibility test.
