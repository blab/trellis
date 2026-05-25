"""Generate a single phylogenetic tree and write it to Auspice JSON."""

import argparse
from pathlib import Path

import numpy as np

from trellis.auspice_io import write_auspice_json
from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.ligand import create_ligand
from trellis.phylogeny import generate_phylogeny
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
    parser.add_argument("--min-active", type=int, default=1)
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
        min_active_lineages=args.min_active,
        max_active_lineages=args.max_active,
        max_total_nodes=args.max_nodes,
        rng=rng,
        fitness_cache=cache,
    )

    n_tips = phy.metadata["n_tips"]
    print(f"tree: {len(phy.nodes)} nodes, {n_tips} tips")
    print(f"writing to {args.output}")
    write_auspice_json(phy, args.output)


if __name__ == "__main__":
    main()
