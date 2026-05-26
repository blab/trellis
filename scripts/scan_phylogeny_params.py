"""Parameter scan for phylogeny generation — find (beta, psi, max_active) that yield ~5000 tips."""

import sys
import time

import numpy as np

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.ligand import create_ligand
from trellis.phylogeny import generate_phylogeny
from trellis.sswm import generate_start_sequence


def main():
    mj = load_mj_matrix()
    ligand = create_ligand("KEMN", anchor=(0, -1))
    print("enumerating conformations (chain_length=12)...", flush=True)
    db = enumerate_conformations(12, ligand)
    print(f"  {db.n_conformations} conformations", flush=True)

    rng = np.random.default_rng(0)
    start_dna = generate_start_sequence(
        12, ligand, mj, min_fitness=0.0, temperature=1.0, rng=rng, db=db,
    )
    print(f"start sequence: {start_dna}\n", flush=True)

    betas = [0.2, 0.3, 0.4, 0.5]
    psis = [0.03, 0.05, 0.08]
    max_actives = [100, 150, 200, 300]
    n_replicates = 3

    header = f"{'beta':>6} {'psi':>6} {'max_a':>6} | {'tips (median)':>13} {'tips (range)':>16} {'nodes':>7} {'time':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for beta in betas:
        for psi in psis:
            for max_active in max_actives:
                tip_counts = []
                node_counts = []
                t0 = time.time()
                for seed in range(n_replicates):
                    phy = generate_phylogeny(
                        start_dna=start_dna,
                        ligand=ligand,
                        mj_matrix=mj,
                        db=db,
                        n_steps=100,
                        Ne=50.0,
                        beta=beta,
                        psi=psi,
                        temperature=1.0,
                        min_active_lineages=1,
                        max_active_lineages=max_active,
                        max_total_nodes=100000,
                        rng=np.random.default_rng(seed + 100),
                        fitness_cache=FitnessCache(),
                    )
                    tip_counts.append(phy.metadata["n_tips"])
                    node_counts.append(phy.metadata["n_nodes"])
                elapsed = time.time() - t0

                median_tips = int(np.median(tip_counts))
                median_nodes = int(np.median(node_counts))
                lo, hi = min(tip_counts), max(tip_counts)
                print(
                    f"{beta:>6.2f} {psi:>6.2f} {max_active:>6d} | "
                    f"{median_tips:>13d} {lo:>7d}-{hi:<7d} {median_nodes:>7d} {elapsed:>5.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
