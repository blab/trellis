"""Validate mean-field pruning against exact enumeration.

Compares fitness values (fraction_folded × -native_binding_energy) between
exact scoring (min_contacts=0) and pruned scoring with Bloom-style
mean-field Z correction at several min_contacts thresholds.
"""

import numpy as np
from scipy.stats import spearmanr

from trellis.energy import AA_ALPHABET, load_mj_matrix
from trellis.fold_enum import enumerate_conformations, fold
from trellis.ligand import create_ligand


def random_sequences(n: int, length: int, rng: np.random.Generator) -> list[str]:
    aa_list = list(AA_ALPHABET)
    return ["".join(rng.choice(aa_list, size=length)) for _ in range(n)]


def main():
    chain_length = 16
    n_sequences = 1000
    seed = 42
    thresholds = [3, 4, 5, 6, 7, 8]

    mj = load_mj_matrix()
    rng = np.random.default_rng(seed)
    lig_seq = "".join(rng.choice(list(AA_ALPHABET), size=4))
    lig = create_ligand(lig_seq, anchor=(0, -1))
    sequences = random_sequences(n_sequences, chain_length, rng)

    print(f"Chain length: {chain_length}")
    print(f"Ligand: {lig_seq}")
    print(f"Sequences: {n_sequences}")
    print()

    db_exact = enumerate_conformations(chain_length, lig)
    n_total = db_exact.n_conformations
    print(f"Total conformations: {n_total:,}")
    print()

    exact_fitness = []
    exact_frac = []
    for seq in sequences:
        r = fold(seq, mj, lig, db=db_exact, recover_conformation=False)
        exact_fitness.append(r.fraction_folded * (-r.native_binding_energy))
        exact_frac.append(r.fraction_folded)
    exact_fitness = np.array(exact_fitness)
    exact_frac = np.array(exact_frac)

    print(f"{'threshold':<12} {'stored':>10} {'pruned':>10} {'%stored':>8} "
          f"{'RMSE_fit':>10} {'RMSE_f(T)':>10} {'MaxErr_fit':>11} {'Spearman':>10}")
    print("-" * 94)

    for mc in thresholds:
        db_pruned = enumerate_conformations(chain_length, lig, min_contacts=mc)
        n_stored = db_pruned.n_conformations
        n_pruned = db_pruned.pruned_counts.sum()

        pruned_fitness = []
        pruned_frac = []
        for seq in sequences:
            r = fold(seq, mj, lig, db=db_pruned, recover_conformation=False)
            pruned_fitness.append(r.fraction_folded * (-r.native_binding_energy))
            pruned_frac.append(r.fraction_folded)
        pruned_fitness = np.array(pruned_fitness)
        pruned_frac = np.array(pruned_frac)

        rmse_fit = np.sqrt(np.mean((exact_fitness - pruned_fitness) ** 2))
        rmse_frac = np.sqrt(np.mean((exact_frac - pruned_frac) ** 2))
        max_err = np.max(np.abs(exact_fitness - pruned_fitness))
        rho, _ = spearmanr(exact_fitness, pruned_fitness)

        print(f"{mc:<12} {n_stored:>10,} {n_pruned:>10,} {100*n_stored/(n_stored+n_pruned):>7.1f}% "
              f"{rmse_fit:>10.6f} {rmse_frac:>10.6f} {max_err:>11.6f} {rho:>10.6f}")

    print()
    print("Pruned counts by intra-contact number (last threshold):")
    for n, count in enumerate(db_pruned.pruned_counts):
        if count > 0:
            print(f"  n={n}: {count:,}")


if __name__ == "__main__":
    main()
