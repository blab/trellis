"""Compare old and new fitness metrics across random sequences.

Old: fitness = -ensemble_binding_energy
New: fitness = fraction_folded * (-native_binding_energy)

Computes Spearman rank correlation to confirm the two metrics produce
equivalent sequence rankings for SSWM trajectory generation.
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

    mj = load_mj_matrix()
    lig = create_ligand("FWYL", anchor=(0, -1))
    rng = np.random.default_rng(seed)
    sequences = random_sequences(n_sequences, chain_length, rng)

    db = enumerate_conformations(chain_length, lig, min_contacts=0)
    print(f"Comparing fitness metrics: {chain_length}-mer, {n_sequences} sequences, "
          f"FWYL ligand")
    print(f"Database: {db.n_conformations:,} conformations\n")

    old_fitness = []
    new_fitness = []
    for seq in sequences:
        r = fold(seq, mj, lig, db=db, recover_conformation=False)
        old_fitness.append(-r.ensemble_binding_energy)
        new_fitness.append(r.fraction_folded * (-r.native_binding_energy))

    old_fitness = np.array(old_fitness)
    new_fitness = np.array(new_fitness)

    rho, pval = spearmanr(old_fitness, new_fitness)
    pearson = np.corrcoef(old_fitness, new_fitness)[0, 1]

    print(f"{'Metric':<30} {'Old (-⟨E_bind⟩)':>18} {'New (f×-BE_nat)':>18}")
    print("-" * 68)
    print(f"{'Mean':<30} {old_fitness.mean():>18.3f} {new_fitness.mean():>18.3f}")
    print(f"{'Std':<30} {old_fitness.std():>18.3f} {new_fitness.std():>18.3f}")
    print(f"{'Min':<30} {old_fitness.min():>18.3f} {new_fitness.min():>18.3f}")
    print(f"{'Max':<30} {old_fitness.max():>18.3f} {new_fitness.max():>18.3f}")
    print()
    print(f"Spearman rank correlation:  {rho:.6f}  (p = {pval:.2e})")
    print(f"Pearson correlation:        {pearson:.6f}")
    print()

    if rho > 0.95:
        print(f"PASS: Spearman rho = {rho:.4f} > 0.95")
    else:
        print(f"FAIL: Spearman rho = {rho:.4f} <= 0.95")

    # Stratified analysis by fraction_folded
    frac_folded = np.array([
        fold(seq, mj, lig, db=db, recover_conformation=False).fraction_folded
        for seq in sequences
    ])
    print(f"\n--- Stratified by fraction_folded ---")
    print(f"{'Threshold':<20} {'N seqs':>8} {'Spearman rho':>14}")
    print("-" * 44)
    for threshold in [0.0, 0.01, 0.05, 0.10, 0.20]:
        mask = frac_folded >= threshold
        n = mask.sum()
        if n > 10:
            rho_sub, _ = spearmanr(old_fitness[mask], new_fitness[mask])
            print(f"f >= {threshold:<15.2f} {n:>8} {rho_sub:>14.4f}")


if __name__ == "__main__":
    main()
