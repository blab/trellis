"""Print the SSWM fixation-probability distribution for a reference DNA sequence.

Two modes:

  Full distribution (default):
      python scripts/pfix_distribution.py \\
          --reference GCTTGT... --ligand KEMN --Ne 50

  Single target:
      python scripts/pfix_distribution.py \\
          --reference GCTTGT... --target GCTTGA... --ligand KEMN --Ne 50
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


def _aa_change(ref_dna: str, mut_dna: str) -> str:
    ref_aa = translate(ref_dna)
    mut_aa = translate(mut_dna)
    pos = next(i for i in range(len(ref_dna)) if ref_dna[i] != mut_dna[i])
    aa_pos = pos // 3
    if ref_aa[aa_pos] == mut_aa[aa_pos]:
        return "syn"
    return f"{ref_aa[aa_pos]}{aa_pos + 1}{mut_aa[aa_pos]}"


def _nuc_change(ref_dna: str, mut_dna: str) -> str:
    pos = next(i for i in range(len(ref_dna)) if ref_dna[i] != mut_dna[i])
    return f"{ref_dna[pos]}{pos + 1}{mut_dna[pos]}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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

    ref_aa = translate(args.reference)
    ref_result = compute_fitness_aa(ref_aa, ligand, mj, args.temperature, db=db)
    cache.put(ref_aa, ref_result)

    if args.target:
        p = pfix_for_target(
            args.reference, args.target, ligand, mj,
            args.Ne, args.temperature, cache, db,
        )
        if p is None:
            print("error: target is not a single-nucleotide neighbor of reference")
            return
        mut_type = classify_mutation(args.reference, args.target)
        if args.json:
            print(json.dumps({
                "reference": args.reference,
                "target": args.target,
                "pfix": p,
                "mutation_type": mut_type,
                "reference_fitness": ref_result.fitness,
            }, indent=2))
        else:
            print(f"reference fitness: {ref_result.fitness:.4f}")
            print(f"target pfix:       {p:.6f}")
            print(f"mutation type:     {mut_type}")
        return

    mutant_dnas, probs = compute_sswm_probabilities(
        args.reference, ref_result.fitness, ligand, mj,
        args.Ne, args.temperature, cache, db,
    )
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
            "reference_fitness": ref_result.fitness,
            "n_mutations": len(mutant_dnas),
            "n_nonzero": n_nonzero,
            "n_above_1pct": n_above_1pct,
            "distribution": entries,
        }, indent=2))
        return

    print(f"reference fitness: {ref_result.fitness:.4f}")
    print(f"mutations: {len(mutant_dnas)} total, "
          f"{n_nonzero} nonzero, {n_above_1pct} above 1%")
    print()
    print(f"{'pfix':>10}  {'type':<14}  {'aa_change':<10}  nuc_change")
    print("-" * 60)
    for i in order[:args.top]:
        if probs[i] == 0:
            break
        mut_type = classify_mutation(args.reference, mutant_dnas[i])
        aa = _aa_change(args.reference, mutant_dnas[i])
        nuc = _nuc_change(args.reference, mutant_dnas[i])
        print(f"{probs[i]:10.6f}  {mut_type:<14}  {aa:<10}  {nuc}")


if __name__ == "__main__":
    main()
