"""Generate an SSWM trajectory and write a JSON file for the D3 dashboard.

Runs a short SSWM trajectory, re-folds each unique amino-acid sequence
encountered to capture its native conformation and binding contacts, and
writes everything to a single JSON file consumed by
``viz/index.html``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from trellis.energy import AA_ALPHABET, AA_INDEX, load_mj_matrix
from trellis.fold_enum import enumerate_conformations, fold as fold_enum
from trellis.lattice import get_contacts
from trellis.ligand import Ligand, binding_contacts, create_ligand
from trellis.sswm import generate_start_sequence, generate_trajectory


def parse_anchor(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"ligand-anchor must be 'x,y', got {value!r}"
        )
    return (int(parts[0]), int(parts[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-codons", type=int, default=10)
    p.add_argument("--ligand-sequence", type=str, default=None)
    p.add_argument("--ligand-anchor", type=parse_anchor, default=(0, -1))
    p.add_argument("--ligand-direction", type=str, default="horizontal",
                   choices=["horizontal", "vertical"])
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--Ne", type=float, default=100.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-fitness", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="viz/viz_trajectory_data.json")
    return p.parse_args()


def fold_unique_sequences(
    aa_sequences: list[str],
    ligand: Ligand,
    mj: np.ndarray,
    db,
    temperature: float,
) -> dict[str, dict]:
    """Fold unique AA sequences with conformation recovery for visualization."""
    results: dict[str, dict] = {}
    for aa in set(aa_sequences):
        fold_result = fold_enum(
            aa, mj, ligand, temperature, db=db, recover_conformation=True,
        )
        conf = fold_result.native_conformation
        i_contacts = get_contacts(conf)
        b_contacts = binding_contacts(conf, ligand)
        results[aa] = {
            "conformation": [list(pos) for pos in conf],
            "native_energy": float(fold_result.native_energy),
            "native_binding_energy": float(fold_result.native_binding_energy),
            "fraction_folded": float(fold_result.fraction_folded),
            "ensemble_binding_energy": float(fold_result.ensemble_binding_energy),
            "intra_contacts": [[i, j] for i, j in i_contacts],
            "intra_contact_energies": [
                float(mj[AA_INDEX[aa[i]], AA_INDEX[aa[j]]]) for i, j in i_contacts
            ],
            "binding_contacts": [[i, k] for i, k in b_contacts],
            "binding_contact_energies": [
                float(mj[AA_INDEX[aa[i]], AA_INDEX[ligand.sequence[k]]])
                for i, k in b_contacts
            ],
        }
    return results


def main() -> None:
    args = parse_args()
    mj = load_mj_matrix()
    rng = np.random.default_rng(args.seed)
    if args.ligand_sequence is None:
        args.ligand_sequence = "".join(rng.choice(list(AA_ALPHABET), size=4))
    print(f"ligand sequence: {args.ligand_sequence}")
    ligand = create_ligand(
        args.ligand_sequence,
        anchor=args.ligand_anchor,
        direction=args.ligand_direction,
    )

    db = enumerate_conformations(args.n_codons, ligand)
    print(f"enumerated {db.n_conformations:,} conformations")

    start_dna = generate_start_sequence(
        args.n_codons, ligand, mj,
        min_fitness=args.min_fitness,
        temperature=args.temperature,
        rng=rng,
        db=db,
    )
    trajectory = generate_trajectory(
        start_dna, ligand, mj,
        n_steps=args.n_steps,
        Ne=args.Ne,
        temperature=args.temperature,
        rng=rng,
        db=db,
    )

    fold_results = fold_unique_sequences(
        trajectory.aa_sequences, ligand, mj, db, args.temperature,
    )

    steps = []
    for i in range(len(trajectory.dna_sequences)):
        step = {
            "step": i,
            "dna_sequence": trajectory.dna_sequences[i],
            "aa_sequence": trajectory.aa_sequences[i],
            "fitness": float(trajectory.fitness_values[i]),
            "mutation_type": trajectory.mutation_types[i - 1] if i > 0 else None,
        }
        steps.append(step)

    data = {
        "metadata": {
            "n_codons": args.n_codons,
            "n_steps": trajectory.metadata["n_steps_completed"],
            "Ne": args.Ne,
            "temperature": args.temperature,
            "seed": args.seed,
            "ligand_sequence": args.ligand_sequence,
        },
        "ligand": {
            "sequence": ligand.sequence,
            "positions": [list(pos) for pos in ligand.positions],
        },
        "steps": steps,
        "conformations": fold_results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"wrote {args.output}")
    print(f"  steps: {len(steps)}")
    print(f"  unique AA sequences: {len(fold_results)}")
    print(f"  fitness: {steps[0]['fitness']:.3f} -> {steps[-1]['fitness']:.3f}")


if __name__ == "__main__":
    main()
