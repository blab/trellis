"""Generate an SSWM trajectory and write a JSON file for the D3 dashboard.

Runs a short SSWM trajectory, re-folds each unique amino-acid sequence
encountered to capture its native conformation and binding contacts, and
writes everything to a single JSON file consumed by
``viz/trajectory_dashboard.html``.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from trellis.cache import FitnessCache
from trellis.energy import AA_INDEX, load_mj_matrix
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
    p.add_argument("--ligand-sequence", type=str, default="FWYL")
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


def conformations_from_cache(
    aa_sequences: list[str],
    cache: FitnessCache,
    ligand: Ligand,
    mj: np.ndarray,
) -> dict[str, dict]:
    """Extract conformation data for unique AA sequences from the cache."""
    results: dict[str, dict] = {}
    for aa in set(aa_sequences):
        cached = cache.get(aa)
        if cached is None or cached.fold_result is None:
            continue
        fold_result = cached.fold_result
        conf = fold_result.native_conformation
        i_contacts = get_contacts(conf)
        b_contacts = binding_contacts(conf, ligand)
        results[aa] = {
            "conformation": [list(pos) for pos in conf],
            "native_energy": float(fold_result.native_energy),
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
    ligand = create_ligand(
        args.ligand_sequence,
        anchor=args.ligand_anchor,
        direction=args.ligand_direction,
    )
    rng = np.random.default_rng(args.seed)

    start_dna = generate_start_sequence(
        args.n_codons, ligand, mj,
        min_fitness=args.min_fitness,
        temperature=args.temperature,
        rng=rng,
    )
    cache = FitnessCache()
    trajectory = generate_trajectory(
        start_dna, ligand, mj,
        n_steps=args.n_steps,
        Ne=args.Ne,
        temperature=args.temperature,
        rng=rng,
        fitness_cache=cache,
    )

    fold_results = conformations_from_cache(
        trajectory.aa_sequences, cache, ligand, mj,
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
