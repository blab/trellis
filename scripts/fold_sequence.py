"""Fold a single sequence and print the result."""

import argparse
import json
import sys
import time

from trellis.energy import load_mj_matrix
from trellis.fold_bb import fold
from trellis.genetic_code import translate
from trellis.ligand import create_ligand


def parse_anchor(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"ligand-anchor must be 'x,y', got {value!r}"
        )
    return (int(parts[0]), int(parts[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--aa", type=str, help="amino acid sequence (use - for stdin)")
    group.add_argument("--dna", type=str, help="DNA sequence (use - for stdin)")
    p.add_argument("--ligand-sequence", type=str, default=None)
    p.add_argument("--ligand-anchor", type=parse_anchor, default=(0, -1))
    p.add_argument("--ligand-direction", type=str, default="horizontal",
                   choices=["horizontal", "vertical"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--json", action="store_true", help="JSON output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.aa is not None:
        raw = args.aa if args.aa != "-" else sys.stdin.readline().strip()
        aa_sequence = raw.upper()
        dna_sequence = None
    else:
        raw = args.dna if args.dna != "-" else sys.stdin.readline().strip()
        dna_sequence = raw.upper()
        aa_sequence = translate(dna_sequence)

    mj = load_mj_matrix()
    ligand = None
    if args.ligand_sequence:
        ligand = create_ligand(
            args.ligand_sequence,
            anchor=args.ligand_anchor,
            direction=args.ligand_direction,
        )

    t0 = time.perf_counter()
    result = fold(aa_sequence, mj, ligand, args.temperature)
    elapsed = time.perf_counter() - t0

    if args.json:
        data = {
            "aa_sequence": aa_sequence,
            "native_energy": result.native_energy,
            "partition_function": result.partition_function,
            "n_conformations_enumerated": result.n_conformations_enumerated,
            "native_conformation": [list(pos) for pos in result.native_conformation],
            "temperature": args.temperature,
            "wall_seconds": round(elapsed, 3),
        }
        if dna_sequence:
            data["dna_sequence"] = dna_sequence
        if ligand:
            data["ensemble_binding_energy"] = result.ensemble_binding_energy
            data["fitness"] = -result.ensemble_binding_energy
            data["ligand_sequence"] = args.ligand_sequence
        print(json.dumps(data, indent=2))
    else:
        print(f"sequence:       {aa_sequence}")
        if dna_sequence:
            print(f"dna:            {dna_sequence}")
        print(f"native_energy:  {result.native_energy:.3f}")
        if ligand:
            print(f"binding_energy: {result.ensemble_binding_energy:.3f}")
            print(f"fitness:        {-result.ensemble_binding_energy:.3f}")
        print(f"Z:              {result.partition_function:.4e}")
        print(f"conformations:  {result.n_conformations_enumerated}")
        conf_str = " ".join(
            f"({x},{y})" for x, y in result.native_conformation
        )
        print(f"conformation:   {conf_str}")
        print(f"time:           {elapsed:.2f}s")


if __name__ == "__main__":
    main()
