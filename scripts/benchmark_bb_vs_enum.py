"""Head-to-head benchmark: branch-and-bound vs exhaustive pre-enumeration."""

import argparse
import time

import numpy as np

from trellis.energy import AA_ALPHABET, load_mj_matrix
from trellis.fold_bb import fold as fold_bb
from trellis.fold_enum import enumerate_conformations, fold as fold_enum
from trellis.ligand import create_ligand


def parse_anchor(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"must be 'x,y', got {value!r}")
    return (int(parts[0]), int(parts[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain-length", type=int, default=10)
    p.add_argument("--n-sequences", type=int, default=20)
    p.add_argument("--ligand-sequence", type=str, default=None)
    p.add_argument("--ligand-anchor", type=parse_anchor, default=(0, -1))
    p.add_argument("--ligand-direction", type=str, default="horizontal",
                   choices=["horizontal", "vertical"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true",
                   help="run across chain lengths 6, 8, 10, 12, 14")
    return p.parse_args()


def random_sequences(n_seq: int, chain_length: int, rng) -> list[str]:
    aa = list(AA_ALPHABET)
    return ["".join(rng.choice(aa, size=chain_length)) for _ in range(n_seq)]


def benchmark_one(chain_length, n_sequences, ligand, mj, rng, warmup_done=False):
    seqs = random_sequences(n_sequences, chain_length, rng)

    # Pre-enumeration: enumerate
    t0 = time.perf_counter()
    db = enumerate_conformations(chain_length, ligand)
    t_enum = time.perf_counter() - t0

    # Warm up numba JIT on first call (excluded from timing)
    if not warmup_done:
        fold_enum(seqs[0], mj, ligand, db=db)

    # Branch-and-bound: batch fold
    t0 = time.perf_counter()
    bb_results = [fold_bb(s, mj, ligand) for s in seqs]
    t_bb = time.perf_counter() - t0

    # Pre-enumeration + numba: batch fold
    t0 = time.perf_counter()
    en_results = [fold_enum(s, mj, ligand, db=db, recover_conformation=False) for s in seqs]
    t_score = time.perf_counter() - t0

    # Correctness check
    n_match = 0
    for r_bb, r_en in zip(bb_results, en_results):
        if abs(r_bb.native_energy - r_en.native_energy) < 1e-8:
            n_match += 1

    return {
        "chain_length": chain_length,
        "n_sequences": n_sequences,
        "n_conformations": db.n_conformations,
        "t_bb_batch": t_bb,
        "t_enum": t_enum,
        "t_score_batch": t_score,
        "t_enum_total": t_enum + t_score,
        "t_bb_per_seq": t_bb / n_sequences,
        "t_score_per_seq": t_score / n_sequences,
        "n_match": n_match,
    }


def print_result(r):
    print(f"\nChain length: {r['chain_length']}, Sequences: {r['n_sequences']}, "
          f"Conformations: {r['n_conformations']:,}")
    print(f"{'':30s} {'Branch-and-bound':>18s} {'Enum + numba':>18s}")
    print(f"{'':30s} {'─' * 18:>18s} {'─' * 18:>18s}")
    print(f"{'Enumeration time':30s} {'N/A':>18s} {r['t_enum']:>17.3f}s")
    print(f"{'Per-sequence fold':30s} {r['t_bb_per_seq']:>17.3f}s {r['t_score_per_seq']:>17.3f}s")
    print(f"{'Batch fold (all sequences)':30s} {r['t_bb_batch']:>17.3f}s {r['t_score_batch']:>17.3f}s")
    print(f"{'Total (enum + scoring)':30s} {r['t_bb_batch']:>17.3f}s {r['t_enum_total']:>17.3f}s")
    if r["t_bb_batch"] > 0:
        speedup_score = r["t_bb_batch"] / r["t_score_batch"] if r["t_score_batch"] > 0 else float("inf")
        speedup_total = r["t_bb_batch"] / r["t_enum_total"] if r["t_enum_total"] > 0 else float("inf")
        print(f"{'Speedup (scoring only)':30s} {'1.0×':>18s} {speedup_score:>17.1f}×")
        print(f"{'Speedup (incl. enumeration)':30s} {'1.0×':>18s} {speedup_total:>17.1f}×")
    print(f"\nCorrectness: {r['n_match']}/{r['n_sequences']} sequences match")


def print_sweep(results):
    print(f"\n{'n':>4s} {'conformations':>14s} {'B&B batch':>12s} "
          f"{'enum':>10s} {'score batch':>12s} {'total':>12s} "
          f"{'speedup':>10s} {'correct':>8s}")
    print("─" * 86)
    for r in results:
        speedup = r["t_bb_batch"] / r["t_enum_total"] if r["t_enum_total"] > 0 else 0
        print(f"{r['chain_length']:>4d} {r['n_conformations']:>14,} "
              f"{r['t_bb_batch']:>11.3f}s {r['t_enum']:>9.3f}s "
              f"{r['t_score_batch']:>11.3f}s {r['t_enum_total']:>11.3f}s "
              f"{speedup:>9.1f}× {r['n_match']}/{r['n_sequences']}")


def main():
    args = parse_args()
    mj = load_mj_matrix()
    ligand = None
    if args.ligand_sequence:
        ligand = create_ligand(
            args.ligand_sequence,
            anchor=args.ligand_anchor,
            direction=args.ligand_direction,
        )

    lig_str = args.ligand_sequence or "none"
    print(f"Ligand: {lig_str}")

    # Warm up numba JIT before any timed runs
    print("warming up numba JIT ...", flush=True)
    warmup_db = enumerate_conformations(4, ligand)
    fold_enum("ACDE", mj, ligand, db=warmup_db)

    if args.sweep:
        rng = np.random.default_rng(args.seed)
        results = []
        for n in [6, 8, 10, 12, 14]:
            print(f"  benchmarking {n}-mer ...", flush=True)
            r = benchmark_one(n, args.n_sequences, ligand, mj, rng, warmup_done=True)
            results.append(r)
        print_sweep(results)
    else:
        rng = np.random.default_rng(args.seed)
        r = benchmark_one(args.chain_length, args.n_sequences, ligand, mj, rng, warmup_done=True)
        print_result(r)


if __name__ == "__main__":
    main()
