"""Single-worker pipeline benchmark for trajectory generation."""

import argparse
import subprocess
import time
from datetime import date

import numpy as np

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations, fold as fold_enum
from trellis.ligand import create_ligand
from trellis.sswm import generate_start_sequence, generate_trajectory


def generate_one(child_seed, chain_length, ligand, mj, n_steps, Ne, mu,
                 temperature, min_fitness):
    """Single trajectory — mirrors _generate_one in generate_trajectories.py."""
    rng = np.random.default_rng(child_seed)
    cache = FitnessCache()
    db = enumerate_conformations(chain_length, ligand)
    start_dna = generate_start_sequence(
        chain_length, ligand, mj,
        min_fitness=min_fitness, temperature=temperature, rng=rng, db=db,
    )
    generate_trajectory(
        start_dna, ligand, mj,
        n_steps=n_steps, Ne=Ne, mu=mu, temperature=temperature,
        rng=rng, fitness_cache=cache, db=db,
    )


def parse_anchor(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"must be 'x,y', got {value!r}")
    return (int(parts[0]), int(parts[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain-length", type=int, default=16)
    p.add_argument("--ligand-sequence", type=str, default="FWYL")
    p.add_argument("--ligand-anchor", type=parse_anchor, default=(0, -1))
    p.add_argument("--ligand-direction", type=str, default="horizontal",
                   choices=["horizontal", "vertical"])
    p.add_argument("--n-trajectories", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--Ne", type=float, default=1000.0)
    p.add_argument("--mu", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-fitness", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_commit_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    args = parse_args()
    mj = load_mj_matrix()
    ligand = create_ligand(
        args.ligand_sequence,
        anchor=args.ligand_anchor,
        direction=args.ligand_direction,
    )

    print(f"Pipeline benchmark: {args.chain_length}-mer + {args.ligand_sequence} ligand")
    print(f"  Seed: {args.seed}")
    print(f"  Trajectories: {args.n_trajectories} x {args.n_steps} steps")

    # Warm up numba JIT (excluded from timing)
    print("\nWarming up numba JIT ...", flush=True)
    warmup_db = enumerate_conformations(4, ligand)
    fold_enum("ACDE", mj, ligand, db=warmup_db)

    # Phase 1: standalone enumeration
    print("\nPhase 1 — enumeration")
    t0 = time.perf_counter()
    db = enumerate_conformations(args.chain_length, ligand)
    t_enum = time.perf_counter() - t0
    print(f"  Conformations: {db.n_conformations:,}")
    print(f"  Time: {t_enum:.1f}s")

    # Phase 2: full trajectory generation (mirrors _generate_one per call)
    print(f"\nPhase 2 — {args.n_trajectories} trajectories (full generate_one each)")
    ss = np.random.SeedSequence(args.seed)
    child_seeds = ss.spawn(args.n_trajectories)

    t0 = time.perf_counter()
    for i in range(args.n_trajectories):
        generate_one(
            child_seeds[i], args.chain_length, ligand, mj,
            args.n_steps, args.Ne, args.mu, args.temperature, args.min_fitness,
        )
        print(f"  {i + 1}/{args.n_trajectories} complete", flush=True)
    t_traj = time.perf_counter() - t0

    t_per_traj = t_traj / args.n_trajectories
    t_per_step = t_traj / (args.n_trajectories * args.n_steps)
    print(f"  Total: {t_traj:.1f}s")
    print(f"  Per trajectory: {t_per_traj:.1f}s")
    print(f"  Per step: {t_per_step:.2f}s")

    # Markdown row for BENCHMARK.md
    commit = get_commit_short()
    today = date.today().isoformat()
    print(f"\n| {today} | {commit} | {t_enum:.1f}s "
          f"| {t_traj:.1f}s | {t_per_step:.2f}s | |")


if __name__ == "__main__":
    main()
