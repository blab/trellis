"""Bulk-generate SSWM trajectories and package as compressed FASTA shards."""

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.ligand import create_ligand
from trellis.sswm import generate_start_sequence, generate_trajectory
from trellis.trajectory_io import (
    package_shards,
    train_test_split,
    write_trajectory_fasta,
)


def _generate_one(kwargs: dict):
    """Worker function for a single trajectory (must be module-level for pickling)."""
    mj = load_mj_matrix()
    ligand = create_ligand(
        kwargs["ligand_sequence"],
        anchor=tuple(kwargs["ligand_anchor"]),
        direction=kwargs["ligand_direction"],
    )
    rng = np.random.default_rng(kwargs["child_seed"])
    cache = FitnessCache()
    db = enumerate_conformations(kwargs["chain_length"], ligand)

    start_dna = generate_start_sequence(
        kwargs["chain_length"],
        ligand,
        mj,
        min_fitness=kwargs["min_fitness"],
        temperature=kwargs["temperature"],
        rng=rng,
        db=db,
    )
    trajectory = generate_trajectory(
        start_dna,
        ligand,
        mj,
        n_steps=kwargs["n_steps"],
        Ne=kwargs["Ne"],
        mu=kwargs["mu"],
        temperature=kwargs["temperature"],
        rng=rng,
        fitness_cache=cache,
        db=db,
    )
    return trajectory


def parse_anchor(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"ligand-anchor must be 'x,y', got {value!r}"
        )
    return (int(parts[0]), int(parts[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trajectories", type=int, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--chain-length", type=int, default=20)
    p.add_argument("--ligand-sequence", type=str, default="FWYL")
    p.add_argument("--ligand-anchor", type=parse_anchor, default=(0, -1))
    p.add_argument("--ligand-direction", type=str, default="horizontal",
                   choices=["horizontal", "vertical"])
    p.add_argument("--Ne", type=float, default=1000.0)
    p.add_argument("--mu", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-fitness", type=float, default=0.0)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--max-per-shard", type=int, default=10000)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    n = args.n_trajectories
    n_workers = args.n_workers or os.cpu_count() or 1

    if args.output_dir is None:
        subdir = f"trellis-{args.chain_length}aa-{args.ligand_sequence}"
        output_dir = Path("results") / subdir
    else:
        output_dir = Path(args.output_dir)

    ss = np.random.SeedSequence(args.seed)
    child_seeds = ss.spawn(n)

    work_items = [
        {
            "child_seed": child_seeds[i],
            "chain_length": args.chain_length,
            "ligand_sequence": args.ligand_sequence,
            "ligand_anchor": list(args.ligand_anchor),
            "ligand_direction": args.ligand_direction,
            "n_steps": args.n_steps,
            "Ne": args.Ne,
            "mu": args.mu,
            "temperature": args.temperature,
            "min_fitness": args.min_fitness,
        }
        for i in range(n)
    ]

    print(f"generating {n} trajectories with {n_workers} workers ...")
    trajectories = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_generate_one, w): i for i, w in enumerate(work_items)
        }
        for future in as_completed(futures):
            idx = futures[future]
            traj = future.result()
            trajectories.append((idx, traj))
            done = len(trajectories)
            if done % max(1, n // 10) == 0 or done == n:
                print(f"  {done}/{n} complete")

    trajectories.sort(key=lambda x: x[0])
    trajectories = [t for _, t in trajectories]

    split_rng = np.random.default_rng(ss.spawn(1)[0])
    train, test = train_test_split(
        trajectories, test_fraction=args.test_fraction, rng=split_rng,
    )
    print(f"train: {len(train)}, test: {len(test)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "train"
        test_dir = Path(tmpdir) / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        for i, traj in enumerate(train):
            write_trajectory_fasta(traj, train_dir / f"traj_{i:06d}.fasta", str(i))
        for i, traj in enumerate(test):
            write_trajectory_fasta(traj, test_dir / f"traj_{i:06d}.fasta", str(i))

        train_shards = package_shards(
            train_dir, output_dir, split="train", max_per_shard=args.max_per_shard,
        )
        test_shards = package_shards(
            test_dir, output_dir, split="test", max_per_shard=args.max_per_shard,
        )

    elapsed = time.perf_counter() - t0

    metadata = {
        "chain_length": args.chain_length,
        "dna_length": args.chain_length * 3,
        "ligand_sequence": args.ligand_sequence,
        "ligand_anchor": list(args.ligand_anchor),
        "ligand_direction": args.ligand_direction,
        "Ne": args.Ne,
        "mu": args.mu,
        "temperature": args.temperature,
        "n_trajectories": n,
        "n_steps": args.n_steps,
        "seed": args.seed,
        "min_fitness": args.min_fitness,
        "test_fraction": args.test_fraction,
        "n_train": len(train),
        "n_test": len(test),
        "wall_seconds": round(elapsed, 1),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"\nwrote {len(train_shards)} train shard(s), {len(test_shards)} test shard(s)")
    print(f"metadata: {meta_path}")
    print(f"total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
