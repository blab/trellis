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
from trellis.energy import AA_ALPHABET, load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.ligand import create_ligand
from trellis.sswm import generate_start_sequence, generate_trajectory
from trellis.trajectory_io import (
    package_shards,
    train_test_split,
    write_trajectory_fasta,
)


def _generate_batch(kwargs: dict) -> list[tuple[int, object]]:
    """Worker function for a batch of trajectories (enumerates once per worker)."""
    mj = load_mj_matrix()
    ligand = create_ligand(
        kwargs["ligand_sequence"],
        anchor=tuple(kwargs["ligand_anchor"]),
        direction=kwargs["ligand_direction"],
    )
    db = enumerate_conformations(kwargs["chain_length"], ligand)

    results = []
    for idx, child_seed in zip(kwargs["indices"], kwargs["child_seeds"]):
        rng = np.random.default_rng(child_seed)
        cache = FitnessCache()
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
        results.append((idx, trajectory))
    return results


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
    p.add_argument("--n-random-ligands", type=int, default=None,
                   help="Generate N random ligand sequences (overrides --ligand-sequence)")
    p.add_argument("--ligand-length", type=int, default=4,
                   help="Length of random ligand sequences (used with --n-random-ligands)")
    p.add_argument("--Ne", type=float, default=1000.0)
    p.add_argument("--mu", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min-fitness", type=float, default=0.0)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--max-per-shard", type=int, default=10000)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _generate_random_ligands(n: int, length: int, rng: np.random.Generator) -> list[str]:
    """Generate *n* unique random ligand sequences of given *length*."""
    ligands: list[str] = []
    seen: set[str] = set()
    while len(ligands) < n:
        seq = "".join(rng.choice(list(AA_ALPHABET), size=length))
        if seq not in seen:
            seen.add(seq)
            ligands.append(seq)
    return ligands


def _run_single_ligand(args, ss, n_workers) -> None:
    """Original mode: all workers share one ligand."""
    n = args.n_trajectories

    if args.output_dir is None:
        subdir = f"trellis-{args.chain_length}aa-{args.ligand_sequence}"
        output_dir = Path("results") / subdir
    else:
        output_dir = Path(args.output_dir)

    child_seeds = ss.spawn(n)

    indices = list(range(n))
    batch_size = (n + n_workers - 1) // n_workers
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append({
            "indices": indices[start:end],
            "child_seeds": child_seeds[start:end],
            "chain_length": args.chain_length,
            "ligand_sequence": args.ligand_sequence,
            "ligand_anchor": list(args.ligand_anchor),
            "ligand_direction": args.ligand_direction,
            "n_steps": args.n_steps,
            "Ne": args.Ne,
            "mu": args.mu,
            "temperature": args.temperature,
            "min_fitness": args.min_fitness,
        })

    print(f"generating {n} trajectories with {n_workers} workers ...")
    trajectories = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_generate_batch, b) for b in batches]
        for future in as_completed(futures):
            batch_results = future.result()
            trajectories.extend(batch_results)
            done = len(trajectories)
            if done % max(1, n // 10) == 0 or done == n:
                print(f"  {done}/{n} complete")

    trajectories.sort(key=lambda x: x[0])
    trajectories = [t for _, t in trajectories]

    _write_output(trajectories, output_dir, args, ss, ligand_sequence=args.ligand_sequence)


def _run_random_ligands(args, ss, n_workers) -> None:
    """Multi-ligand mode: one batch per random ligand."""
    n_ligands = args.n_random_ligands
    n = args.n_trajectories

    ligand_rng = np.random.default_rng(ss.spawn(1)[0])
    ligands = _generate_random_ligands(n_ligands, args.ligand_length, ligand_rng)

    base_output = Path(args.output_dir) if args.output_dir else Path("results")

    batches = []
    for i, lig_seq in enumerate(ligands):
        ligand_seeds = ss.spawn(n)
        batches.append({
            "indices": list(range(n)),
            "child_seeds": ligand_seeds,
            "chain_length": args.chain_length,
            "ligand_sequence": lig_seq,
            "ligand_anchor": list(args.ligand_anchor),
            "ligand_direction": args.ligand_direction,
            "n_steps": args.n_steps,
            "Ne": args.Ne,
            "mu": args.mu,
            "temperature": args.temperature,
            "min_fitness": args.min_fitness,
        })

    total = n_ligands * n
    print(f"generating {total} trajectories ({n_ligands} ligands × {n} each) "
          f"with {n_workers} workers ...")

    results_by_ligand: dict[str, list[tuple[int, object]]] = {lig: [] for lig in ligands}
    done_count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_ligand = {}
        for batch, lig_seq in zip(batches, ligands):
            fut = executor.submit(_generate_batch, batch)
            future_to_ligand[fut] = lig_seq
        for future in as_completed(future_to_ligand):
            lig_seq = future_to_ligand[future]
            batch_results = future.result()
            results_by_ligand[lig_seq].extend(batch_results)
            done_count += len(batch_results)
            print(f"  {done_count}/{total} complete (ligand {lig_seq} done)")

    for lig_seq in ligands:
        traj_list = results_by_ligand[lig_seq]
        traj_list.sort(key=lambda x: x[0])
        trajectories = [t for _, t in traj_list]

        output_dir = base_output / f"trellis-{args.chain_length}aa-{lig_seq}"
        _write_output(trajectories, output_dir, args, ss, ligand_sequence=lig_seq)

    print(f"\n{n_ligands} ligand directories written under {base_output}")


def _write_output(
    trajectories: list,
    output_dir: Path,
    args,
    ss: np.random.SeedSequence,
    ligand_sequence: str,
) -> None:
    """Split, shard, and write metadata for a set of trajectories."""
    split_rng = np.random.default_rng(ss.spawn(1)[0])
    train, test = train_test_split(
        trajectories, test_fraction=args.test_fraction, rng=split_rng,
    )

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

    metadata = {
        "chain_length": args.chain_length,
        "dna_length": args.chain_length * 3,
        "ligand_sequence": ligand_sequence,
        "ligand_anchor": list(args.ligand_anchor),
        "ligand_direction": args.ligand_direction,
        "Ne": args.Ne,
        "mu": args.mu,
        "temperature": args.temperature,
        "n_trajectories": len(trajectories),
        "n_steps": args.n_steps,
        "seed": args.seed,
        "min_fitness": args.min_fitness,
        "test_fraction": args.test_fraction,
        "n_train": len(train),
        "n_test": len(test),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"  {output_dir}: {len(train_shards)} train, {len(test_shards)} test shard(s)")


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    n_workers = args.n_workers or os.cpu_count() or 1
    ss = np.random.SeedSequence(args.seed)

    if args.n_random_ligands is not None:
        _run_random_ligands(args, ss, n_workers)
    else:
        _run_single_ligand(args, ss, n_workers)

    elapsed = time.perf_counter() - t0
    print(f"total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
