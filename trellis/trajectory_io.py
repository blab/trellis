"""FASTA and tar.zst output for SSWM trajectories."""

import tarfile
from pathlib import Path

import numpy as np
import zstandard as zstd

from trellis.sswm import Trajectory


def _hamming_distance(a: str, b: str) -> int:
    """Count positions where strings *a* and *b* differ."""
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def write_trajectory_fasta(
    trajectory: Trajectory,
    output_path: str | Path,
    trajectory_id: str,
) -> None:
    """Write a single trajectory as a FASTA file.

    Each step becomes a record ``>NODE_XXXX|cumulative_hamming|branch_hamming``
    with the DNA sequence.  The final node is named ``TIP_{trajectory_id}``.
    """
    output_path = Path(output_path)
    seqs = trajectory.dna_sequences
    n = len(seqs)
    lines: list[str] = []
    cumulative = 0
    for i, dna in enumerate(seqs):
        if i > 0:
            branch = _hamming_distance(seqs[i - 1], dna)
            cumulative += branch
        else:
            branch = 0
        if i == n - 1:
            name = f"TIP_{trajectory_id}"
        else:
            name = f"NODE_{i:04d}"
        lines.append(f">{name}|{cumulative}|{branch}")
        lines.append(dna)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def package_shards(
    trajectory_dir: str | Path,
    output_dir: str | Path,
    split: str = "train",
    max_per_shard: int = 10000,
) -> list[Path]:
    """Package FASTA files into compressed tar.zst shards.

    Reads all ``.fasta`` files from *trajectory_dir*, groups them into
    shards of at most *max_per_shard* files, and writes each shard as
    ``forwards-{split}-{NNN}.tar.zst`` in *output_dir*.
    """
    trajectory_dir = Path(trajectory_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(trajectory_dir.glob("*.fasta"))
    if not fasta_files:
        return []

    shard_paths: list[Path] = []
    for shard_idx in range(0, len(fasta_files), max_per_shard):
        chunk = fasta_files[shard_idx : shard_idx + max_per_shard]
        shard_num = shard_idx // max_per_shard
        shard_name = f"forwards-{split}-{shard_num:03d}.tar.zst"
        shard_path = output_dir / shard_name

        cctx = zstd.ZstdCompressor()
        with open(shard_path, "wb") as fh:
            with cctx.stream_writer(fh) as zst_writer:
                with tarfile.open(mode="w|", fileobj=zst_writer) as tar:
                    for fasta_path in chunk:
                        tar.add(str(fasta_path), arcname=fasta_path.name)

        shard_paths.append(shard_path)
    return shard_paths


def train_test_split(
    trajectories: list[Trajectory],
    test_fraction: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Split trajectories into train and test sets.

    Holds out entire trajectories so the test set contains genuinely
    unseen evolutionary lineages.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(trajectories)
    n_test = max(1, round(n * test_fraction)) if n > 1 else 0
    indices = np.arange(n)
    rng.shuffle(indices)
    test_idx = set(indices[:n_test].tolist())
    train = [t for i, t in enumerate(trajectories) if i not in test_idx]
    test = [t for i, t in enumerate(trajectories) if i in test_idx]
    return train, test
