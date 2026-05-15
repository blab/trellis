"""Tests for trajectory_io — FASTA writing, tar.zst sharding, train/test split."""

import tarfile

import numpy as np
import pytest
import zstandard as zstd

from trellis.sswm import Trajectory
from trellis.trajectory_io import (
    _hamming_distance,
    package_shards,
    train_test_split,
    write_trajectory_fasta,
)

# Three-step trajectory with known Hamming distances:
#   step 0→1: pos 15 G→A  (branch=1, cumulative=1)
#   step 1→2: pos 9  G→A  (branch=1, cumulative=2)
_TRAJ_DNA = [
    "GCTTGTGATGAATTTGGT",
    "GCTTGTGATGAATTTAGT",
    "GCTTGTGATAAATTTAGT",
]


def _make_trajectory(dna_seqs=None):
    if dna_seqs is None:
        dna_seqs = list(_TRAJ_DNA)
    n = len(dna_seqs)
    return Trajectory(
        dna_sequences=dna_seqs,
        aa_sequences=["X" * (len(dna_seqs[0]) // 3)] * n,
        fitness_values=[0.0] * n,
        mutation_types=["nonsynonymous"] * (n - 1),
        metadata={},
    )


def _parse_fasta(text):
    """Parse FASTA text into list of (name, sequence) tuples."""
    records = []
    name = None
    for line in text.strip().split("\n"):
        if line.startswith(">"):
            name = line[1:]
        else:
            records.append((name, line))
    return records


# --- Hamming distance ---


def test_hamming_identical():
    assert _hamming_distance("ACGT", "ACGT") == 0


def test_hamming_one_diff():
    assert _hamming_distance("ACGT", "ACGA") == 1


def test_hamming_all_diff():
    assert _hamming_distance("AAAA", "TTTT") == 4


# --- write_trajectory_fasta ---


def test_fasta_round_trip(tmp_path):
    traj = _make_trajectory()
    path = tmp_path / "traj.fasta"
    write_trajectory_fasta(traj, path, "42")
    records = _parse_fasta(path.read_text())
    assert len(records) == 3
    for (_, seq), expected in zip(records, _TRAJ_DNA):
        assert seq == expected


def test_fasta_tip_name(tmp_path):
    traj = _make_trajectory()
    path = tmp_path / "traj.fasta"
    write_trajectory_fasta(traj, path, "42")
    records = _parse_fasta(path.read_text())
    last_name = records[-1][0]
    assert last_name.startswith("TIP_42|")
    for name, _ in records[:-1]:
        assert name.startswith("NODE_")


def test_fasta_hamming_distances(tmp_path):
    traj = _make_trajectory()
    path = tmp_path / "traj.fasta"
    write_trajectory_fasta(traj, path, "0")
    records = _parse_fasta(path.read_text())
    headers = [r[0] for r in records]
    # NODE_0000|0|0
    assert headers[0].endswith("|0|0")
    # NODE_0001|1|1
    assert "|1|1" in headers[1]
    # TIP_0|2|1
    assert "|2|1" in headers[2]


def test_fasta_single_sequence(tmp_path):
    traj = _make_trajectory(dna_seqs=["ACGTACGT"])
    path = tmp_path / "traj.fasta"
    write_trajectory_fasta(traj, path, "solo")
    records = _parse_fasta(path.read_text())
    assert len(records) == 1
    assert records[0][0] == "TIP_solo|0|0"
    assert records[0][1] == "ACGTACGT"


def test_fasta_creates_parent_dirs(tmp_path):
    traj = _make_trajectory(dna_seqs=["ACGT"])
    path = tmp_path / "a" / "b" / "traj.fasta"
    write_trajectory_fasta(traj, path, "0")
    assert path.exists()


# --- package_shards ---


def _read_shard(shard_path):
    """Decompress a tar.zst shard and return {filename: content}."""
    dctx = zstd.ZstdDecompressor()
    with open(shard_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(mode="r|", fileobj=reader) as tar:
                contents = {}
                for member in tar:
                    f = tar.extractfile(member)
                    if f is not None:
                        contents[member.name] = f.read().decode()
    return contents


def _write_n_fastas(tmp_path, n):
    """Write n dummy FASTA files and return the directory."""
    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    for i in range(n):
        traj = _make_trajectory(dna_seqs=[f"ACGT{i:04d}ACGT"])
        write_trajectory_fasta(traj, fasta_dir / f"traj_{i:04d}.fasta", str(i))
    return fasta_dir


def test_shards_round_trip(tmp_path):
    fasta_dir = _write_n_fastas(tmp_path, 3)
    out_dir = tmp_path / "shards"
    paths = package_shards(fasta_dir, out_dir)
    assert len(paths) == 1
    contents = _read_shard(paths[0])
    assert len(contents) == 3
    for fname in contents:
        assert fname.endswith(".fasta")


def test_shards_multiple(tmp_path):
    fasta_dir = _write_n_fastas(tmp_path, 5)
    out_dir = tmp_path / "shards"
    paths = package_shards(fasta_dir, out_dir, max_per_shard=2)
    assert len(paths) == 3
    sizes = [len(_read_shard(p)) for p in paths]
    assert sizes == [2, 2, 1]


def test_shards_naming(tmp_path):
    fasta_dir = _write_n_fastas(tmp_path, 1)
    out_dir = tmp_path / "shards"
    paths = package_shards(fasta_dir, out_dir, split="train")
    assert paths[0].name == "forwards-train-000.tar.zst"
    paths_test = package_shards(fasta_dir, out_dir, split="test")
    assert paths_test[0].name == "forwards-test-000.tar.zst"


def test_shards_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    out_dir = tmp_path / "shards"
    paths = package_shards(empty_dir, out_dir)
    assert paths == []


# --- train_test_split ---


def test_split_sizes():
    trajs = [_make_trajectory(dna_seqs=["ACGT"]) for _ in range(100)]
    train, test = train_test_split(trajs, test_fraction=0.1, rng=np.random.default_rng(0))
    assert len(train) == 90
    assert len(test) == 10


def test_split_no_overlap():
    trajs = [_make_trajectory(dna_seqs=[f"ACGT{i:04d}"]) for i in range(20)]
    train, test = train_test_split(trajs, test_fraction=0.2, rng=np.random.default_rng(0))
    train_seqs = {t.dna_sequences[0] for t in train}
    test_seqs = {t.dna_sequences[0] for t in test}
    assert len(train_seqs & test_seqs) == 0
    assert len(train) + len(test) == 20


def test_split_deterministic():
    trajs = [_make_trajectory(dna_seqs=[f"ACGT{i:04d}"]) for i in range(50)]
    train1, test1 = train_test_split(trajs, rng=np.random.default_rng(99))
    train2, test2 = train_test_split(trajs, rng=np.random.default_rng(99))
    assert [t.dna_sequences for t in test1] == [t.dna_sequences for t in test2]


def test_split_single():
    trajs = [_make_trajectory(dna_seqs=["ACGT"])]
    train, test = train_test_split(trajs, test_fraction=0.1, rng=np.random.default_rng(0))
    assert len(train) == 1
    assert len(test) == 0
