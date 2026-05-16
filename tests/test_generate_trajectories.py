"""Integration test for generate_trajectories.py pipeline."""

import json
from unittest.mock import patch

from scripts.generate_trajectories import main


def test_generate_trajectories_end_to_end(tmp_path):
    """Run the full pipeline with minimal parameters."""
    args = [
        "--n-trajectories", "2",
        "--chain-length", "6",
        "--n-steps", "5",
        "--n-workers", "1",
        "--seed", "42",
        "--output-dir", str(tmp_path),
    ]
    with patch("sys.argv", ["generate_trajectories.py"] + args):
        main()

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["chain_length"] == 6
    assert meta["n_steps"] == 5
    assert meta["n_train"] + meta["n_test"] == 2

    train_shards = list(tmp_path.glob("*-train-*.tar.zst"))
    assert len(train_shards) >= 1


def test_generate_trajectories_random_ligands(tmp_path):
    """Run the pipeline with random ligands (multi-ligand mode)."""
    args = [
        "--n-random-ligands", "2",
        "--ligand-length", "3",
        "--n-trajectories", "2",
        "--chain-length", "6",
        "--n-steps", "3",
        "--n-workers", "1",
        "--seed", "7",
        "--output-dir", str(tmp_path),
    ]
    with patch("sys.argv", ["generate_trajectories.py"] + args):
        main()

    ligand_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(ligand_dirs) == 2

    for lig_dir in ligand_dirs:
        meta = json.loads((lig_dir / "metadata.json").read_text())
        assert meta["chain_length"] == 6
        assert meta["n_trajectories"] == 2
        assert len(meta["ligand_sequence"]) == 3
        train_shards = list(lig_dir.glob("*-train-*.tar.zst"))
        assert len(train_shards) >= 1
