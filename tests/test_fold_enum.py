"""Tests for fold_enum — exhaustive pre-enumeration folding."""

import tempfile

import numpy as np
import pytest

from trellis.energy import load_mj_matrix
from trellis.fold_bb import fold as fold_bb
from trellis.fold_enum import (
    ConformationDatabase,
    enumerate_conformations,
    fold,
    load_database,
    save_database,
)
from trellis.lattice import enumerate_saws
from trellis.ligand import create_ligand


# ---------------------------------------------------------------------------
# ConformationDatabase structure
# ---------------------------------------------------------------------------

def test_db_counts_no_ligand():
    for n in [4, 6, 8]:
        db = enumerate_conformations(n)
        expected = sum(1 for _ in enumerate_saws(n, reduce_symmetry=True))
        assert db.n_conformations == expected
        assert db.reduced_symmetry is True


def test_db_counts_with_ligand():
    lig = create_ligand("FW", anchor=(0, -1))
    for n in [4, 6, 8]:
        db = enumerate_conformations(n, lig)
        expected = sum(
            1 for c in enumerate_saws(n, reduce_symmetry=False)
            if not (lig.sites & set(c))
        )
        assert db.n_conformations == expected
        assert db.reduced_symmetry is False


def test_db_offsets_shape():
    db = enumerate_conformations(6)
    assert db.contact_offsets.shape == (db.n_conformations + 1,)
    assert db.binding_offsets.shape == (db.n_conformations + 1,)
    assert db.contact_offsets[0] == 0
    assert db.contact_offsets[-1] == len(db.contact_pairs)


def test_db_contact_pairs_shape():
    db = enumerate_conformations(8)
    assert db.contact_pairs.ndim == 2
    assert db.contact_pairs.shape[1] == 2


def test_db_ligand_stored():
    lig = create_ligand("FWYL", anchor=(0, -1))
    db = enumerate_conformations(6, lig)
    assert db.ligand is lig


# ---------------------------------------------------------------------------
# Cross-validation: fold_enum vs fold_bb.py (with ligand)
# ---------------------------------------------------------------------------

@pytest.fixture
def mj():
    return load_mj_matrix()


@pytest.fixture
def ligand():
    return create_ligand("FW", anchor=(0, -1))


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_fold_matches_bb_with_ligand(n, mj, ligand):
    seq = "ACDEFGHIKLMNPQRSTVWY"[:n]
    db = enumerate_conformations(n, ligand)
    r_bb = fold_bb(seq, mj, ligand)
    r_en = fold(seq, mj, ligand, db=db)
    assert r_en.native_energy == pytest.approx(r_bb.native_energy)
    assert r_en.partition_function == pytest.approx(r_bb.partition_function, rel=1e-10)
    assert r_en.ensemble_binding_energy == pytest.approx(r_bb.ensemble_binding_energy, rel=1e-10)
    assert r_en.n_conformations_enumerated == r_bb.n_conformations_enumerated


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_fold_matches_bb_no_ligand(n, mj):
    seq = "ACDEFGHIKLMNPQRSTVWY"[:n]
    db = enumerate_conformations(n)
    r_bb = fold_bb(seq, mj)
    r_en = fold(seq, mj, db=db)
    assert r_en.native_energy == pytest.approx(r_bb.native_energy)
    # B&B prunes subtrees with negligible Boltzmann weight, so Z can differ
    # slightly from exhaustive enumeration. rel=1e-5 accommodates this.
    assert r_en.partition_function == pytest.approx(r_bb.partition_function, rel=1e-5)


def test_fold_different_sequences_same_db(mj, ligand):
    db = enumerate_conformations(6, ligand)
    for seq in ["ACDEFG", "FWYLMK", "GGGGGG"]:
        r_bb = fold_bb(seq, mj, ligand)
        r_en = fold(seq, mj, ligand, db=db)
        assert r_en.native_energy == pytest.approx(r_bb.native_energy)
        assert r_en.ensemble_binding_energy == pytest.approx(r_bb.ensemble_binding_energy, rel=1e-10)


# ---------------------------------------------------------------------------
# Native conformation correctness
# ---------------------------------------------------------------------------

def test_native_conformation_energy(mj, ligand):
    from trellis.energy import conformation_energy
    from trellis.ligand import binding_energy

    seq = "ACDEFGHIKL"
    db = enumerate_conformations(10, ligand)
    r = fold(seq, mj, ligand, db=db)
    e_intra = conformation_energy(seq, r.native_conformation, mj)
    e_bind = binding_energy(seq, r.native_conformation, ligand, mj)
    assert e_intra + e_bind == pytest.approx(r.native_energy)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fold_single_residue(mj):
    db = enumerate_conformations(1)
    r = fold("A", mj, db=db)
    assert r.native_conformation == ((0, 0),)
    assert r.native_energy == 0.0
    assert r.n_conformations_enumerated == 1


def test_fold_two_residues(mj):
    db = enumerate_conformations(2)
    r_bb = fold_bb("AC", mj)
    r_en = fold("AC", mj, db=db)
    assert r_en.native_energy == pytest.approx(r_bb.native_energy)
    assert r_en.partition_function == pytest.approx(r_bb.partition_function)


def test_fold_without_db_warns(mj):
    with pytest.warns(UserWarning, match="ConformationDatabase"):
        fold("ACDEFG", mj)


def test_fold_wrong_chain_length(mj):
    db = enumerate_conformations(6)
    with pytest.raises(ValueError, match="chain_length"):
        fold("ACDE", mj, db=db)


def test_fold_empty_sequence(mj):
    with pytest.raises(ValueError, match="at least 1"):
        fold("", mj)


def test_fold_zero_temperature(mj):
    db = enumerate_conformations(6)
    with pytest.raises(ValueError, match="temperature"):
        fold("ACDEFG", mj, temperature=0.0, db=db)


# ---------------------------------------------------------------------------
# Temperature variation
# ---------------------------------------------------------------------------

def test_temperature_matches_bb(mj, ligand):
    seq = "ACDEFGHIKL"
    db = enumerate_conformations(10, ligand)
    for temp in [0.5, 1.0, 2.0]:
        r_bb = fold_bb(seq, mj, ligand, temperature=temp)
        r_en = fold(seq, mj, ligand, temperature=temp, db=db)
        assert r_en.native_energy == pytest.approx(r_bb.native_energy)
        assert r_en.ensemble_binding_energy == pytest.approx(r_bb.ensemble_binding_energy, rel=1e-9)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_no_ligand(mj):
    db = enumerate_conformations(8)
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        save_database(db, f.name)
        db2 = load_database(f.name)
    assert db2.chain_length == db.chain_length
    assert db2.n_conformations == db.n_conformations
    assert db2.reduced_symmetry == db.reduced_symmetry
    assert db2.ligand is None
    r1 = fold("ACDEFGHK", mj, db=db)
    r2 = fold("ACDEFGHK", mj, db=db2)
    assert r1.native_energy == pytest.approx(r2.native_energy)
    assert r1.partition_function == pytest.approx(r2.partition_function)


def test_save_load_with_ligand(mj):
    lig = create_ligand("FW", anchor=(0, -1))
    db = enumerate_conformations(8, lig)
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        save_database(db, f.name)
        db2 = load_database(f.name)
    assert db2.chain_length == db.chain_length
    assert db2.n_conformations == db.n_conformations
    assert db2.ligand is not None
    assert db2.ligand.sequence == lig.sequence
    assert db2.ligand.positions == lig.positions
    r1 = fold("ACDEFGHK", mj, lig, db=db)
    r2 = fold("ACDEFGHK", mj, db2.ligand, db=db2)
    assert r1.native_energy == pytest.approx(r2.native_energy)
    assert r1.ensemble_binding_energy == pytest.approx(r2.ensemble_binding_energy)
