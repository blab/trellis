import pytest

from trellis.energy import AA_INDEX, load_mj_matrix
from trellis.ligand import Ligand, binding_contacts, binding_energy, create_ligand


# ---------------------------------------------------------------------------
# Ligand creation
# ---------------------------------------------------------------------------

def test_create_ligand_horizontal():
    lig = create_ligand("FWYL")
    assert lig.sequence == "FWYL"
    assert lig.positions == ((0, -1), (1, -1), (2, -1), (3, -1))
    assert lig.sites == frozenset(lig.positions)


def test_create_ligand_horizontal_custom_anchor():
    lig = create_ligand("AC", anchor=(5, 3))
    assert lig.positions == ((5, 3), (6, 3))


def test_create_ligand_vertical():
    lig = create_ligand("FW", anchor=(-1, 0), direction="vertical")
    assert lig.positions == ((-1, 0), (-1, 1))
    assert lig.sites == frozenset(lig.positions)


def test_create_ligand_single_residue():
    lig = create_ligand("F", anchor=(0, -1))
    assert lig.positions == ((0, -1),)
    assert len(lig.sites) == 1


def test_create_ligand_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        create_ligand("FW", direction="diagonal")


def test_create_ligand_empty_sequence_raises():
    with pytest.raises(ValueError, match="non-empty"):
        create_ligand("")


def test_create_ligand_invalid_amino_acid_raises():
    with pytest.raises(ValueError, match="invalid amino acid"):
        create_ligand("FXW")


# ---------------------------------------------------------------------------
# Binding contacts
# ---------------------------------------------------------------------------

def test_binding_contacts_single():
    conf = ((0, 0), (1, 0), (2, 0))
    lig = create_ligand("F", anchor=(0, -1))
    assert binding_contacts(conf, lig) == [(0, 0)]


def test_binding_contacts_none_distant():
    conf = ((0, 0), (1, 0), (2, 0))
    lig = create_ligand("F", anchor=(0, -5))
    assert binding_contacts(conf, lig) == []


def test_binding_contacts_multiple():
    # U-shape going down and back:
    #   0 - 1
    #       |
    #   5   2
    #   |   |
    #   4 - 3
    # Ligand "FW" at (0,-1) and (1,-1), below residues 4 and 3.
    conf = ((0, 2), (1, 2), (1, 1), (1, 0), (0, 0), (0, 1))
    lig = create_ligand("FW", anchor=(0, -1))
    contacts = binding_contacts(conf, lig)
    # Residue 4 at (0,0) is adjacent to ligand 0 at (0,-1): distance 1
    # Residue 3 at (1,0) is adjacent to ligand 1 at (1,-1): distance 1
    assert contacts == [(3, 1), (4, 0)]


def test_binding_contacts_protein_adjacent_to_multiple_ligand_residues():
    conf = ((0, 0), (1, 0), (2, 0))
    lig = create_ligand("FW", anchor=(0, -1))
    # Residue 0 at (0,0) adjacent to ligand 0 at (0,-1): yes
    # Residue 1 at (1,0) adjacent to ligand 0 at (0,-1): distance 2, no
    # Residue 1 at (1,0) adjacent to ligand 1 at (1,-1): distance 1, yes
    assert binding_contacts(conf, lig) == [(0, 0), (1, 1)]


# ---------------------------------------------------------------------------
# Binding energy
# ---------------------------------------------------------------------------

def test_binding_energy_single_contact():
    mj = load_mj_matrix()
    conf = ((0, 0), (1, 0), (2, 0))
    lig = create_ligand("F", anchor=(0, -1))
    # Only contact: protein residue 0 ("A") with ligand residue 0 ("F")
    expected = mj[AA_INDEX["A"], AA_INDEX["F"]]
    assert binding_energy("ACD", conf, lig, mj) == pytest.approx(expected)


def test_binding_energy_zero_distant():
    mj = load_mj_matrix()
    conf = ((0, 0), (1, 0), (2, 0))
    lig = create_ligand("FWYL", anchor=(100, 100))
    assert binding_energy("ACD", conf, lig, mj) == 0.0


def test_binding_energy_multiple_contacts():
    mj = load_mj_matrix()
    conf = ((0, 0), (1, 0), (2, 0))
    lig = create_ligand("FW", anchor=(0, -1))
    # Contacts: (0, 0) and (1, 1)  => A-F + C-W
    expected = mj[AA_INDEX["A"], AA_INDEX["F"]] + mj[AA_INDEX["C"], AA_INDEX["W"]]
    assert binding_energy("ACD", conf, lig, mj) == pytest.approx(expected)


def test_binding_energy_length_mismatch_raises():
    mj = load_mj_matrix()
    conf = ((0, 0), (1, 0))
    lig = create_ligand("F", anchor=(0, -1))
    with pytest.raises(ValueError):
        binding_energy("ACD", conf, lig, mj)
