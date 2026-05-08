import numpy as np
import pytest

from trellis.energy import (
    AA_ALPHABET,
    AA_INDEX,
    conformation_energy,
    load_mj_matrix,
    max_contact_energy,
    partition_function,
)
from trellis.lattice import enumerate_saws


def test_mj_matrix_shape_and_symmetry():
    mj = load_mj_matrix()
    assert mj.shape == (20, 20)
    assert np.allclose(mj, mj.T)


def test_mj_matrix_min_at_phe_phe():
    mj = load_mj_matrix()
    f = AA_INDEX["F"]
    assert np.unravel_index(np.argmin(mj), mj.shape) == (f, f)
    assert mj[f, f] == pytest.approx(-6.85)


def test_mj_matrix_value_range():
    mj = load_mj_matrix()
    # Most attractive: F-F = -6.85; most repulsive: K-K = +0.13.
    assert mj.min() == pytest.approx(-6.85)
    assert mj.max() == pytest.approx(0.13)
    k = AA_INDEX["K"]
    assert mj[k, k] == pytest.approx(0.13)
    # Spot-checks against the source dict.
    assert mj[AA_INDEX["C"], AA_INDEX["C"]] == pytest.approx(-5.44)
    assert mj[AA_INDEX["A"], AA_INDEX["G"]] == pytest.approx(-2.15)
    assert mj[AA_INDEX["L"], AA_INDEX["I"]] == pytest.approx(-6.17)


def test_load_caches():
    assert load_mj_matrix() is load_mj_matrix()


def test_load_returns_immutable():
    mj = load_mj_matrix()
    with pytest.raises(ValueError):
        mj[0, 0] = 0.0


def test_conformation_energy_simple_square():
    mj = load_mj_matrix()
    # 2x2 square; only contact is (0, 3).
    conf = ((0, 0), (1, 0), (1, 1), (0, 1))
    energy = conformation_energy("AAAC", conf, mj)
    assert energy == pytest.approx(mj[AA_INDEX["A"], AA_INDEX["C"]])


def test_conformation_energy_no_contacts():
    mj = load_mj_matrix()
    conf = ((0, 0), (1, 0), (2, 0), (3, 0))
    assert conformation_energy("FWYL", conf, mj) == 0.0
    assert conformation_energy("AAAA", conf, mj) == 0.0


def test_conformation_energy_u_shape():
    mj = load_mj_matrix()
    # Same U-shape as test_lattice.test_get_contacts_multiple_in_lex_order;
    # contacts are (0, 5) and (2, 5).
    conf = ((0, 2), (1, 2), (1, 1), (1, 0), (0, 0), (0, 1))
    seq = "ACDEFG"
    expected = (
        mj[AA_INDEX[seq[0]], AA_INDEX[seq[5]]]
        + mj[AA_INDEX[seq[2]], AA_INDEX[seq[5]]]
    )
    assert conformation_energy(seq, conf, mj) == pytest.approx(expected)


def test_conformation_energy_length_mismatch_raises():
    mj = load_mj_matrix()
    conf = ((0, 0), (1, 0), (1, 1), (0, 1))
    with pytest.raises(ValueError):
        conformation_energy("AAA", conf, mj)


def test_max_contact_energy_zero_remaining():
    mj = load_mj_matrix()
    assert max_contact_energy(0, mj) == 0.0
    assert max_contact_energy(-1, mj) == 0.0


def test_max_contact_energy_uses_min_entry():
    mj = load_mj_matrix()
    # n_remaining = 1: 2*1 + 1 = 3 contacts at min_mj each.
    assert max_contact_energy(1, mj) == pytest.approx(mj.min() * 3)
    # n_remaining = 5: 2*5 + 1 = 11 contacts at min_mj each.
    assert max_contact_energy(5, mj) == pytest.approx(mj.min() * 11)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_max_contact_energy_is_valid_lower_bound(k):
    mj = load_mj_matrix()
    seq = "ACDEFG"  # n = 6
    n = len(seq)
    bound = max_contact_energy(n - k, mj)
    for conf in enumerate_saws(n):
        partial_e = conformation_energy(seq[:k], conf[:k], mj)
        total_e = conformation_energy(seq, conf, mj)
        remaining_e = total_e - partial_e
        # Bound is a lower bound (most negative); actual remaining energy
        # must be no more negative than the bound.
        assert bound <= remaining_e + 1e-9, (
            f"bound {bound} > remaining_e {remaining_e} for conf={conf}, k={k}"
        )


def test_partition_function_normalization():
    energies = [-1.0, -2.0, -3.0]
    Z = partition_function(energies, temperature=1.0)
    probs = np.exp(-np.asarray(energies) / 1.0) / Z
    assert probs.sum() == pytest.approx(1.0)


def test_partition_function_high_temperature_limit():
    # As T -> infinity, exp(-E/T) -> 1 for every conformation.
    energies = [-5.0, -2.0, 0.0, 1.0]
    Z = partition_function(energies, temperature=1e9)
    assert Z == pytest.approx(len(energies), rel=1e-6)


def test_partition_function_low_temperature_limit():
    # As T -> 0, the sum is dominated by the most negative energy.
    # T=0.1 keeps exp(-E/T) within float64 range while still giving a
    # ~13 order-of-magnitude gap between the ground state and next.
    energies = [-5.0, -2.0, 0.0, 1.0]
    T = 0.1
    Z = partition_function(energies, temperature=T)
    expected = np.exp(-min(energies) / T)
    assert Z == pytest.approx(expected, rel=1e-6)


def test_energy_consistent_under_chain_reversal():
    mj = load_mj_matrix()
    seq = "ACDEF"
    n = len(seq)
    rev_seq = seq[::-1]
    for conf in enumerate_saws(n):
        rev_conf = tuple(reversed(conf))
        assert conformation_energy(seq, conf, mj) == pytest.approx(
            conformation_energy(rev_seq, rev_conf, mj)
        )
