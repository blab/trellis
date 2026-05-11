import time
from math import exp

import numpy as np
import pytest

from trellis.energy import conformation_energy, load_mj_matrix, partition_function
from trellis.fold import FoldResult, fold
from trellis.lattice import enumerate_saws, is_self_avoiding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exhaustive_native_energy(sequence, mj):
    """Brute-force minimum energy over all symmetry-reduced SAWs."""
    best = float("inf")
    for conf in enumerate_saws(len(sequence)):
        e = conformation_energy(sequence, conf, mj)
        if e < best:
            best = e
    return best


def exhaustive_Z(sequence, mj, temperature=1.0):
    """Brute-force partition function with symmetry correction."""
    Z_reduced = 0.0
    for conf in enumerate_saws(len(sequence)):
        e = conformation_energy(sequence, conf, mj)
        Z_reduced += exp(-e / temperature)
    return Z_reduced * 8 - 4


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

def test_fold_result_fields():
    mj = load_mj_matrix()
    result = fold("AC", mj)
    assert isinstance(result, FoldResult)
    assert hasattr(result, "native_conformation")
    assert hasattr(result, "native_energy")
    assert hasattr(result, "partition_function")
    assert hasattr(result, "ensemble_binding_energy")
    assert hasattr(result, "n_conformations_enumerated")


def test_ligand_raises():
    mj = load_mj_matrix()
    with pytest.raises(NotImplementedError):
        fold("ACDEF", mj, ligand=object())


def test_ensemble_binding_energy_zero_without_ligand():
    mj = load_mj_matrix()
    result = fold("ACDEFG", mj)
    assert result.ensemble_binding_energy == 0.0


# ---------------------------------------------------------------------------
# Correctness via exhaustive comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", ["ACDEFG", "ACDEFGHI", "ACDEFGHIKL"])
def test_native_energy_matches_exhaustive(seq):
    mj = load_mj_matrix()
    result = fold(seq, mj)
    expected = exhaustive_native_energy(seq, mj)
    assert result.native_energy == pytest.approx(expected)


@pytest.mark.parametrize("seq", ["ACDEFG", "ACDEFGHI", "ACDEFGHIKL"])
def test_partition_function_matches_exhaustive(seq):
    mj = load_mj_matrix()
    result = fold(seq, mj)
    expected = exhaustive_Z(seq, mj)
    assert result.partition_function == pytest.approx(expected, rel=1e-9)


def test_n_conformations_matches_for_small_n():
    mj = load_mj_matrix()
    result = fold("ACDEFG", mj)
    reduced_count = sum(1 for _ in enumerate_saws(6))
    assert result.n_conformations_enumerated == reduced_count


def test_n_conformations_at_most_reduced_count():
    mj = load_mj_matrix()
    seq = "ACDEFGHIKL"
    result = fold(seq, mj)
    reduced_count = sum(1 for _ in enumerate_saws(len(seq)))
    assert result.n_conformations_enumerated <= reduced_count


# ---------------------------------------------------------------------------
# Native conformation validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", ["ACDEFG", "ACDEFGHI"])
def test_native_conformation_is_valid_saw(seq):
    mj = load_mj_matrix()
    result = fold(seq, mj)
    conf = result.native_conformation
    n = len(seq)
    assert len(conf) == n
    assert conf[0] == (0, 0)
    assert conf[1] == (1, 0)
    assert is_self_avoiding(conf)
    for i in range(n - 1):
        dx = abs(conf[i + 1][0] - conf[i][0])
        dy = abs(conf[i + 1][1] - conf[i][1])
        assert dx + dy == 1


def test_native_conformation_has_reported_energy():
    mj = load_mj_matrix()
    seq = "ACDEFGHI"
    result = fold(seq, mj)
    recomputed = conformation_energy(seq, result.native_conformation, mj)
    assert recomputed == pytest.approx(result.native_energy)


# ---------------------------------------------------------------------------
# Symmetry and invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq", ["ACDEFG", "FWYLKR"])
def test_energy_invariant_under_sequence_reversal(seq):
    mj = load_mj_matrix()
    fwd = fold(seq, mj)
    rev = fold(seq[::-1], mj)
    assert fwd.native_energy == pytest.approx(rev.native_energy)


# ---------------------------------------------------------------------------
# Temperature effects
# ---------------------------------------------------------------------------

def test_high_temperature_Z_approaches_total_conformations():
    mj = load_mj_matrix()
    seq = "ACDEFG"
    n = len(seq)
    result = fold(seq, mj, temperature=1000.0)
    reduced_count = sum(1 for _ in enumerate_saws(n))
    expected_Z = 8 * reduced_count - 4
    assert result.partition_function == pytest.approx(expected_Z, rel=5e-3)


def test_low_temperature_Z_dominated_by_native():
    mj = load_mj_matrix()
    seq = "ACDEFGHI"
    T = 0.1
    result = fold(seq, mj, temperature=T)
    expected = 8 * exp(-result.native_energy / T)
    assert result.partition_function == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fold_single_residue():
    mj = load_mj_matrix()
    result = fold("A", mj)
    assert result.native_conformation == ((0, 0),)
    assert result.native_energy == 0.0
    assert result.partition_function == 1.0
    assert result.n_conformations_enumerated == 1


def test_fold_two_residues():
    mj = load_mj_matrix()
    result = fold("AC", mj)
    assert result.native_conformation == ((0, 0), (1, 0))
    assert result.native_energy == 0.0
    assert result.partition_function == pytest.approx(4.0)
    assert result.n_conformations_enumerated == 1


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fold_20mer_under_30_seconds():
    mj = load_mj_matrix()
    seq = "ACDEFGHIKLMNPQRSTVWY"
    t0 = time.perf_counter()
    result = fold(seq, mj)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30, f"20-mer fold took {elapsed:.1f}s (target <30s)"
    assert result.native_energy < 0
    assert result.n_conformations_enumerated > 0
