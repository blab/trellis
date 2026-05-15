"""Tests for cache — FitnessCache."""

from math import inf

import pytest

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fitness import FitnessResult, compute_fitness_aa
from trellis.ligand import create_ligand


def _make_result(aa: str, fitness: float = 1.0) -> FitnessResult:
    return FitnessResult(
        fitness=fitness, fold_result=None,
        aa_sequence=aa, dna_sequence="",
    )


# ---------------------------------------------------------------------------
# Construction and basic operations
# ---------------------------------------------------------------------------

def test_empty_cache():
    cache = FitnessCache()
    assert len(cache) == 0
    s = cache.stats()
    assert s["entries"] == 0
    assert s["hits"] == 0
    assert s["misses"] == 0


def test_put_and_get():
    cache = FitnessCache()
    r = _make_result("ACDE", 2.5)
    cache.put("ACDE", r)
    assert cache.get("ACDE") is r


def test_get_missing():
    cache = FitnessCache()
    assert cache.get("XXXX") is None


def test_contains():
    cache = FitnessCache()
    assert "ACDE" not in cache
    cache.put("ACDE", _make_result("ACDE"))
    assert "ACDE" in cache


def test_len():
    cache = FitnessCache()
    cache.put("AA", _make_result("AA"))
    cache.put("BB", _make_result("BB"))
    cache.put("CC", _make_result("CC"))
    assert len(cache) == 3


def test_put_overwrites():
    cache = FitnessCache()
    r1 = _make_result("ACDE", 1.0)
    r2 = _make_result("ACDE", 2.0)
    cache.put("ACDE", r1)
    cache.put("ACDE", r2)
    assert cache.get("ACDE") is r2
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# Hit/miss tracking
# ---------------------------------------------------------------------------

def test_stats_miss():
    cache = FitnessCache()
    _ = "ACDE" in cache
    s = cache.stats()
    assert s["misses"] == 1
    assert s["hits"] == 0


def test_stats_hit():
    cache = FitnessCache()
    cache.put("ACDE", _make_result("ACDE"))
    _ = "ACDE" in cache
    s = cache.stats()
    assert s["hits"] == 1
    assert s["misses"] == 0


def test_stats_hit_rate():
    cache = FitnessCache()
    cache.put("A", _make_result("A"))
    # 3 hits
    _ = "A" in cache
    _ = "A" in cache
    _ = "A" in cache
    # 1 miss
    _ = "B" in cache
    s = cache.stats()
    assert s["hit_rate"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Integration with compute_fitness_aa
# ---------------------------------------------------------------------------

def test_cache_stores_fold_result():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    r = compute_fitness_aa("ACDEFG", lig, mj)
    cache = FitnessCache()
    cache.put("ACDEFG", r)
    cached = cache.get("ACDEFG")
    assert cached.fold_result is not None
    assert cached.fold_result.native_conformation is not None
    assert cached.fitness == r.fitness


def test_cache_stop_codon():
    cache = FitnessCache()
    r = FitnessResult(fitness=-inf, fold_result=None, aa_sequence="A*C", dna_sequence="")
    cache.put("A*C", r)
    cached = cache.get("A*C")
    assert cached.fitness == -inf
    assert cached.fold_result is None
