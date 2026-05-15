from math import exp, inf

import numpy as np
import pytest

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fitness import compute_fitness_aa
from trellis.genetic_code import classify_mutation, translate
from trellis.ligand import create_ligand
from trellis.sswm import (
    Trajectory,
    fixation_probability,
    generate_start_sequence,
    generate_trajectory,
)

# DNA encoding ACDEFG (6 AA = 18 nt)
_DNA_6 = "GCTTGTGATGAATTTGGT"


# ---------------------------------------------------------------------------
# Fixation probability
# ---------------------------------------------------------------------------

def test_fixation_neutral():
    Ne = 1000
    assert fixation_probability(0.0, Ne) == pytest.approx(1.0 / (2 * Ne))


def test_fixation_beneficial_weak():
    s = 0.01
    Ne = 100
    expected = (1 - exp(-2 * s)) / (1 - exp(-2 * Ne * s))
    assert fixation_probability(s, Ne) == pytest.approx(expected)


def test_fixation_beneficial_strong():
    s = 5.0
    Ne = 1000
    p = fixation_probability(s, Ne)
    assert p == pytest.approx(1.0 - exp(-2 * s), rel=1e-6)


def test_fixation_deleterious():
    p = fixation_probability(-0.01, 1000)
    assert 0 < p < 1e-5


def test_fixation_lethal():
    assert fixation_probability(-inf, 1000) == 0.0


def test_fixation_probability_range():
    for s in [-1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0]:
        p = fixation_probability(s, 1000)
        assert 0 <= p <= 1


# ---------------------------------------------------------------------------
# Trajectory structure
# ---------------------------------------------------------------------------

@pytest.fixture
def short_trajectory():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    rng = np.random.default_rng(42)
    return generate_trajectory(
        _DNA_6, lig, mj, n_steps=5, Ne=100, rng=rng,
    )


def test_trajectory_lengths(short_trajectory):
    t = short_trajectory
    n = len(t.mutation_types)
    assert len(t.dna_sequences) == n + 1
    assert len(t.aa_sequences) == n + 1
    assert len(t.fitness_values) == n + 1


def test_trajectory_sequences_valid(short_trajectory):
    t = short_trajectory
    dna_len = len(t.dna_sequences[0])
    for dna, aa in zip(t.dna_sequences, t.aa_sequences):
        assert len(dna) == dna_len
        assert translate(dna) == aa


def test_trajectory_fitness_values_match(short_trajectory):
    t = short_trajectory
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    for aa, fitness in zip(t.aa_sequences, t.fitness_values):
        expected = compute_fitness_aa(aa, lig, mj).fitness
        assert fitness == pytest.approx(expected)


def test_trajectory_mutation_types_correct(short_trajectory):
    t = short_trajectory
    for i, mut_type in enumerate(t.mutation_types):
        expected = classify_mutation(t.dna_sequences[i], t.dna_sequences[i + 1])
        assert mut_type == expected


def test_trajectory_consecutive_differ_by_one(short_trajectory):
    t = short_trajectory
    for i in range(len(t.mutation_types)):
        diffs = sum(
            1 for a, b in zip(t.dna_sequences[i], t.dna_sequences[i + 1])
            if a != b
        )
        assert diffs == 1


# ---------------------------------------------------------------------------
# SSWM behavior
# ---------------------------------------------------------------------------

def test_trajectory_no_stop_codons(short_trajectory):
    for aa in short_trajectory.aa_sequences:
        assert "*" not in aa


def test_trajectory_deterministic_with_seed():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    t1 = generate_trajectory(
        _DNA_6, lig, mj, n_steps=3, Ne=100,
        rng=np.random.default_rng(99),
    )
    t2 = generate_trajectory(
        _DNA_6, lig, mj, n_steps=3, Ne=100,
        rng=np.random.default_rng(99),
    )
    assert t1.dna_sequences == t2.dna_sequences


def test_trajectory_metadata(short_trajectory):
    m = short_trajectory.metadata
    assert "Ne" in m
    assert "mu" in m
    assert "temperature" in m
    assert "ligand_sequence" in m


def test_trajectory_cache_populated():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    cache = FitnessCache()
    generate_trajectory(
        _DNA_6, lig, mj, n_steps=3, Ne=100,
        rng=np.random.default_rng(42), fitness_cache=cache,
    )
    assert len(cache) > 1
    stats = cache.stats()
    assert stats["entries"] > 1
    assert stats["hits"] >= 0


# ---------------------------------------------------------------------------
# Start sequence generation
# ---------------------------------------------------------------------------

def test_generate_start_sequence_valid_dna():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    dna = generate_start_sequence(
        6, lig, mj, min_fitness=0.0, rng=np.random.default_rng(42),
    )
    assert len(dna) == 18
    aa = translate(dna)
    assert "*" not in aa


def test_generate_start_sequence_meets_fitness():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    threshold = 1.0
    dna = generate_start_sequence(
        6, lig, mj, min_fitness=threshold, rng=np.random.default_rng(42),
    )
    aa = translate(dna)
    r = compute_fitness_aa(aa, lig, mj)
    assert r.fitness >= threshold


def test_generate_start_sequence_raises_on_failure():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    with pytest.raises(RuntimeError, match="failed to find"):
        generate_start_sequence(
            6, lig, mj, min_fitness=1e6, max_attempts=10,
            rng=np.random.default_rng(42),
        )
