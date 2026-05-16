from math import inf

import pytest

from trellis.energy import load_mj_matrix
from trellis.fitness import FitnessResult, compute_fitness, compute_fitness_aa
from trellis.fold_bb import FoldResult, fold
from trellis.ligand import create_ligand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# DNA encoding ACDEFG (6 AA)
DNA_ACDEFG_1 = "GCTTGTGATGAATTTGGT"
# Synonymous DNA encoding the same ACDEFG
DNA_ACDEFG_2 = "GCCTGCGACGAGTTCGGC"
# DNA with internal stop codon: ACD*FG
DNA_STOP = "GCTTGTGATTAGTTTGGT"
# DNA encoding FWYLKR (6 AA)
DNA_FWYLKR = "TTTTGGTATCTTAAACGT"
# DNA encoding ACDEFGHI (8 AA)
DNA_ACDEFGHI = "GCTTGTGATGAATTTGGTCATATT"


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_fitness_result_fields():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    result = compute_fitness(DNA_ACDEFG_1, lig, mj)
    assert isinstance(result, FitnessResult)
    assert isinstance(result.fitness, float)
    assert isinstance(result.fold_result, FoldResult)
    assert result.aa_sequence == "ACDEFG"
    assert result.dna_sequence == DNA_ACDEFG_1


# ---------------------------------------------------------------------------
# Stop codons
# ---------------------------------------------------------------------------

def test_stop_codon_gives_neg_inf():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    result = compute_fitness(DNA_STOP, lig, mj)
    assert result.fitness == -inf
    assert result.fold_result is None
    assert "*" in result.aa_sequence


# ---------------------------------------------------------------------------
# Synonymous mutations
# ---------------------------------------------------------------------------

def test_synonymous_mutations_identical_fitness():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    r1 = compute_fitness(DNA_ACDEFG_1, lig, mj)
    r2 = compute_fitness(DNA_ACDEFG_2, lig, mj)
    assert r1.fitness == r2.fitness
    assert r1.aa_sequence == r2.aa_sequence


# ---------------------------------------------------------------------------
# Fitness values
# ---------------------------------------------------------------------------

def test_fitness_positive_for_binding_sequence():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    result = compute_fitness(DNA_ACDEFG_1, lig, mj)
    assert result.fitness > 0


def test_fitness_near_zero_for_distant_ligand():
    mj = load_mj_matrix()
    lig = create_ligand("FWYL", anchor=(100, 100))
    result = compute_fitness(DNA_ACDEFG_1, lig, mj)
    assert result.fitness == pytest.approx(0.0)


def test_fitness_matches_manual_fold():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    result = compute_fitness(DNA_ACDEFGHI, lig, mj)
    manual = fold("ACDEFGHI", mj, ligand=lig)
    assert result.fitness == pytest.approx(-manual.ensemble_binding_energy)
    assert result.fold_result.native_energy == pytest.approx(manual.native_energy)


# ---------------------------------------------------------------------------
# AA convenience function
# ---------------------------------------------------------------------------

def test_compute_fitness_aa():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    dna_result = compute_fitness(DNA_ACDEFG_1, lig, mj)
    aa_result = compute_fitness_aa("ACDEFG", lig, mj)
    assert aa_result.fitness == dna_result.fitness
    assert aa_result.dna_sequence == ""


# ---------------------------------------------------------------------------
# Fitness varies across sequences
# ---------------------------------------------------------------------------

def test_fitness_varies_across_sequences():
    mj = load_mj_matrix()
    lig = create_ligand("FW", anchor=(0, -1))
    seqs = [DNA_ACDEFG_1, DNA_FWYLKR, DNA_ACDEFGHI]
    fitnesses = [compute_fitness(dna, lig, mj).fitness for dna in seqs]
    assert len(set(fitnesses)) == len(fitnesses)
