"""Fitness as ensemble-averaged ligand binding energy."""

from dataclasses import dataclass
from math import inf

import numpy as np

from trellis.fold import FoldResult, fold
from trellis.genetic_code import translate
from trellis.ligand import Ligand


@dataclass(frozen=True)
class FitnessResult:
    """Result of a fitness evaluation."""

    fitness: float
    fold_result: FoldResult | None
    aa_sequence: str
    dna_sequence: str


def compute_fitness(
    dna_sequence: str,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    temperature: float = 1.0,
) -> FitnessResult:
    """Compute fitness for a DNA sequence.

    Translates DNA to amino acids, folds with the ligand, and returns
    fitness = −⟨E_bind⟩ (higher = fitter).  Stop codons yield −inf.
    """
    aa_sequence = translate(dna_sequence)
    if "*" in aa_sequence:
        return FitnessResult(
            fitness=-inf,
            fold_result=None,
            aa_sequence=aa_sequence,
            dna_sequence=dna_sequence,
        )
    fold_result = fold(aa_sequence, mj_matrix, ligand, temperature)
    return FitnessResult(
        fitness=-fold_result.ensemble_binding_energy,
        fold_result=fold_result,
        aa_sequence=aa_sequence,
        dna_sequence=dna_sequence,
    )


def compute_fitness_aa(
    aa_sequence: str,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    temperature: float = 1.0,
) -> FitnessResult:
    """Compute fitness directly from an amino acid sequence."""
    fold_result = fold(aa_sequence, mj_matrix, ligand, temperature)
    return FitnessResult(
        fitness=-fold_result.ensemble_binding_energy,
        fold_result=fold_result,
        aa_sequence=aa_sequence,
        dna_sequence="",
    )
