"""Fitness as ensemble-averaged ligand binding energy."""

from dataclasses import dataclass
from math import inf

import numpy as np

from trellis.fold_bb import FoldResult, fold as fold_bb
from trellis.fold_enum import ConformationDatabase, fold as fold_enum, fold_batch
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
    db: ConformationDatabase | None = None,
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
    if db is not None:
        fold_result = fold_enum(
            aa_sequence, mj_matrix, ligand, temperature,
            db=db, recover_conformation=False,
        )
    else:
        fold_result = fold_bb(aa_sequence, mj_matrix, ligand, temperature)
    return FitnessResult(
        fitness=fold_result.fraction_folded * (-fold_result.native_binding_energy),
        fold_result=fold_result,
        aa_sequence=aa_sequence,
        dna_sequence=dna_sequence,
    )


def compute_fitness_aa(
    aa_sequence: str,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    temperature: float = 1.0,
    db: ConformationDatabase | None = None,
) -> FitnessResult:
    """Compute fitness directly from an amino acid sequence."""
    if db is not None:
        fold_result = fold_enum(
            aa_sequence, mj_matrix, ligand, temperature,
            db=db, recover_conformation=False,
        )
    else:
        fold_result = fold_bb(aa_sequence, mj_matrix, ligand, temperature)
    return FitnessResult(
        fitness=fold_result.fraction_folded * (-fold_result.native_binding_energy),
        fold_result=fold_result,
        aa_sequence=aa_sequence,
        dna_sequence="",
    )


def compute_fitness_batch(
    aa_sequences: list[str],
    ligand: Ligand,
    mj_matrix: np.ndarray,
    temperature: float = 1.0,
    db: ConformationDatabase | None = None,
) -> list[FitnessResult]:
    """Compute fitness for multiple AA sequences in one batch."""
    fold_results = fold_batch(aa_sequences, mj_matrix, ligand, temperature, db=db)
    return [
        FitnessResult(
            fitness=fr.fraction_folded * (-fr.native_binding_energy),
            fold_result=fr,
            aa_sequence=aa,
            dna_sequence="",
        )
        for aa, fr in zip(aa_sequences, fold_results)
    ]
