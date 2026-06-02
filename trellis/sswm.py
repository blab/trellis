"""SSWM (strong-selection weak-mutation) trajectory generation."""

from dataclasses import dataclass, field
from math import exp, inf

import numpy as np

from trellis.cache import FitnessCache
from trellis.fitness import FitnessResult, compute_fitness_aa, compute_fitness_batch
from trellis.fold_enum import ConformationDatabase
from trellis.genetic_code import (
    CODON_TABLE,
    classify_mutation,
    mutant_aa_sequences,
    single_nt_mutations,
    translate,
)
from trellis.ligand import Ligand

SENSE_CODONS = [c for c, aa in CODON_TABLE.items() if aa != "*"]


@dataclass
class Trajectory:
    """An SSWM evolutionary trajectory through DNA sequence space."""

    dna_sequences: list[str]
    aa_sequences: list[str]
    fitness_values: list[float]
    mutation_types: list[str]
    metadata: dict = field(default_factory=dict)


def fixation_probability(s: float, Ne: float) -> float:
    """Kimura (1962) fixation probability for selection coefficient *s*."""
    if s == -inf:
        return 0.0
    if s == 0.0:
        return 1.0 / (2.0 * Ne)
    x = 2.0 * Ne * s
    if x > 500:
        return 1.0 - exp(-2.0 * s)
    if x < -500:
        return 0.0
    return (1.0 - exp(-2.0 * s)) / (1.0 - exp(-x))


def compute_sswm_probabilities(
    dna: str,
    current_fitness: float,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    Ne: float,
    temperature: float,
    fitness_cache: FitnessCache,
    db: ConformationDatabase | None = None,
) -> tuple[list[str], np.ndarray]:
    """Normalized SSWM fixation probabilities for all single-nt neighbors.

    Enumerates every single-nucleotide mutant of ``dna``, folds each
    unique mutant AA sequence (consulting / populating ``fitness_cache``),
    and computes the Kimura fixation probability for each mutant relative
    to ``current_fitness``. Probabilities are normalized to sum to 1.

    Returns
    -------
    mutant_dnas : list[str]
        Mutant DNA sequences, one per single-nt neighbor (~3 * len(dna)).
    probs : np.ndarray
        Normalized fixation probability for each mutant. If every neighbor
        is lethal, returns an all-zero array (caller can detect via
        ``probs.sum() == 0``).
    """
    aa_groups = mutant_aa_sequences(dna)

    uncached = []
    for aa_seq in aa_groups:
        if aa_seq not in fitness_cache:
            if "*" in aa_seq:
                fitness_cache.put(aa_seq, FitnessResult(
                    fitness=-inf, fold_result=None,
                    aa_sequence=aa_seq, dna_sequence="",
                ))
            else:
                uncached.append(aa_seq)

    if uncached and db is not None:
        batch_results = compute_fitness_batch(
            uncached, ligand, mj_matrix, temperature, db=db,
        )
        for aa_seq, result in zip(uncached, batch_results):
            fitness_cache.put(aa_seq, result)
    elif uncached:
        for aa_seq in uncached:
            r = compute_fitness_aa(
                aa_seq, ligand, mj_matrix, temperature, db=db,
            )
            fitness_cache.put(aa_seq, r)

    mutant_dnas = [m for m, _, _, _ in single_nt_mutations(dna)]
    probs = np.array([
        fixation_probability(
            fitness_cache.get(translate(m)).fitness - current_fitness, Ne
        )
        for m in mutant_dnas
    ])
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return mutant_dnas, probs


def pfix_for_target(
    reference_dna: str,
    target_dna: str,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    Ne: float,
    temperature: float,
    fitness_cache: FitnessCache,
    db: ConformationDatabase | None = None,
) -> float | None:
    """Normalized P_fix for the specific reference -> target mutation.

    Returns the probability that ``target_dna`` would be the next
    substitution under SSWM dynamics, i.e. its fixation probability
    normalized by the sum over all single-nt neighbors of
    ``reference_dna``.

    Returns ``None`` if ``target_dna`` is not a single-nucleotide neighbor
    of ``reference_dna``. Returns ``0.0`` if every neighbor is lethal.
    """
    if len(reference_dna) != len(target_dna):
        return None
    diffs = sum(1 for a, b in zip(reference_dna, target_dna) if a != b)
    if diffs != 1:
        return None

    ref_aa = translate(reference_dna)
    if ref_aa not in fitness_cache:
        r = compute_fitness_aa(ref_aa, ligand, mj_matrix, temperature, db=db)
        fitness_cache.put(ref_aa, r)
    current_fitness = fitness_cache.get(ref_aa).fitness

    mutant_dnas, probs = compute_sswm_probabilities(
        reference_dna, current_fitness, ligand, mj_matrix,
        Ne, temperature, fitness_cache, db,
    )
    try:
        idx = mutant_dnas.index(target_dna)
    except ValueError:
        return None
    return float(probs[idx])


def generate_trajectory(
    start_dna: str,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    n_steps: int = 100,
    Ne: float = 1000.0,
    mu: float = 1e-6,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
    fitness_cache: FitnessCache | None = None,
    db: ConformationDatabase | None = None,
) -> Trajectory:
    """Generate a single SSWM trajectory."""
    if rng is None:
        rng = np.random.default_rng()
    if fitness_cache is None:
        fitness_cache = FitnessCache()

    current_dna = start_dna
    current_aa = translate(current_dna)
    if current_aa not in fitness_cache:
        r = compute_fitness_aa(current_aa, ligand, mj_matrix, temperature, db=db)
        fitness_cache.put(current_aa, r)
    current_fitness = fitness_cache.get(current_aa).fitness

    dna_seqs = [current_dna]
    aa_seqs = [current_aa]
    fitness_vals = [current_fitness]
    mut_types: list[str] = []

    for _ in range(n_steps):
        mutant_dnas, probs = compute_sswm_probabilities(
            current_dna, current_fitness, ligand, mj_matrix,
            Ne, temperature, fitness_cache, db,
        )

        if probs.sum() == 0:
            break

        idx = rng.choice(len(mutant_dnas), p=probs)
        chosen_dna = mutant_dnas[idx]
        chosen_aa = translate(chosen_dna)
        chosen_fitness = fitness_cache.get(chosen_aa).fitness
        mut_type = classify_mutation(current_dna, chosen_dna)

        current_dna = chosen_dna
        current_aa = chosen_aa
        current_fitness = chosen_fitness

        dna_seqs.append(current_dna)
        aa_seqs.append(current_aa)
        fitness_vals.append(current_fitness)
        mut_types.append(mut_type)

    metadata = {
        "Ne": Ne,
        "mu": mu,
        "temperature": temperature,
        "ligand_sequence": ligand.sequence,
        "ligand_positions": ligand.positions,
        "n_steps_requested": n_steps,
        "n_steps_completed": len(mut_types),
    }
    return Trajectory(
        dna_sequences=dna_seqs,
        aa_sequences=aa_seqs,
        fitness_values=fitness_vals,
        mutation_types=mut_types,
        metadata=metadata,
    )


def generate_start_sequence(
    n_codons: int,
    ligand: Ligand,
    mj_matrix: np.ndarray,
    min_fitness: float = 0.0,
    max_attempts: int = 10000,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
    db: ConformationDatabase | None = None,
) -> str:
    """Generate a random DNA sequence with fitness >= *min_fitness*."""
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(max_attempts):
        codons = rng.choice(SENSE_CODONS, size=n_codons)
        dna = "".join(codons)
        aa = translate(dna)
        r = compute_fitness_aa(aa, ligand, mj_matrix, temperature, db=db)
        if r.fitness >= min_fitness:
            return dna
    raise RuntimeError(
        f"failed to find sequence with fitness >= {min_fitness} "
        f"after {max_attempts} attempts"
    )
