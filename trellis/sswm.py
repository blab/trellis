"""SSWM (strong-selection weak-mutation) trajectory generation."""

from dataclasses import dataclass, field
from math import exp, inf

import numpy as np

from trellis.cache import FitnessCache
from trellis.fitness import FitnessResult, compute_fitness_aa
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
) -> Trajectory:
    """Generate a single SSWM trajectory."""
    if rng is None:
        rng = np.random.default_rng()
    if fitness_cache is None:
        fitness_cache = FitnessCache()

    current_dna = start_dna
    current_aa = translate(current_dna)
    if current_aa not in fitness_cache:
        r = compute_fitness_aa(current_aa, ligand, mj_matrix, temperature)
        fitness_cache.put(current_aa, r)
    current_fitness = fitness_cache.get(current_aa).fitness

    dna_seqs = [current_dna]
    aa_seqs = [current_aa]
    fitness_vals = [current_fitness]
    mut_types: list[str] = []

    for _ in range(n_steps):
        aa_groups = mutant_aa_sequences(current_dna)
        for aa_seq in aa_groups:
            if aa_seq not in fitness_cache:
                if "*" in aa_seq:
                    fitness_cache.put(aa_seq, FitnessResult(
                        fitness=-inf, fold_result=None,
                        aa_sequence=aa_seq, dna_sequence="",
                    ))
                else:
                    r = compute_fitness_aa(
                        aa_seq, ligand, mj_matrix, temperature,
                    )
                    fitness_cache.put(aa_seq, r)

        mutations = single_nt_mutations(current_dna)
        weights = []
        for mutant_dna, _, _, _ in mutations:
            mutant_aa = translate(mutant_dna)
            mutant_fitness = fitness_cache.get(mutant_aa).fitness
            s = mutant_fitness - current_fitness
            weights.append(fixation_probability(s, Ne))

        total = sum(weights)
        if total == 0:
            break

        probs = np.array(weights)
        probs /= probs.sum()
        idx = rng.choice(len(mutations), p=probs)
        chosen_dna = mutations[idx][0]
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
) -> str:
    """Generate a random DNA sequence with fitness >= *min_fitness*."""
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(max_attempts):
        codons = rng.choice(SENSE_CODONS, size=n_codons)
        dna = "".join(codons)
        aa = translate(dna)
        r = compute_fitness_aa(aa, ligand, mj_matrix, temperature)
        if r.fitness >= min_fitness:
            return dna
    raise RuntimeError(
        f"failed to find sequence with fitness >= {min_fitness} "
        f"after {max_attempts} attempts"
    )
