"""Branch-and-bound folding for lattice proteins.

Finds the native (lowest-energy) conformation of an amino acid sequence
on the 2D square lattice, while accumulating the Boltzmann partition
function Z in a single pass.  Energy pruning skips subtrees that cannot
improve on the current best, and the pruned conformations have negligible
Boltzmann weight so Z remains accurate.

Without a ligand the symmetry reduction matches ``lattice._walk_reduced``:
residue 0 is fixed at (0, 0), residue 1 at (1, 0), and the first
off-axis step is forced to +y.  The reported partition function is
corrected via ``Z_full = Z_reduced * 8 − 4``.

With a ligand the 8-fold dihedral symmetry is broken, so the folder
uses unreduced enumeration (all walks from the origin) and no symmetry
correction.
"""

from dataclasses import dataclass
from math import exp, inf

import numpy as np

from trellis.energy import AA_INDEX
from trellis.lattice import MOVES, Conformation
from trellis.ligand import Ligand, binding_energy as ligand_binding_energy


@dataclass(frozen=True)
class FoldResult:
    """Result of branch-and-bound folding."""

    native_conformation: Conformation
    native_energy: float
    partition_function: float
    ensemble_binding_energy: float
    n_conformations_enumerated: int


def fold(
    sequence: str,
    mj_matrix: np.ndarray,
    ligand: Ligand | None = None,
    temperature: float = 1.0,
) -> FoldResult:
    """Fold *sequence* on the 2D square lattice via branch-and-bound.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes).
    mj_matrix : np.ndarray
        20×20 Miyazawa-Jernigan contact-energy matrix.
    ligand : Ligand or None
        Fixed ligand on the lattice for protein-ligand binding.
    temperature : float
        Boltzmann temperature in reduced units (default 1.0).

    Returns
    -------
    FoldResult
    """
    n = len(sequence)
    if n < 1:
        raise ValueError("sequence must have at least 1 residue")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    indices = [AA_INDEX[aa] for aa in sequence]

    ligand_sites: frozenset[tuple[int, int]] = (
        ligand.sites if ligand is not None else frozenset()
    )

    if n == 1:
        e_bind = 0.0
        if ligand is not None:
            e_bind = ligand_binding_energy(sequence, ((0, 0),), ligand, mj_matrix)
        return FoldResult(
            native_conformation=((0, 0),),
            native_energy=e_bind,
            partition_function=1.0,
            ensemble_binding_energy=e_bind,
            n_conformations_enumerated=1,
        )

    mj_min = float(mj_matrix.min())
    bounds = [mj_min * (2 * r + 1) if r > 0 else 0.0 for r in range(n + 1)]
    binding_bound = (4 * len(ligand.sequence) * mj_min) if ligand is not None else 0.0

    best_energy = inf
    best_conf: list[tuple[int, int]] | None = None
    Z_sum = 0.0
    binding_weighted_sum = 0.0
    n_enumerated = 0

    def _recurse(energy: float, depth: int, y_locked: bool) -> None:
        nonlocal best_energy, best_conf, Z_sum, binding_weighted_sum, n_enumerated

        if depth == n:
            e_bind = 0.0
            if ligand is not None:
                e_bind = ligand_binding_energy(
                    sequence, tuple(path), ligand, mj_matrix
                )
            total = energy + e_bind
            boltz = exp(-total / temperature)
            Z_sum += boltz
            binding_weighted_sum += e_bind * boltz
            n_enumerated += 1
            if total < best_energy:
                best_energy = total
                best_conf = list(path)
            return

        if energy + bounds[n - depth] + binding_bound >= best_energy:
            return

        last_x, last_y = path[-1]
        for dx, dy in MOVES:
            if not y_locked and dy == -1:
                continue
            new_x = last_x + dx
            new_y = last_y + dy
            new_pos = (new_x, new_y)
            if new_pos in occupied or new_pos in ligand_sites:
                continue

            delta = 0.0
            for i in range(depth - 1):
                ix, iy = path[i]
                if abs(ix - new_x) + abs(iy - new_y) == 1:
                    delta += mj_matrix[indices[i], indices[depth]]

            path.append(new_pos)
            occupied.add(new_pos)
            _recurse(energy + delta, depth + 1, y_locked or dy == 1)
            occupied.remove(new_pos)
            path.pop()

    if ligand is not None:
        path: list[tuple[int, int]] = [(0, 0)]
        occupied: set[tuple[int, int]] = {(0, 0)}
        _recurse(0.0, 1, True)
        Z_full = Z_sum
    else:
        path = [(0, 0), (1, 0)]
        occupied = {(0, 0), (1, 0)}
        _recurse(0.0, 2, False)
        # Symmetry: unreduced = 8 * reduced - 4.  The straight-line walk
        # has only 4 rotational images (it is its own x-axis reflection)
        # and its energy is 0, contributing exp(0/T) = 1 per image.
        Z_full = Z_sum * 8 - 4

    ensemble_binding = binding_weighted_sum / Z_sum if Z_sum > 0 else 0.0

    return FoldResult(
        native_conformation=tuple(best_conf),  # type: ignore[arg-type]
        native_energy=best_energy,
        partition_function=Z_full,
        ensemble_binding_energy=ensemble_binding,
        n_conformations_enumerated=n_enumerated,
    )
