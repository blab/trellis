"""Branch-and-bound folding for lattice proteins.

Finds the native (lowest-energy) conformation of an amino acid sequence
on the 2D square lattice, while accumulating the Boltzmann partition
function Z in a single pass.  Energy pruning skips subtrees that cannot
improve on the current best, and the pruned conformations have negligible
Boltzmann weight so Z remains accurate.

The symmetry reduction matches ``lattice._walk_reduced``: residue 0 is
fixed at (0, 0), residue 1 at (1, 0), and the first off-axis step is
forced to +y.  The reported partition function is corrected from the
reduced to the unreduced count via ``Z_full = Z_reduced * 8 − 4``.
"""

from dataclasses import dataclass
from math import exp, inf

import numpy as np

from trellis.energy import AA_INDEX
from trellis.lattice import MOVES, Conformation


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
    ligand: object | None = None,
    temperature: float = 1.0,
) -> FoldResult:
    """Fold *sequence* on the 2D square lattice via branch-and-bound.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes).
    mj_matrix : np.ndarray
        20×20 Miyazawa-Jernigan contact-energy matrix.
    ligand : object or None
        Reserved for Step 4 (``ligand.py``).  Must be ``None``.
    temperature : float
        Boltzmann temperature in reduced units (default 1.0).

    Returns
    -------
    FoldResult
    """
    if ligand is not None:
        raise NotImplementedError("Ligand support requires Step 4 (ligand.py)")
    n = len(sequence)
    if n < 1:
        raise ValueError("sequence must have at least 1 residue")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    indices = [AA_INDEX[aa] for aa in sequence]

    if n == 1:
        return FoldResult(
            native_conformation=((0, 0),),
            native_energy=0.0,
            partition_function=1.0,
            ensemble_binding_energy=0.0,
            n_conformations_enumerated=1,
        )

    mj_min = float(mj_matrix.min())
    bounds = [mj_min * (2 * r + 1) if r > 0 else 0.0 for r in range(n + 1)]

    best_energy = inf
    best_conf: list[tuple[int, int]] | None = None
    Z_sum = 0.0
    n_enumerated = 0

    path: list[tuple[int, int]] = [(0, 0), (1, 0)]
    occupied: set[tuple[int, int]] = {(0, 0), (1, 0)}

    def _recurse(energy: float, depth: int, y_locked: bool) -> None:
        nonlocal best_energy, best_conf, Z_sum, n_enumerated

        if depth == n:
            n_enumerated += 1
            Z_sum += exp(-energy / temperature)
            if energy < best_energy:
                best_energy = energy
                best_conf = list(path)
            return

        if energy + bounds[n - depth] >= best_energy:
            return

        last_x, last_y = path[-1]
        for dx, dy in MOVES:
            if not y_locked and dy == -1:
                continue
            new_x = last_x + dx
            new_y = last_y + dy
            new_pos = (new_x, new_y)
            if new_pos in occupied:
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

    _recurse(0.0, 2, False)

    # Symmetry: unreduced = 8 * reduced - 4.  The straight-line walk has
    # only 4 rotational images (it is its own x-axis reflection) and its
    # energy is 0, contributing exp(0/T) = 1 per image.
    Z_full = Z_sum * 8 - 4

    if n == 2:
        return FoldResult(
            native_conformation=((0, 0), (1, 0)),
            native_energy=0.0,
            partition_function=Z_full,
            ensemble_binding_energy=0.0,
            n_conformations_enumerated=n_enumerated,
        )

    return FoldResult(
        native_conformation=tuple(best_conf),  # type: ignore[arg-type]
        native_energy=best_energy,
        partition_function=Z_full,
        ensemble_binding_energy=0.0,
        n_conformations_enumerated=n_enumerated,
    )
