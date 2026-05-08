"""Miyazawa-Jernigan contact energies for lattice-protein conformations.

Provides the MJ 1985 contact-potential matrix loader plus three energy
helpers used downstream:

- ``conformation_energy`` — total contact energy of a sequence on a
  conformation, summed over all topological contacts from
  ``trellis.lattice.get_contacts``.
- ``max_contact_energy`` — loose lower bound on additional energy from
  unplaced residues, used by the branch-and-bound folder in Step 3.
- ``partition_function`` — naive ``Σ exp(-E/T)`` for testing on small
  chains. Production folding accumulates Z directly inside the B&B loop.

The MJ matrix is loaded from ``data/mj_matrix.csv``. The values are
taken from the ``miyazawa_jernigan`` dictionary in
``jbloomlab/latticeproteins/src/interactions.py``
(https://github.com/jbloomlab/latticeproteins), which in turn cites
Table V of:

    Miyazawa, S. & Jernigan, R. L. (1985). Estimation of effective
    interresidue contact energies from protein crystal structures:
    Quasi-chemical approximation. Macromolecules 18:534-552.

See ``data/README.md`` for the full citation and the source commit SHA.
"""

import csv
from functools import cache
from pathlib import Path
from typing import Iterable

import numpy as np

from trellis.lattice import Conformation, get_contacts

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}

DEFAULT_MJ_PATH = Path(__file__).resolve().parent.parent / "data" / "mj_matrix.csv"


@cache
def load_mj_matrix(path: str | Path = DEFAULT_MJ_PATH) -> np.ndarray:
    """Load the 20×20 MJ contact-energy matrix from ``path``.

    Returns a read-only ``float64`` ``np.ndarray`` indexed by
    ``AA_INDEX[aa]``. Repeat calls with the same path return the same
    cached array object.
    """
    path = Path(path)
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0][1:]
    if tuple(header) != tuple(AA_ALPHABET):
        raise ValueError(
            f"MJ CSV header {header!r} does not match expected AA order {AA_ALPHABET!r}"
        )
    body = rows[1:]
    if len(body) != 20:
        raise ValueError(f"MJ CSV must have 20 data rows, got {len(body)}")
    matrix = np.zeros((20, 20), dtype=np.float64)
    for i, row in enumerate(body):
        if row[0] != AA_ALPHABET[i]:
            raise ValueError(
                f"MJ CSV row {i} label {row[0]!r} does not match {AA_ALPHABET[i]!r}"
            )
        matrix[i] = [float(x) for x in row[1:]]
    if not np.allclose(matrix, matrix.T):
        raise ValueError("MJ matrix is not symmetric")
    matrix.flags.writeable = False
    return matrix


def conformation_energy(
    sequence: str,
    conformation: Conformation,
    mj_matrix: np.ndarray,
) -> float:
    """Total contact energy of ``sequence`` on ``conformation``."""
    if len(sequence) != len(conformation):
        raise ValueError(
            f"sequence length {len(sequence)} != conformation length {len(conformation)}"
        )
    indices = [AA_INDEX[aa] for aa in sequence]
    energy = 0.0
    for i, j in get_contacts(conformation):
        energy += mj_matrix[indices[i], indices[j]]
    return float(energy)


def max_contact_energy(n_remaining: int, mj_matrix: np.ndarray) -> float:
    """Loose lower bound on energy from contacts of ``n_remaining`` unplaced residues.

    Used as a branch-and-bound pruning bound: if the current partial
    energy plus this bound is already worse than the best complete
    energy found so far, the branch can be pruned.

    The bound assumes each unplaced interior residue can form up to 2
    new contacts (4 lattice neighbors minus 2 chain bonds) and the
    C-terminus can form up to 3, with each contact contributing
    ``mj_matrix.min()``. This double-counts contacts between two
    unplaced residues, which keeps the bound a valid lower bound while
    making it slightly looser.
    """
    if n_remaining <= 0:
        return 0.0
    return float(mj_matrix.min()) * (2 * n_remaining + 1)


def partition_function(
    energies: Iterable[float],
    temperature: float = 1.0,
) -> float:
    """Compute ``Z = Σ exp(-E_i / T)`` over the supplied energies."""
    e = np.asarray(list(energies), dtype=np.float64)
    return float(np.exp(-e / temperature).sum())
