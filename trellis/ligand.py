"""Fixed ligand on the 2D lattice and protein-ligand binding energy.

A ``Ligand`` occupies fixed lattice sites adjacent to the protein's
starting position.  During branch-and-bound folding the protein walk
must not overlap ligand sites; at each complete conformation the
binding energy (sum of MJ contacts between protein and ligand residues)
is accumulated into the Boltzmann-weighted ensemble average.
"""

from dataclasses import dataclass

import numpy as np

from trellis.energy import AA_INDEX
from trellis.lattice import Conformation


@dataclass(frozen=True)
class Ligand:
    """A fixed ligand on the 2D square lattice."""

    sequence: str
    positions: tuple[tuple[int, int], ...]
    sites: frozenset[tuple[int, int]]


def create_ligand(
    sequence: str,
    anchor: tuple[int, int] = (0, -1),
    direction: str = "horizontal",
) -> Ligand:
    """Place a ligand on the lattice starting at *anchor*.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the ligand (single-letter codes).
    anchor : tuple[int, int]
        Lattice position of the first ligand residue.
    direction : ``"horizontal"`` or ``"vertical"``
        Axis along which subsequent residues are placed.
    """
    if not sequence:
        raise ValueError("ligand sequence must be non-empty")
    for aa in sequence:
        if aa not in AA_INDEX:
            raise ValueError(f"invalid amino acid {aa!r} in ligand sequence")
    if direction not in ("horizontal", "vertical"):
        raise ValueError(f"direction must be 'horizontal' or 'vertical', got {direction!r}")

    if direction == "horizontal":
        positions = tuple((anchor[0] + i, anchor[1]) for i in range(len(sequence)))
    else:
        positions = tuple((anchor[0], anchor[1] + i) for i in range(len(sequence)))

    return Ligand(sequence=sequence, positions=positions, sites=frozenset(positions))


def binding_contacts(
    conformation: Conformation,
    ligand: Ligand,
) -> list[tuple[int, int]]:
    """Return protein-ligand contact pairs ``(protein_idx, ligand_idx)``.

    A contact exists when a protein residue and a ligand residue are
    lattice-adjacent (Manhattan distance 1).  Pairs are returned in
    lexicographic order.
    """
    contacts: list[tuple[int, int]] = []
    for i, (px, py) in enumerate(conformation):
        for k, (lx, ly) in enumerate(ligand.positions):
            if abs(px - lx) + abs(py - ly) == 1:
                contacts.append((i, k))
    return contacts


def binding_energy(
    protein_sequence: str,
    conformation: Conformation,
    ligand: Ligand,
    mj_matrix: np.ndarray,
) -> float:
    """Sum of MJ contact energies between protein and ligand residues."""
    if len(protein_sequence) != len(conformation):
        raise ValueError(
            f"sequence length {len(protein_sequence)} != conformation length {len(conformation)}"
        )
    energy = 0.0
    for i, k in binding_contacts(conformation, ligand):
        energy += mj_matrix[AA_INDEX[protein_sequence[i]], AA_INDEX[ligand.sequence[k]]]
    return float(energy)
