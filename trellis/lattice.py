"""2D square lattice and self-avoiding walks.

Geometric foundation for lattice-protein conformations. Provides the
``Conformation`` type alias, a depth-first SAW enumerator with optional
8-fold dihedral symmetry reduction, and helpers for contact and site
queries used by the energy, folding, and ligand modules.
"""

from typing import Iterator

Conformation = tuple[tuple[int, int], ...]

MOVES: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))


def enumerate_saws(n: int, *, reduce_symmetry: bool = True) -> Iterator[Conformation]:
    """Enumerate self-avoiding walks of ``n`` residues on the 2D square lattice.

    With ``reduce_symmetry=True`` (default), residue 0 is fixed at (0, 0)
    and residue 1 at (1, 0), and the first off-axis step is forced to +y.
    This removes the 8-fold dihedral symmetry of the lattice.

    With ``reduce_symmetry=False``, all SAWs starting at the origin are
    enumerated. The count then matches OEIS A001411 indexed by step count
    (steps = n - 1).
    """
    if n < 1:
        return
    if n == 1:
        yield ((0, 0),)
        return

    if reduce_symmetry:
        path = [(0, 0), (1, 0)]
        occupied = {(0, 0), (1, 0)}
        yield from _walk_reduced(path, occupied, n, y_locked=False)
    else:
        path = [(0, 0)]
        occupied = {(0, 0)}
        yield from _walk_unreduced(path, occupied, n)


def _walk_reduced(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    n: int,
    y_locked: bool,
) -> Iterator[Conformation]:
    if len(path) == n:
        yield tuple(path)
        return
    last_x, last_y = path[-1]
    for dx, dy in MOVES:
        # Reflection symmetry across the x-axis: until the walk first leaves
        # y=0, only allow +y deviations. This collapses each up/down mirror
        # pair to a single representative.
        if not y_locked and dy == -1:
            continue
        new_pos = (last_x + dx, last_y + dy)
        if new_pos in occupied:
            continue
        path.append(new_pos)
        occupied.add(new_pos)
        yield from _walk_reduced(path, occupied, n, y_locked or dy == 1)
        occupied.remove(new_pos)
        path.pop()


def _walk_unreduced(
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    n: int,
) -> Iterator[Conformation]:
    if len(path) == n:
        yield tuple(path)
        return
    last_x, last_y = path[-1]
    for dx, dy in MOVES:
        new_pos = (last_x + dx, last_y + dy)
        if new_pos in occupied:
            continue
        path.append(new_pos)
        occupied.add(new_pos)
        yield from _walk_unreduced(path, occupied, n)
        occupied.remove(new_pos)
        path.pop()


def get_contacts(conformation: Conformation) -> list[tuple[int, int]]:
    """Return non-bonded lattice-adjacent residue pairs ``(i, j)`` with ``j > i + 1``.

    Pairs are returned in lexicographic order.
    """
    contacts: list[tuple[int, int]] = []
    n = len(conformation)
    for i in range(n):
        xi, yi = conformation[i]
        for j in range(i + 2, n):
            xj, yj = conformation[j]
            if abs(xi - xj) + abs(yi - yj) == 1:
                contacts.append((i, j))
    return contacts


def is_self_avoiding(conformation: Conformation) -> bool:
    """True iff no two residues occupy the same lattice site."""
    return len(set(conformation)) == len(conformation)


def occupied_sites(conformation: Conformation) -> set[tuple[int, int]]:
    """Set of lattice sites occupied by the conformation."""
    return set(conformation)
