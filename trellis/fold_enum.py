"""Exhaustive pre-enumeration folding for lattice proteins.

Pre-enumerate all self-avoiding walks for a given chain length and
(optional) ligand once, storing contact lists in CSR-like arrays.
Scoring a new sequence then reduces to summing MJ lookups over
precomputed contacts — no geometry discovery needed.

This is faster than branch-and-bound (``fold_bb.py``) when many sequences
are folded on the same lattice+ligand configuration, which is exactly
the SSWM use case.
"""

from dataclasses import dataclass
from math import exp, inf
import warnings

import numba
import numpy as np

from trellis.energy import AA_INDEX
from trellis.fold_bb import FoldResult
from trellis.lattice import Conformation, enumerate_saws
from trellis.ligand import Ligand

_GRID_SIZE = 41
_GRID_OFFSET = 20
_MOVES = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)


@numba.njit(cache=True)
def _enumerate_numba(
    chain_length,
    ligand_grid,
    ligand_pos,
    n_ligand,
    use_symmetry,
    count_only,
    contact_pairs,
    contact_offsets,
    binding_pairs,
    binding_offsets,
):
    """Enumerate SAWs via explicit-stack DFS and extract contacts.

    Returns (n_conformations, n_contact_pairs, n_binding_pairs).
    """
    grid = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=numba.boolean)
    path_x = np.zeros(chain_length, dtype=np.int32)
    path_y = np.zeros(chain_length, dtype=np.int32)

    # Place residue 0 at origin
    path_x[0] = 0
    path_y[0] = 0
    grid[_GRID_OFFSET, _GRID_OFFSET] = True

    if use_symmetry and chain_length >= 2:
        path_x[1] = 1
        path_y[1] = 0
        grid[1 + _GRID_OFFSET, _GRID_OFFSET] = True
        start_depth = 2
    elif not use_symmetry:
        start_depth = 1
    else:
        # chain_length == 1, reduced: single residue at origin
        start_depth = chain_length

    conf_idx = 0
    contact_cursor = 0
    binding_cursor = 0

    # Handle case where pre-placed residues already form a complete walk
    if start_depth >= chain_length:
        # Extract contacts for the pre-placed path
        n_new_contacts = 0
        for i in range(chain_length):
            for j in range(i + 2, chain_length):
                if abs(path_x[i] - path_x[j]) + abs(path_y[i] - path_y[j]) == 1:
                    if not count_only:
                        contact_pairs[contact_cursor + n_new_contacts, 0] = i
                        contact_pairs[contact_cursor + n_new_contacts, 1] = j
                    n_new_contacts += 1
        contact_cursor += n_new_contacts

        n_new_binding = 0
        for i in range(chain_length):
            for k in range(n_ligand):
                if abs(path_x[i] - ligand_pos[k, 0]) + abs(path_y[i] - ligand_pos[k, 1]) == 1:
                    if not count_only:
                        binding_pairs[binding_cursor + n_new_binding, 0] = i
                        binding_pairs[binding_cursor + n_new_binding, 1] = k
                    n_new_binding += 1
        binding_cursor += n_new_binding

        if not count_only:
            contact_offsets[conf_idx + 1] = contact_cursor
            binding_offsets[conf_idx + 1] = binding_cursor
        conf_idx += 1
        return conf_idx, contact_cursor, binding_cursor

    # DFS with explicit stack
    move_idx = np.zeros(chain_length, dtype=np.int32)
    y_locked_stack = np.zeros(chain_length, dtype=numba.boolean)

    if use_symmetry:
        y_locked_stack[start_depth] = False
    else:
        y_locked_stack[start_depth] = True  # no symmetry filtering

    depth = start_depth
    move_idx[depth] = 0

    while depth >= start_depth:
        if move_idx[depth] >= 4:
            move_idx[depth] = 0
            depth -= 1
            if depth >= start_depth:
                gx = path_x[depth] + _GRID_OFFSET
                gy = path_y[depth] + _GRID_OFFSET
                grid[gx, gy] = False
            continue

        m = move_idx[depth]
        move_idx[depth] += 1

        dx = _MOVES[m, 0]
        dy = _MOVES[m, 1]

        if use_symmetry and not y_locked_stack[depth] and dy == -1:
            continue

        nx = path_x[depth - 1] + dx
        ny = path_y[depth - 1] + dy
        gx = nx + _GRID_OFFSET
        gy = ny + _GRID_OFFSET

        if gx < 0 or gx >= _GRID_SIZE or gy < 0 or gy >= _GRID_SIZE:
            continue

        if grid[gx, gy] or ligand_grid[gx, gy]:
            continue

        # Place residue at depth
        path_x[depth] = nx
        path_y[depth] = ny
        grid[gx, gy] = True

        if depth + 1 == chain_length:
            # Leaf: complete conformation — extract contacts
            n_new_contacts = 0
            for i in range(chain_length):
                for j in range(i + 2, chain_length):
                    if abs(path_x[i] - path_x[j]) + abs(path_y[i] - path_y[j]) == 1:
                        if not count_only:
                            contact_pairs[contact_cursor + n_new_contacts, 0] = i
                            contact_pairs[contact_cursor + n_new_contacts, 1] = j
                        n_new_contacts += 1
            contact_cursor += n_new_contacts

            n_new_binding = 0
            for i in range(chain_length):
                for k in range(n_ligand):
                    if abs(path_x[i] - ligand_pos[k, 0]) + abs(path_y[i] - ligand_pos[k, 1]) == 1:
                        if not count_only:
                            binding_pairs[binding_cursor + n_new_binding, 0] = i
                            binding_pairs[binding_cursor + n_new_binding, 1] = k
                        n_new_binding += 1
            binding_cursor += n_new_binding

            if not count_only:
                contact_offsets[conf_idx + 1] = contact_cursor
                binding_offsets[conf_idx + 1] = binding_cursor
            conf_idx += 1

            grid[gx, gy] = False
            continue

        # Advance to next depth
        if use_symmetry:
            y_locked_stack[depth + 1] = y_locked_stack[depth] or (dy == 1)
        else:
            y_locked_stack[depth + 1] = True
        move_idx[depth + 1] = 0
        depth += 1

    return conf_idx, contact_cursor, binding_cursor


@dataclass
class ConformationDatabase:
    """Pre-enumerated conformations for a given chain length and ligand."""

    chain_length: int
    ligand: Ligand | None
    n_conformations: int
    contact_pairs: np.ndarray       # (total_contacts, 2) int32
    contact_offsets: np.ndarray     # (n_conformations + 1,) int64
    binding_pairs: np.ndarray       # (total_binding_contacts, 2) int32
    binding_offsets: np.ndarray     # (n_conformations + 1,) int64
    reduced_symmetry: bool          # True if SAWs were symmetry-reduced


def enumerate_conformations(
    chain_length: int,
    ligand: Ligand | None = None,
) -> ConformationDatabase:
    """Build a ConformationDatabase for all SAWs of *chain_length*.

    With a ligand, walks that collide with ligand sites are skipped and
    unreduced enumeration is used (symmetry is broken by the ligand).
    Without a ligand, symmetry-reduced enumeration is used.
    """
    use_reduced = ligand is None

    ligand_grid = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=np.bool_)
    if ligand is not None:
        ligand_pos = np.array(ligand.positions, dtype=np.int32)
        n_ligand = len(ligand.positions)
        for x, y in ligand.positions:
            ligand_grid[x + _GRID_OFFSET, y + _GRID_OFFSET] = True
    else:
        ligand_pos = np.empty((0, 2), dtype=np.int32)
        n_ligand = 0

    # Pass 1: count conformations and contacts
    dummy = np.empty((0, 2), dtype=np.int32)
    dummy_off = np.empty(0, dtype=np.int64)
    n_conf, n_contacts, n_binding = _enumerate_numba(
        chain_length, ligand_grid, ligand_pos, n_ligand,
        use_reduced, True, dummy, dummy_off, dummy, dummy_off,
    )

    # Pass 2: allocate exact arrays and fill
    contact_pairs = np.empty((n_contacts, 2), dtype=np.int32)
    contact_offsets = np.zeros(n_conf + 1, dtype=np.int64)
    binding_pairs = np.empty((n_binding, 2), dtype=np.int32)
    binding_offsets = np.zeros(n_conf + 1, dtype=np.int64)

    _enumerate_numba(
        chain_length, ligand_grid, ligand_pos, n_ligand,
        use_reduced, False,
        contact_pairs, contact_offsets, binding_pairs, binding_offsets,
    )

    return ConformationDatabase(
        chain_length=chain_length,
        ligand=ligand,
        n_conformations=n_conf,
        contact_pairs=contact_pairs,
        contact_offsets=contact_offsets,
        binding_pairs=binding_pairs,
        binding_offsets=binding_offsets,
        reduced_symmetry=use_reduced,
    )


def _recover_native_conformation(
    best_idx: int,
    chain_length: int,
    ligand: Ligand | None,
    reduced_symmetry: bool,
) -> Conformation:
    """Re-enumerate SAWs and return the conformation at index *best_idx*."""
    ligand_sites = ligand.sites if ligand is not None else frozenset()
    idx = 0
    for conf in enumerate_saws(chain_length, reduce_symmetry=reduced_symmetry):
        if ligand_sites and ligand_sites & set(conf):
            continue
        if idx == best_idx:
            return conf
        idx += 1
    raise RuntimeError(f"conformation index {best_idx} out of range")


@numba.njit(cache=True)
def _score_conformations(
    aa_indices: np.ndarray,
    lig_indices: np.ndarray,
    mj_matrix: np.ndarray,
    contact_pairs: np.ndarray,
    contact_offsets: np.ndarray,
    binding_pairs: np.ndarray,
    binding_offsets: np.ndarray,
    temperature: float,
) -> tuple[float, float, float, int]:
    """Score all conformations. Returns (best_energy, Z_sum, binding_weighted_sum, best_idx)."""
    best_energy = np.inf
    best_idx = -1
    Z_sum = 0.0
    binding_weighted_sum = 0.0
    n_conf = len(contact_offsets) - 1

    for k in range(n_conf):
        e_intra = 0.0
        for c in range(contact_offsets[k], contact_offsets[k + 1]):
            i = contact_pairs[c, 0]
            j = contact_pairs[c, 1]
            e_intra += mj_matrix[aa_indices[i], aa_indices[j]]

        e_bind = 0.0
        for c in range(binding_offsets[k], binding_offsets[k + 1]):
            i = binding_pairs[c, 0]
            l = binding_pairs[c, 1]
            e_bind += mj_matrix[aa_indices[i], lig_indices[l]]

        total = e_intra + e_bind
        boltz = np.exp(-total / temperature)
        Z_sum += boltz
        binding_weighted_sum += e_bind * boltz

        if total < best_energy:
            best_energy = total
            best_idx = k

    return best_energy, Z_sum, binding_weighted_sum, best_idx


def fold(
    sequence: str,
    mj_matrix: np.ndarray,
    ligand: Ligand | None = None,
    temperature: float = 1.0,
    db: ConformationDatabase | None = None,
    recover_conformation: bool = True,
) -> FoldResult:
    """Fold *sequence* using pre-enumerated conformations.

    If *db* is None, conformations are enumerated on the fly (slow for
    repeated calls — pass a prebuilt database instead).

    Set *recover_conformation* to False to skip the costly re-enumeration
    of SAWs to find native coordinates. The returned
    ``native_conformation`` will be ``()``.  Use this when only fitness
    (ensemble binding energy) is needed.
    """
    n = len(sequence)
    if n < 1:
        raise ValueError("sequence must have at least 1 residue")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if db is None:
        warnings.warn(
            "fold_enum.fold() called without a ConformationDatabase — "
            "building one on the fly. For repeated calls, precompute with "
            "enumerate_conformations().",
            stacklevel=2,
        )
        db = enumerate_conformations(n, ligand)

    if db.chain_length != n:
        raise ValueError(
            f"database chain_length {db.chain_length} != sequence length {n}"
        )

    aa_idx = np.array([AA_INDEX[aa] for aa in sequence], dtype=np.int32)
    lig_idx = (
        np.array([AA_INDEX[aa] for aa in ligand.sequence], dtype=np.int32)
        if ligand is not None
        else np.empty(0, dtype=np.int32)
    )

    best_energy, Z_sum, binding_weighted_sum, best_idx = _score_conformations(
        aa_idx, lig_idx, mj_matrix,
        db.contact_pairs, db.contact_offsets,
        db.binding_pairs, db.binding_offsets,
        temperature,
    )

    if db.reduced_symmetry:
        Z_full = Z_sum * 8 - 4
    else:
        Z_full = Z_sum

    ensemble_binding = binding_weighted_sum / Z_sum if Z_sum > 0 else 0.0

    if recover_conformation:
        native_conf = _recover_native_conformation(
            best_idx, db.chain_length, db.ligand, db.reduced_symmetry,
        )
    else:
        native_conf = ()

    return FoldResult(
        native_conformation=native_conf,
        native_energy=best_energy,
        partition_function=Z_full,
        ensemble_binding_energy=ensemble_binding,
        n_conformations_enumerated=db.n_conformations,
    )


def save_database(db: ConformationDatabase, path: str) -> None:
    """Save a ConformationDatabase to disk as a compressed .npz file."""
    meta = {
        "chain_length": np.array(db.chain_length),
        "n_conformations": np.array(db.n_conformations),
        "reduced_symmetry": np.array(db.reduced_symmetry),
        "contact_pairs": db.contact_pairs,
        "contact_offsets": db.contact_offsets,
        "binding_pairs": db.binding_pairs,
        "binding_offsets": db.binding_offsets,
    }
    if db.ligand is not None:
        meta["ligand_sequence"] = np.array(list(db.ligand.sequence))
        meta["ligand_positions"] = np.array(db.ligand.positions, dtype=np.int32)
    np.savez_compressed(path, **meta)


def load_database(path: str) -> ConformationDatabase:
    """Load a ConformationDatabase from a .npz file."""
    data = np.load(path, allow_pickle=False)

    ligand = None
    if "ligand_sequence" in data:
        from trellis.ligand import Ligand
        seq = "".join(data["ligand_sequence"])
        positions = tuple(tuple(int(x) for x in row) for row in data["ligand_positions"])
        ligand = Ligand(
            sequence=seq,
            positions=positions,
            sites=frozenset(positions),
        )

    return ConformationDatabase(
        chain_length=int(data["chain_length"]),
        ligand=ligand,
        n_conformations=int(data["n_conformations"]),
        contact_pairs=data["contact_pairs"],
        contact_offsets=data["contact_offsets"],
        binding_pairs=data["binding_pairs"],
        binding_offsets=data["binding_offsets"],
        reduced_symmetry=bool(data["reduced_symmetry"]),
    )
