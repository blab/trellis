"""Exhaustive pre-enumeration folding for lattice proteins.

Pre-enumerate all self-avoiding walks for a given chain length and
(optional) ligand once, storing contact lists in CSR-like arrays.
Scoring a new sequence then reduces to summing MJ lookups over
precomputed contacts — no geometry discovery needed.

This is faster than branch-and-bound (``fold.py``) when many sequences
are folded on the same lattice+ligand configuration, which is exactly
the SSWM use case.
"""

from dataclasses import dataclass
from math import exp, inf
import warnings

import numpy as np

from trellis.energy import AA_INDEX
from trellis.fold import FoldResult
from trellis.lattice import Conformation, enumerate_saws, get_contacts
from trellis.ligand import Ligand, binding_contacts


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
    ligand_sites = ligand.sites if ligand is not None else frozenset()

    all_contact_pairs: list[tuple[int, int]] = []
    all_binding_pairs: list[tuple[int, int]] = []
    contact_offsets = [0]
    binding_offsets = [0]

    for conf in enumerate_saws(chain_length, reduce_symmetry=use_reduced):
        if ligand_sites and ligand_sites & set(conf):
            continue

        contacts = get_contacts(conf)
        all_contact_pairs.extend(contacts)
        contact_offsets.append(len(all_contact_pairs))

        if ligand is not None:
            bcontacts = binding_contacts(conf, ligand)
            all_binding_pairs.extend(bcontacts)
        binding_offsets.append(len(all_binding_pairs))

    n_conf = len(contact_offsets) - 1

    cp = np.array(all_contact_pairs, dtype=np.int32).reshape(-1, 2) if all_contact_pairs else np.empty((0, 2), dtype=np.int32)
    bp = np.array(all_binding_pairs, dtype=np.int32).reshape(-1, 2) if all_binding_pairs else np.empty((0, 2), dtype=np.int32)

    return ConformationDatabase(
        chain_length=chain_length,
        ligand=ligand,
        n_conformations=n_conf,
        contact_pairs=cp,
        contact_offsets=np.array(contact_offsets, dtype=np.int64),
        binding_pairs=bp,
        binding_offsets=np.array(binding_offsets, dtype=np.int64),
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


def fold(
    sequence: str,
    mj_matrix: np.ndarray,
    ligand: Ligand | None = None,
    temperature: float = 1.0,
    db: ConformationDatabase | None = None,
) -> FoldResult:
    """Fold *sequence* using pre-enumerated conformations.

    If *db* is None, conformations are enumerated on the fly (slow for
    repeated calls — pass a prebuilt database instead).
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

    aa_indices = [AA_INDEX[aa] for aa in sequence]
    lig_indices = [AA_INDEX[aa] for aa in ligand.sequence] if ligand is not None else []

    best_energy = inf
    best_idx = -1
    Z_sum = 0.0
    binding_weighted_sum = 0.0

    cp = db.contact_pairs
    co = db.contact_offsets
    bp = db.binding_pairs
    bo = db.binding_offsets

    for k in range(db.n_conformations):
        e_intra = 0.0
        for c in range(co[k], co[k + 1]):
            i, j = cp[c]
            e_intra += mj_matrix[aa_indices[i], aa_indices[j]]

        e_bind = 0.0
        for c in range(bo[k], bo[k + 1]):
            i, l = bp[c]
            e_bind += mj_matrix[aa_indices[i], lig_indices[l]]

        total = e_intra + e_bind
        boltz = exp(-total / temperature)
        Z_sum += boltz
        binding_weighted_sum += e_bind * boltz

        if total < best_energy:
            best_energy = total
            best_idx = k

    if db.reduced_symmetry:
        Z_full = Z_sum * 8 - 4
    else:
        Z_full = Z_sum

    ensemble_binding = binding_weighted_sum / Z_sum if Z_sum > 0 else 0.0

    native_conf = _recover_native_conformation(
        best_idx, db.chain_length, db.ligand, db.reduced_symmetry,
    )

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
