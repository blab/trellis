# Exhaustive Pre-Enumeration: Alternative Folding Strategy

Trevor Bedford — 2026-05-07

## Motivation

The main implementation plan uses branch-and-bound with energy pruning to fold each sequence independently (Step 3, `fold.py`). Jesse Bloom's [latticeproteins](https://github.com/jbloomlab/latticeproteins) package takes a fundamentally different approach: pre-enumerate all self-avoiding walks for a given chain length once, store them, then for each new sequence just score every stored conformation.

The key insight is that the set of possible conformations is purely geometric — it depends on chain length and lattice obstacles (ligand sites), not on the amino acid sequence. You enumerate once, then scoring each new sequence is a linear scan over stored conformations computing contact energies. For our SSWM use case where we fold ~60 unique AA sequences per step on the same lattice with the same ligand, this could be significantly faster than re-discovering the geometry 60 times via branch-and-bound.

This document describes how to implement the pre-enumeration strategy for a head-to-head speed comparison against branch-and-bound after Steps 1–3 of the main plan are complete.

## Architecture

Add a single new module alongside `fold.py`:

```
trellis/
├── trellis/
│   ├── fold.py                 # Branch-and-bound (existing, from main plan Steps 1–3)
│   ├── fold_enum.py            # Exhaustive pre-enumeration (this document)
│   └── ...
├── scripts/
│   ├── benchmark_folding.py    # Head-to-head speed comparison
│   └── ...
```

Both `fold.py` and `fold_enum.py` should expose the same `FoldResult` return type so that downstream code (`fitness.py`, `sswm.py`) can use either interchangeably. The entry point for each is a `fold()` function with the same signature.

## Core idea

**Bloom's approach, adapted for ligand binding:**

1. Enumerate all self-avoiding walks of length N on the 2D lattice, excluding ligand-occupied sites
2. For each walk, precompute two things:
   - The intra-protein contact list: pairs (i, j) where |i-j| > 1 and residues are lattice-adjacent
   - The protein-ligand contact list: pairs (protein_residue_index, ligand_residue_index) that are lattice-adjacent
3. Store all walks with their contact lists in a `ConformationDatabase`
4. To fold a sequence: iterate over all stored conformations, compute E_intra + E_binding for each using the MJ matrix, accumulate Z and ⟨E_binding⟩

Steps 1–3 are a one-time cost. Step 4 is per-sequence and should be fast — each conformation requires only summing MJ lookups over its (precomputed) contact list.

## Implementation: `fold_enum.py`

### Data structures

```python
@dataclass
class ConformationDatabase:
    """Pre-enumerated conformations for a given chain length and ligand configuration."""
    chain_length: int
    ligand: Ligand | None
    n_conformations: int

    # Stored as numpy arrays for fast vectorized scoring
    # contact_pairs: (total_contacts, 2) array of (residue_i, residue_j) indices
    # contact_offsets: (n_conformations + 1,) array — conformation k has contacts
    #   contact_pairs[contact_offsets[k]:contact_offsets[k+1]]
    contact_pairs: np.ndarray
    contact_offsets: np.ndarray

    # Same structure for protein-ligand contacts
    # binding_pairs: (total_binding_contacts, 2) array of (protein_residue_idx, ligand_residue_idx)
    binding_pairs: np.ndarray
    binding_offsets: np.ndarray
```

This CSR-like (compressed sparse row) layout stores all contact lists in contiguous arrays. Each conformation's contacts are a slice of the global array, indexed by offsets. This is both memory-efficient and numba-friendly.

### Key functions

- `enumerate_conformations(chain_length: int, ligand: Ligand | None = None) -> ConformationDatabase`: Enumerate all SAWs of the given length, avoiding ligand-occupied sites. Precompute intra-protein and protein-ligand contact lists. Fix first residue at origin and first step to (1,0) for symmetry reduction (factor of 8). Return a `ConformationDatabase`.

- `fold(sequence: str, mj_matrix: np.ndarray, ligand: Ligand | None = None, temperature: float = 1.0, db: ConformationDatabase | None = None) -> FoldResult`: Fold a sequence using pre-enumerated conformations. If `db` is None, call `enumerate_conformations` first (and warn about performance). Returns the same `FoldResult` as `fold.py` — native conformation, native energy, partition function, ensemble-averaged binding energy, conformations enumerated.

- `save_database(db: ConformationDatabase, path: str)`: Serialize to disk (numpy `.npz` format).

- `load_database(path: str) -> ConformationDatabase`: Load from disk.

### Enumeration algorithm

```
function enumerate_conformations(chain_length, ligand):
    contact_pairs_list = []
    binding_pairs_list = []
    contact_offsets = [0]
    binding_offsets = [0]

    ligand_sites = ligand.sites if ligand else set()

    function enumerate(path, occupied, depth):
        if depth == chain_length:
            # Complete walk — extract contacts
            intra_contacts = []
            for i in range(chain_length):
                for j in range(i + 2, chain_length):
                    if manhattan_distance(path[i], path[j]) == 1:
                        intra_contacts.append((i, j))

            bind_contacts = []
            if ligand:
                for i in range(chain_length):
                    for k in range(len(ligand.sequence)):
                        if manhattan_distance(path[i], ligand.positions[k]) == 1:
                            bind_contacts.append((i, k))

            contact_pairs_list.extend(intra_contacts)
            contact_offsets.append(contact_offsets[-1] + len(intra_contacts))
            binding_pairs_list.extend(bind_contacts)
            binding_offsets.append(binding_offsets[-1] + len(bind_contacts))
            return

        for move in MOVES:
            new_pos = path[-1] + move
            if new_pos not in occupied and new_pos not in ligand_sites:
                enumerate(path + [new_pos], occupied | {new_pos}, depth + 1)

    # Symmetry reduction: fix first residue at origin, first step to (1,0)
    start = [(0,0), (1,0)]
    enumerate(start, {(0,0), (1,0)}, 2)

    return ConformationDatabase(
        chain_length=chain_length,
        ligand=ligand,
        n_conformations=len(contact_offsets) - 1,
        contact_pairs=np.array(contact_pairs_list, dtype=np.int32),
        contact_offsets=np.array(contact_offsets, dtype=np.int64),
        binding_pairs=np.array(binding_pairs_list, dtype=np.int32),
        binding_offsets=np.array(binding_offsets, dtype=np.int64),
    )
```

### Scoring algorithm

```
function fold(sequence, mj_matrix, ligand, temperature, db):
    aa_indices = [AA_TO_INDEX[aa] for aa in sequence]
    ligand_indices = [AA_TO_INDEX[aa] for aa in ligand.sequence] if ligand else []

    best_energy = +infinity
    best_conf_idx = -1
    Z = 0.0
    binding_weighted_sum = 0.0

    for k in range(db.n_conformations):
        # Compute intra-protein energy from precomputed contacts
        e_intra = 0.0
        for c in range(db.contact_offsets[k], db.contact_offsets[k+1]):
            i, j = db.contact_pairs[c]
            e_intra += mj_matrix[aa_indices[i], aa_indices[j]]

        # Compute binding energy from precomputed ligand contacts
        e_bind = 0.0
        for c in range(db.binding_offsets[k], db.binding_offsets[k+1]):
            i, k_lig = db.binding_pairs[c]
            e_bind += mj_matrix[aa_indices[i], ligand_indices[k_lig]]

        total = e_intra + e_bind
        boltzmann = exp(-total / temperature)
        Z += boltzmann
        binding_weighted_sum += e_bind * boltzmann

        if total < best_energy:
            best_energy = total
            best_conf_idx = k

    # Multiply Z and binding_weighted_sum by 8 for symmetry
    ensemble_binding = binding_weighted_sum / Z
    return FoldResult(
        native_conformation=reconstruct_coords(best_conf_idx, db),  # optional
        native_energy=best_energy,
        partition_function=Z * 8,
        ensemble_binding_energy=ensemble_binding,
        n_conformations_enumerated=db.n_conformations,
    )
```

### Numba acceleration of scoring

The scoring inner loop is the hot path and should be JIT-compiled:

```python
@numba.njit
def score_all_conformations(
    aa_indices,           # (chain_length,) int32
    ligand_indices,       # (ligand_length,) int32
    mj_matrix,            # (20, 20) float64
    contact_pairs,        # (total_contacts, 2) int32
    contact_offsets,      # (n_conformations + 1,) int64
    binding_pairs,        # (total_binding_contacts, 2) int32
    binding_offsets,      # (n_conformations + 1,) int64
    temperature,          # float64
) -> tuple[float, float, float, int]:
    """Returns (best_energy, Z, binding_weighted_sum, best_conf_idx)."""
    ...
```

This is a tight loop over precomputed integer arrays with MJ matrix lookups — ideal for numba. No recursion, no set operations, no Python objects. Expected to be very fast.

### Memory estimates

For a 20-residue chain on a 2D square lattice:

| Quantity | Estimate | Notes |
|----------|----------|-------|
| SAWs (symmetry-reduced) | ~60M | Without ligand; ligand reduces this |
| Avg contacts per conformation | ~4–6 | For 20-mers |
| Total contact pairs | ~300M | 60M × 5 avg |
| Contact pairs array | ~2.4 GB | 300M × 2 × 4 bytes (int32) |
| Contact offsets array | ~480 MB | 60M × 8 bytes (int64) |
| Binding pairs (similar scale) | ~1–2 GB | Fewer binding than intra contacts |
| **Total** | **~5–7 GB** | Without ligand |

With a 4-residue ligand blocking lattice sites, the SAW count drops substantially (the ligand blocks walk paths near the origin). Rough estimate: 30-50% reduction, bringing total memory to ~3–5 GB.

This is feasible on machines with 16+ GB RAM. For the 8×H100 server (which will have hundreds of GB), it's trivial. For a laptop with 16 GB, it's tight but workable.

### Disk storage

The `ConformationDatabase` should be saved to disk after enumeration so it doesn't need to be recomputed. Use numpy's `.npz` format:

```python
def save_database(db, path):
    np.savez_compressed(path,
        contact_pairs=db.contact_pairs,
        contact_offsets=db.contact_offsets,
        binding_pairs=db.binding_pairs,
        binding_offsets=db.binding_offsets,
        chain_length=db.chain_length,
        # ligand info as metadata
    )
```

Expected compressed size on disk: ~1–2 GB. Loading from disk is fast (numpy memory-mapping).

### Optional: storing coordinates

The scoring algorithm above doesn't need full (x,y) coordinates — it only needs precomputed contact lists. However, coordinates are useful for:

1. `visualize_conformation.py` (rendering the native conformation)
2. Debugging / validation

Two options:

**Option A: Don't store coordinates.** Reconstruct coordinates on demand by re-running the SAW enumeration and stopping at the k-th walk. This is slow but rarely needed (only for visualization of the single native conformation).

**Option B: Store coordinates for all conformations.** Adds ~4.8 GB (60M × 20 × 2 × 4 bytes). Total memory ~10–12 GB. More convenient but expensive.

**Recommendation:** Option A. Store only contact lists for scoring. When coordinates are needed (visualization, debugging), either re-enumerate to find the k-th walk, or maintain a small cache of "interesting" conformations (native states of recently folded sequences).

## Benchmark script: `scripts/benchmark_folding.py`

### Purpose

Head-to-head comparison of branch-and-bound (`fold.py`) vs exhaustive pre-enumeration (`fold_enum.py`).

### Usage

```bash
# Run benchmark with default settings
python scripts/benchmark_folding.py --chain-length 20 --n-sequences 100

# Include ligand
python scripts/benchmark_folding.py --chain-length 20 --n-sequences 100 --ligand-sequence FWYL

# Shorter chains for quick validation
python scripts/benchmark_folding.py --chain-length 12 --n-sequences 50
```

### What it measures

1. **Enumeration time** (pre-enumeration only): wall time to build the `ConformationDatabase` for the given chain length and ligand
2. **Per-sequence fold time**: wall time to fold a single random sequence, averaged over `--n-sequences` random sequences. Measured for both strategies
3. **Batch fold time**: wall time to fold all `--n-sequences` sequences. This is the SSWM-relevant metric — how long does it take to evaluate a full mutational neighborhood?
4. **Memory usage**: peak RSS for each strategy
5. **Correctness check**: verify both strategies produce identical `FoldResult` values (same native energy, same Z, same ⟨E_binding⟩ within floating-point tolerance)

### Output

Print a comparison table:

```
Chain length: 20, Ligand: FWYL, Sequences: 100

                        Branch-and-bound    Pre-enumeration
Enumeration time        N/A                 XXX sec
Per-sequence fold       XXX sec             XXX sec
Batch fold (100 seq)    XXX sec             XXX sec
Peak memory             XXX MB              XXX MB
Speedup (batch)         1.0×                XXX×

Correctness: ALL PASSED (100/100 sequences match)
```

### Scaling sweep

Additionally, run a sweep over chain lengths 8, 10, 12, 14, 16, 18, 20 to show how each strategy scales. This helps identify the crossover point where pre-enumeration becomes advantageous.

```bash
python scripts/benchmark_folding.py --sweep --n-sequences 20
```

## Implementation order

This should be implemented after Steps 1–3 of the main plan are complete and tested:

1. **`fold_enum.py` — enumeration**: Implement `enumerate_conformations`. Validate by comparing SAW counts against `lattice.py`'s `enumerate_saws` for small chains (n ≤ 12). Verify contact lists match `lattice.py`'s `get_contacts`.

2. **`fold_enum.py` — scoring**: Implement the numba-accelerated `score_all_conformations`. Validate by comparing `FoldResult` against `fold.py` (branch-and-bound) on small chains where both are fast.

3. **`fold_enum.py` — persistence**: Implement `save_database` / `load_database`.

4. **`scripts/benchmark_folding.py`**: Run the head-to-head comparison. Start with small chains (12–14) to validate correctness, then scale up to 20.

5. **Decision**: Based on benchmark results, choose the default folding strategy for `fitness.py`. The winner becomes the default; the loser remains available as an alternative.

## Expected outcome

Branch-and-bound will likely be faster for one-off folds of individual sequences (no enumeration overhead). Pre-enumeration will likely be faster for batch folds (many sequences on the same lattice), which is the SSWM use case. The crossover point depends on how many sequences you fold before the enumeration cost is amortized.

If pre-enumeration wins decisively for the batch case (which I expect), `fitness.py` should accept an optional `ConformationDatabase` parameter and construct it once at the start of trajectory generation, passing it through to all subsequent fold calls. This replaces the per-sequence branch-and-bound with a single upfront enumeration plus fast per-sequence scoring.
