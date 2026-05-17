# Plan: Numba-Accelerated SAW Enumeration for 20-mers

## Problem

`enumerate_conformations()` is the bottleneck preventing 20-mer lattice proteins. It iterates through all self-avoiding walks in pure Python — recursive generators yielding tuples, set operations for collision detection, calling `get_contacts()` per conformation, appending to Python lists. For 16-mers (~3.3M conformations) this takes 36 seconds. For 20-mers (~177M conformations, 53× more), it would take ~30 minutes or more.

Jesse Bloom's `latticeproteins` solved this with C extensions. We'll use numba instead, which integrates cleanly with the existing `_score_conformations` JIT path.

The scoring loop (`_score_conformations`) is already numba-JIT and fast. Only the enumeration needs rewriting.

## Strategy

Replace `enumerate_conformations()` internals with a single `@numba.njit` function that:

1. Enumerates all SAWs using an explicit stack (no Python recursion)
2. Uses a numpy boolean grid for O(1) occupied-site and ligand-site checks (no Python sets)
3. Computes intra-protein contacts and protein-ligand contacts inline during walk generation (no separate `get_contacts()` / `binding_contacts()` calls)
4. Writes contact pairs directly into pre-allocated numpy arrays in CSR layout

The output arrays are the same ones that `_score_conformations` already consumes — `contact_pairs`, `contact_offsets`, `binding_pairs`, `binding_offsets`. No downstream changes needed.

## Data structures inside the numba function

### Occupied-site grid

A 2D boolean array centered so that any 20-mer walk fits. A 20-residue chain starting at (0,0) can reach at most ±19 in any direction. With ligand positions potentially at small negative y, a 41×41 grid centered at (20,20) covers everything:

```python
GRID_SIZE = 41
GRID_OFFSET = 20  # (0,0) maps to grid[20][20]

grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=numba.boolean)
```

Check: `grid[x + GRID_OFFSET, y + GRID_OFFSET]` — True means occupied. This replaces `new_pos in occupied` (Python set lookup) with a single array index.

### Ligand grid

A separate boolean array marking ligand-occupied sites:

```python
ligand_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=numba.boolean)
for pos in ligand_positions:
    ligand_grid[pos[0] + GRID_OFFSET, pos[1] + GRID_OFFSET] = True
```

Built once before the enumeration loop. Replaces `new_pos in ligand_sites`.

### Ligand positions array

For computing binding contacts, need actual (x,y) coordinates of ligand residues:

```python
ligand_pos = np.array(ligand_positions, dtype=np.int32)  # (n_ligand, 2)
n_ligand = len(ligand_positions)
```

### Walk path

Fixed-size array for the current walk:

```python
path_x = np.zeros(chain_length, dtype=np.int32)
path_y = np.zeros(chain_length, dtype=np.int32)
```

### Explicit stack for DFS

Replace Python recursion with an explicit stack. At each depth, we try moves 0–3 (the four directions). The stack tracks which move to try next at each depth:

```python
move_idx = np.zeros(chain_length, dtype=np.int32)  # next move to try at each depth
```

The DFS loop:
- Start at depth=start_depth (1 for unreduced with ligand, 2 for reduced without)
- At each depth, try `move_idx[depth]` through 3
- If a move is valid (not occupied, not ligand site, satisfies symmetry constraints), place the residue, compute new contacts, advance depth
- When depth == chain_length, record the conformation's contacts
- When all moves exhausted at a depth, backtrack

### Output arrays — dynamic growth

The total number of conformations and contacts isn't known in advance. Two approaches:

**Option A: Two-pass.** First pass counts conformations and contacts (no writes). Second pass allocates exact arrays and fills them. Doubles enumeration time but avoids reallocation. Simple and predictable.

**Option B: Pre-allocate with generous upper bounds, then trim.** Estimate upper bounds from known SAW counts (OEIS A001411), allocate, fill, then `np.resize` at the end. Faster but needs reliable upper bounds.

**Recommendation: Option A (two-pass).** The enumeration itself is the expensive part, but with numba it should complete in minutes even for 20-mers. Doubling a few-minute computation is acceptable. The code is much simpler — no risk of buffer overflows, no guessing at upper bounds, and the first pass is slightly faster than the second (no array writes). The two passes can share the same core DFS logic with a `count_only` flag.

Actually, an even cleaner variant: **single-pass with dynamically grown numpy arrays**. Numba doesn't support Python lists or dynamic arrays natively, but you can pre-allocate a large buffer and track a write cursor. If the buffer fills, the function returns what it has plus a flag indicating it was truncated, and the Python wrapper doubles the buffer and restarts. In practice with a 1.5× overestimate of the upper bound, this never triggers.

**Final recommendation: two-pass.** Simplest, most robust, and "2× a few minutes" is still far better than "hours in Python."

## Implementation

### New function: `_enumerate_numba`

```python
@numba.njit(cache=True)
def _enumerate_numba(
    chain_length: int,
    ligand_grid: np.ndarray,       # (GRID_SIZE, GRID_SIZE) bool
    ligand_pos: np.ndarray,        # (n_ligand, 2) int32, or (0, 2) if no ligand
    n_ligand: int,
    use_symmetry: bool,
    count_only: bool,
    # Output arrays (ignored if count_only=True):
    contact_pairs: np.ndarray,     # (max_contacts, 2) int32
    contact_offsets: np.ndarray,   # (max_conformations + 1,) int64
    binding_pairs: np.ndarray,     # (max_binding, 2) int32
    binding_offsets: np.ndarray,   # (max_binding + 1,) int64
) -> tuple[int, int, int]:
    """Enumerate SAWs and extract contacts.

    Returns (n_conformations, n_contact_pairs, n_binding_pairs).
    """
```

### DFS loop (pseudocode)

```
MOVES = [(0,1), (0,-1), (1,0), (-1,0)]
GRID_OFFSET = 20

# Initialize grid, path
grid[:,:] = False
path_x[0] = 0; path_y[0] = 0
grid[GRID_OFFSET, GRID_OFFSET] = True

if use_symmetry:
    # Fix residue 1 at (1,0)
    path_x[1] = 1; path_y[1] = 0
    grid[1 + GRID_OFFSET, GRID_OFFSET] = True
    start_depth = 2
    y_locked = False
else:
    start_depth = 1
    y_locked = True  # no symmetry filtering

# move_idx tracks which direction to try next at each depth
move_idx[:] = 0
depth = start_depth

# Also track y_locked per depth for symmetry reduction
y_locked_stack = np.zeros(chain_length, dtype=numba.boolean)
y_locked_stack[start_depth] = y_locked

conf_idx = 0
contact_cursor = 0
binding_cursor = 0

while depth >= start_depth:
    if move_idx[depth] >= 4:
        # Backtrack
        move_idx[depth] = 0
        depth -= 1
        if depth >= start_depth:
            # Un-place residue at depth
            grid[path_x[depth] + GRID_OFFSET, path_y[depth] + GRID_OFFSET] = False
        continue

    m = move_idx[depth]
    move_idx[depth] += 1  # advance for next iteration at this depth

    dx, dy = MOVES[m]

    # Symmetry check
    if use_symmetry and not y_locked_stack[depth] and dy == -1:
        continue

    nx = path_x[depth - 1] + dx
    ny = path_y[depth - 1] + dy
    gx = nx + GRID_OFFSET
    gy = ny + GRID_OFFSET

    # Bounds check (shouldn't be needed with GRID_SIZE=41, but safe)
    if gx < 0 or gx >= GRID_SIZE or gy < 0 or gy >= GRID_SIZE:
        continue

    # Collision check
    if grid[gx, gy] or ligand_grid[gx, gy]:
        continue

    # Place residue
    path_x[depth] = nx
    path_y[depth] = ny
    grid[gx, gy] = True

    if depth + 1 == chain_length:
        # Complete conformation — extract contacts
        if not count_only:
            # Intra-protein contacts: check all (i, j) with j > i+1
            for i in range(chain_length):
                for j in range(i + 2, chain_length):
                    if abs(path_x[i] - path_x[j]) + abs(path_y[i] - path_y[j]) == 1:
                        contact_pairs[contact_cursor, 0] = i
                        contact_pairs[contact_cursor, 1] = j
                        contact_cursor += 1
            contact_offsets[conf_idx + 1] = contact_cursor

            # Binding contacts: check protein residues vs ligand residues
            for i in range(chain_length):
                for k in range(n_ligand):
                    if abs(path_x[i] - ligand_pos[k, 0]) + abs(path_y[i] - ligand_pos[k, 1]) == 1:
                        binding_pairs[binding_cursor, 0] = i
                        binding_pairs[binding_cursor, 1] = k
                        binding_cursor += 1
            binding_offsets[conf_idx + 1] = binding_cursor

        conf_idx += 1

        # Un-place and continue (don't advance depth)
        grid[gx, gy] = False
        continue

    # Advance depth
    if use_symmetry:
        y_locked_stack[depth + 1] = y_locked_stack[depth] or (dy == 1)
    move_idx[depth + 1] = 0
    depth += 1
```

### Contact extraction optimization

The inner contact extraction loop is O(n²) per conformation. For 20-mers, that's 190 pair checks × 177M conformations = ~34 billion checks. This is a lot, but each check is a single Manhattan distance computation (2 subtracts + 2 abs + 1 add + 1 compare) — very fast in numba.

An optimization: maintain a running contact count during the walk. When placing residue at depth `d`, check only `d` against residues 0..d-2 for new contacts (not all pairs). Store cumulative contact count per depth on a stack. On backtrack, restore. This reduces work from O(n²) per complete conformation to O(n) per step, amortized across all conformations. Same for binding contacts.

This is a meaningful win for 20-mers. Implement it:

```
# When placing residue at depth d:
new_contacts = 0
for i in range(d - 1):  # skip d-1 (chain neighbor)
    if abs(path_x[i] - nx) + abs(path_y[i] - ny) == 1:
        if not count_only:
            contact_pairs[contact_cursor + new_contacts, 0] = i
            contact_pairs[contact_cursor + new_contacts, 1] = d
        new_contacts += 1
contact_count_stack[d] = contact_count_stack[d-1] + new_contacts

# Similarly for binding contacts
new_binding = 0
for k in range(n_ligand):
    if abs(nx - ligand_pos[k, 0]) + abs(ny - ligand_pos[k, 1]) == 1:
        if not count_only:
            binding_pairs[binding_cursor + new_binding, 0] = d
            binding_pairs[binding_cursor + new_binding, 1] = k
        new_binding += 1
binding_count_stack[d] = binding_count_stack[d-1] + new_binding

# Advance cursors
contact_cursor += new_contacts
binding_cursor += new_binding
```

On backtrack at depth d:
```
# Retract contact/binding cursors
n_retract_contacts = contact_count_stack[d] - contact_count_stack[d-1]
contact_cursor -= n_retract_contacts
n_retract_binding = binding_count_stack[d] - binding_count_stack[d-1]
binding_cursor -= n_retract_binding
```

Wait — this doesn't work for the CSR layout. The contact_pairs array needs contacts grouped by conformation with offsets marking boundaries. If we're writing contacts incrementally during the walk and retracting on backtrack, the cursor position at `depth == chain_length` is the right end of that conformation's contacts, but intermediate writes for non-complete conformations get overwritten on backtrack.

Actually, it does work. The contact_pairs cursor only advances as we go deeper and retracts on backtrack. When we reach a complete conformation (`depth + 1 == chain_length`), the cursor reflects all contacts accumulated along the current path. We record `contact_offsets[conf_idx + 1] = contact_cursor` and increment `conf_idx`. The contacts for this conformation are `contact_pairs[contact_offsets[conf_idx] : contact_offsets[conf_idx+1]]`. Then we backtrack (un-place the last residue, retract the cursor), and the next complete conformation will write its contacts starting at the retracted cursor position... which overlaps.

No — that's wrong. After recording a complete conformation, we need the contacts to persist. But on backtrack from depth `chain_length-1`, we retract the cursor, overwriting those contacts.

**Fix:** At leaf nodes, don't retract. Only retract on backtrack from non-leaf depths. Since the leaf (depth == chain_length - 1) is always immediately followed by backtrack (we un-place and continue), we need to handle it differently: at a leaf, commit the contacts (advance conf_idx, record offset) but do NOT retract the contact cursor for the leaf's contributions. Instead, only retract contacts added at depths < chain_length - 1.

This is getting complex. Simpler approach:

**At leaf nodes, compute all contacts from scratch (O(n²) per conformation) and write them sequentially.** Don't try to maintain incremental contacts during the walk. For 20-mers, the O(n²) = 190 checks per conformation is tiny compared to the walk enumeration itself. The 34 billion total checks will execute in ~30 seconds in numba (each check is ~1 ns). This is negligible compared to the walk enumeration.

**Use the incremental approach only for counting** (pass 1): maintain a running contact count on a stack to know how many contacts each depth adds, so we can compute `total_contacts` and `total_binding` for array allocation without doing O(n²) per conformation. Actually even simpler: in the count-only pass, just count conformations. Estimate contacts per conformation from the count pass as `avg_contacts ≈ 5` for 20-mers and multiply by 1.5 for safety margin. If the arrays fill up in pass 2, fall back to re-running with larger arrays.

**Simplest correct approach:** Two-pass. Pass 1 counts conformations AND total contacts (doing the O(n²) contact extraction but not writing). Pass 2 allocates exact arrays and writes. Both passes use the same DFS logic. The `count_only` flag controls whether arrays are written.

## Modified `enumerate_conformations`

```python
def enumerate_conformations(
    chain_length: int,
    ligand: Ligand | None = None,
) -> ConformationDatabase:
    use_reduced = ligand is None

    # Build ligand grid and positions array
    ligand_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    if ligand is not None:
        ligand_pos = np.array(ligand.positions, dtype=np.int32)
        n_ligand = len(ligand.positions)
        for x, y in ligand.positions:
            ligand_grid[x + GRID_OFFSET, y + GRID_OFFSET] = True
    else:
        ligand_pos = np.empty((0, 2), dtype=np.int32)
        n_ligand = 0

    # Pass 1: count conformations and contacts
    dummy = np.empty((0, 2), dtype=np.int32)
    dummy_offsets = np.empty(0, dtype=np.int64)
    n_conf, n_contacts, n_binding = _enumerate_numba(
        chain_length, ligand_grid, ligand_pos, n_ligand,
        use_symmetry=use_reduced,
        count_only=True,
        contact_pairs=dummy, contact_offsets=dummy_offsets,
        binding_pairs=dummy, binding_offsets=dummy_offsets,
    )

    # Pass 2: allocate exact arrays and fill
    contact_pairs = np.empty((n_contacts, 2), dtype=np.int32)
    contact_offsets = np.zeros(n_conf + 1, dtype=np.int64)
    binding_pairs = np.empty((n_binding, 2), dtype=np.int32)
    binding_offsets = np.zeros(n_conf + 1, dtype=np.int64)

    _enumerate_numba(
        chain_length, ligand_grid, ligand_pos, n_ligand,
        use_symmetry=use_reduced,
        count_only=False,
        contact_pairs=contact_pairs, contact_offsets=contact_offsets,
        binding_pairs=binding_pairs, binding_offsets=binding_offsets,
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
```

## Validation

### Correctness tests

For chain lengths 6, 8, 10, 12 (fast enough to cross-check):

1. **Conformation count.** `_enumerate_numba` with `count_only=True` must produce the same conformation count as the current Python `enumerate_conformations`. Test both with and without ligand.

2. **Contact arrays.** The CSR arrays from `_enumerate_numba` must produce identical `_score_conformations` results (native energy, Z, ⟨E_bind⟩) as the current Python path, for several random sequences. This is the definitive correctness test — if scoring produces the same answers, the contact lists are correct.

3. **SAW count vs OEIS.** Without ligand (symmetry-reduced), verify the conformation count matches OEIS A001411 / 8 (approximately, accounting for the straight-line walk edge case).

### Performance tests

Time `enumerate_conformations` for chain lengths 12, 14, 16, 18, 20:

| Chain | Python (current) | Numba (expected) |
|-------|-----------------|------------------|
| 12 | ~0.7s | <0.1s |
| 14 | ~6s | <0.5s |
| 16 | ~36s | <3s |
| 18 | ~5 min (est.) | <30s |
| 20 | ~30 min (est.) | <5 min |

The numba version should be 10–50× faster than pure Python. The exact speedup depends on how much overhead comes from Python object creation (tuples, sets, generator yields) in the current code vs. the raw DFS work.

## Changes to existing code

### `fold_enum.py`

- Add `GRID_SIZE = 41` and `GRID_OFFSET = 20` module constants
- Add `_enumerate_numba()` function with `@numba.njit(cache=True)`
- Replace body of `enumerate_conformations()` with the two-pass wrapper above
- Keep `_recover_native_conformation()` as-is (only called for visualization, doesn't need to be fast)
- Keep `_score_conformations()`, `fold()`, `save_database()`, `load_database()` unchanged

### No other files change

The `ConformationDatabase` dataclass, `fold()`, `_score_conformations()`, `save_database()`, `load_database()` — all stay exactly the same. The output arrays have the same format. Downstream code (`fitness.py`, `sswm.py`, `generate_trajectories.py`) sees no change.

## Numba constraints to follow

- No Python objects inside `@njit`: no strings, no tuples, no sets, no lists, no dicts
- All arrays must be numpy with explicit dtypes
- No `enumerate()`, no generators, no closures
- Recursion is supported in numba but explicit stack is faster and avoids stack overflow for deep recursion
- `np.bool_` for boolean arrays (not Python `bool`)
- Use `numba.int32`, `numba.int64` for typed local variables if needed
- The `MOVES` array should be defined as a module-level numpy constant:
  ```python
  _MOVES = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)
  ```

## Implementation order

1. Add `_enumerate_numba()` to `fold_enum.py` — the core numba DFS with explicit stack, grid-based collision, inline contact extraction, two-pass support via `count_only` flag
2. Replace `enumerate_conformations()` body with the two-pass wrapper
3. Run existing tests (`pytest tests/`) — everything should pass with identical results
4. Add a benchmark in `scripts/benchmark.py` that times enumeration for chain lengths 12–20
5. If 20-mer enumeration completes in <10 minutes, update `scripts/generate_trajectories.py` default `--chain-length` from 16 to 20 and update `production-run-timing.md`

## Memory estimate for 20-mer database

| Array | 20-mer estimate | Size |
|-------|----------------|------|
| contact_pairs (~5 avg × 177M) | ~885M pairs | ~7.1 GB |
| contact_offsets | 177M + 1 | ~1.4 GB |
| binding_pairs (~2 avg × 177M) | ~354M pairs | ~2.8 GB |
| binding_offsets | 177M + 1 | ~1.4 GB |
| **Total** | | **~12.7 GB** |

This fits on machines with 16+ GB RAM. The pass-1 count-only enumeration uses negligible memory (just the grid and path arrays). Pass-2 allocation requires the full ~13 GB. The saved `.npz` will compress significantly (contact indices have structure).
