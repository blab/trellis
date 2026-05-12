# Plan: Implement `trellis/ligand.py` and integrate into `fold.py`

## Context

Steps 1–3 (lattice geometry, MJ energy, branch-and-bound folding) are complete. Step 4 adds a fixed ligand on the lattice and protein-ligand binding energy, then wires it into `fold()` so it returns a meaningful `ensemble_binding_energy`. Required before Step 5 (fitness = −⟨E_binding⟩). Spec: `notes/lattice-protein-implementation-plan.md:200–238`.

## Architecture

```
┌──────────────┐
│  ligand.py   │  NEW — Ligand dataclass, create_ligand, binding_contacts, binding_energy
│  (standalone) │
└──────┬───────┘
       │ imported by
       ▼
┌──────────────┐
│   fold.py    │  MODIFIED — ligand collision, binding energy in B&B, unreduced enumeration
│              │  when ligand present, ensemble_binding_energy accumulation
└──────────────┘
```

Key design constraint: the ligand breaks 8-fold dihedral symmetry. When `ligand is not None`, `fold()` must use **unreduced enumeration** (all walks from origin, no y-lock, no `Z × 8 − 4` correction). The existing `_recurse` already supports this — `y_locked=True` disables the symmetry filter.

---

## 1. Create `trellis/ligand.py` (~60 lines)

### `Ligand` dataclass
```python
@dataclass(frozen=True)
class Ligand:
    sequence: str
    positions: tuple[tuple[int, int], ...]
    sites: frozenset[tuple[int, int]]       # frozenset for immutability + hashability
```
Spec says `set`, but `frozenset` is better for a frozen dataclass. `sites` is the same points as `positions`, stored as a set for O(1) collision lookup.

### `create_ligand(sequence, anchor=(0,-1), direction="horizontal") -> Ligand`
- Validate: `direction` in `{"horizontal", "vertical"}`, `sequence` non-empty, all chars in `AA_INDEX`.
- Horizontal: `(anchor[0]+i, anchor[1])` for each residue i.
- Vertical: `(anchor[0], anchor[1]+i)` for each residue i.

### `binding_contacts(conformation, ligand) -> list[tuple[int, int]]`
- O(n×k) loop: for each protein residue `i` and ligand residue `k`, check Manhattan distance == 1.
- Return `(i, k)` pairs in lexicographic order.
- Same pattern as `lattice.py:89` `get_contacts()` but inter-molecular.

### `binding_energy(protein_sequence, conformation, ligand, mj_matrix) -> float`
- Sum `mj_matrix[AA_INDEX[protein_seq[i]], AA_INDEX[ligand.sequence[k]]]` over contacts.
- Same pattern as `energy.py:74` `conformation_energy()`.

### Imports
```python
from trellis.lattice import Conformation
from trellis.energy import AA_INDEX
```
`numpy` imported for `mj_matrix` type annotation only; computation is pure Python (sum of floats).

---

## 2. Modify `trellis/fold.py`

Changes listed in execution order through the file:

### 2a. Add import (after line 21)
```python
from trellis.ligand import Ligand, binding_energy as ligand_binding_energy
```
Alias avoids shadowing the local energy variable names.

### 2b. Update type annotation (line 38)
`ligand: object | None = None` → `ligand: Ligand | None = None`

### 2c. Update docstring (lines 49–50)
Change "Reserved for Step 4" to describe the ligand parameter.

### 2d. Replace NotImplementedError (lines 58–59)
```python
ligand_sites = ligand.sites if ligand is not None else frozenset()
```

### 2e. Handle n=1 with ligand (lines 67–74)
If `ligand is not None`, compute binding energy for single residue at (0,0) and return accordingly. Without ligand, unchanged.

### 2f. Add binding pruning bound (after line 77)
```python
binding_bound = (4 * len(ligand.sequence) * mj_min) if ligand is not None else 0.0
```
Each of the k ligand residues has at most 4 lattice neighbors → max `4k` binding contacts, each ≥ `mj_min`.

### 2g. Add accumulator (line 82)
Add `binding_weighted_sum = 0.0` and include in the `nonlocal` declaration (line 88).

### 2h. Split start path by ligand presence (lines 84–86)
```python
if ligand is not None:
    path = [(0, 0)]
    occupied = {(0, 0)} | ligand_sites   # ligand sites excluded from walk
else:
    path = [(0, 0), (1, 0)]
    occupied = {(0, 0), (1, 0)}
```

Wait — actually, ligand sites should NOT go in `occupied` at init. They should be checked separately so the collision message is clear and so `occupied` only tracks protein positions. Instead, add ligand collision check inline (step 2k).

Revised:
```python
if ligand is not None:
    path = [(0, 0)]
    occupied = {(0, 0)}
else:
    path = [(0, 0), (1, 0)]
    occupied = {(0, 0), (1, 0)}
```

### 2i. Update base case (lines 90–96)
When `depth == n`:
```python
e_bind = 0.0
if ligand is not None:
    e_bind = ligand_binding_energy(sequence, tuple(path), ligand, mj_matrix)
total = energy + e_bind
boltz = exp(-total / temperature)
Z_sum += boltz
binding_weighted_sum += e_bind * boltz
n_enumerated += 1
if total < best_energy:
    best_energy = total
    best_conf = list(path)
```
`best_energy` now tracks total energy (intra + binding). Boltzmann weight uses total energy.

### 2j. Update pruning condition (line 98)
```python
if energy + bounds[n - depth] + binding_bound >= best_energy:
```
`binding_bound` is 0.0 without ligand → no-ligand path unchanged.

### 2k. Add ligand collision check (line 108)
```python
if new_pos in occupied or new_pos in ligand_sites:
```

### 2l. Start recursion and symmetry correction (lines 123–128)
```python
if ligand is not None:
    _recurse(0.0, 1, True)       # y_locked=True → all moves allowed (unreduced)
    Z_full = Z_sum                # no symmetry correction
else:
    _recurse(0.0, 2, False)
    Z_full = Z_sum * 8 - 4
```

### 2m. Remove n=2 special case (lines 130–137)
The n=2 early return exists because the general recursion starts at depth=2, making n=2 a trivial base case. With ligand (depth=1 start), n=2 works through general recursion. Without ligand, the general return path at line 139 already produces the correct result for n=2 (`best_conf=[(0,0),(1,0)]`, `best_energy=0.0`, `Z_full=4`). Remove lines 130–137.

### 2n. Compute ensemble binding energy and update return
```python
ensemble_binding = binding_weighted_sum / Z_sum if Z_sum > 0 else 0.0
```
Divide by `Z_sum` (not `Z_full`) — both numerator and denominator sum over the same walk set.

Replace hardcoded `0.0` in FoldResult with `ensemble_binding`.

---

## 3. Create `tests/test_ligand.py` (~100 lines)

### Creation tests
- `test_create_ligand_horizontal` — "FWYL" at (0,-1): check positions, sites.
- `test_create_ligand_vertical` — "FW" at (-1,0) vertical: check positions.
- `test_create_ligand_single_residue` — "F" at (0,-1): one position.
- `test_create_ligand_invalid_direction` — `ValueError` for "diagonal".

### Contact tests
- `test_binding_contacts_single` — protein `((0,0),(1,0),(2,0))`, ligand "F" at (0,-1): contact `[(0,0)]`.
- `test_binding_contacts_none` — distant ligand: empty.
- `test_binding_contacts_multiple` — U-shaped protein near 2-residue ligand.

### Energy tests
- `test_binding_energy_single_contact` — equals one MJ entry.
- `test_binding_energy_zero_distant` — 0.0 when no contacts.
- `test_binding_energy_multiple` — sum of MJ entries.

---

## 4. Update `tests/test_fold.py`

### Remove
- `test_ligand_raises` (line 50–53) — ligand is now supported.

### Add helper
```python
def exhaustive_fold_with_ligand(sequence, mj, ligand, temperature=1.0):
```
Enumerate all **unreduced** SAWs (`reduce_symmetry=False`), skip those overlapping `ligand.sites`, compute intra + binding energy, accumulate Z and binding_weighted_sum. Return `(native_energy, Z, ensemble_binding)`.

### Add tests
- `test_fold_with_ligand_matches_exhaustive` — n=6 and n=8. Compare native energy, Z, ensemble_binding against brute-force. **Key correctness test.**
- `test_fold_with_ligand_native_energy_includes_binding` — recompute `conformation_energy + binding_energy` for native conformation, assert matches.
- `test_fold_with_ligand_fewer_conformations` — ligand blocks sites, so `n_conformations_enumerated` < unreduced SAW count.
- `test_fold_with_distant_ligand_zero_binding` — ligand at (100,100), assert `ensemble_binding_energy == 0.0`.
- `test_fold_with_ligand_nonzero_binding` — fold with nearby ligand, assert `ensemble_binding_energy < 0`.

### Existing tests unchanged
All 19 remaining tests (no-ligand path) must continue to pass with no modifications.

---

## Execution order

1. Create `trellis/ligand.py`
2. Create `tests/test_ligand.py` → run `pytest tests/test_ligand.py -v`
3. Modify `trellis/fold.py` (all changes in §2)
4. Update `tests/test_fold.py` (remove old test, add new tests)
5. Run `pytest tests/test_fold.py -v` → all pass
6. Run `pytest tests/ -v` → full suite green

## Verification

- `pytest tests/test_ligand.py -v` — all ligand module tests pass
- `pytest tests/test_fold.py -v` — all fold tests pass (existing no-ligand + new ligand)
- `pytest tests/ -v` — full suite green, no regressions