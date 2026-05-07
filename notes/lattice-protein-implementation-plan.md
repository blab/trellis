# Lattice Protein Synthetic Data: Implementation Plan

Trevor Bedford — 2026-05-07

## Motivation

Evolutionary models needs synthetic evaluation data where the true fitness landscape is known. Lattice proteins provide this: a 20-residue amino acid chain on a 2D square lattice with exact native state determination via branch-and-bound enumeration, a Miyazawa-Jernigan contact potential for energy, and ligand binding as a fitness proxy. Evolutionary trajectories are generated via strong-selection weak-mutation (SSWM) dynamics at the DNA level (60 nt codon sequences), producing training data directly compatible with the [trajectories](https://github.com/blab/trajectories) repo output format.

## Architecture overview

```
trellis/
├── trellis/
│   ├── __init__.py
│   ├── lattice.py              # Step 1: 2D lattice, self-avoiding walks, contacts
│   ├── energy.py               # Step 2: MJ matrix, conformation energy, partition function
│   ├── fold.py                 # Step 3: Branch-and-bound folding with energy pruning
│   ├── ligand.py               # Step 4: Ligand placement, protein-ligand contacts, binding energy
│   ├── fitness.py              # Step 5: Fitness as ensemble-averaged ligand binding
│   ├── genetic_code.py         # Step 6: Codon table, DNA ↔ AA translation, mutation enumeration
│   ├── sswm.py                 # Step 7: SSWM trajectory generation at DNA level
│   ├── trajectory_io.py        # Step 8: Output in trajectories repo FASTA format
│   └── cache.py                # Step 9: Fitness cache (sequence → fitness + native state)
├── scripts/
│   ├── fold_sequence.py           # Fold a single sequence, print results
│   ├── visualize_conformation.py  # Render a conformation on the 2D lattice
│   ├── analyze_landscape.py       # Landscape statistics and visualization
│   ├── validate_folding.py        # Correctness checks against known results
│   └── generate_trajectories.py   # Main CLI entry point
├── tests/
│   ├── test_lattice.py
│   ├── test_energy.py
│   ├── test_fold.py
│   ├── test_ligand.py
│   ├── test_fitness.py
│   ├── test_genetic_code.py
│   ├── test_sswm.py
│   └── test_trajectory_io.py
├── data/
│   └── mj_matrix.csv           # Miyazawa-Jernigan contact potential (20×20)
├── pyproject.toml
└── README.md
```

The package is pure Python with one performance-critical inner loop (branch-and-bound walk enumeration in `fold.py`) that should use numba for JIT compilation. All other modules are standard Python with numpy.

## Step 1: 2D lattice and self-avoiding walks — `lattice.py`

### Purpose

Represent a 2D square lattice and enumerate self-avoiding walks (SAWs) on it. This is the geometric foundation for everything else.

### Data structures

```python
# A conformation is a tuple of (x, y) coordinates, one per residue
# conformation[i] = (x, y) position of residue i on the lattice
Conformation = tuple[tuple[int, int], ...]

# Directions on the 2D square lattice
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
```

### Key functions

- `enumerate_saws(n: int) -> Iterator[Conformation]`: Generate all self-avoiding walks of length `n` on the 2D square lattice. Fix the first residue at the origin (0, 0) and the first step to (1, 0) to remove rotational/reflective symmetry (reduces enumeration by factor of 8). Yield conformations as tuples of coordinates.

- `get_contacts(conformation: Conformation) -> list[tuple[int, int]]`: Return all topological contacts — pairs (i, j) where |i - j| > 1 and residues i, j are lattice-adjacent (Manhattan distance = 1). These are the non-bonded contacts that contribute to energy.

- `is_self_avoiding(conformation: Conformation) -> bool`: Check that no two residues occupy the same lattice site. Used for validation.

- `occupied_sites(conformation: Conformation) -> set[tuple[int, int]]`: Return the set of lattice sites occupied by the conformation. Used in Step 4 for ligand collision detection.

### Implementation notes

- The SAW enumerator is a depth-first recursive generator. At each step, try all 4 lattice directions, skip any that would revisit an occupied site.
- For 20 residues, raw SAW count is ~4.7 × 10^8. With symmetry reduction (factor of 8) this is ~5.9 × 10^7. With energy pruning (Step 3) this becomes very manageable.
- The SAW enumerator here is the unpruned version for testing. The production folding code in Step 3 will integrate pruning directly into the walk generation.
- `enumerate_saws` is primarily for testing and validation of small chains (n ≤ 12). For production folding of 20-residue chains, use the branch-and-bound folder in `fold.py` which integrates energy pruning.

### Tests

- Verify SAW counts against known values: n=6 → 36, n=8 → 88, n=10 → 200 (after symmetry reduction). Use OEIS A001411 for reference, divided by 8.
- Verify contact extraction on hand-constructed conformations.
- Verify symmetry reduction: the full enumeration count should be 8× the reduced count.

## Step 2: Energy model — `energy.py`

### Purpose

Compute conformation energies using the Miyazawa-Jernigan (MJ) contact potential.

### Data

The MJ matrix is a 20×20 symmetric matrix of pairwise residue-residue contact energies (in units of kT). Use the original 1996 Miyazawa-Jernigan matrix (Table V from Miyazawa & Jernigan, *J Mol Biol* 256:623-644, 1996). Store as a CSV file in `data/mj_matrix.csv` with single-letter amino acid codes as row/column headers.

### Key functions

- `load_mj_matrix(path: str) -> np.ndarray`: Load the 20×20 MJ matrix. Return as a numpy array indexed by integer residue codes (A=0, C=1, ..., Y=19 in alphabetical order). Cache on first load.

- `conformation_energy(sequence: str, conformation: Conformation, mj_matrix: np.ndarray) -> float`: Compute the total internal energy: sum of MJ contacts `mj_matrix[aa_i, aa_j]` for all contacts (i, j) in the conformation.

- `max_contact_energy(sequence: str, n_remaining: int, mj_matrix: np.ndarray) -> float`: Compute the most favorable (most negative) energy possible from contacts involving the remaining `n_remaining` residues. This is the bound for branch-and-bound pruning. For each unplaced residue, the maximum new contacts it can form is 2 (for interior residues) or 3 (for the terminal residue), and each contact contributes at most `min(mj_matrix)`. This gives a loose but fast lower bound.

- `partition_function(sequence: str, conformations_and_energies: list[tuple[Conformation, float]], temperature: float) -> float`: Compute Z = Σ exp(-E_i / kT) over all conformations. Temperature in reduced units (kT). In practice, Z is accumulated during branch-and-bound traversal in `fold.py`; this standalone function is for testing on small chains.

### Implementation notes

- All energies are in reduced units where the MJ values are already in kT.
- The MJ matrix has all negative entries (favorable contacts). The most negative entry (~-7.0 for Cys-Cys) provides the tightest bound for pruning.
- A tighter bound can be computed by considering the actual amino acid identities of unplaced residues and the maximum contacts each can form, but the loose bound is usually sufficient for 20-mers.

### Tests

- Verify energy calculation on a known conformation with hand-computed contacts.
- Verify that the bound function always returns a value ≤ the true remaining energy for randomly generated partial conformations.
- Verify partition function normalization: Boltzmann probabilities should sum to 1.

## Step 3: Branch-and-bound folding — `fold.py`

### Purpose

Enumerate conformations of a lattice protein via branch-and-bound, accumulating the partition function Z and the Boltzmann-weighted ensemble average of ligand binding energy. This is the performance-critical inner loop.

### Key functions

- `fold(sequence: str, mj_matrix: np.ndarray, ligand: Ligand | None = None, temperature: float = 1.0) -> FoldResult`: Enumerate conformations via branch-and-bound. Returns a `FoldResult` containing: native conformation, native energy, partition function, ensemble-averaged binding energy, and total conformations enumerated.

- `FoldResult`: Dataclass with fields `native_conformation`, `native_energy`, `partition_function`, `ensemble_binding_energy` (= ⟨E_binding⟩, the Boltzmann-weighted average binding energy across all conformations), `n_conformations_enumerated`.

### Algorithm

The key insight: during branch-and-bound traversal, we already visit every conformation to accumulate Z. We simultaneously accumulate a weighted sum of binding energies at negligible extra cost.

```
function fold(sequence, mj_matrix, ligand):
    best_energy = +infinity
    best_conformation = None
    Z = 0.0                    # partition function
    binding_weighted_sum = 0.0  # Σ E_binding(i) * exp(-E_total(i) / kT)

    function branch_and_bound(partial_conformation, current_energy, depth):
        if depth == len(sequence):
            # Complete conformation
            e_bind = 0.0
            if ligand is not None:
                e_bind = binding_energy(sequence, partial_conformation, ligand, mj_matrix)
            total_energy = current_energy + e_bind
            boltzmann = exp(-total_energy / kT)
            Z += boltzmann
            binding_weighted_sum += e_bind * boltzmann
            if total_energy < best_energy:
                best_energy = total_energy
                best_conformation = partial_conformation
            return

        # Pruning: can remaining residues possibly beat current best?
        bound = current_energy + max_contact_energy(sequence, len(sequence) - depth, mj_matrix)
        if bound >= best_energy:
            return  # prune

        for move in MOVES:
            new_pos = last_pos + move
            if new_pos not in occupied_sites and (ligand is None or new_pos not in ligand.sites):
                new_energy = current_energy + new_contacts_energy(...)
                branch_and_bound(partial_conformation + [new_pos], new_energy, depth + 1)

    # Fix first residue at origin, first step to (1,0) for symmetry reduction
    branch_and_bound([(0,0), (1,0)], initial_energy, depth=2)
    # Multiply Z and weighted sums by 8 to account for symmetry reduction
    ensemble_binding = binding_weighted_sum / Z  # ⟨E_binding⟩
    return FoldResult(best_conformation, best_energy, Z * 8, ensemble_binding, ...)
```

The ensemble-averaged binding energy ⟨E_binding⟩ = Σ_i E_binding(i) · p(i) where p(i) = exp(−E_total(i) / kT) / Z and E_total = E_intra + E_binding. This naturally handles the full thermodynamics: a protein that spends most of its time in conformations contacting the ligand gets a strong (negative) ⟨E_binding⟩; an unstable protein spread across many non-binding conformations gets a weak ⟨E_binding⟩ — no separate stability criterion needed.

### Implementation notes

**Pruning and the partition function.** Aggressive pruning skips conformations that can't beat the current best energy. These pruned conformations have very high energy and therefore negligible Boltzmann weights — their contribution to both Z and ⟨E_binding⟩ is vanishingly small. The single-pass approach (prune aggressively, accumulate Z and binding sums only for visited conformations) is therefore a safe approximation. If validation against exhaustive enumeration on small chains (n ≤ 12) shows discrepancies in ⟨E_binding⟩, fall back to a two-pass approach: first pass finds native energy with pruning, second pass enumerates without pruning to compute exact Z and ⟨E_binding⟩.

**Numba acceleration.** The inner loop should be JIT-compiled with numba. Key requirements:
- Represent conformations as fixed-size numpy arrays rather than lists of tuples.
- Represent occupied sites as a 2D boolean grid (e.g., 41×41 centered at (20,20) to accommodate walks up to length 20 in any direction).
- Pass the MJ matrix as a numpy array.
- The recursive function should be a numba `@njit` function.
- The binding energy calculation must also be numba-compatible (no Python objects in the hot loop).

**Expected performance.** For 20-residue chains with MJ energy pruning, expect to enumerate ~10^5 to 10^6 conformations (vs. ~10^7 unpruned), taking ~1-10 seconds per sequence with numba. This means folding the ~60 unique AA neighbors of a sequence takes ~1-10 minutes per SSWM step. With fitness caching (Step 9), this drops dramatically as the SSWM revisits sequences.

### Tests

- Verify native state of well-known HP model sequences against published results (e.g., the 20-residue HP benchmark sequence from Dill et al.).
- Verify that fold results are symmetric: rotating/reflecting the sequence gives the same energy.
- Verify partition function and ⟨E_binding⟩ by comparing to exhaustive enumeration on short (n=10) chains. The pruned single-pass result should match within floating-point tolerance.
- Verify that ⟨E_binding⟩ is more negative for sequences whose native state contacts the ligand than for sequences whose native state does not.
- Benchmark: 20-residue MJ folding should complete in <30 seconds without numba, <5 seconds with numba.

## Step 4: Ligand binding — `ligand.py`

### Purpose

Define a fixed ligand on the lattice and compute protein-ligand binding energies.

### Data structures

```python
@dataclass
class Ligand:
    sequence: str                           # AA sequence of ligand (e.g., "FWCY")
    positions: tuple[tuple[int, int], ...]  # Fixed lattice positions of ligand residues
    sites: set[tuple[int, int]]             # Set of occupied sites (for collision detection)
```

### Key functions

- `create_ligand(sequence: str, anchor: tuple[int, int], direction: str = "horizontal") -> Ligand`: Place a ligand on the lattice starting at `anchor`, extending in the given direction. Ligand residues are placed on consecutive lattice sites. Direction is "horizontal" (extending along x-axis) or "vertical" (extending along y-axis).

- `binding_energy(protein_sequence: str, conformation: Conformation, ligand: Ligand, mj_matrix: np.ndarray) -> float`: Compute the sum of MJ contact energies between protein residues and ligand residues that are lattice-adjacent. A protein residue at position (x1, y1) contacts a ligand residue at position (x2, y2) if their Manhattan distance is 1.

- `binding_contacts(conformation: Conformation, ligand: Ligand) -> list[tuple[int, int]]`: Return pairs (protein_residue_index, ligand_residue_index) that are in contact. Useful for analysis.

### Ligand placement strategy

Place the ligand adjacent to the lattice origin so that protein walks naturally encounter it. A reasonable default: for a 4-residue ligand, place it at positions (0, -1), (1, -1), (2, -1), (3, -1) — a horizontal line one step below the protein's starting position. The protein starts at (0, 0) with first step to (1, 0), so the ligand is immediately adjacent.

The protein's self-avoiding walk must not occupy any ligand site. This is enforced in the branch-and-bound by treating ligand sites as occupied.

### Ligand composition

Use a ligand with strong interaction potential — hydrophobic residues create a binding patch that rewards complementary protein surfaces. A default ligand of "FWYL" (Phe-Trp-Tyr-Leu) provides strong hydrophobic interactions. The ligand sequence is a configurable parameter.

### Tests

- Verify binding energy calculation on hand-constructed protein-ligand configurations.
- Verify that protein conformations that overlap ligand sites are correctly excluded.
- Verify that moving the protein away from the ligand yields zero binding energy.

## Step 5: Fitness function — `fitness.py`

### Purpose

Compute fitness for a DNA sequence as the ensemble-averaged ligand binding energy of the encoded protein. This is a thin wrapper around `fold()` — translate DNA to AA, fold in the presence of the ligand, return ⟨E_binding⟩.

### Key functions

- `compute_fitness(dna_sequence: str, ligand: Ligand, mj_matrix: np.ndarray, temperature: float = 1.0) -> FitnessResult`: The main fitness function. Translates DNA to AA, folds with ligand, returns fitness as −⟨E_binding⟩ (negated so higher = fitter).

- `FitnessResult`: Dataclass with fields: `fitness` (float), `fold_result` (FoldResult), `aa_sequence` (str), `dna_sequence` (str).

### Fitness definition

```python
def compute_fitness(dna_sequence, ligand, mj_matrix, temperature=1.0):
    aa_sequence = translate(dna_sequence)
    if '*' in aa_sequence:  # stop codon
        return FitnessResult(fitness=-inf, ...)

    fold_result = fold(aa_sequence, mj_matrix, ligand, temperature)
    fitness = -fold_result.ensemble_binding_energy  # negate: more negative binding = higher fitness

    return FitnessResult(fitness=fitness, fold_result=fold_result, ...)
```

No separate stability criterion is needed. The ensemble average handles stability naturally: a protein that doesn't fold stably is spread across many conformations, most of which don't contact the ligand, yielding a weak ⟨E_binding⟩ and therefore low fitness. A protein that folds stably into a ligand-binding conformation concentrates its Boltzmann weight there, yielding a strong ⟨E_binding⟩. A protein that bounces between two low-energy conformations that both bind the ligand also scores well — the physics handles it correctly.

### Tests

- Verify that stop codons give -inf fitness.
- Verify that synonymous mutations give identical fitness.
- Verify that a sequence whose native state contacts the ligand has higher fitness than one whose native state does not.
- Verify that fitness varies continuously: single-AA mutations produce a range of fitness changes, not a binary foldable/unfoldable split.
- Generate fitness statistics for 1000 random 60 nt sequences to characterize the landscape.

## Step 6: Genetic code — `genetic_code.py`

### Purpose

Handle translation between DNA codons and amino acids, and enumerate all possible single-nucleotide mutations.

### Key functions

- `translate(dna_sequence: str) -> str`: Translate a DNA sequence (length must be divisible by 3) to amino acids using the standard genetic code. Return `*` for stop codons.

- `single_nt_mutations(dna_sequence: str) -> list[tuple[str, int, str, str]]`: Enumerate all possible single-nucleotide mutations. Return list of (mutant_dna_sequence, position, ref_base, alt_base). For a 60 nt sequence, this is 180 mutations (60 positions × 3 alternative bases).

- `classify_mutation(dna_ref: str, dna_alt: str) -> str`: Classify a mutation as "synonymous", "nonsynonymous", or "nonsense" (introduces stop codon).

- `mutant_aa_sequences(dna_sequence: str) -> dict[str, list[str]]`: Group all single-nt mutants by their resulting AA sequence. This is important for efficiency — many DNA mutations map to the same AA sequence and thus the same fitness, so we only need to fold each unique AA sequence once.

### Implementation notes

- The standard genetic code has 61 sense codons and 3 stop codons.
- For a 60 nt sequence (20 codons), roughly 1/3 of single-nt mutations are synonymous, 2/3 are nonsynonymous, and a small fraction introduce stop codons.
- `mutant_aa_sequences` is the key efficiency function: it deduplicates the folding workload for SSWM. If 10 different DNA mutations all produce the same AA change, we fold once and assign the same fitness to all 10.

### Tests

- Verify translation against known sequences.
- Verify that known synonymous mutations (e.g., third-position transitions in 4-fold degenerate codons) are correctly classified.
- Verify mutation count: 60 nt × 3 alt bases = 180 mutations total.

## Step 7: SSWM trajectory generation — `sswm.py`

### Purpose

Generate evolutionary trajectories through DNA sequence space under the SSWM regime.

### Algorithm

```
function generate_trajectory(start_dna, ligand, mj_matrix, n_steps, Ne, mu):
    trajectory = [start_dna]
    current_dna = start_dna
    current_fitness = compute_fitness(current_dna, ...)

    for step in range(n_steps):
        # Enumerate all single-nt neighbors
        mutations = single_nt_mutations(current_dna)

        # Compute fitness for all unique AA sequences (deduplicated)
        unique_aa = mutant_aa_sequences(current_dna)
        aa_fitness = {aa: compute_fitness(dna, ...) for aa, dna_list in unique_aa}

        # Compute fixation probabilities (Kimura 1962)
        candidates = []
        for mutant_dna, pos, ref, alt in mutations:
            mutant_fitness = aa_fitness[translate(mutant_dna)]
            s = mutant_fitness - current_fitness  # selection coefficient
            if s == 0:
                p_fix = 1.0 / (2 * Ne)  # neutral
            else:
                x = 2 * Ne * s
                p_fix = (1 - exp(-2*s)) / (1 - exp(-x))  # Kimura formula
            rate = mu * p_fix  # substitution rate
            candidates.append((mutant_dna, rate, mutant_fitness))

        # SSWM: total rate determines waiting time; pick mutation proportional to rate
        total_rate = sum(rate for _, rate, _ in candidates)
        if total_rate == 0:
            break  # stuck — all mutations lethal or neutral with tiny fixation prob

        # Sample next mutation proportional to substitution rate
        chosen = random.choices(candidates, weights=[r for _, r, _ in candidates])[0]
        current_dna = chosen[0]
        current_fitness = chosen[2]
        trajectory.append(current_dna)

    return trajectory
```

### Key functions

- `generate_trajectory(start_dna: str, ligand: Ligand, mj_matrix: np.ndarray, n_steps: int = 100, Ne: float = 1000, mu: float = 1e-6, temperature: float = 1.0, rng: np.random.Generator = None, cache: FitnessCache = None) -> Trajectory`: Generate a single SSWM trajectory.

- `generate_start_sequence(ligand: Ligand, mj_matrix: np.ndarray, min_fitness: float, max_attempts: int = 10000, rng: np.random.Generator = None) -> str`: Generate a random DNA sequence with at least `min_fitness`. This is the starting point for SSWM. Strategy: generate random 60 nt sequences (avoiding stop codons), compute fitness, keep the first that passes the threshold. The `min_fitness` threshold ensures trajectories start from sequences that already have some ligand binding — otherwise SSWM would spend many steps in flat (zero-binding) fitness landscape before finding anything interesting.

- `Trajectory`: Dataclass with fields: `dna_sequences` (list[str]), `aa_sequences` (list[str]), `fitness_values` (list[float]), `mutation_types` (list[str] — "synonymous"/"nonsynonymous"/"nonsense" for each step), `metadata` (dict with Ne, mu, ligand info, etc.).

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_steps` | 100 | Steps per trajectory |
| `Ne` | 1000 | Effective population size. Higher = more deterministic (beneficial mutations fix more reliably) |
| `mu` | 1e-6 | Per-site per-generation mutation rate. Only affects relative waiting times, not trajectory path under SSWM |
| `temperature` | 1.0 | Folding temperature in reduced units |

### Parallelization

Trajectory generation is embarrassingly parallel — each trajectory is independent. Use `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` to generate trajectories in parallel across CPU cores. The `generate_trajectories` (plural) function should accept `n_trajectories` and `n_workers` parameters.

### Tests

- Verify that synonymous mutations have fixation probability 1/(2Ne).
- Verify that strongly beneficial mutations have fixation probability ≈ 2s.
- Verify that stop codon mutations are never accepted.
- Verify trajectory statistics: mean fitness should increase over time (directional selection), with the rate depending on Ne.
- Generate a small batch (10 trajectories, 20 steps) and verify output format.

## Step 8: Output formatting — `trajectory_io.py`

### Purpose

Write SSWM trajectories in the exact FASTA format expected by the `trajectories` repo preprocessing pipeline, and package into compressed tar.zst shards for S3 upload.

### Output format

Each trajectory is a FASTA file matching the forwards trajectory format:

```
>NODE_0000|0|0
ATCGATCG...
>NODE_0001|3|3
ATCAATCG...
>NODE_0002|5|2
ATCAATCA...
```

Where:
- Node names are sequential: `NODE_0000`, `NODE_0001`, ..., with the final node named `TIP_{trajectory_id}`
- `cumulative_hamming` is the total DNA Hamming distance from the start sequence
- `branch_hamming` is the DNA Hamming distance from the previous sequence
- Sequences are DNA (not AA)

### Pairwise format

Additionally generate pairwise trajectory files (tip-to-tip pairs sampled from different trajectories that share the same starting sequence):

```
>TipA|0|0
ATCGATCG...
>TipB|12|12
ATCAATCG...
```

### Key functions

- `write_trajectory_fasta(trajectory: Trajectory, output_path: str, trajectory_id: str)`: Write a single trajectory as a FASTA file.

- `write_pairwise_fasta(seq_a: str, seq_b: str, output_path: str, pair_id: str)`: Write a pairwise trajectory file.

- `package_shards(trajectory_dir: str, output_dir: str, max_per_shard: int = 10000)`: Package FASTA files into compressed tar.zst shards matching the S3 naming convention (`forwards-train-000.tar.zst`, etc.).

- `train_test_split(trajectories: list[Trajectory], test_fraction: float = 0.1) -> tuple[list, list]`: Split trajectories into train and test sets. Since these are synthetic (no phylogenetic tree to excise clades from), use a simpler strategy: hold out entire trajectories that share starting sequences, so the test set contains genuinely unseen evolutionary lineages starting from unseen proteins.

### Dataset naming convention

Use the prefix `lattice-` for all synthetic datasets:

```
lattice-20aa-4lig/          # 20-residue protein, 4-residue ligand
  forwards-train-000.tar.zst
  forwards-test-000.tar.zst
  pairwise-train-000.tar.zst
  pairwise-test-000.tar.zst
  metadata.json             # Ligand sequence, positions, Ne, mu, temperature, etc.
```

### Tests

- Verify FASTA output parses correctly with BioPython.
- Verify Hamming distances in headers match actual sequence differences.
- Verify that the output is compatible with the `trajectories` repo's JSONL preprocessing.

## Step 9: Fitness cache — `cache.py`

### Purpose

Cache fitness evaluations to avoid redundant folding. Many SSWM trajectories will visit the same or similar sequences, and deduplication across AA sequences (Step 6) means the cache key is the AA sequence, not the DNA sequence.

### Key functions

- `FitnessCache`: Class wrapping a dict mapping AA sequences to `FitnessResult` objects. Thread-safe for parallel trajectory generation.
  - `get(aa_sequence: str) -> FitnessResult | None`
  - `put(aa_sequence: str, result: FitnessResult)`
  - `stats() -> dict`: Cache hit rate, total entries, etc.

### Implementation notes

- Cache key is the AA sequence (20 characters), not the DNA sequence (60 characters), since synonymous DNA variants share fitness.
- For parallel generation, use `multiprocessing.Manager().dict()` or a shared-memory approach.
- The cache can grow large if many unique AA sequences are explored. For 10,000 trajectories of 100 steps each, expect up to ~10^6 unique AA sequences (though with significant overlap). At ~1 KB per FitnessResult, this is ~1 GB — manageable in memory.
- Optionally persist cache to disk (pickle or sqlite) to resume interrupted runs.

## Fold a single sequence — `scripts/fold_sequence.py`

A diagnostic script for folding a single sequence and printing the result. This is the script you'll reach for constantly during development — "does this sequence actually fold the way I think it does?" Also the fastest way to sanity-check numba compilation (first call is slow, second is fast).

### Usage

```bash
# Fold an amino acid sequence
python scripts/fold_sequence.py --aa ACDEFGHIKLMNPQRSTVWY

# Translate and fold a DNA sequence
python scripts/fold_sequence.py --dna GCTTGTGATGAATTTGGTCATATCAAACTTATGAATCCGCAAAGAACTTCTGTTTGGTAC

# Fold with a ligand
python scripts/fold_sequence.py --aa ACDEFGHIKLMNPQRSTVWY --ligand-sequence FWYL

# Read sequence from stdin
echo "ACDEFGHIKLMNPQRSTVWY" | python scripts/fold_sequence.py --aa -
```

### Output

Prints to stdout:

- AA sequence (and DNA if provided)
- Native energy (E_intra, and E_intra + E_binding if ligand present)
- Ensemble-averaged binding energy ⟨E_binding⟩ (if ligand present)
- Fitness (−⟨E_binding⟩)
- Native conformation coordinates
- Number of conformations enumerated
- Wall time

Optionally `--json` flag for machine-readable output.

## Visualize a conformation — `scripts/visualize_conformation.py`

Renders the native conformation on the 2D lattice as a matplotlib figure. Invaluable for debugging — when fold results look wrong, seeing the conformation instantly tells you whether the issue is in energy calculation, contact detection, or the walk itself. Also useful for paper figures eventually.

### Usage

```bash
# Fold and visualize in one step
python scripts/visualize_conformation.py --aa ACDEFGHIKLMNPQRSTVWY -o conformation.png

# With ligand
python scripts/visualize_conformation.py --aa ACDEFGHIKLMNPQRSTVWY --ligand-sequence FWYL -o conformation.png

# Interactive display (no save)
python scripts/visualize_conformation.py --aa ACDEFGHIKLMNPQRSTVWY --show
```

### Visual elements

- Residues as colored circles (color by amino acid property — hydrophobic, polar, charged — or by single-letter identity)
- Chain backbone bonds as solid lines between consecutive residues
- Non-bonded contacts as dashed lines between contacting residue pairs
- Ligand residues in a distinct color/shape, positioned at their fixed lattice sites
- Residue index labels inside or adjacent to each circle
- Grid lines showing the lattice

## Main CLI — `scripts/generate_trajectories.py`

### Usage

```bash
# Generate 1000 trajectories with default parameters
python scripts/generate_trajectories.py \
    --n-trajectories 1000 \
    --n-steps 100 \
    --chain-length 20 \
    --ligand-sequence FWYL \
    --Ne 1000 \
    --output-dir output/lattice-20aa-4lig/ \
    --n-workers 8 \
    --seed 42

# Generate with custom ligand placement
python scripts/generate_trajectories.py \
    --n-trajectories 1000 \
    --ligand-sequence FWY \
    --ligand-anchor 0,-1 \
    --ligand-direction horizontal \
    --output-dir output/lattice-20aa-3lig/
```

### Output

```
output/lattice-20aa-4lig/
├── forwards-train-000.tar.zst
├── forwards-test-000.tar.zst
├── pairwise-train-000.tar.zst
├── pairwise-test-000.tar.zst
├── metadata.json           # Full parameter record
└── landscape_stats.json    # Fitness distribution, synonymous fraction, etc.
```

### Metadata

`metadata.json` records all parameters needed to reproduce the dataset:

```json
{
    "chain_length": 20,
    "dna_length": 60,
    "ligand_sequence": "FWYL",
    "ligand_positions": [[0, -1], [1, -1], [2, -1], [3, -1]],
    "Ne": 1000,
    "mu": 1e-6,
    "temperature": 1.0,
    "n_trajectories": 1000,
    "n_steps": 100,
    "seed": 42,
    "n_start_sequences": 50,
    "min_start_fitness": 0.5,
    "test_fraction": 0.1
}
```

## Landscape analysis — `scripts/analyze_landscape.py`

A diagnostic script to characterize the fitness landscape before generating trajectories. Run this first to tune parameters.

### Outputs

- **Fitness distribution** of 10,000 random sequences: what's the distribution of ⟨E_binding⟩? What fraction have non-trivial binding (fitness above some threshold)? What's the fitness range?
- **Synonymous mutation fraction**: of all single-nt mutations from a set of 100 random sequences with non-trivial fitness, what fraction are synonymous vs. nonsynonymous vs. nonsense?
- **Fitness effect distribution (DFE)**: histogram of Δfitness for nonsynonymous mutations. Should show a mix of deleterious, neutral, and beneficial mutations.
- **Epistasis check**: for a few pairs of mutations, verify that Δfitness(AB) ≠ Δfitness(A) + Δfitness(B).
- **Neutral network connectivity**: from a high-fitness sequence, how many synonymous neighbors have similar fitness? This affects whether SSWM can traverse the landscape via neutral drift.

These diagnostics determine whether the parameter settings produce a scientifically interesting landscape.

## Implementation order

Build and test each step sequentially. Each step should be fully tested before moving to the next.

1. **`lattice.py`** — Geometry and self-avoiding walks
2. **`energy.py`** — MJ matrix and energy computation
3. **`fold.py`** — Branch-and-bound folding (the hard part; get this right and fast)
4. **`scripts/fold_sequence.py`** — CLI for folding individual sequences (useful immediately once Step 3 works)
5. **`ligand.py`** — Ligand placement and binding energy
6. **`scripts/visualize_conformation.py`** — Lattice conformation renderer (useful for debugging Steps 3–5)
7. **`fitness.py`** — Combined fitness function
8. **`genetic_code.py`** — DNA/AA translation and mutation enumeration
9. **`sswm.py`** — SSWM trajectory generation
10. **`trajectory_io.py`** — Output formatting compatible with trajectories repo
11. **`cache.py`** — Fitness caching for efficiency
12. **`scripts/analyze_landscape.py`** — Landscape diagnostics
13. **`scripts/generate_trajectories.py`** — Main CLI
14. **`scripts/validate_folding.py`** — Correctness validation against known results

Steps 1–3 are the core computational engine. Steps 4 and 6 are diagnostic scripts that become useful the moment their underlying library code works. Steps 5–8 build the evolutionary model on top. Steps 9–11 generate the data. Steps 12–14 are the remaining user-facing scripts.

## Dependencies

```
numpy
numba
zstandard        # for tar.zst shard packaging
biopython        # for FASTA I/O (optional, could use plain text)
click            # CLI argument parsing
matplotlib       # for landscape analysis plots (optional)
```

## Performance targets

| Operation | Target time | Notes |
|-----------|-------------|-------|
| Fold one 20-residue sequence (no ligand) | <5 sec | With numba, MJ pruning |
| Fold one 20-residue sequence (with ligand) | <10 sec | Ligand constrains walk space |
| One SSWM step (fold ~60 unique AA neighbors) | <10 min | With AA deduplication |
| One trajectory (100 steps) | <3 hours | With caching reducing later steps |
| 1000 trajectories (8 workers) | <2 days | With shared cache |

These are rough estimates. Actual performance depends heavily on pruning efficiency, cache hit rates, and hardware. The `analyze_landscape.py` script should include timing benchmarks.

## Evaluation targets

Once trajectories are generated and the model is fine-tuned on them, the primary evaluation questions are:

1. **Mutation prediction**: Given a trajectory prefix, does the model assign higher probability to the true next mutation than to random alternatives?
2. **Fitness correlation**: Do model-predicted mutation probabilities correlate with the known fitness effects? (This is the lattice protein analog of the nucleotide frequency baseline.)
3. **Synonymous vs. nonsynonymous**: Does the model correctly predict that synonymous mutations are more likely to be accepted (they're effectively neutral)?
4. **Epistasis**: Does the predicted probability of mutation A change appropriately when mutation B is present vs. absent?
5. **Multi-step forecasting**: Given a trajectory prefix, can the model predict multiple future steps that track the fitness gradient?
