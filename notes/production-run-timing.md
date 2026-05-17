# Production Run Timing and 20-mer Feasibility Analysis

## Why 16-mers (current baseline)

The computational cost of exhaustive SAW enumeration and scoring grows exponentially with chain length. For 2D lattice proteins with a 4-mer ligand (unreduced enumeration):

| Chain length | Conformations | Enumeration (Python) | Per SSWM step | Per trajectory (100 steps) |
|--------------|---------------|---------------------|---------------|---------------------------|
| 16           | ~3.3M         | ~36s                | ~0.92s        | ~92s                      |
| 18           | ~23M          | ~4 min (est.)       | ~6.5s         | ~650s                     |
| 20           | ~177M         | ~30 min (est.)      | ~49s          | ~82 min                   |

SAW counts follow OEIS A001411 (by step count = residues - 1). With a 4-site ligand, roughly half of unreduced SAWs are filtered for collisions.

At 16 residues, a single worker can complete 1000 trajectories × 100 steps in ~26 hours, making batch production tractable on a 32-core server.

## Bloom et al. (2005) — the 18-mer precedent

Reference: Bloom, Silberg, Wilke, Drummond, Adami & Arnold. "Thermodynamic prediction of protein neutrality." *Biophysical Journal* 86:2758-2764. [arXiv:q-bio/0401038](https://arxiv.org/abs/q-bio/0401038)

Key facts:
- **Chain length: 18** (not 20 as sometimes cited)
- **Total conformations: 5.81M** (symmetry-reduced)
- **Explicitly scored: only 795K** (conformations with >4 contacts, ~14% of total)
- **Mean-field approximation** for the remaining 5.01M low-contact conformations
- **Per-sequence time: 0.3s** on a ~2 GHz processor (2003 hardware)
- **Energy model: Miyazawa-Jernigan** with 20 amino acids (same as ours)

The critical insight: Bloom did not brute-force score all 5.81M conformations for every sequence. Low-contact conformations (≤4 contacts) were handled collectively via a mean-field formula, reducing per-sequence workload by ~7×.

## Scoring throughput analysis

Our numba JIT scoring loop (`_score_conformations`) processes conformations at ~470M conformations/second per sequence evaluation:

```
16-mer: 3.3M conformations / 0.007s per sequence = 470M/s
```

Projected for longer chains (all conformations scored, no pruning):

| Chain | Conformations | Per sequence | Per SSWM step (~130 seqs) | Per trajectory |
|-------|---------------|-------------|---------------------------|----------------|
| 16    | 3.3M          | 7 ms        | 0.92s                     | 92s            |
| 18    | ~23M          | 49 ms       | 6.4s                      | 640s           |
| 20    | ~177M         | 377 ms      | 49s                       | 82 min         |

## Numba enumeration plan assessment

The plan in `numba-enumeration-plan.md` proposes rewriting `enumerate_conformations()` as a numba JIT function with:
- Explicit DFS stack (no Python recursion/generators)
- Boolean grid for O(1) collision checks (no Python sets)
- Two-pass: count then allocate+fill
- Inline contact extraction at leaf nodes

**Expected enumeration speedup:**

| Chain | Python (current) | Numba (estimated) |
|-------|-----------------|-------------------|
| 16    | 36s             | ~3s               |
| 18    | ~4 min          | ~20s              |
| 20    | ~30 min         | ~3-5 min          |

**Verdict: the plan is sound but addresses the wrong bottleneck for production.** Enumeration is a one-time startup cost per worker. For a 1000-trajectory run, even 30 minutes of enumeration is negligible compared to 57 days of scoring. The numba enumeration helps developer iteration time and benchmarking, but doesn't change production feasibility.

## What would make 20-mers production-feasible

### Option 1: Bloom-style contact-count pruning

Only explicitly score conformations with ≥k contacts. Low-contact conformations contribute little to the Boltzmann ensemble at T=1.0 because their energies are near zero (no stabilizing contacts).

For Bloom's 18-mers: 14% of conformations had >4 contacts. If a similar fraction applies to 20-mers (~15%), we'd score ~26M conformations instead of 177M:

| Metric | Full scoring | Pruned (15%) |
|--------|-------------|--------------|
| Per sequence | 377 ms | ~55 ms |
| Per SSWM step | 49s | ~7.2s |
| Per trajectory (100 steps) | 82 min | ~12 min |
| Per ligand (1000 traj) | 57 days | **8.3 days** |

With 32 workers/ligands in parallel: **~8 days wall time** per batch of 32 ligands.

This requires:
1. During enumeration, count contacts and only store conformations with ≥k contacts in the scored database
2. Track the count and mean Boltzmann contribution of pruned conformations for partition function correction
3. Validate that the approximation doesn't materially change fitness rankings (compare against full scoring for 16-mers)

### Option 2: Accept longer runs at full accuracy

With 32 workers, 32 ligands, 1000 trajectories:
- 20-mer, full scoring: **~57 days** per batch (impractical)
- 20-mer, reduced to 100 trajectories: **~5.7 days** per batch (marginal)

### Option 3: 18-mers as a middle ground

| Metric | 16-mer | 18-mer | 20-mer |
|--------|--------|--------|--------|
| Conformations | 3.3M | ~23M | ~177M |
| Per trajectory | 92s | ~640s | ~82 min |
| Per ligand (1000 traj) | 26 hr | 7.4 days | 57 days |
| 32 ligands (32 workers) | 26 hr | 7.4 days | 57 days |

18-mers at full accuracy take ~7 days per batch — feasible for a single large run.

## Memory constraints for 20-mers

The ConformationDatabase for 20-mers consumes ~13 GB RAM per worker:

| Array | Size estimate |
|-------|--------------|
| contact_pairs (~5 avg × 177M) | ~7.1 GB |
| contact_offsets (177M + 1) | ~1.4 GB |
| binding_pairs (~2 avg × 177M) | ~2.8 GB |
| binding_offsets (177M + 1) | ~1.4 GB |
| **Total** | **~12.7 GB** |

On the AMD EPYC 9354P server (assuming 256 GB RAM):
- 32 workers × 13 GB = **416 GB** — exceeds 256 GB
- 16 workers × 13 GB = **208 GB** — fits in 256 GB
- With pruning (15% of conformations): 32 workers × ~2 GB = **64 GB** — easily fits

Memory alone limits full-scoring 20-mer parallelism to ~16 workers on a 256 GB machine.

## Production plan: current 16-mer configuration

For immediate production with validated code:

```bash
python scripts/generate_trajectories.py \
    --n-random-ligands 32 --ligand-length 4 \
    --chain-length 16 --n-trajectories 1000 --n-steps 100 \
    --n-workers 32 --seed 42
```

| Metric | Estimate |
|--------|----------|
| Enumeration per worker | ~36s |
| Trajectories per worker | 1000 × 100 steps |
| Per-step time | ~0.92s |
| Per-worker wall time | ~26 hours |
| Total wall time (32 ligands, 32 workers) | ~26 hours |
| Output | 32 ligand directories, 32K total trajectories |

For 64 ligands with 32 workers (two rounds):
- Total wall time: ~52 hours

## Recommendations

1. **Ship 16-mers now.** The infrastructure is ready and wall times are reasonable.
2. **Implement numba enumeration** as a developer-experience improvement (faster iteration, faster benchmarks) and a prerequisite for longer chains.
3. **To reach 20-mers, additionally implement Bloom-style pruning** — score only high-contact conformations, use mean-field correction for the rest. Validate against full 16-mer scoring first.
4. **18-mers are an intermediate target** that works with full accuracy at ~7 days per 32-ligand batch, requiring no approximations but needing numba enumeration for acceptable startup time.
