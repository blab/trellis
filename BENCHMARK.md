# Benchmark

Single-worker pipeline benchmark for trajectory generation.
Run with `python scripts/benchmark.py` (defaults: 16-mer, FWYL ligand, 3 trajectories x 100 steps).

## 16-mers

| Parameter | Value |
|-----------|-------|
| Chain length | 16 |
| Ligand | FWYL (anchor (0,-1), horizontal) |
| Trajectories | 3 |
| Steps per trajectory | 100 |
| Ne | 1000 |
| Temperature | 1.0 |
| Seed | 42 |

### Results

| Date | Commit | Enumeration | 3 traj (100 steps) | Per step | Notes |
|------|--------|-------------|---------------------|----------|-------|
| 2026-05-16 | a7e87db | 36.1s | 384.9s | 1.28s | baseline |
