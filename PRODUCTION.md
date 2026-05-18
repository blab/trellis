# Production

## Single ligand

Initial production runs focus on many trajectories for a single ligand. Running `generate_trajectories.py` with fixed `--ligand-sequence` should cause trajectories to be split over `--n-workers`.

On Conatus server

```
pip install --user --break-system-packages .
```

```
python3 scripts/generate_trajectories.py \
    --ligand-sequence KEMN --chain-length 18 \
    --n-trajectories 1000 --n-steps 100 \
    --n-workers 32 --Ne 50 --seed 101
```
