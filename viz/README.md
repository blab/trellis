# Trajectory visualization

Interactive D3 dashboard for inspecting an SSWM trajectory: a fitness-vs-step
plot and small-multiple lattice diagrams of each native conformation.

## Usage

1. Generate trajectory data:

   ```bash
   python scripts/generate_trajectory.py
   ```

   This runs from defaults (`--n-codons 10 --n-steps 30 --Ne 100
   --temperature 1.0 --seed 42 --ligand-sequence FWYL`) and writes
   `results/trajectory_data.json`.

   Shorter chains (`--n-codons 8`) run in seconds, while 12-codons takes a 
   few minutes because every unique AA neighbour of every step is folded.

2. Serve the repo root over HTTP (needed because the dashboard fetches the
   JSON file — `file://` URLs would be blocked by CORS):

   ```bash
   python -m http.server 8000
   ```

3. Open <http://localhost:8000/viz/trajectory_dashboard.html>.

## What's shown

- **Metadata bar** — parameters from the CLI run.
- **Fitness trajectory** — line plus dot per step. Dot color encodes the
  mutation that led to that step: gray (start), blue (synonymous), red
  (nonsynonymous). Dots with a black outline correspond to the lattice
  panels below.
- **Lattice conformations** — up to 24 panels sampled across the
  trajectory. Each shows the native conformation: hydrophobic / polar /
  charged residues as colored circles, ligand as purple squares, backbone
  as solid lines, intra-protein contacts as dashed gray, protein-ligand
  contacts as dashed orange (line width scaled by MJ contact energy).
- **Step detail** — text line under the lattice grid updates on hover.

Hovering a trajectory dot highlights the nearest lattice panel and vice
versa; tooltip shows step number, fitness (and Δ), AA sequence, mutation
type, and binding-contact count.

## Regenerating after parameter changes

Edit-and-rerun loop: change the CLI args, rerun
`scripts/generate_trajectory.py`, then reload the browser tab. The HTML
re-fetches the JSON on every load.
