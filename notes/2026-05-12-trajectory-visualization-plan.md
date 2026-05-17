# Plan: Trajectory Visualization Dashboard

## Overview

Build a two-part visualization system: a Python script that runs a short SSWM trajectory and writes a JSON data file, and a standalone D3 HTML dashboard that reads the JSON and renders an interactive display. The pattern follows the VAE dashboard (`vae_dashboard_d3.html`): Python generates data, HTML/JS renders it, no server needed.

## Architecture

```
scripts/visualize_trajectory.py    → viz/viz_trajectory_data.json
viz/trajectory_dashboard.html      ← reads viz_trajectory_data.json
```

Open `viz/trajectory_dashboard.html` in a browser, it loads `../viz/viz_trajectory_data.json` via fetch.

## 1. Python script: `scripts/visualize_trajectory.py`

### Purpose

Generate a short SSWM trajectory, re-fold each unique AA sequence to get native conformations, and write everything to a single JSON file.

### CLI interface

```bash
python scripts/visualize_trajectory.py \
    --n-codons 12 \
    --ligand-sequence FWYL \
    --ligand-anchor 0,-1 \
    --n-steps 50 \
    --Ne 1000 \
    --temperature 1.0 \
    --seed 42 \
    --output viz/viz_trajectory_data.json
```

Defaults: `--n-codons 12` (shorter chain for speed — 20 residues would be slow without pre-enumeration), `--n-steps 50`, `--Ne 1000`, `--temperature 1.0`.

### Implementation

```python
import argparse
import json

from trellis.energy import AA_INDEX, load_mj_matrix
from trellis.fitness import compute_fitness_aa
from trellis.fold import fold
from trellis.genetic_code import classify_mutation, translate
from trellis.lattice import get_contacts
from trellis.ligand import Ligand, binding_contacts, create_ligand
from trellis.sswm import generate_start_sequence, generate_trajectory


def main():
    args = parse_args()
    mj = load_mj_matrix()
    ligand = create_ligand(args.ligand_sequence, anchor=tuple(args.ligand_anchor))
    rng = np.random.default_rng(args.seed)

    # Generate trajectory
    start_dna = generate_start_sequence(
        args.n_codons, ligand, mj, min_fitness=0.0, temperature=args.temperature, rng=rng
    )
    trajectory = generate_trajectory(
        start_dna, ligand, mj,
        n_steps=args.n_steps, Ne=args.Ne, temperature=args.temperature, rng=rng
    )

    # Re-fold each unique AA sequence to get native conformations
    unique_aa = set(trajectory.aa_sequences)
    fold_results = {}
    for aa in unique_aa:
        result = fold(aa, mj, ligand, args.temperature)
        b_contacts = binding_contacts(result.native_conformation, ligand)
        i_contacts = get_contacts(result.native_conformation)
        fold_results[aa] = {
            "conformation": [list(pos) for pos in result.native_conformation],
            "native_energy": result.native_energy,
            "ensemble_binding_energy": result.ensemble_binding_energy,
            "intra_contacts": [[i, j] for i, j in i_contacts],
            "intra_contact_energies": [
                float(mj[AA_INDEX[aa[i]], AA_INDEX[aa[j]]]) for i, j in i_contacts
            ],
            "binding_contacts": [[i, k] for i, k in b_contacts],
            "binding_contact_energies": [
                float(mj[AA_INDEX[aa[i]], AA_INDEX[ligand.sequence[k]]]) for i, k in b_contacts
            ],
        }

    # Assemble output
    data = {
        "metadata": {
            "n_codons": args.n_codons,
            "n_steps": trajectory.metadata["n_steps_completed"],
            "Ne": args.Ne,
            "temperature": args.temperature,
            "seed": args.seed,
            "ligand_sequence": args.ligand_sequence,
        },
        "ligand": {
            "sequence": ligand.sequence,
            "positions": [list(pos) for pos in ligand.positions],
        },
        "steps": [],  # per-step data
        "conformations": fold_results,  # keyed by AA sequence
    }

    for i in range(len(trajectory.dna_sequences)):
        step = {
            "step": i,
            "dna_sequence": trajectory.dna_sequences[i],
            "aa_sequence": trajectory.aa_sequences[i],
            "fitness": trajectory.fitness_values[i],
        }
        if i > 0:
            step["mutation_type"] = trajectory.mutation_types[i - 1]
        else:
            step["mutation_type"] = None
        data["steps"].append(step)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
```

### JSON output structure

```json
{
  "metadata": {
    "n_codons": 12,
    "n_steps": 50,
    "Ne": 1000,
    "temperature": 1.0,
    "seed": 42,
    "ligand_sequence": "FWYL"
  },
  "ligand": {
    "sequence": "FWYL",
    "positions": [[0, -1], [1, -1], [2, -1], [3, -1]]
  },
  "steps": [
    {
      "step": 0,
      "dna_sequence": "GCTTGT...",
      "aa_sequence": "ACDEF...",
      "fitness": 3.42,
      "mutation_type": null
    },
    {
      "step": 1,
      "dna_sequence": "GCTTGC...",
      "aa_sequence": "ACDEF...",
      "fitness": 3.42,
      "mutation_type": "synonymous"
    }
  ],
  "conformations": {
    "ACDEF...": {
      "conformation": [[0, 0], [1, 0], [2, 0], ...],
      "native_energy": -15.3,
      "ensemble_binding_energy": -8.7,
      "intra_contacts": [[0, 3], [1, 4]],
      "intra_contact_energies": [-3.45, -2.10],
      "binding_contacts": [[0, 0], [2, 1]],
      "binding_contact_energies": [-4.20, -3.85]
    }
  }
}
```

Note: `conformations` is keyed by AA sequence. Multiple steps can share the same AA sequence (synonymous mutations). The dashboard looks up the conformation by `step.aa_sequence`.

## 2. HTML dashboard: `viz/trajectory_dashboard.html`

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Trellis: SSWM Trajectory Viewer                                     │
│                                                                      │
│  Metadata summary (n_codons, n_steps, Ne, T, ligand, seed)          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Fitness Trajectory                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ●───●───●───●───●───●       (line + dots)                    │  │
│  │                               color by mutation type           │  │
│  │                               hover highlights dot +           │  │
│  │                               corresponding small multiple     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Lattice Conformations                                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ...  │
│  │ Step 0  │ │ Step 5  │ │ Step 10 │ │ Step 15 │ │ Step 20 │       │
│  │  ○─○    │ │  ○─○    │ │  ○─○    │ │  ○─○    │ │  ○─○    │       │
│  │  │ │    │ │  │ │    │ │  │ │    │ │  │ │    │ │  │ │    │       │
│  │  ○─○    │ │  ○─○    │ │  ○─○    │ │  ○─○    │ │  ○─○    │       │
│  │ ■ ■ ■ ■ │ │ ■ ■ ■ ■ │ │ ■ ■ ■ ■ │ │ ■ ■ ■ ■ │ │ ■ ■ ■ ■ │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│  (6 per row × 4 rows max = 24 panels, evenly sampled from traj)    │
│                                                                      │
│  Step Detail (updates on hover)                                      │
│  Step 15: AA = ACDEF... | Fitness = 5.32 | Mutation: nonsynonymous  │
└──────────────────────────────────────────────────────────────────────┘
```

### Dependencies

Load from CDN (same pattern as the VAE dashboard):

```html
<script type="module">
    import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
</script>
```

No Observable Plot needed — this is pure D3 since the lattice diagrams are custom SVG.

### Styling

Follow the VAE dashboard conventions:

- White `container` with `max-width: 1200px`, light gray `#f5f5f5` background
- System font stack (`-apple-system, BlinkMacSystemFont, ...`)
- Section headers with `border-bottom: 1px solid #e1e4e8`
- `plot-container` class for chart areas with `background: #fafafa`
- Metrics row with CSS grid for metadata summary
- Tooltip div positioned absolutely, shown/hidden on hover

### Data loading

```javascript
const DATA_PATH = "../viz/viz_trajectory_data.json";
let data = null;

async function loadData() {
    const response = await fetch(DATA_PATH);
    data = await response.json();
    renderAll();
}
```

### Panel 1: Fitness trajectory

**Container:** SVG, width 1140px (container width minus padding), height 250px.

**Axes:** x = step number (0 to n_steps), y = fitness. Standard D3 linear scales with axis labels "Step" and "Fitness (−⟨E_bind⟩)".

**Line:** `d3.line()` connecting all steps. Stroke color: `#999`, thin (1px).

**Dots:** Circle at each step, radius 4px. Color by `mutation_type`:
- `null` (step 0): `#999` gray
- `"synonymous"`: `#4393c3` blue
- `"nonsynonymous"`: `#d6604d` red
- `"nonsense"`: should not appear in a valid trajectory, but `#000` black as fallback

**Highlighted dots:** The ~24 steps shown as small multiples get slightly larger dots (radius 6px) with a darker stroke to indicate they have a corresponding lattice panel below.

**Hover behavior:** On mouseover of any dot, show a tooltip with step number, fitness, AA sequence, mutation type. If the dot corresponds to a small multiple panel, highlight that panel with a border change.

**Legend:** Small inline legend below the x-axis showing the three mutation type colors.

### Panel 2: Lattice conformation small multiples

**Step sampling:** Given `n_steps + 1` total frames and a maximum of 24 panels, compute the stride: `stride = Math.max(1, Math.floor(n_steps / 23))`. Always include step 0 and the final step. Sample at `[0, stride, 2*stride, ..., n_steps]`, capped at 24.

**Grid layout:** 6 columns, up to 4 rows. Each panel is a square SVG, approximately 170×170px (allowing for gaps in a 1140px-wide container with 6 columns).

**Lattice coordinate mapping:** Each panel has its own viewport computed from the conformation coordinates + ligand positions. Compute the bounding box of all points (protein + ligand), add 1 unit padding, then map lattice coordinates to pixel coordinates with `d3.scaleLinear`. Ensure square aspect ratio.

**Drawing order (back to front):**

1. **Grid lines** (optional, very faint): light gray lines at integer lattice positions. Only draw within the bounding box. Stroke: `#eee`, 0.5px. This helps orient the eye but shouldn't dominate.

2. **Intra-protein contacts (non-bonded):** Dashed lines between residue pairs from `intra_contacts` in the JSON. Line width scaled by absolute MJ energy from `intra_contact_energies`: `strokeWidth = scale(|energy|)` where scale maps [0, 7] → [0.5, 3]. Color: `#ccc`. These show the internal stabilizing contacts.

3. **Protein-ligand contacts:** Dashed lines between protein residue and ligand residue from `binding_contacts` in the JSON. Line width scaled by absolute MJ energy from `binding_contact_energies`, same scale. Color: `#e6550d` (orange) — distinct from intra-protein contacts to highlight binding.

4. **Protein backbone:** Solid lines between consecutive residues (chain bonds). Stroke: `#333`, 2px.

5. **Ligand residues:** Squares (not circles) at ligand positions. Fill: `#756bb1` (purple). Side length matching protein residue diameter. Single-letter AA label inside or adjacent.

6. **Protein residues:** Circles at conformation positions. Fill by amino acid hydrophobicity: hydrophobic (A, F, I, L, M, V, W, Y) in `#fc8d59` (warm orange), polar (C, G, N, P, Q, S, T) in `#91bfdb` (cool blue), charged (D, E, H, K, R) in `#fee090` (yellow). Radius 8px. Thin stroke `#333`. Single-letter AA label inside each circle (font-size 8px, centered).

7. **Panel label:** "Step N" below each panel, with fitness value. Font-size 11px, gray.

**Intra-protein contacts and energies:** The JSON includes `intra_contacts` (list of `[i, j]` pairs from `get_contacts(native_conformation)`) and `intra_contact_energies` (the MJ energy for each pair), computed in Python by `visualize_trajectory.py`. Similarly, `binding_contacts` and `binding_contact_energies` are precomputed. This keeps the JS simple — it reads the energy value for each contact and maps directly to line width, with no need for the MJ matrix in the browser.

Add to the `fold_results` dict in `visualize_trajectory.py`:

```python
from trellis.lattice import get_contacts
intra = get_contacts(result.native_conformation)
fold_results[aa]["intra_contacts"] = [[i, j] for i, j in intra]
fold_results[aa]["intra_contact_energies"] = [
    float(mj[AA_INDEX[aa[i]], AA_INDEX[aa[j]]]) for i, j in intra
]
fold_results[aa]["binding_contact_energies"] = [
    float(mj[AA_INDEX[aa[i]], AA_INDEX[ligand.sequence[k]]]) for i, k in contacts
]
```

### Hover interaction between panels

**Trajectory → lattice:** When hovering a dot in the fitness trajectory that corresponds to a sampled step, highlight the matching small multiple panel (add a colored border, e.g., 2px solid `#0969da`). If the dot is not a sampled step, highlight the nearest sampled panel.

**Lattice → trajectory:** When hovering a small multiple panel, highlight the corresponding dot in the fitness trajectory (increase radius, add stroke). Show the tooltip with step details.

**Implementation:** Use D3 event handlers (`.on("mouseover", ...)` and `.on("mouseout", ...)`). Give each small multiple panel a `data-step` attribute and each trajectory dot a `data-step` attribute for cross-referencing.

### Step detail bar

A single line of text below the small multiples that updates on hover:

```
Step 15  |  AA: ACDEFGHIKLMN  |  Fitness: 5.32  |  ΔFitness: +0.15  |  Mutation: nonsynonymous  |  Binding contacts: 3
```

Implemented as a `<div>` with `id="step-detail"`, updated in the mouseover handler.

### Tooltip

Follow the VAE dashboard tooltip pattern:

```html
<div id="tooltip"></div>
```

Positioned absolutely near the cursor. Shows on trajectory dot hover:
- Step number
- Fitness (and Δ from previous step)
- AA sequence (truncated if long, with full on click/expand)
- DNA mutation position and type (if not step 0)
- Binding energy (from conformations lookup)

Styled to match `#eval-tooltip` from the VAE dashboard.

## 3. Color scheme summary

| Element | Color | Hex |
|---------|-------|-----|
| Synonymous mutation dot | Blue | `#4393c3` |
| Nonsynonymous mutation dot | Red | `#d6604d` |
| Step 0 dot | Gray | `#999` |
| Protein backbone | Dark gray | `#333` |
| Intra-protein contact | Light gray dashed | `#ccc` |
| Protein-ligand contact | Orange dashed | `#e6550d` |
| Ligand residues | Purple square | `#756bb1` |
| Hydrophobic residues | Warm orange circle | `#fc8d59` |
| Polar residues | Cool blue circle | `#91bfdb` |
| Charged residues | Yellow circle | `#fee090` |
| Highlight border | Blue | `#0969da` |

## 4. File placement

```
trellis/
├── scripts/
│   └── visualize_trajectory.py
├── viz/
│   └── trajectory_dashboard.html
└── results/                         # gitignored
    └── viz_trajectory_data.json
```

The `viz/` directory is new. The `results/` directory is for generated artifacts and should be in `.gitignore`. To use: run the script, then serve the repo root with `python -m http.server` and open `viz/trajectory_dashboard.html` in the browser.

## 5. Implementation order

1. **`scripts/visualize_trajectory.py`** — Python script with argparse CLI. Generate trajectory, re-fold, write JSON with all fields (conformations, intra contacts, binding contacts, energies). Test by running with `--n-codons 8 --n-steps 20` and inspecting the JSON.

2. **`viz/trajectory_dashboard.html`** — Start with the skeleton: HTML structure, CSS (copied/adapted from VAE dashboard), data loading via fetch, metadata display.

3. **Fitness trajectory panel** — D3 line + dots with mutation type coloring. Tooltip on hover. This is a standard line chart.

4. **Lattice small multiples** — Step sampling logic, bounding box computation, coordinate scaling, drawing layers (grid → contacts → backbone → residues → labels). This is the bulk of the D3 work.

5. **Hover interaction** — Wire up cross-panel highlighting: trajectory dots ↔ lattice panels. Step detail bar updates.

6. **Polish** — Legend, edge cases (trajectory shorter than 24 steps → show all steps), SVG export button (following VAE dashboard pattern).

## 6. Testing

- Run `visualize_trajectory.py` with `--n-codons 8 --n-steps 20 --seed 42` — should complete in under a minute and produce valid JSON.
- Open `viz/trajectory_dashboard.html` in browser (via `python -m http.server` from repo root) — should render without errors.
- Verify fitness trajectory shows increasing trend (SSWM should climb fitness).
- Verify lattice panels show protein avoiding ligand sites.
- Verify binding contacts (orange dashed lines) appear between protein and ligand.
- Verify hover interaction works bidirectionally.
- Verify synonymous steps show identical conformations (same AA → same panel content).
