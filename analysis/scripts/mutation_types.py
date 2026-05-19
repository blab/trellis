"""Plot proportion of nonsynonymous mutations at each step across trajectories."""

import argparse
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd

from trellis.genetic_code import classify_mutation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch_dir", type=Path)
    return p.parse_args()


def _parse_fasta(text: str) -> list[str]:
    seqs = []
    for line in text.strip().split("\n"):
        if not line.startswith(">"):
            seqs.append(line.strip())
    return seqs


def _read_shard(path: Path) -> list[list[str]]:
    trajectories = []
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(mode="r|", fileobj=reader) as tar:
                for member in tar:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    seqs = _parse_fasta(f.read().decode())
                    if seqs:
                        trajectories.append(seqs)
    return trajectories


def load_trajectories(batch_dir: Path) -> list[list[str]]:
    trajs = []
    for shard_path in sorted(batch_dir.glob("forwards-*.tar.zst")):
        trajs.extend(_read_shard(shard_path))
    print(f"loaded {len(trajs)} trajectories")
    return trajs


def analyze_mutation_types(trajs: list[list[str]]) -> dict:
    n_steps = len(trajs[0]) - 1
    n_trajs = len(trajs)

    # 1 = nonsynonymous, 0 = synonymous, for each trajectory and step
    is_nonsyn = np.zeros((n_trajs, n_steps), dtype=float)

    for i, traj in enumerate(trajs):
        for t in range(1, len(traj)):
            mut_type = classify_mutation(traj[t - 1], traj[t])
            if mut_type == "nonsynonymous":
                is_nonsyn[i, t - 1] = 1.0

    return {
        "mean": is_nonsyn.mean(axis=0),
        "std": is_nonsyn.std(axis=0),
        "n_steps": n_steps,
    }


def plot_mutation_types(data: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    steps = np.arange(1, data["n_steps"] + 1)
    mean = data["mean"]

    ax.plot(steps, mean, color="#4e79a7", label="Nonsynonymous")
    ax.plot(steps, 1 - mean, color="#e15759", label="Synonymous")

    ax.set_xlabel("Step")
    ax.set_ylabel("Proportion of mutations")
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.set_title("Mutation type proportions over evolutionary time")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    args = parse_args()

    trajs = load_trajectories(args.batch_dir)

    fig_dir = Path("analysis/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("analyzing mutation types...")
    data = analyze_mutation_types(trajs)
    plot_mutation_types(data, fig_dir / "mutation_types.png")


if __name__ == "__main__":
    main()
