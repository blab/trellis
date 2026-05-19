"""Summary analyses for a completed batch of trellis trajectories."""

import argparse
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd

from trellis.genetic_code import translate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch_dir", type=Path)
    p.add_argument("--step-interval", type=int, default=10)
    return p.parse_args()


def _parse_fasta(text: str) -> list[str]:
    """Parse a FASTA string into a list of sequences."""
    seqs = []
    for line in text.strip().split("\n"):
        if not line.startswith(">"):
            seqs.append(line.strip())
    return seqs


def _read_shard(path: Path) -> list[list[str]]:
    """Read a tar.zst shard, return list of trajectories (each a list of DNA strings)."""
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


def load_trajectories(batch_dir: Path) -> tuple[list[list[str]], list[list[str]]]:
    """Read all shards, return (train_trajectories, test_trajectories)."""
    train, test = [], []
    for shard_path in sorted(batch_dir.glob("forwards-*.tar.zst")):
        trajs = _read_shard(shard_path)
        if "-train-" in shard_path.name:
            train.extend(trajs)
        else:
            test.extend(trajs)
    print(f"loaded {len(train)} train + {len(test)} test trajectories")
    return train, test


def pairwise_hamming(sequences: list[str]) -> np.ndarray:
    """Compute condensed pairwise Hamming distances for equal-length strings."""
    arr = np.array([list(s) for s in sequences], dtype="U1")
    n = len(arr)
    dists = []
    for i in range(n):
        diff = arr[i + 1 :] != arr[i]
        dists.append(diff.sum(axis=1))
    return np.concatenate(dists)


def cross_hamming_min(seqs_a: list[str], seqs_b: list[str]) -> int:
    """Minimum Hamming distance between any pair from two sequence sets."""
    arr_a = np.array([list(s) for s in seqs_a], dtype="U1")
    arr_b = np.array([list(s) for s in seqs_b], dtype="U1")
    min_dist = arr_a.shape[1]
    for row in arr_a:
        dists = (arr_b != row).sum(axis=1)
        min_dist = min(min_dist, dists.min())
    return int(min_dist)


def analyze_convergence(
    train: list[list[str]],
    test: list[list[str]],
    step_interval: int,
) -> dict:
    all_trajs = train + test
    n_steps = len(all_trajs[0]) - 1
    sampled_steps = list(range(0, n_steps + 1, step_interval))

    results = {"dna": {}, "aa": {}}
    for level in ("dna", "aa"):
        step_stats = []
        hist_steps = [0, 25, 50, 75, 100]
        hist_data = {}

        for step in sampled_steps:
            if level == "dna":
                seqs = [t[step] for t in all_trajs]
            else:
                seqs = [translate(t[step]) for t in all_trajs]

            dists = pairwise_hamming(seqs)
            stats = {
                "step": step,
                "mean": float(dists.mean()),
                "min": int(dists.min()),
                "p5": float(np.percentile(dists, 5)),
            }
            step_stats.append(stats)

            if step in hist_steps:
                hist_data[step] = dists

        # Train vs test minimum at final step
        if level == "dna":
            train_final = [t[n_steps] for t in train]
            test_final = [t[n_steps] for t in test]
        else:
            train_final = [translate(t[n_steps]) for t in train]
            test_final = [translate(t[n_steps]) for t in test]
        cross_min = cross_hamming_min(train_final, test_final)

        results[level] = {
            "step_stats": step_stats,
            "hist_data": hist_data,
            "cross_min": cross_min,
        }

    return results


def analyze_diversity(all_trajs: list[list[str]]) -> dict:
    n_steps = len(all_trajs[0])

    # Panel A & B: cumulative unique AAs across trajectories
    global_seen: set[str] = set()
    cumulative_counts = []
    marginal_counts = []
    for traj in all_trajs:
        aa_seqs = {translate(dna) for dna in traj}
        new = aa_seqs - global_seen
        marginal_counts.append(len(new))
        global_seen |= aa_seqs
        cumulative_counts.append(len(global_seen))

    # Panel C: unique AAs within each trajectory up to step t
    per_traj_unique = np.zeros((len(all_trajs), n_steps), dtype=int)
    for i, traj in enumerate(all_trajs):
        seen: set[str] = set()
        for t, dna in enumerate(traj):
            seen.add(translate(dna))
            per_traj_unique[i, t] = len(seen)

    return {
        "cumulative_counts": cumulative_counts,
        "marginal_counts": marginal_counts,
        "per_traj_mean": per_traj_unique.mean(axis=0),
        "per_traj_std": per_traj_unique.std(axis=0),
        "n_steps": n_steps,
    }


def plot_convergence(data: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    hist_colors = {0: "#4e79a7", 25: "#59a14f", 50: "#edc948",
                   75: "#f28e2b", 100: "#e15759"}

    for row, level in enumerate(("dna", "aa")):
        level_data = data[level]
        seq_label = "DNA" if level == "dna" else "AA"
        seq_len = None

        # Panel A: histograms at selected steps
        ax = axes[row, 0]
        for step, dists in sorted(level_data["hist_data"].items()):
            if seq_len is None:
                seq_len = int(dists.max()) + 5
            ax.hist(dists, bins=30, alpha=0.5, label=f"step {step}",
                    color=hist_colors[step], density=True)
        ax.set_xlabel(f"Pairwise {seq_label} Hamming distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(f"{seq_label} distance distributions")

        # Panel B: summary stats over steps
        ax = axes[row, 1]
        steps = [s["step"] for s in level_data["step_stats"]]
        means = [s["mean"] for s in level_data["step_stats"]]
        mins = [s["min"] for s in level_data["step_stats"]]
        p5s = [s["p5"] for s in level_data["step_stats"]]
        ax.plot(steps, means, "-", label="mean", color="#4e79a7")
        ax.plot(steps, mins, "--", label="min", color="#e15759")
        ax.plot(steps, p5s, ":", label="5th percentile", color="#f28e2b")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Pairwise {seq_label} Hamming distance")
        ax.legend(fontsize=8)
        ax.set_title(f"{seq_label} distance over time")

        # Panel C: train vs test cross minimum
        ax = axes[row, 2]
        cross_min = level_data["cross_min"]
        ax.bar([0], [cross_min], color="#4e79a7", width=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([f"Step {len(level_data['step_stats']) * 10 - 10}"])
        ax.set_ylabel(f"Min {seq_label} Hamming distance")
        ax.set_title(f"Train–test min {seq_label} distance = {cross_min}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_diversity(data: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: cumulative unique AAs
    ax = axes[0]
    ax.plot(data["cumulative_counts"], color="#4e79a7", linewidth=0.8)
    ax.set_xlabel("Trajectory index")
    ax.set_ylabel("Cumulative unique AA sequences")
    ax.set_title("Cumulative AA diversity")

    # Panel B: marginal contribution
    ax = axes[1]
    ax.scatter(range(len(data["marginal_counts"])), data["marginal_counts"],
               s=2, alpha=0.3, color="#4e79a7")
    ax.set_xlabel("Trajectory index")
    ax.set_ylabel("New unique AA sequences")
    ax.set_title("Marginal contribution per trajectory")

    # Panel C: within-trajectory unique AAs over steps
    ax = axes[2]
    steps = np.arange(data["n_steps"])
    mean = data["per_traj_mean"]
    std = data["per_traj_std"]
    ax.plot(steps, mean, color="#4e79a7")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="#4e79a7")
    ax.set_xlabel("Step")
    ax.set_ylabel("Unique AA sequences")
    ax.set_title("Within-trajectory AA diversity (mean ± SD)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    args = parse_args()

    train, test = load_trajectories(args.batch_dir)

    fig_dir = Path("analysis/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("analyzing convergence...")
    conv = analyze_convergence(train, test, args.step_interval)
    plot_convergence(conv, fig_dir / "convergence.png")

    print("analyzing diversity...")
    div = analyze_diversity(train + test)
    plot_diversity(div, fig_dir / "sequence_diversity.png")


if __name__ == "__main__":
    main()
