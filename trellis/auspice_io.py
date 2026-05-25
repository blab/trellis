"""Auspice v2 JSON serialization for Phylogeny objects."""

import json
from pathlib import Path

from trellis.phylogeny import Phylogeny


def phylogeny_to_auspice(phy: Phylogeny, title: str | None = None) -> dict:
    """Convert a Phylogeny to an Auspice v2 JSON dict."""
    children_of: dict[int, list[int]] = {n.id: [] for n in phy.nodes}
    for n in phy.nodes:
        if n.parent_id is not None:
            children_of[n.parent_id].append(n.id)

    tip_counts: dict[int, int] = {}

    def count_tips(node_id: int) -> int:
        if not children_of[node_id]:
            tip_counts[node_id] = 1
        else:
            tip_counts[node_id] = sum(count_tips(c) for c in children_of[node_id])
        return tip_counts[node_id]

    count_tips(phy.root_id)

    def build_subtree(node_id: int) -> dict:
        node = phy.nodes[node_id]
        parent = phy.nodes[node.parent_id] if node.parent_id is not None else None

        mutations = []
        if parent is not None:
            for i, (a, b) in enumerate(zip(parent.dna, node.dna)):
                if a != b:
                    mutations.append(f"{a}{i + 1}{b}")

        name = f"TIP_{node_id:04d}" if node.is_tip else f"NODE_{node_id:04d}"
        subtree = {
            "name": name,
            "node_attrs": {"div": node.depth},
            "branch_attrs": {"mutations": {"nuc": mutations}},
        }
        if children_of[node_id]:
            ordered = sorted(children_of[node_id], key=lambda c: tip_counts[c])
            subtree["children"] = [build_subtree(c) for c in ordered]
        return subtree

    default_title = f"Trellis phylogeny | ligand={phy.metadata.get('ligand_sequence', '?')}"
    auspice = {
        "version": "v2",
        "meta": {
            "title": title or default_title,
            "extensions": {"trellis": phy.metadata},
        },
        "tree": build_subtree(phy.root_id),
    }
    return auspice


def write_auspice_json(
    phy: Phylogeny, path: str | Path, title: str | None = None
) -> None:
    """Write a phylogeny to disk as Auspice v2 JSON."""
    auspice = phylogeny_to_auspice(phy, title=title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(auspice, f, indent=2)
        f.write("\n")
