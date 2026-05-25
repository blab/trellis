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

    # Precompute cumulative syn/nonsyn counts via BFS from root.
    cumulative_syn: dict[int, int] = {phy.root_id: 0}
    cumulative_nonsyn: dict[int, int] = {phy.root_id: 0}
    stack = [phy.root_id]
    while stack:
        nid = stack.pop()
        node = phy.nodes[nid]
        for child_id in children_of[nid]:
            child = phy.nodes[child_id]
            branch_syn = 0
            branch_nonsyn = 0
            for j in range(len(node.aa)):
                if node.aa[j] != child.aa[j]:
                    branch_nonsyn += 1
                elif node.dna[j*3:(j+1)*3] != child.dna[j*3:(j+1)*3]:
                    branch_syn += 1
            cumulative_syn[child_id] = cumulative_syn[nid] + branch_syn
            cumulative_nonsyn[child_id] = cumulative_nonsyn[nid] + branch_nonsyn
            stack.append(child_id)

    def build_subtree(node_id: int) -> dict:
        node = phy.nodes[node_id]
        parent = phy.nodes[node.parent_id] if node.parent_id is not None else None

        nuc_mutations = []
        aa_mutations = []
        if parent is not None:
            for i, (a, b) in enumerate(zip(parent.dna, node.dna)):
                if a != b:
                    nuc_mutations.append(f"{a}{i + 1}{b}")
            for j, (a, b) in enumerate(zip(parent.aa, node.aa)):
                if a != b:
                    aa_mutations.append(f"{a}{j + 1}{b}")

        name = f"TIP_{node_id:04d}" if node.is_tip else f"NODE_{node_id:04d}"
        subtree = {
            "name": name,
            "node_attrs": {
                "div": node.depth,
                "fitness": {"value": node.fitness},
                "nonsyn_muts": {"value": cumulative_nonsyn[node_id]},
                "syn_muts": {"value": cumulative_syn[node_id]},
            },
            "branch_attrs": {
                "mutations": {"nuc": nuc_mutations, "protein": aa_mutations},
            },
        }
        if children_of[node_id]:
            ordered = sorted(children_of[node_id], key=lambda c: tip_counts[c])
            subtree["children"] = [build_subtree(c) for c in ordered]
        return subtree

    root = phy.nodes[phy.root_id]
    default_title = f"Trellis phylogeny | ligand={phy.metadata.get('ligand_sequence', '?')}"
    auspice = {
        "version": "v2",
        "meta": {
            "title": title or default_title,
            "panels": ["tree", "entropy"],
            "colorings": [
                {"key": "fitness", "title": "Fitness", "type": "continuous"},
                {"key": "nonsyn_muts", "title": "Nonsynonymous mutations", "type": "continuous"},
                {"key": "syn_muts", "title": "Synonymous mutations", "type": "continuous"},
            ],
            "genome_annotations": {
                "nuc": {"start": 1, "end": len(root.dna), "type": "source", "strand": "+"},
                "protein": {"start": 1, "end": len(root.dna), "type": "CDS", "strand": "+"},
            },
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
