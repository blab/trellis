"""Tests for Auspice v2 JSON serialization."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from trellis.auspice_io import phylogeny_to_auspice, write_auspice_json
from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.ligand import create_ligand
from trellis.phylogeny import generate_phylogeny
from trellis.sswm import generate_start_sequence


@pytest.fixture
def phy():
    mj = load_mj_matrix()
    ligand = create_ligand("FW", anchor=(0, -1))
    db = enumerate_conformations(10, ligand)
    rng = np.random.default_rng(42)
    start_dna = generate_start_sequence(
        10, ligand, mj, min_fitness=0.0, temperature=1.0, rng=rng, db=db,
    )
    return generate_phylogeny(
        start_dna=start_dna, ligand=ligand, mj_matrix=mj, db=db,
        n_steps=20, Ne=50.0, beta=0.2, psi=0.05,
        temperature=1.0, min_active_lineages=1,
        max_active_lineages=50, max_total_nodes=500,
        rng=np.random.default_rng(42), fitness_cache=FitnessCache(),
    )


def test_top_level_structure(phy):
    auspice = phylogeny_to_auspice(phy)
    assert auspice["version"] == "v2"
    assert "meta" in auspice
    assert "tree" in auspice
    assert "title" in auspice["meta"]
    assert "trellis" in auspice["meta"]["extensions"]


def test_custom_title(phy):
    auspice = phylogeny_to_auspice(phy, title="Custom Title")
    assert auspice["meta"]["title"] == "Custom Title"


def test_tree_root_name(phy):
    auspice = phylogeny_to_auspice(phy)
    assert auspice["tree"]["name"] == "NODE_0000"


def test_div_matches_depth(phy):
    auspice = phylogeny_to_auspice(phy)

    def check_div(subtree, expected_depth=0):
        assert subtree["node_attrs"]["div"] == expected_depth
        for child in subtree.get("children", []):
            check_div(child, expected_depth + 1)

    check_div(auspice["tree"])


def test_mutations_match_sequences(phy):
    auspice = phylogeny_to_auspice(phy)
    node_by_name = {f"{'TIP' if n.is_tip else 'NODE'}_{n.id:04d}": n for n in phy.nodes}

    def check_mutations(subtree, parent_dna=None):
        name = subtree["name"]
        node = node_by_name[name]
        mutations = subtree["branch_attrs"]["mutations"]["nuc"]

        if parent_dna is not None:
            expected = []
            for i, (a, b) in enumerate(zip(parent_dna, node.dna)):
                if a != b:
                    expected.append(f"{a}{i + 1}{b}")
            assert mutations == expected
        else:
            assert mutations == []

        for child in subtree.get("children", []):
            check_mutations(child, node.dna)

    check_mutations(auspice["tree"])


def test_tips_have_no_children(phy):
    auspice = phylogeny_to_auspice(phy)

    def check_tips(subtree):
        if subtree["name"].startswith("TIP_"):
            assert "children" not in subtree
        for child in subtree.get("children", []):
            check_tips(child)

    check_tips(auspice["tree"])


def test_write_and_reload(phy):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        write_auspice_json(phy, path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
    assert loaded["version"] == "v2"
    assert loaded["meta"]["extensions"]["trellis"]["n_nodes"] == len(phy.nodes)


def test_node_count_matches(phy):
    auspice = phylogeny_to_auspice(phy)

    def count_nodes(subtree):
        return 1 + sum(count_nodes(c) for c in subtree.get("children", []))

    assert count_nodes(auspice["tree"]) == len(phy.nodes)
