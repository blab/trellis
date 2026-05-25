"""Tests for phylogeny — tree generation via Yule-with-sampling on SSWM."""

import numpy as np
import pytest

from trellis.cache import FitnessCache
from trellis.energy import load_mj_matrix
from trellis.fold_enum import enumerate_conformations
from trellis.genetic_code import translate
from trellis.ligand import create_ligand
from trellis.phylogeny import Phylogeny, PhylogenyNode, generate_phylogeny
from trellis.sswm import generate_start_sequence


@pytest.fixture
def setup():
    mj = load_mj_matrix()
    ligand = create_ligand("FW", anchor=(0, -1))
    db = enumerate_conformations(10, ligand)
    rng = np.random.default_rng(42)
    start_dna = generate_start_sequence(
        10, ligand, mj, min_fitness=0.0, temperature=1.0, rng=rng, db=db,
    )
    return mj, ligand, db, start_dna


def _generate(setup, seed=42, **kwargs):
    mj, ligand, db, start_dna = setup
    defaults = dict(
        n_steps=20, Ne=50.0, beta=0.08, psi=0.05,
        temperature=1.0, min_active_lineages=1,
        max_active_lineages=50, max_total_nodes=500,
    )
    defaults.update(kwargs)
    return generate_phylogeny(
        start_dna=start_dna, ligand=ligand, mj_matrix=mj, db=db,
        rng=np.random.default_rng(seed), fitness_cache=FitnessCache(),
        **defaults,
    )


def test_root_structure(setup):
    phy = _generate(setup)
    root = phy.nodes[phy.root_id]
    assert root.parent_id is None
    assert root.depth == 0
    assert not root.is_tip


def test_node_ids_unique(setup):
    phy = _generate(setup)
    ids = [n.id for n in phy.nodes]
    assert len(ids) == len(set(ids))


def test_parent_ids_valid(setup):
    phy = _generate(setup)
    node_ids = {n.id for n in phy.nodes}
    for n in phy.nodes:
        if n.parent_id is not None:
            assert n.parent_id in node_ids
            assert n.parent_id < n.id


def test_tips_have_no_children(setup):
    phy = _generate(setup)
    children_of = {n.id: [] for n in phy.nodes}
    for n in phy.nodes:
        if n.parent_id is not None:
            children_of[n.parent_id].append(n.id)
    for n in phy.nodes:
        if n.is_tip:
            assert children_of[n.id] == []


def test_non_tips_have_children(setup):
    phy = _generate(setup)
    children_of = {n.id: [] for n in phy.nodes}
    for n in phy.nodes:
        if n.parent_id is not None:
            children_of[n.parent_id].append(n.id)
    for n in phy.nodes:
        if not n.is_tip:
            assert len(children_of[n.id]) > 0


def test_deterministic_with_seed(setup):
    phy1 = _generate(setup, seed=99)
    phy2 = _generate(setup, seed=99)
    assert len(phy1.nodes) == len(phy2.nodes)
    for n1, n2 in zip(phy1.nodes, phy2.nodes):
        assert n1.dna == n2.dna
        assert n1.is_tip == n2.is_tip


def test_max_total_nodes_cap(setup):
    phy = _generate(setup, max_total_nodes=50, beta=0.3, n_steps=50)
    # Cap is checked at start of each depth; daughters created that depth may overshoot
    # slightly, but the tree won't grow unboundedly.
    assert len(phy.nodes) <= 50 + 50  # max_total_nodes + max_active_lineages


def test_max_active_lineages_cap(setup):
    phy = _generate(setup, max_active_lineages=5, beta=0.5, n_steps=30)
    children_of = {n.id: [] for n in phy.nodes}
    for n in phy.nodes:
        if n.parent_id is not None:
            children_of[n.parent_id].append(n.id)
    for depth in range(1, 31):
        active_at_depth = [
            n for n in phy.nodes
            if n.depth == depth and not n.is_tip
        ]
        assert len(active_at_depth) <= 5


def test_tips_across_depths(setup):
    phy = _generate(setup, n_steps=20, beta=0.3, psi=0.1, min_active_lineages=2)
    tip_depths = [n.depth for n in phy.nodes if n.is_tip]
    assert len(set(tip_depths)) > 1


def test_min_active_prevents_sampling_extinction(setup):
    phy = _generate(setup, n_steps=15, beta=0.1, psi=0.5, min_active_lineages=2)
    assert max(n.depth for n in phy.nodes if n.is_tip) > 5


def test_sequences_valid(setup):
    phy = _generate(setup)
    for n in phy.nodes:
        assert len(n.dna) == 30
        assert all(c in "ACGT" for c in n.dna)
        assert n.aa == translate(n.dna)
        assert "*" not in n.aa


def test_metadata_fields(setup):
    phy = _generate(setup)
    assert phy.metadata["n_steps"] == 20
    assert phy.metadata["Ne"] == 50.0
    assert phy.metadata["beta"] == 0.08
    assert phy.metadata["psi"] == 0.05
    assert phy.metadata["min_active_lineages"] == 1
    assert phy.metadata["n_nodes"] == len(phy.nodes)
    assert phy.metadata["n_tips"] == sum(1 for n in phy.nodes if n.is_tip)
