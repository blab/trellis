import pytest

from trellis.lattice import (
    enumerate_saws,
    get_contacts,
    is_self_avoiding,
    occupied_sites,
)

# OEIS A001411: number of self-avoiding walks on the 2D square lattice
# starting at the origin, indexed by number of steps. A walk of n residues
# has n - 1 steps, so a SAW count for n residues is A001411[n - 1].
A001411 = [1, 4, 12, 36, 100, 284, 780, 2172, 5916, 16268, 44100, 120292]


@pytest.mark.parametrize("n", list(range(1, 12)))
def test_unreduced_saw_counts_match_oeis(n):
    count = sum(1 for _ in enumerate_saws(n, reduce_symmetry=False))
    assert count == A001411[n - 1]


@pytest.mark.parametrize("n", list(range(2, 12)))
def test_reduced_saw_counts_match_unreduced(n):
    # The 8-fold dihedral orbit collapses to 1 reduced walk for each
    # off-axis SAW. The unique fully-on-axis straight walk has only 4
    # rotational images (it's its own x-axis reflection), so:
    #   unreduced = 8 * (reduced - 1) + 4 = 8 * reduced - 4.
    #
    # NOTE: the reduced counts cited in the design notes (n=8 -> 88,
    # n=10 -> 200) appear to be placeholders and do not match this
    # relation; the OEIS-based formula here is the correct reference.
    reduced_count = sum(1 for _ in enumerate_saws(n, reduce_symmetry=True))
    assert 8 * reduced_count - 4 == A001411[n - 1]


@pytest.mark.parametrize("n", [1, 2, 3, 6, 9])
def test_reduced_saws_well_formed(n):
    seen: set = set()
    for conf in enumerate_saws(n, reduce_symmetry=True):
        assert len(conf) == n
        assert conf[0] == (0, 0)
        if n >= 2:
            assert conf[1] == (1, 0)
        assert is_self_avoiding(conf)
        assert conf not in seen
        seen.add(conf)


def test_get_contacts_simple_square():
    # 2x2 square: residues 0 and 3 are adjacent and non-bonded.
    conf = ((0, 0), (1, 0), (1, 1), (0, 1))
    assert get_contacts(conf) == [(0, 3)]


def test_get_contacts_excludes_bonded_neighbors():
    conf = ((0, 0), (1, 0), (2, 0), (3, 0))
    assert get_contacts(conf) == []


def test_get_contacts_multiple_in_lex_order():
    # U-shape; residue 5 is adjacent to both 0 and 2:
    #   0 - 1
    #   |   |
    #   5   2
    #   |   |
    #   4 - 3
    conf = ((0, 2), (1, 2), (1, 1), (1, 0), (0, 0), (0, 1))
    contacts = get_contacts(conf)
    assert contacts == [(0, 5), (2, 5)]


def test_is_self_avoiding_true():
    conf = ((0, 0), (1, 0), (1, 1), (0, 1))
    assert is_self_avoiding(conf) is True


def test_is_self_avoiding_false():
    conf = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
    assert is_self_avoiding(conf) is False


def test_occupied_sites():
    conf = ((0, 0), (1, 0), (1, 1))
    assert occupied_sites(conf) == {(0, 0), (1, 0), (1, 1)}
    assert len(occupied_sites(conf)) == len(conf)
