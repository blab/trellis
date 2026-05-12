import pytest

from trellis.genetic_code import (
    CODON_TABLE,
    NUCLEOTIDES,
    classify_mutation,
    mutant_aa_sequences,
    single_nt_mutations,
    translate,
)


# ---------------------------------------------------------------------------
# Codon table structure
# ---------------------------------------------------------------------------

def test_codon_table_has_64_entries():
    assert len(CODON_TABLE) == 64


def test_codon_table_has_3_stops():
    stops = [c for c, aa in CODON_TABLE.items() if aa == "*"]
    assert sorted(stops) == ["TAA", "TAG", "TGA"]


def test_codon_table_covers_20_amino_acids():
    aas = {aa for aa in CODON_TABLE.values() if aa != "*"}
    assert len(aas) == 20


def test_codon_table_keys_are_valid_triplets():
    for codon in CODON_TABLE:
        assert len(codon) == 3
        assert all(base in NUCLEOTIDES for base in codon)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codon,expected", [
    ("ATG", "M"),
    ("TTT", "F"),
    ("TGG", "W"),
    ("TAA", "*"),
    ("TAG", "*"),
    ("TGA", "*"),
    ("GCT", "A"),
    ("AAA", "K"),
])
def test_translate_known_codons(codon, expected):
    assert translate(codon) == expected


def test_translate_full_sequence():
    dna = "GCTTGTGATGAATTTGGTCATATTAAACTTATGAATCCGCAACGTTCTACTGTTTGGTAT"
    aa = translate(dna)
    assert len(aa) == 20
    assert aa == "ACDEFGHIKLMNPQRSTVWY"


def test_translate_stop_codon_in_middle():
    assert translate("ATGTAAATG") == "M*M"


def test_translate_empty():
    assert translate("") == ""


def test_translate_not_divisible_by_3():
    with pytest.raises(ValueError, match="not divisible by 3"):
        translate("ATGA")


def test_translate_invalid_base():
    with pytest.raises(ValueError, match="invalid codon"):
        translate("AXG")


# ---------------------------------------------------------------------------
# Single-nucleotide mutations
# ---------------------------------------------------------------------------

# DNA encoding ACDEFG (6 AA = 18 nt)
_DNA_SHORT = "GCTTGTGATGAATTTGGT"
# DNA encoding ACDEFGHIKLMNPQRSTVWY (20 AA = 60 nt)
_DNA_FULL = "GCTTGTGATGAATTTGGTCATATTAAACTTATGAATCCGCAACGTTCTACTGTTTGGTAT"


def test_single_nt_mutations_count():
    assert len(single_nt_mutations(_DNA_FULL)) == 180


def test_single_nt_mutations_count_short():
    assert len(single_nt_mutations(_DNA_SHORT)) == 18 * 3


def test_single_nt_mutations_all_differ_at_one_position():
    for mutant, pos, ref, alt in single_nt_mutations(_DNA_SHORT):
        diffs = [i for i in range(len(_DNA_SHORT)) if mutant[i] != _DNA_SHORT[i]]
        assert diffs == [pos]


def test_single_nt_mutations_ref_alt_correct():
    for mutant, pos, ref, alt in single_nt_mutations(_DNA_SHORT):
        assert ref == _DNA_SHORT[pos]
        assert alt == mutant[pos]
        assert ref != alt


def test_single_nt_mutations_no_duplicates():
    mutants = [m for m, _, _, _ in single_nt_mutations(_DNA_SHORT)]
    assert len(mutants) == len(set(mutants))


def test_single_nt_mutations_invalid_base_raises():
    with pytest.raises(ValueError, match="invalid base"):
        single_nt_mutations("ACXG")


# ---------------------------------------------------------------------------
# Mutation classification
# ---------------------------------------------------------------------------

def test_classify_synonymous():
    # GCT → GCC: both Ala
    assert classify_mutation("GCT", "GCC") == "synonymous"


def test_classify_nonsynonymous():
    # GCT (Ala) → TCT (Ser)
    assert classify_mutation("GCT", "TCT") == "nonsynonymous"


def test_classify_nonsense():
    # TAT (Tyr) → TAA (Stop)
    assert classify_mutation("TAT", "TAA") == "nonsense"


def test_classify_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        classify_mutation("GCT", "GCTA")


def test_classify_multiple_differences_raises():
    with pytest.raises(ValueError, match="exactly 1 position"):
        classify_mutation("GCT", "TCA")


# ---------------------------------------------------------------------------
# Mutant AA sequences (deduplication)
# ---------------------------------------------------------------------------

def test_mutant_aa_sequences_total_mutants():
    groups = mutant_aa_sequences(_DNA_SHORT)
    total = sum(len(v) for v in groups.values())
    assert total == len(_DNA_SHORT) * 3


def test_mutant_aa_sequences_synonymous_grouped():
    wt_aa = translate(_DNA_SHORT)
    groups = mutant_aa_sequences(_DNA_SHORT)
    assert wt_aa in groups
    for dna in groups[wt_aa]:
        assert translate(dna) == wt_aa


def test_mutant_aa_sequences_excludes_wildtype_dna():
    groups = mutant_aa_sequences(_DNA_SHORT)
    for dna_list in groups.values():
        assert _DNA_SHORT not in dna_list


def test_mutant_aa_sequences_keys_are_valid_proteins():
    groups = mutant_aa_sequences(_DNA_SHORT)
    for aa in groups:
        assert len(aa) == len(_DNA_SHORT) // 3
