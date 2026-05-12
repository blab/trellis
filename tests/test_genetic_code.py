import pytest

from trellis.genetic_code import CODON_TABLE, NUCLEOTIDES, translate


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
