"""Standard genetic code: translation, mutation enumeration, classification."""

NUCLEOTIDES = "ACGT"

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate(dna_sequence: str) -> str:
    """Translate a DNA sequence to amino acids using the standard genetic code.

    Stop codons are translated to ``"*"`` and included in the output
    (not truncated).
    """
    if len(dna_sequence) % 3 != 0:
        raise ValueError(
            f"DNA sequence length {len(dna_sequence)} is not divisible by 3"
        )
    codons = []
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i : i + 3]
        if codon not in CODON_TABLE:
            raise ValueError(f"invalid codon {codon!r} at position {i}")
        codons.append(CODON_TABLE[codon])
    return "".join(codons)


def single_nt_mutations(
    dna_sequence: str,
) -> list[tuple[str, int, str, str]]:
    """Enumerate all single-nucleotide mutations of *dna_sequence*.

    Returns a list of ``(mutant_dna, position, ref_base, alt_base)``
    tuples, ordered by position then by ``NUCLEOTIDES`` order.
    """
    for i, base in enumerate(dna_sequence):
        if base not in NUCLEOTIDES:
            raise ValueError(f"invalid base {base!r} at position {i}")
    seq = list(dna_sequence)
    mutations: list[tuple[str, int, str, str]] = []
    for i, ref in enumerate(seq):
        for alt in NUCLEOTIDES:
            if alt == ref:
                continue
            seq[i] = alt
            mutations.append(("".join(seq), i, ref, alt))
            seq[i] = ref
    return mutations


def classify_mutation(dna_ref: str, dna_alt: str) -> str:
    """Classify a single-nucleotide mutation.

    Returns ``"synonymous"``, ``"nonsynonymous"``, or ``"nonsense"``.
    """
    if len(dna_ref) != len(dna_alt):
        raise ValueError("sequences must be the same length")
    if len(dna_ref) % 3 != 0:
        raise ValueError(
            f"sequence length {len(dna_ref)} is not divisible by 3"
        )
    diffs = [i for i in range(len(dna_ref)) if dna_ref[i] != dna_alt[i]]
    if len(diffs) != 1:
        raise ValueError(
            f"sequences must differ at exactly 1 position, got {len(diffs)}"
        )
    aa_ref = translate(dna_ref)
    aa_alt = translate(dna_alt)
    if aa_ref == aa_alt:
        return "synonymous"
    if "*" in aa_alt and "*" not in aa_ref:
        return "nonsense"
    return "nonsynonymous"


def mutant_aa_sequences(
    dna_sequence: str,
) -> dict[str, list[str]]:
    """Group single-nt mutants by their translated AA sequence.

    Returns ``{aa_sequence: [mutant_dna, ...]}``.  The wildtype DNA is
    not included in any value list.  Synonymous mutants appear under
    the wildtype AA key.
    """
    groups: dict[str, list[str]] = {}
    for mutant_dna, _, _, _ in single_nt_mutations(dna_sequence):
        aa = translate(mutant_dna)
        groups.setdefault(aa, []).append(mutant_dna)
    return groups
