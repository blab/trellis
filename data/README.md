# `data/`

## `mj_matrix.csv`

20×20 Miyazawa-Jernigan contact-energy matrix in kT-like reduced units.
Rows and columns are the 20 standard amino acids in alphabetical
one-letter code order: `A C D E F G H I K L M N P Q R S T V W Y`.
The matrix is symmetric.

### Source

Values are taken from the `miyazawa_jernigan` dictionary in
[`jbloomlab/latticeproteins/src/interactions.py`](https://github.com/jbloomlab/latticeproteins/blob/master/src/interactions.py),
fetched at commit
[`de1316a`](https://github.com/jbloomlab/latticeproteins/blob/de1316a66139030c931a4cbdb35f521561686ff0/src/interactions.py)
on 2026-05-07.

### Citation

The values originate from Table V of:

> Miyazawa, S. & Jernigan, R. L. (1985). "Estimation of effective
> interresidue contact energies from protein crystal structures:
> Quasi-chemical approximation." *Macromolecules* 18:534–552.

Per the source file, the table comments indicate the values are taken
from "Table V of the paper, upper half and diagonal," and the dict
in the source mirrors them across the diagonal to give a full
symmetric 20×20 matrix.

### Range

- Most attractive (minimum): `F-F = -6.85`
- Most repulsive (maximum): `K-K = +0.13`
- Most entries are negative; a small number of like-charged or
  mismatched-polarity pairs are mildly positive.
