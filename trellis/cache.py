"""Fitness cache mapping amino acid sequences to FitnessResult objects."""

from trellis.fitness import FitnessResult


class FitnessCache:
    """Cache of fitness evaluations keyed by amino acid sequence.

    Stores full ``FitnessResult`` objects (including ``FoldResult``) so
    that downstream code can access native conformations without
    re-folding.
    """

    def __init__(self) -> None:
        self._data: dict[str, FitnessResult] = {}
        self._hits: int = 0
        self._misses: int = 0

    def __contains__(self, aa_sequence: str) -> bool:
        found = aa_sequence in self._data
        if found:
            self._hits += 1
        else:
            self._misses += 1
        return found

    def __len__(self) -> int:
        return len(self._data)

    def get(self, aa_sequence: str) -> FitnessResult | None:
        return self._data.get(aa_sequence)

    def put(self, aa_sequence: str, result: FitnessResult) -> None:
        self._data[aa_sequence] = result

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }
