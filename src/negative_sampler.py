"""
NegativeSampler: builds a unigram distribution raised to the 0.75 power
and efficiently draws negative samples for Word2Vec training.
"""

import numpy as np
from collections import Counter
from typing import Dict, List


class NegativeSampler:
    """
    Pre-computes a smoothed unigram distribution and samples negative
    word IDs for Skip-gram Negative Sampling (SGNS).

    The probability of drawing word w as a negative sample is:

        P_neg(w)  ∝  count(w) ^ 0.75

    Parameters
    ----------
    word_counts : Counter
        Mapping  word -> raw frequency count (from DataProcessor.word_counts).
    word2id : dict
        Mapping  word -> integer ID.
    table_size : int
        Size of the internal sampling table (larger = more accurate but
        uses more memory).  Default: 1e7.
    """

    SMOOTHING_EXPONENT: float = 0.75

    def __init__(
        self,
        word_counts: Counter,
        word2id: Dict[str, int],
        table_size: int = 10_000_000,
    ) -> None:
        self.word2id = word2id
        self.table_size = table_size
        self._table: np.ndarray = self._build_table(word_counts)

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _build_table(self, word_counts: Counter) -> np.ndarray:
        """
        Build an integer array of word IDs whose relative frequency
        matches the smoothed unigram distribution.

        Each entry in the table is a word ID; we fill proportional
        regions of the table so that the fraction of cells equal to id *i*
        is  P_neg(w_i).
        """
        # Gather IDs and smoothed counts in a deterministic order
        ids = sorted(self.word2id.values())
        id_to_word = {v: k for k, v in self.word2id.items()}

        smoothed = np.array(
            [word_counts[id_to_word[i]] ** self.SMOOTHING_EXPONENT for i in ids],
            dtype=np.float64,
        )
        smoothed /= smoothed.sum()  # normalize to a proper distribution

        # Build the table by filling stretches of IDs proportionally
        table = np.zeros(self.table_size, dtype=np.int32)
        cumulative = 0.0
        j = 0
        for idx, prob in zip(ids, smoothed):
            cumulative += prob
            fill_until = int(cumulative * self.table_size)
            while j < fill_until and j < self.table_size:
                table[j] = idx
                j += 1

        # Fill any remaining slots with the last word (rounding artefact)
        if j < self.table_size:
            table[j:] = ids[-1]

        return table

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_negative_samples(self, target_id: int, count: int) -> List[int]:
        """
        Draw *count* negative-sample word IDs from the smoothed unigram
        distribution, **excluding** the target word itself.

        Parameters
        ----------
        target_id : int
            The positive (center or context) word ID to exclude.
        count : int
            Number of negative samples required.

        Returns
        -------
        List[int] of length *count*.
        """
        negatives: List[int] = []
        while len(negatives) < count:
            # Sample a random position in the pre-built table
            idx = np.random.randint(0, self.table_size)
            sampled_id = int(self._table[idx])
            if sampled_id != target_id:
                negatives.append(sampled_id)
        return negatives

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NegativeSampler(vocab_size={len(self.word2id)}, "
            f"table_size={self.table_size})"
        )
