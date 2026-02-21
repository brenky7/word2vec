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

    def get_negative_samples_batch(
        self, center_ids: np.ndarray, count: int
    ) -> np.ndarray:
        """
        Draw *count* negative samples for every entry in *center_ids*
        in a single vectorised call, returning a (B, count) int32 array.

        Each row i is sampled from the smoothed unigram distribution
        and is guaranteed not to contain center_ids[i].  A per-row
        fallback fires only when the over-sampled buffer happens to be
        entirely composed of the target word (practically impossible for
        large vocabularies).

        Parameters
        ----------
        center_ids : np.ndarray, shape (B,)
        count      : int  – negatives per sample (K)

        Returns
        -------
        np.ndarray of shape (B, count), dtype int32.
        """
        B = len(center_ids)
        oversample = count * 2 + 4  # buffer to absorb target collisions

        # One bulk random draw: (B, oversample) table indices
        indices    = np.random.randint(0, self.table_size, (B, oversample))
        candidates = self._table[indices]           # (B, oversample)

        result = np.zeros((B, count), dtype=np.int32)
        for i in range(B):
            valid = candidates[i][candidates[i] != center_ids[i]]
            if len(valid) >= count:
                result[i] = valid[:count]
            else:
                # Rare path: fall back to the single-sample method
                result[i] = np.array(
                    self.get_negative_samples(int(center_ids[i]), count),
                    dtype=np.int32,
                )
        return result

    def get_negative_samples(self, target_id: int, count: int) -> List[int]:
        """
        Draw *count* negative-sample word IDs from the smoothed unigram
        distribution, **excluding** the target word itself.

        Uses a single vectorised NumPy call to sample a small batch and
        filters in one pass, avoiding a Python while-loop on the hot path.
        A Python fallback loop runs only when the oversampled batch does
        not contain enough valid samples (rare for large vocabularies).

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
        # Over-sample by a small factor to absorb target_id collisions.
        # 2× + 4 is sufficient unless target_id dominates the table.
        indices = np.random.randint(0, self.table_size, count * 2 + 4)
        candidates = self._table[indices]
        valid = candidates[candidates != target_id]

        if len(valid) >= count:
            return valid[:count].tolist()

        # Rare fallback: target_id was very frequent and filled the sample.
        result: List[int] = valid.tolist()
        while len(result) < count:
            idx = np.random.randint(0, self.table_size)
            sid = int(self._table[idx])
            if sid != target_id:
                result.append(sid)
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NegativeSampler(vocab_size={len(self.word2id)}, "
            f"table_size={self.table_size})"
        )
