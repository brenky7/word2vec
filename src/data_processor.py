"""
DataProcessor: handles tokenization, vocabulary building,
rare-word filtering, and subsampling of frequent words.
"""

import re
import math
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional


class DataProcessor:
    """
    Preprocesses raw text for Word2Vec training.

    Parameters
    raw_text : str
        The corpus to process.
    min_count : int
        Words appearing fewer than this many times are discarded (default 5).
    subsample_threshold : float
        The t constant in the subsampling formula P(w) = 1 - sqrt(t / f(w)).
    window_size : int
        Half-window used when generating skip-gram pairs (default 5).
    """

    def __init__(
        self,
        raw_text: str,
        min_count: int = 5,
        subsample_threshold: float = 1e-3,
        window_size: int = 5,
    ) -> None:
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        self.window_size = window_size

        # --- Step 1 ---
        self.tokens: List[str] = self._tokenize(raw_text)
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self._build_vocabulary()

        # Encode the full corpus (rare words already removed from vocab)
        self.encoded: List[int] = self._encode(self.tokens)

        # Pre-compute per-word discard probability for subsampling
        self._discard_probs: Dict[int, float] = self._compute_discard_probs()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """
        Lowercase the text and strip every character that is not a
        letter or an apostrophe, then split on whitespace.
        """
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = text.split()
        return tokens

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _build_vocabulary(self) -> None:
        """
        Count word frequencies, discard rare words (count < min_count),
        then assign a unique integer ID to each surviving word.
        """
        counts = Counter(self.tokens)

        # Filter rare words
        filtered = {w: c for w, c in counts.items() if c >= self.min_count}

        # Sort by frequency (most common first) for determinism
        sorted_words = sorted(filtered.items(), key=lambda x: -x[1])

        self.word2id = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        self.word_counts = Counter({word: cnt for word, cnt in sorted_words})

    def _encode(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to integer IDs, silently dropping any
        token that was filtered out of the vocabulary.
        """
        return [self.word2id[t] for t in tokens if t in self.word2id]

    # ------------------------------------------------------------------
    # Subsampling
    # ------------------------------------------------------------------

    def _compute_discard_probs(self) -> Dict[int, float]:
        """
        Pre-compute P_discard(w) = 1 - sqrt(t / f(w))  for every word,
        where f(w) is the relative frequency of word w.

        A token is kept with probability  keep_prob = 1 - P_discard(w).
        Values are clamped to [0, 1].
        """
        total = sum(self.word_counts.values())
        probs: Dict[int, float] = {}
        for word, wid in self.word2id.items():
            freq = self.word_counts[word] / total
            p_discard = 1.0 - math.sqrt(self.subsample_threshold / freq)
            probs[wid] = max(0.0, min(1.0, p_discard))
        return probs

    def subsample(self, encoded: Optional[List[int]] = None) -> List[int]:
        """
        Apply subsampling to an encoded sequence (defaults to self.encoded).
        Each word w is discarded with probability P_discard(w).
        Returns the filtered list of IDs.
        """
        if encoded is None:
            encoded = self.encoded
        return [
            wid
            for wid in encoded
            if random.random() > self._discard_probs.get(wid, 0.0)
        ]

    # ------------------------------------------------------------------
    # Training-pair generation
    # ------------------------------------------------------------------

    def generate_training_data(
        self, apply_subsampling: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Materialise all (center_id, context_id) pairs into a list.
        Prefer stream_training_pairs() for large corpora to avoid
        allocating the full list in memory.
        """
        return list(self.stream_training_pairs(apply_subsampling=apply_subsampling))

    def stream_training_pairs(
        self, apply_subsampling: bool = True
    ):
        """
        Generator version of generate_training_data.

        Yields (center_id, context_id) pairs one at a time without
        materialising the full list, which for large corpora (e.g. text8)
        avoids allocating hundreds of millions of tuples in memory.

        The dynamic window size is re-sampled for every center word,
        matching the original Word2Vec behaviour.

        Parameters
        ----------
        apply_subsampling : bool
            Whether to subsample frequent words before generating pairs.
        """
        corpus = self.subsample() if apply_subsampling else self.encoded
        n = len(corpus)

        for center_pos, center_id in enumerate(corpus):
            # Dynamic window size: sample uniformly in [1, window_size]
            dynamic_window = random.randint(1, self.window_size)

            start = max(0, center_pos - dynamic_window)
            end = min(n, center_pos + dynamic_window + 1)

            for ctx_pos in range(start, end):
                if ctx_pos != center_pos:
                    yield center_id, corpus[ctx_pos]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DataProcessor(vocab_size={self.vocab_size}, "
            f"corpus_length={len(self.encoded)}, "
            f"min_count={self.min_count})"
        )
