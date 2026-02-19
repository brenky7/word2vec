"""
Word2Vec: Skip-gram model with Negative Sampling, implemented with NumPy.
"""

import numpy as np
from typing import List, Tuple, Dict


class Word2Vec:
    """
    Skip-gram Word2Vec with Negative Sampling (SGNS).

    Two embedding matrices are maintained:

    * W1  (vocab_size, embedding_dim) - input / center-word embeddings
    * W2  (vocab_size, embedding_dim) - output / context-word embeddings

    Parameters
    ----------
    vocab_size : int
        Number of words in the vocabulary.
    embedding_dim : int
        Dimensionality of each word vector.
    learning_rate : float
        SGD learning-rate (default 0.025).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        learning_rate: float = 0.025,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate

        # Weight initialisation with small random uniform values
        init_range = 0.5 / embedding_dim
        self.W1: np.ndarray = np.random.uniform(
            -init_range, init_range, (vocab_size, embedding_dim)
        )
        self.W2: np.ndarray = np.random.uniform(
            -init_range, init_range, (vocab_size, embedding_dim)
        )

    # ------------------------------------------------------------------
    # Numerically stable sigmoid
    # ------------------------------------------------------------------

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Element-wise sigmoid with numerical stability.

        Uses the identity:
            sigma(x) = exp(x) / (1 + exp(x))   for x >= 0
            sigma(x) = 1 / (1 + exp(-x))        for x <  0

        This avoids both overflow (large positive x) and underflow
        (large negative x).
        """
        result = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        center_id: int,
        context_id: int,
        negative_ids: List[int],
    ) -> Tuple[float, np.ndarray]:
        """
        Compute sigmoid predictions for one positive pair and its
        associated negative samples.

        Returns
        -------
        pos_pred : float
            sigma( W1[center] · W2[context] ) - should approach 1.
        neg_preds : np.ndarray of shape (len(negative_ids),)
            sigma( W1[center] · W2[neg_i] ) - should approach 0.
        """
        center_vec = self.W1[center_id]          # (embedding_dim,)
        context_vec = self.W2[context_id]        # (embedding_dim,)
        neg_vecs = self.W2[negative_ids]         # (K, embedding_dim)

        pos_score = np.dot(center_vec, context_vec)          # scalar
        neg_scores = neg_vecs @ center_vec                   # (K,)

        pos_pred = float(self.sigmoid(np.array([pos_score]))[0])
        neg_preds = self.sigmoid(neg_scores)                 # (K,)

        return pos_pred, neg_preds

    # ------------------------------------------------------------------
    # Train step (forward + backward + SGD update)
    # ------------------------------------------------------------------

    def train_step(
        self,
        center_id: int,
        context_id: int,
        negative_ids: List[int],
    ) -> float:
        """
        Execute one SGNS update for a single (center, context) pair.

        Loss (binary cross-entropy with negative samples):

            L = -log sigma(v_c · v_ctx)  -  Σ_k log sigma(-v_c · v_neg_k)

        Gradients:

            ∂L/∂v_ctx   = (sigma(v_c·v_ctx) - 1) · v_c   →  error_pos * center
            ∂L/∂v_neg_k = sigma(v_c·v_neg_k) · v_c        →  error_neg_k * center
            ∂L/∂v_c     = error_pos * v_ctx  +  Σ_k error_neg_k * v_neg_k

        Parameters
        ----------
        center_id : int
        context_id : int
        negative_ids : List[int]

        Returns
        -------
        loss : float   (for monitoring)
        """
        pos_pred, neg_preds = self.forward(center_id, context_id, negative_ids)

        # ---- errors (prediction - label) ----
        # label for context word  = 1  → error = pred - 1
        # label for negative words = 0  → error = pred - 0 = pred
        error_pos: float = pos_pred - 1.0           # scalar
        error_neg: np.ndarray = neg_preds           # (K,)  already pred - 0

        # ---- cache frequently used vectors ----
        center_vec = self.W1[center_id].copy()      # (D,)
        context_vec = self.W2[context_id].copy()    # (D,)
        neg_vecs = self.W2[negative_ids].copy()     # (K, D)

        # ---- gradient w.r.t. center embedding ----
        # accumulate contributions from positive and all negative words
        grad_center = (
            error_pos * context_vec                     # (D,)
            + error_neg @ neg_vecs                      # (D,)  = Σ_k e_k * v_neg_k
        )

        # ---- gradients w.r.t. output (context/negative) embeddings ----
        grad_context = error_pos * center_vec           # (D,)
        grad_neg = np.outer(error_neg, center_vec)      # (K, D)

        # ---- SGD updates ----
        self.W1[center_id] -= self.lr * grad_center
        self.W2[context_id] -= self.lr * grad_context
        self.W2[negative_ids] -= self.lr * grad_neg

        # ---- loss (binary cross-entropy) ----
        eps = 1e-10  # prevent log(0)
        loss = (
            -np.log(pos_pred + eps)
            - np.sum(np.log(1.0 - neg_preds + eps))
        )
        return float(loss)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def get_most_similar(
        self,
        word: str,
        word2id: Dict[str, int],
        id2word: Dict[int, str],
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find the *top_n* words most similar to *word* by cosine similarity
        over the input (W1) embedding matrix.

        Parameters
        ----------
        word : str
        word2id : dict - from DataProcessor.word2id
        id2word : dict - from DataProcessor.id2word
        top_n : int

        Returns
        -------
        List of (word, cosine_similarity) tuples, sorted descending.
        """
        if word not in word2id:
            raise KeyError(f"'{word}' is not in the vocabulary.")

        query_id = word2id[word]
        query_vec = self.W1[query_id]                    # (D,)

        # Cosine similarity against every row in W1
        norms = np.linalg.norm(self.W1, axis=1)          # (V,)
        query_norm = np.linalg.norm(query_vec)

        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1e-10, norms)
        query_norm = max(query_norm, 1e-10)

        similarities = (self.W1 @ query_vec) / (safe_norms * query_norm)  # (V,)

        # Exclude the query word itself
        similarities[query_id] = -np.inf

        # Top-N indices (descending)
        top_indices = np.argsort(similarities)[::-1][:top_n]

        return [(id2word[int(i)], float(similarities[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Word2Vec(vocab_size={self.vocab_size}, "
            f"embedding_dim={self.embedding_dim}, "
            f"lr={self.lr})"
        )
