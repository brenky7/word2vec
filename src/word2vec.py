"""
Word2Vec: Skip-gram model with Negative Sampling, implemented with NumPy.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict


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
        # ---- read weight rows exactly once (copies protect gradients) ----
        center_vec  = self.W1[center_id].copy()      # (D,)
        context_vec = self.W2[context_id].copy()     # (D,)
        neg_vecs    = self.W2[negative_ids].copy()   # (K, D)

        # ---- forward pass (inlined) ----
        # Compute all scores in a single array so sigmoid is called once.
        pos_score  = np.dot(center_vec, context_vec)   # scalar
        neg_scores = neg_vecs @ center_vec             # (K,)
        all_scores = np.empty(1 + len(negative_ids))
        all_scores[0]  = pos_score
        all_scores[1:] = neg_scores
        all_preds = self.sigmoid(all_scores)           # (1+K,)
        pos_pred  = all_preds[0]                       # scalar
        neg_preds = all_preds[1:]                      # (K,)

        # ---- errors (prediction − label) ----
        # label for context word   = 1  →  error = pred − 1
        # label for negative words = 0  →  error = pred − 0 = pred
        error_pos: float       = float(pos_pred) - 1.0   # scalar
        error_neg: np.ndarray  = neg_preds                # (K,)

        # ---- gradients ----
        # Center word: accumulate from positive pair and all K negatives.
        grad_center  = error_pos * context_vec + error_neg @ neg_vecs  # (D,)
        # Context word: pull toward / push away from center.
        grad_context = error_pos * center_vec                           # (D,)
        # Negative words: each pushed away from center independently.
        grad_neg     = np.outer(error_neg, center_vec)                  # (K, D)

        # ---- SGD updates ----
        self.W1[center_id]       -= self.lr * grad_center
        self.W2[context_id]      -= self.lr * grad_context
        self.W2[negative_ids]    -= self.lr * grad_neg

        # ---- loss (binary cross-entropy) ----
        eps = 1e-10
        loss = (
            -np.log(float(pos_pred) + eps)
            - np.sum(np.log(1.0 - neg_preds + eps))
        )
        return float(loss)

    def train_step_batch(
        self,
        center_ids:  np.ndarray,  # (B,)
        context_ids: np.ndarray,  # (B,)
        neg_ids:     np.ndarray,  # (B, K)
    ) -> float:
        """
        Execute one mini-batch SGNS update.

        Shapes
        ------
        B = batch size,  K = negatives per pair,  D = embedding_dim

        Forward
        -------
        pos_scores[i]    = W1[c_i] · W2[ctx_i]              scalar per sample
        neg_scores[i, k] = W1[c_i] · W2[neg_ids[i, k]]      K scalars per sample

        Gradients (per sample i, averaged over batch via np.add.at)
        ---------
        ∂L/∂W1[c_i]      = e_pos[i] · ctx_vec[i]  +  Σ_k e_neg[i,k] · neg_vec[i,k]
        ∂L/∂W2[ctx_i]    = e_pos[i] · ctr_vec[i]
        ∂L/∂W2[neg_i, k] = e_neg[i,k] · ctr_vec[i]

        where  e_pos = σ(pos_score) − 1   (label = 1)
               e_neg = σ(neg_score)       (label = 0)

        Parameter updates use np.add.at so that duplicate IDs within
        a batch accumulate correctly instead of being silently overwritten
        by plain indexed assignment (W[ids] -= grad).

        Returns
        -------
        Mean per-pair loss for the batch (float, for monitoring).
        """
        B, K = neg_ids.shape
        D    = self.embedding_dim

        # ---- gather embeddings (no .copy() needed – we build fresh grad arrays) ----
        center_vecs  = self.W1[center_ids]    # (B, D)
        context_vecs = self.W2[context_ids]   # (B, D)
        neg_vecs     = self.W2[neg_ids]       # (B, K, D)

        # ---- forward: compute all scores in one shot, call sigmoid once ----
        pos_scores = np.einsum('bd,bd->b', center_vecs, context_vecs)   # (B,)
        neg_scores = np.einsum('bd,bkd->bk', center_vecs, neg_vecs)     # (B, K)

        all_scores        = np.empty((B, 1 + K), dtype=np.float64)
        all_scores[:, 0]  = pos_scores
        all_scores[:, 1:] = neg_scores
        all_preds  = self.sigmoid(all_scores)   # (B, 1+K)
        pos_preds  = all_preds[:, 0]            # (B,)
        neg_preds  = all_preds[:, 1:]           # (B, K)

        # ---- errors (prediction − label) ----
        error_pos = pos_preds - 1.0             # (B,)    label = 1
        error_neg = neg_preds                   # (B, K)  label = 0

        # ---- gradients ----
        # W1 gradient for each center word
        grad_center = (
            error_pos[:, None] * context_vecs                     # (B, D)
            + np.einsum('bk,bkd->bd', error_neg, neg_vecs)        # (B, D)
        )                                                          # (B, D)

        # W2 gradient for each context word
        grad_context = error_pos[:, None] * center_vecs            # (B, D)

        # W2 gradient for each negative word  →  (B, K, D)
        grad_neg = error_neg[:, :, None] * center_vecs[:, None, :] # (B, K, D)

        # ---- SGD updates with np.add.at (duplicate-safe accumulation) ----
        # Plain  W[ids] -= grad  silently overwrites duplicate rows;
        # np.add.at accumulates all contributions before committing.
        np.add.at(self.W1, center_ids,       -self.lr * grad_center)
        np.add.at(self.W2, context_ids,      -self.lr * grad_context)
        np.add.at(self.W2, neg_ids.ravel(),  -self.lr * grad_neg.reshape(B * K, D))

        # ---- loss (binary cross-entropy, summed then normalised per pair) ----
        eps  = 1e-10
        loss = (
            -np.sum(np.log(pos_preds + eps))
            - np.sum(np.log(1.0 - neg_preds + eps))
        )
        return float(loss) / B   # mean per-pair loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(
        self,
        path_prefix: str,
        id2word: Dict[int, str],
    ) -> None:
        """
        Save the model weights and vocabulary to disk.

        Creates three files:
          {path_prefix}_W1.npy    – input  (center-word) embedding matrix
          {path_prefix}_W2.npy    – output (context-word) embedding matrix
          {path_prefix}_vocab.pkl – id2word dict (pickle)

        Parameters
        ----------
        path_prefix : str
            Common prefix for all output files.  May include a directory
            (which will be created if it does not already exist).
        id2word : Dict[int, str]
            Mapping from integer ID → word string (DataProcessor.id2word).
        """
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        np.save(f"{prefix}_W1.npy", self.W1)
        np.save(f"{prefix}_W2.npy", self.W2)

        with open(f"{prefix}_vocab.pkl", "wb") as fh:
            pickle.dump(id2word, fh, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            f"[save] Model saved to →\n"
            f"        {prefix}_W1.npy\n"
            f"        {prefix}_W2.npy\n"
            f"        {prefix}_vocab.pkl"
        )

    @classmethod
    def load_model(
        cls,
        path_prefix: str,
        learning_rate: float = 0.025,
    ) -> "Word2Vec":
        """
        Load a previously saved model from disk and return a new
        Word2Vec instance with the restored weights and vocabulary.

        Parameters
        ----------
        path_prefix : str
            The same prefix used when save_model() was called.
        learning_rate : float
            Learning rate to set on the restored instance.

        Returns
        -------
        Word2Vec instance with W1, W2 restored and id2word attached.

        Raises
        ------
        FileNotFoundError if any of the three expected files is missing.
        """
        prefix = Path(path_prefix)

        w1_path    = Path(f"{prefix}_W1.npy")
        w2_path    = Path(f"{prefix}_W2.npy")
        vocab_path = Path(f"{prefix}_vocab.pkl")

        for p in (w1_path, w2_path, vocab_path):
            if not p.exists():
                raise FileNotFoundError(f"[load] Expected file not found: {p}")

        W1 = np.load(str(w1_path))
        W2 = np.load(str(w2_path))

        with open(vocab_path, "rb") as fh:
            id2word: Dict[int, str] = pickle.load(fh)

        vocab_size, embedding_dim = W1.shape
        instance = cls(vocab_size, embedding_dim, learning_rate)
        instance.W1      = W1
        instance.W2      = W2
        instance.id2word = id2word  # attached for convenience

        print(
            f"[load] Model loaded from →\n"
            f"        {w1_path}\n"
            f"        {w2_path}\n"
            f"        {vocab_path}\n"
            f"       vocab_size={vocab_size:,}  embedding_dim={embedding_dim}"
        )
        return instance

    def export_to_text(
        self,
        filepath: str,
        id2word: Dict[int, str],
    ) -> None:
        """
        Write the input embeddings (W1) in the standard Word2Vec text format:

            <vocab_size> <embedding_dim>
            <word> <f1> <f2> ... <fD>
            ...

        This format is compatible with tools such as Gensim's
        KeyedVectors.load_word2vec_format().

        Parameters
        ----------
        filepath : str    Path to the output .txt file.
        id2word  : Dict[int, str]
        """
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)

        vocab_size, embedding_dim = self.W1.shape
        with out.open("w", encoding="utf-8") as fh:
            fh.write(f"{vocab_size} {embedding_dim}\n")
            for idx in range(vocab_size):
                word = id2word.get(idx, f"__UNK_{idx}__")
                vec  = " ".join(f"{v:.6f}" for v in self.W1[idx])
                fh.write(f"{word} {vec}\n")

        print(f"[export] Embeddings written to '{out}'  "
              f"({vocab_size:,} words, dim={embedding_dim})")

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
