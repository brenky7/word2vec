"""
train.py - End-to-end training script for the Word2Vec implementation.

Usage
-----
    python src/train.py

Configuration is loaded from a .env file in the project root.
All keys are optional; built-in defaults are used for any missing key.
If TEXT_SOURCE is set but the file cannot be opened, the script
automatically falls back to the built-in dummy corpus.

Supported .env keys
-------------------
  TEXT_SOURCE   – absolute or relative path to a plain-text corpus
  EMBEDDING_DIM – integer  (default 100)
  LEARNING_RATE – float    (default 0.025)
  EPOCHS        – integer  (default 5)
  WINDOW_SIZE   – integer  (default 5)
  NEG_SAMPLES   – integer  (default 5)
  MIN_COUNT     – integer  (default 5)
  SUBSAMPLE_T   – float    (default 1e-3)
  BATCH_SIZE    – integer  (default 128)
  PRINT_EVERY   – integer  (default 1000, counts batches)
  MODEL_SAVE_PATH – path prefix for saved model files (default 'models/word2vec')
  SEED          – integer  (default 42)
"""

import os
import sys
import time
import random
from typing import List
import numpy as np
from pathlib import Path
from dotenv import dotenv_values

# ---------------------------------------------------------------------------
# Make sure the src/ directory is on the path when running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from data_processor import DataProcessor
from negative_sampler import NegativeSampler
from word2vec import Word2Vec

# ===========================================================================
# CONFIG  –  loaded from .env (project root), with built-in defaults
# ===========================================================================

# The .env file is expected one level above src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

_env = dotenv_values(_ENV_PATH)   # empty dict if file is absent

if _ENV_PATH.exists():
    print(f"[config] Loaded .env from '{_ENV_PATH}'")
else:
    print("[config] No .env file found – using built-in defaults.")

def _get(key: str, default: str) -> str:
    """Return value from .env, then OS environment, then the default."""
    return _env.get(key) or os.environ.get(key) or default

TEXT_SOURCE: str  = _get("TEXT_SOURCE",   "")          # empty → dummy
EMBEDDING_DIM: int   = int(_get("EMBEDDING_DIM", "100"))
LEARNING_RATE: float = float(_get("LEARNING_RATE", "0.025"))
EPOCHS: int          = int(_get("EPOCHS",         "5"))
WINDOW_SIZE: int     = int(_get("WINDOW_SIZE",    "5"))
NEG_SAMPLES: int     = int(_get("NEG_SAMPLES",    "5"))
MIN_COUNT: int       = int(_get("MIN_COUNT",      "5"))
SUBSAMPLE_T: float   = float(_get("SUBSAMPLE_T",  "1e-3"))
BATCH_SIZE: int      = int(_get("BATCH_SIZE",     "128"))
PRINT_EVERY: int     = int(_get("PRINT_EVERY",    "1000"))  # in batches
MODEL_SAVE_PATH: str = _get("MODEL_SAVE_PATH",   "models/word2vec")
SEED: int            = int(_get("SEED",            "42"))

# ===========================================================================
# Dummy corpus (used when TEXT_SOURCE == "dummy")
# ===========================================================================

DUMMY_TEXT = """
Machine learning is a field of artificial intelligence that uses statistical
techniques to give computer systems the ability to learn from data without
being explicitly programmed. Deep learning is part of a broader family of
machine learning methods based on artificial neural networks. Neural networks
are computing systems inspired by biological neural networks that constitute
animal brains. The neural network itself is not an algorithm, but rather a
framework for many different machine learning algorithms to work together and
process complex data inputs. Machine learning algorithms are used in a wide
variety of applications, such as email filtering and computer vision, where it
is difficult or infeasible to develop conventional algorithms to perform the
needed tasks. Machine learning is closely related to computational statistics,
which focuses on making predictions using computers. The study of mathematical
optimization delivers methods, theory, and application domains to the field of
machine learning. Data mining is a related field of study, focusing on
exploratory data analysis through unsupervised learning. In its application
across business problems, machine learning is also referred to as predictive
analytics. Machine learning and statistics are closely related fields in terms
of methods, but distinct in their principal goal. Statistics draws population
inferences from a sample, while machine learning finds generalizable predictive
patterns. According to Michael I. Jordan, the ideas of machine learning from
its history and also its potential are those which have shaped and
contributed to the foundations of statistics, data science, and artificial
intelligence. Conventional statistical analyses require the a priori selection
of a model and its fitting to the data. In contrast, machine learning programs
detect patterns in data and adjust program actions accordingly. Machine
learning algorithms are used in a wide variety of applications. In supervised
learning, the algorithm is provided with training data that has input output
pairs and the algorithm learns a function that maps inputs to outputs. In
unsupervised learning, the algorithm is provided with data that has no labels
and the algorithm must find structure in the data on its own. Reinforcement
learning is an area of machine learning concerned with how software agents
ought to take actions in an environment in order to maximize the notion of
cumulative reward. Deep learning uses multiple layers to progressively extract
higher-level features from the raw input. For example, in image processing,
lower layers may identify edges, while higher layers may identify the concepts
relevant to a human such as digits or letters or faces. Deep learning is part
of a broader family of machine learning methods based on artificial neural
networks with representation learning. Learning can be supervised, semi
supervised or unsupervised. Deep learning architectures such as deep neural
networks, recurrent neural networks, convolutional neural networks and
transformers have been applied to fields including computer vision, speech
recognition, natural language processing, audio recognition, social network
filtering, machine translation, bioinformatics, drug design, medical image
analysis, climate science, material inspection and board game programs where
they have produced results comparable to and in some cases surpassing human
expert performance. The term deep learning was first used in the context of
machine learning by Rina Dechter in 1986 and introduced to the machine
learning community by Aizenberg in 2000. Artificial neural networks were
inspired by information processing and distributed communication nodes in
biological systems. Artificial neural networks have various differences from
biological brains. Specifically, neural networks tend to be static and
symbolic, while the biological brain of most living organisms is dynamic and
analog. The adjective deep in deep learning refers to the use of multiple
layers in the network. Early work showed that a linear perceptron cannot be a
universal classifier, but that a network with a nonpolynomial activation
function with one hidden layer of unbounded width can be. Universal
approximation theorem. Artificial intelligence is intelligence demonstrated by
machines, as opposed to the natural intelligence displayed by animals including
humans. Artificial intelligence research has been defined as the field of
study of intelligent agents, which refers to any system that perceives its
environment and takes actions that maximize its chance of achieving its goals.
The term artificial intelligence had previously been used to describe machines
that mimic and display human cognitive skills associated with the human mind.
Natural language processing is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers
and human language, in particular how to program computers to process and
analyze large amounts of natural language data. Challenges in natural language
processing frequently involve speech recognition, natural language
understanding, and natural language generation. The history of natural language
processing generally started in the 1950s although work can be found from
earlier periods. In 1950 Alan Turing published an article titled Computing
Machinery and Intelligence which proposed the Turing test as a criterion of
intelligence, a task that involves the automated interpretation and generation
of natural language, but at the time not articulated as a problem separate from
artificial intelligence.
""" * 10   # repeat to get enough tokens for min_count filtering


# ===========================================================================
# Helpers
# ===========================================================================

def load_text() -> str:
    """
    Try to load the corpus specified by TEXT_SOURCE.
    Falls back to the built-in dummy corpus in case of an error.
    """
    if not TEXT_SOURCE:
        print("[corpus] TEXT_SOURCE not set – using dummy corpus.")
        return DUMMY_TEXT

    try:
        path = Path(TEXT_SOURCE)
        if not path.is_absolute():
            # Resolve relative paths against the project root
            path = _PROJECT_ROOT / path
        print(f"[corpus] Trying to load corpus from '{path}' …")
        text = path.read_text(encoding="utf-8")
        print(f"[corpus] Loaded {len(text):,} characters from '{path.name}'.")
        return text
    except FileNotFoundError:
        print(f"[corpus] WARNING: '{TEXT_SOURCE}' not found – falling back to dummy corpus.")
    except OSError as exc:
        print(f"[corpus] WARNING: Could not read '{TEXT_SOURCE}' ({exc}) – falling back to dummy corpus.")

    return DUMMY_TEXT


def get_batches(stream, batch_size: int):
    """
    Consume a (center_id, context_id) generator and yield mini-batches.

    Each yield is a tuple (center_ids, context_ids) of np.int32 arrays
    of shape (B,) where B <= batch_size.  The final batch may be smaller
    than batch_size if the stream length is not evenly divisible.

    Parameters
    ----------
    stream     : iterable of (int, int) pairs
    batch_size : int
    """
    centers:  List[int] = []
    contexts: List[int] = []
    for center_id, context_id in stream:
        centers.append(center_id)
        contexts.append(context_id)
        if len(centers) == batch_size:
            yield (
                np.array(centers,  dtype=np.int32),
                np.array(contexts, dtype=np.int32),
            )
            centers  = []
            contexts = []
    if centers:
        yield (
            np.array(centers,  dtype=np.int32),
            np.array(contexts, dtype=np.int32),
        )


def lr_schedule(initial_lr: float, step: int, total_steps: int) -> float:
    """Linear decay; never drops below 1 % of the initial value."""
    return max(initial_lr * (1.0 - step / total_steps), initial_lr * 0.01)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load & pre-process text
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1 – Loading and preprocessing text …")
    raw_text = load_text()

    processor = DataProcessor(
        raw_text,
        min_count=MIN_COUNT,
        subsample_threshold=SUBSAMPLE_T,
        window_size=WINDOW_SIZE,
    )
    print(processor)
    print(f"  Raw tokens  : {len(processor.tokens):,}")
    print(f"  Corpus IDs  : {len(processor.encoded):,}")
    print(f"  Vocab size  : {processor.vocab_size:,}")

    # ------------------------------------------------------------------
    # 2. Build negative sampler
    # ------------------------------------------------------------------
    print("\nStep 2 – Building negative sampler …")
    sampler = NegativeSampler(processor.word_counts, processor.word2id)
    print(sampler)

    # ------------------------------------------------------------------
    # 3. Initialise model
    # ------------------------------------------------------------------
    print("\nStep 3 – Initialising Word2Vec model …")
    model = Word2Vec(
        vocab_size=processor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        learning_rate=LEARNING_RATE,
    )
    print(model)

    # ------------------------------------------------------------------
    # 4. Training loop  (mini-batch SGD)
    # ------------------------------------------------------------------
    print(f"\nStep 4 – Training …  (batch_size={BATCH_SIZE})")
    print("=" * 60)

    global_batch = 0   # counts batches (used for lr schedule + PRINT_EVERY)
    total_loss   = 0.0
    total_pairs  = 0

    # Estimate total batches for the lr schedule without materialising pairs.
    # avg pairs/token ≈ WINDOW_SIZE  (dynamic window + both directions cancel)
    approx_pairs_per_epoch  = len(processor.encoded) * WINDOW_SIZE
    approx_total_batches    = (approx_pairs_per_epoch // BATCH_SIZE) * EPOCHS

    for epoch in range(1, EPOCHS + 1):
        epoch_start  = time.time()
        epoch_loss   = 0.0
        epoch_pairs  = 0
        epoch_batches = 0

        # stream_training_pairs yields pairs on-the-fly (no giant list).
        # Subsampling is re-applied at the start of each epoch.
        stream = processor.stream_training_pairs(apply_subsampling=True)

        for center_ids, context_ids in get_batches(stream, BATCH_SIZE):
            B = len(center_ids)

            # Adaptive learning rate (per batch)
            model.lr = lr_schedule(LEARNING_RATE, global_batch, approx_total_batches)

            # Draw (B, K) negative samples in one vectorised call
            neg_ids = sampler.get_negative_samples_batch(center_ids, NEG_SAMPLES)

            # Mini-batch forward + backward + SGD update
            mean_loss = model.train_step_batch(center_ids, context_ids, neg_ids)

            batch_loss    = mean_loss * B
            epoch_loss   += batch_loss
            total_loss   += batch_loss
            epoch_pairs  += B
            total_pairs  += B
            global_batch += 1
            epoch_batches += 1

            if global_batch % PRINT_EVERY == 0:
                avg = total_loss / max(total_pairs, 1)
                print(
                    f"  Epoch {epoch}/{EPOCHS}  "
                    f"batch {global_batch:>8,}  "
                    f"avg loss {avg:.4f}  "
                    f"lr {model.lr:.6f}"
                )

        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch}/{EPOCHS}]  "
            f"batches: {epoch_batches:,}  "
            f"pairs: {epoch_pairs:,}  "
            f"epoch loss: {epoch_loss / max(epoch_pairs, 1):.4f}  "
            f"time: {elapsed:.1f}s"
        )

    print("\nTraining complete.")
    print(f"Total batches : {global_batch:,}")
    print(f"Total pairs   : {total_pairs:,}")
    print(f"Final avg loss : {total_loss / max(total_pairs, 1):.4f}")



    # ------------------------------------------------------------------
    # 5. Quick similarity demo  (5 random words from the vocabulary)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Similarity demo")
    print("=" * 60)

    all_vocab_words = list(processor.word2id.keys())
    demo_words = random.sample(all_vocab_words, min(5, len(all_vocab_words)))

    for word in demo_words:
        results = model.get_most_similar(
            word, processor.word2id, processor.id2word, top_n=5
        )
        neighbours = ", ".join(f"{w} ({s:.3f})" for w, s in results)
        print(f"  {word:>20} →  {neighbours}")

    
    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------

    # Resolve MODEL_SAVE_PATH relative to the project root so the script
    # works correctly regardless of which directory it is launched from.
    save_prefix = str(_PROJECT_ROOT / MODEL_SAVE_PATH)
    model.save_model(save_prefix, processor.id2word)
    model.export_to_text(save_prefix + "_embeddings.txt", processor.id2word)


if __name__ == "__main__":
    main()
