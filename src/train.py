"""
train.py - End-to-end training script for the Word2Vec implementation.

Usage
-----
    python src/train.py

Optional flags (edit the CONFIG section below):
  - TEXT_SOURCE : "dummy" | path to a plain-text file (e.g. text8)
  - EMBEDDING_DIM, LEARNING_RATE, EPOCHS, WINDOW_SIZE, NEG_SAMPLES
"""

import os
import sys
import time
import random
import numpy as np

# ---------------------------------------------------------------------------
# Make sure the src/ directory is on the path when running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from data_processor import DataProcessor
from negative_sampler import NegativeSampler
from word2vec import Word2Vec

# ===========================================================================
# CONFIG  –  edit these to change the training run
# ===========================================================================

# "dummy"  → use the built-in sample paragraph (good for smoke-testing)
# Any other string → treated as a path to a plain-text corpus file (e.g. text8)
TEXT_SOURCE: str = "dummy"

EMBEDDING_DIM: int = 100
LEARNING_RATE: float = 0.025
EPOCHS: int = 5
WINDOW_SIZE: int = 5
NEG_SAMPLES: int = 5        # negative samples per positive pair
MIN_COUNT: int = 5          # rare-word threshold
SUBSAMPLE_T: float = 1e-3   # subsampling threshold
PRINT_EVERY: int = 1000     # print average loss every N steps
SEED: int = 42

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
    if TEXT_SOURCE == "dummy":
        return DUMMY_TEXT
    if not os.path.exists(TEXT_SOURCE):
        raise FileNotFoundError(
            f"Corpus file '{TEXT_SOURCE}' not found. "
            "Set TEXT_SOURCE = 'dummy' to use the built-in sample text."
        )
    print(f"Loading corpus from '{TEXT_SOURCE}' …")
    with open(TEXT_SOURCE, "r", encoding="utf-8") as fh:
        return fh.read()


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
    # 4. Training loop
    # ------------------------------------------------------------------
    print("\nStep 4 – Training …")
    print("=" * 60)

    global_step = 0
    total_loss = 0.0

    # Estimate total steps for the learning-rate schedule
    sample_pairs = processor.generate_training_data(apply_subsampling=True)
    approx_total_steps = len(sample_pairs) * EPOCHS

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Re-generate training pairs each epoch (subsampling is stochastic)
        pairs = processor.generate_training_data(apply_subsampling=True)
        random.shuffle(pairs)

        epoch_loss = 0.0

        for step_in_epoch, (center_id, context_id) in enumerate(pairs):
            # Adaptive learning rate
            model.lr = lr_schedule(LEARNING_RATE, global_step, approx_total_steps)

            # Draw negative samples (exclude center and context)
            neg_ids = sampler.get_negative_samples(center_id, NEG_SAMPLES)

            # One SGD step
            loss = model.train_step(center_id, context_id, neg_ids)

            epoch_loss += loss
            total_loss += loss
            global_step += 1

            if global_step % PRINT_EVERY == 0:
                avg = total_loss / global_step
                print(
                    f"  Epoch {epoch}/{EPOCHS}  "
                    f"step {global_step:>8,}  "
                    f"avg loss {avg:.4f}  "
                    f"lr {model.lr:.6f}"
                )

        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch}/{EPOCHS}]  "
            f"pairs: {len(pairs):,}  "
            f"epoch loss: {epoch_loss / max(len(pairs), 1):.4f}  "
            f"time: {elapsed:.1f}s"
        )

    print("\nTraining complete.")
    print(f"Total steps : {global_step:,}")
    print(f"Final avg loss : {total_loss / max(global_step, 1):.4f}")

    # ------------------------------------------------------------------
    # 5. Quick similarity demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Similarity demo")
    print("=" * 60)

    demo_words = ["learning", "network", "data", "language", "intelligence"]
    for word in demo_words:
        if word not in processor.word2id:
            print(f"  '{word}' not in vocabulary – skipping.")
            continue
        results = model.get_most_similar(
            word, processor.word2id, processor.id2word, top_n=5
        )
        neighbours = ", ".join(f"{w} ({s:.3f})" for w, s in results)
        print(f"  {word:>15} →  {neighbours}")


if __name__ == "__main__":
    main()
