# NumPy Word2Vec Implementation
Author - Peter Brenkus

This repository contains a demonstrative implementation of the Word2Vec algorithm (Skip-gram with Negative Sampling architecture), built using only NumPy. No high-level machine learning frameworks like PyTorch or TensorFlow are used.

## Project Goal
The primary objective is to learn about the core mechanisms behind neural embedding models, including:
-   **Data Preprocessing:** Tokenization, vocabulary building, and subsampling of frequent words.
-   **Skip-gram Architecture:** Generating context-target pairs from a sliding window.
-   **Negative Sampling:** Efficient optimization using binary classification objectives instead of full Softmax.
-   **Backpropagation:** Manual derivation and implementation of gradients and parameter updates (SGD).
