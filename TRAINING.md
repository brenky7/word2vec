# Training Experiments Log

This document tracks the performance of different training strategies implemented in this project. All experiments were conducted on the `text8` dataset (approx. 17M tokens) for **1 epoch**.

## 1. Baseline: Pure Stochastic Gradient Descent (SGD)
**Configuration:**
- **Batch Size:** 1 (Constant)
- **Learning Rate:** 0.025 with linear decay

**Results:**
- **Training Time:** ~111 min (6660s)
- **Total Steps:** 70,215,006
- **Final Avg Loss:** 2.2614
- **Similarity  check:**
overthrew →  makarios (0.957), abdur (0.957), sukarno (0.955), amanullah (0.954), ousted (0.952)
meaningful →  satisfactory (0.951), straightforward (0.950), meaningless (0.950), priori (0.945), verifiable (0.944)
lawsuits →  fairness (0.933), litigation (0.930), contractual (0.927), prosecutions (0.925), copyrights (0.923)
doric →  thracian (0.919), gaulish (0.910), elamite (0.909), dialectal (0.909), galatia (0.907)
rats →  skins (0.921), rabbits (0.916), chickens (0.914), rodents (0.913), larvae (0.910)

- **Observations:** Sharp associations, but slow convergence due to Python loop overhead per sample.

## 2. Optimized: Mini-Batch Gradient Descent
**Configuration:**
- **Batch Size:** 128 (Constant)
- **Learning Rate:** 0.025 with linear decay

**Results:**
- **Training Time:** ~46 min (2792s) **(2.4x Speedup)**
- **Total Steps (Batches):** 548,555
- **Final Avg Loss:** 2.3675



- **Observations:** Significant speedup. Loss is slightly higher and embeddings are slightly less sharp due to gradient averaging. This suggests that a higher learning rate or more epochs are needed to match SGD accuracy.

## 3. Dynamic Batching (Smith's Strategy)
*Goal: Combine the speed of large batches with the generalization of stochastic updates.*

**Configuration:**
- **Batch Size:** Dynamic - starts at 256, increases linearly to 1024
- **Learning Rate:** 0.05 (Constant - No decay)



**Results:**
- **Training Time:** 
- **Final Avg Loss:** 
- **Observations:** 