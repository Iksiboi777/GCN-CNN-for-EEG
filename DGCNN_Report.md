# DGCNN Experiment Report: Dynamic Graph Learning

**Date:** December 14, 2025  
**Model:** Dynamical Graph Convolutional Neural Network (DGCNN)  
**Window Sizes:** 1s & 4s  
**Split Mode:** Subject-Dependent (Session Holdout)  

---

## 1. Motivation: Why DGCNN?
After extensive experimentation with Static GCNs (fixed physical graph), we hit a performance ceiling of ~67% (1s) and ~56% (4s). The error analysis revealed a persistent "Red Stripe" pattern in the heatmaps:
*   **"Standard" Subjects (e.g., 14, 15):** The model learned perfectly.
*   **"Outlier" Subjects (e.g., 2, 12):** The model failed completely and never improved.

**Hypothesis:** The fixed physical graph (based on electrode distance) is valid for "Standard" subjects but invalid for "Outliers" whose functional brain connectivity during emotion processing differs from the physical topology. A **Dynamic Graph** that *learns* the connections is needed.

---

## 2. Implementation Evolution

### 2.1 Version 1: Global Learnable Graph (The "Average" Map)
*   **Concept:** Instead of a fixed Adjacency Matrix $A$, we made $A$ a learnable parameter `self.A = nn.Parameter(...)`.
*   **Mechanism:** The model learns *one* optimal graph structure that minimizes loss across all training subjects.
*   **Result:**
    *   **Performance:** Identical to Static GCN (~67%).
    *   **Diagnosis:** The model learned the "Average Best Graph". This helped the majority but still failed for the outliers (Subject 12), as a single global graph cannot satisfy conflicting connectivity needs.

### 2.2 Version 2: Input-Dependent Dynamic Graph (True DGCNN)
*   **Concept:** The graph structure $A$ is generated *on the fly* for every single sample.
*   **Mechanism:** Self-Attention style.
    $$ A = \text{Softmax}\left( \frac{Q(x) K(x)^T}{\sqrt{d}} \right) $$
    Where $Q$ and $K$ are linear transformations of the input features $x$.
*   **Goal:** Allow the model to "rewire" itself instantly. If Subject 12 blinks (Delta artifact), the model should disconnect the frontal nodes *for that specific second*.
*   **Result:**
    *   **Performance:** Still capped at ~67%.
    *   **Observation:** The model likely overfitted to the "easy" features (Gamma energy) and ignored the subtle connectivity patterns needed for the hard subjects.

### 2.3 Version 3: Learnable Band Attention (SE-Block)
*   **Concept:** Instead of manual Fisher Score weighting (which might be biased), let the model learn which frequency bands to trust.
*   **Mechanism:** A Squeeze-and-Excitation block that predicts a weight vector $w \in [0, 1]^5$ for the 5 bands (Delta, Theta, Alpha, Beta, Gamma) based on the global input.
*   **Result:**
    *   **Performance:** No significant change.
    *   **Conclusion:** The model learned to replicate the standard weights (High Gamma, Low Delta) but could not find a solution for the "Hard" subjects.

---

## 3. Key Findings & Failure Analysis

### 3.1 The "Ceiling" is Robust
We threw three levels of increasing complexity at the problem:
1.  **Static Topology** (GCN)
2.  **Global Dynamic Topology** (DGCNN v1)
3.  **Input-Dependent Topology + Band Attention** (DGCNN v3)

All three converged to the **exact same performance (~67%)** and the **exact same error pattern** (Subjects 2, 4, 12 failing).

### 3.2 The "Random Split" Proof
In Attempt 11, we ran a **Random Split** (mixing sessions).
*   **Result:** ~53% Accuracy (Chance level).
*   **Implication:** This proves **Severe Non-Stationarity**. A "Sad" signal from Session 1 is statistically different from a "Sad" signal from Session 3. The model cannot generalize across time without explicit Domain Adaptation.

### 3.3 Conclusion
The failure is not in the *spatial* modeling (GCN vs DGCNN) or the *spectral* weighting. The failure is due to **Negative Transfer** caused by **Session Drift**. Training on multiple subjects/sessions simultaneously forces the model to learn a "Mean Representation" that works for the majority but sacrifices the outliers.

**Next Step:** Move to **Subject-Independent (LOSO)** training to isolate the generalization gap and apply Domain Adaptation techniques.
