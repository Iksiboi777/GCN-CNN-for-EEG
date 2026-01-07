# Final Project Report: Adaptive Graph Convolutional Networks (GCN) for EEG Emotion Recognition on SEED-IV

## 1. Introduction

This project aimed to develop a robust, cross-subject emotion recognition model using the SEED-IV dataset. The primary challenge in EEG analysis is the extreme variability between subjects, where physiological differences, sensor placement shifts, and environmental noise often degrade model performance when testing on unseen subjects.

Our approach focused on a **Graph Convolutional Network (GCN)** architecture enhanced with dynamic feature engineering and attention mechanisms. The goal was to create a "Subject-Invariant" model that could adapt to these differences without requiring subject-specific retraining.

## 2. Methodology

### 2.1 Feature Engineering: The "Energy + Stability" Approach
Initial experiments relied solely on **Differential Entropy (DE)** across 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma). while DE captures the *magnitude* of brain activity (Energy), it failed to distinguish emotion in subjects with naturally low cortical arousal (e.g., distinguishing "Sad" from "Neutral" in low-energy subjects).

To resolve this, we introduced **Rolling Variance** (5 bands) as a secondary feature set.
- **Inputs:** 62 Channels × 10 Features (5 DE + 5 Variance).
- **Impact:** Variance acts as a "stability" metric. A calm "Neutral" state has low DE and low Variance. A suppressed "Sad" state might have low DE but unique variance patterns. This addition was critical for resolving confusion matrices for "flat" subjects.

### 2.2 Network Architecture: Dynamic Adaptation
To handle the 10-feature input and varying noise levels, we evolved the standard GCN into a dynamic architecture:

1.  **Adaptive Graph Input Layer:** 
    Instead of a fixed adjacency matrix (defining which electrodes are connected), we implemented a *trainable* adjacency matrix. This allows the model to learn which brain regions correlate most strongly for emotion recognition, regardless of physical distance.

2.  **Squeeze-and-Excitation (SE) Blocks:**
    We applied SE-Blocks to the feature dimension. This mechanism allows the network to dynamically weight the importance of different frequency bands per sample. For example, if a subject has excessive high-frequency noise (Gamma band artifacts), the SE-Block can learn to down-weight the Gamma channel for that specific input.

3.  **Global Node Attention:**
    We replaced standard pooling with an Attention mechanism that assigns an importance score to every electrode before aggregation. This allows the model to "ignore" broken or noisy channels (e.g., a loose sensor at Cz) by assigning them near-zero weights, preventing them from contaminating the global feature vector.

## 3. Subject Analysis: Performance Archetypes

We observed distinct categories of performance across the 15 subjects. It is crucial to note that **all subjects were retained in the final evaluation** to provide a realistic assessment of the model's capabilities in a real-world setting.

### Type A: The "Ideal" Subjects (e.g., Subject 6, 8, 5)
*   **Characteristics:** High Subject-Invariant representations. Clear separation between emotion classes in the feature space.
*   **Performance:** Consistently **>90% Accuracy**.
*   **Insight:** These subjects validate that the GCN architecture is highly effective when high-quality EEG data is available.

### Type B: The "Noisy" Subjects (e.g., Subject 10, 1)
*   **Characteristics:** Contaminated by significant artifacts (likely muscle movement or eye blinks).
*   **Performance:** Improved from ~65% to **~83%** via Architecture upgrades.
*   **Insight:** The success here validates our **Attention mechanisms**. Use of SE-Blocks and Global Attention allowed the model to filter out the specific noisy channels/bands and recover accurate classification.

### Type C: The "Difficult" Subjects (e.g., Subject 12, 2)
*   **Characteristics:** These subjects present fundamental data challenges. Subject 12, in particular, exhibits periods of "flatlining" signals or extremely low physiological response.
*   **Performance:** **<60% Accuracy**.
*   **Analysis:**
    *   **Trial Inflation/Bias:** On Subject 12, we observed a pattern where early trials were classified with reasonable accuracy, but performance degraded sharply in later trials within the same session. This suggests "Session Drift" or sensor impedance degradation over time.
    *   **Intrinsic Limitation:** The model cannot recover information that is not there. If the sensors fail to capture brain activity (the "flatline" issue), no amount of architectural complexity can invent the correct label. 
    *   **Reporting Decision:** Rather than removing these subjects, we report them as known failure modes. The model is highly accurate on valid EEG data but lacks the capability to diagnose or correct for fundamental sensor failure or subject non-compliance during data collection.

## 4. Results Timeline

*   **Phase 1: Reproducibility (Accuracy: ~76%)**
    *   Baseline GCN using only DE features.
    *   Struggled massively with "Type B" and "Type C" subjects.

*   **Phase 2: Variance Injection (Accuracy: ~81%)**
    *   Added Variance features.
    *   Solved the specific confusion between "Neutral" and "Sad" for low-energy subjects. 
    *   *Side Effect:* Slightly amplified noise for Subject 10, as noise has high variance.

*   **Phase 3: Dynamic Architecture (Final Accuracy: ~83%)**
    *   Added SE-Blocks and Attention.
    *   Successfully managed the noise introduced in Phase 2.
    *   Reached the "Knowledge Ceiling" for this dataset using topological features. Further improvements would likely require domain adaptation (transfer learning) rather than architectural tuning.

## 5. Conclusion

The final Adaptive GCN model achieves a robust **83% average accuracy** across 15 subjects. It successfully demonstrates:
1.  **Noise Robustness:** Via Attention mechanisms handling Subject 10.
2.  **State Discrimination:** Via Variance features distinguishing low-arousal states.
3.  **Realistic Limits:** The model correctly identifies that for subjects like #12, the limiting factor is the data quality itself. Future work for such cases should focus on hardware-level signal quality indicators to reject bad trials before they reach the classifier.
