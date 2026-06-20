# GCN-DE 1s Experiment Report: Timeline & Findings

**Date:** December 13, 2025  
**Model:** Graph Convolutional Network (GCN)  
**Feature:** Differential Entropy (DE)  
**Window Size:** 1 Second  
**Split Mode:** Subject-Dependent  

---

## 1. Data Configuration & Preprocessing

### 1.1 Window Size Transition
We transitioned from 4-second windows to **1-second windows**.
*   **Goal:** Increase the number of training samples (4x data volume) to help the GCN learn more robust features and potentially capture faster emotional dynamics.
*   **Total Samples:** ~50,000 samples (vs ~12,000 for 4s).

### 1.2 Frequency Band Splits
The underlying EEG data was preprocessed into two distinct frequency ranges before DE feature extraction:
1.  **Standard Bands (1-49 Hz):** Includes Delta, Theta, Alpha, and Beta.
2.  **Gamma Band (50-75 Hz):** Processed separately to isolate high-frequency emotional markers.
This separation ensures that the "Gamma" features are not contaminated by lower-frequency noise or powerline interference (50Hz notch).

---

## 2. Chronological Timeline of Experiments

### Phase 1: The Random Split Baseline (Attempt 1)
*   **Strategy:** **Random Split** (Shuffle all 1s windows from all sessions, 80/20 split).
*   **Result:**
    *   **Test Accuracy:** High (similar to 4s Random Split).
    *   **Diagnosis:** **Temporal Data Leakage**. Adjacent 1s windows from the same trial are nearly identical. The model memorized the specific trial signatures rather than learning generalized emotion features.
    *   **Action:** Discarded this strategy in favor of Session Holdout for stricter validation.

### Phase 2: Session Holdout with Band Weights (Attempt 2)
*   **Strategy:** **Session Holdout** (Train S1+S2, Test S3).
*   **Innovation:** Introduced **Subject-Specific Band Weighting**.
    *   **Logic:** Not all frequency bands are equally useful for every subject.
    *   **Method:** Calculated **Fisher Scores (F-value)** for each band (Delta, Theta, Alpha, Beta, Gamma) per subject to quantify discriminative power.
    *   **Implementation:** Loaded weights from `subject_band_weights.json` and scaled input features before the GCN.
*   **Result:**
    *   **Performance:** More balanced than previous attempts, removing the need for explicit Class Weights.
    *   **Observation:** Error rates were highly subject-specific. Some subjects (14, 15) performed perfectly, while others (2, 4, 12) failed catastrophically.

---

## 3. Deep Dive: The Band Weighting Saga

### 3.1 The Fisher Score Analysis
We analyzed the calculated weights in `subject_band_weights.json` and discovered a clear correlation between "Weight Distribution" and "Model Performance".

#### The "Good" Subjects (14, 15) -> The Gamma Hypothesis
*   **Weights:** High Gamma (~3.0), Low Delta (~0.2).
*   **Outcome:** Perfect performance.
*   **Conclusion:** The F-score correctly identified Gamma as the "Golden Feature" for emotion. The model focused on Gamma and ignored noise.

#### The "Catastrophic" Subject (2) -> The Delta Trap
*   **Weights:** **High Delta (~3.7)**, Low Gamma (~0.9).
*   **Outcome:** High error rate.
*   **Conclusion:** Delta (1-3 Hz) is often contaminated by **EOG (Eye Blinks)**. The F-score found that blinks were "separable" in the training set (e.g., subject blinked more during Negative clips). The model learned to detect blinks, which failed in the Test set (Session 3) where blink patterns changed.

#### The "Drowsy" Subjects (4, 12) -> The Theta Trap
*   **Weights:** **High Theta (~2.5)**.
*   **Outcome:** Poor generalization.
*   **Conclusion:** Theta (4-7 Hz) is a marker for **Fatigue/Drowsiness**. The model learned "Tired vs Awake" instead of "Sad vs Happy". Fatigue levels drift significantly between sessions.

### 3.2 The "Guided" Weighting Intervention
To fix these issues, we implemented a **Safety Cap** on the weights during loading:
1.  **Kill Delta:** Multiplied Delta weights by 0.2 to suppress artifact learning.
2.  **Cap Weights:** Clipped max weights to 2.5 to prevent gradient explosions.
3.  **Lower LR:** Reduced Learning Rate to 0.0001 to accommodate the scaled inputs.

---

## 4. Feature Inspection Findings

### 4.1 The "Energy Hypothesis"
We visualized the raw DE feature distributions (Boxplots) to understand why the model confuses classes.
*   **Positive:** Consistently **High Energy** (Gamma). Easy to classify.
*   **Negative vs Neutral:** **High Overlap**.
    *   For many subjects (e.g., Subject 1, 4), the Gamma energy for Negative and Neutral is identical.
    *   **Implication:** The model cannot distinguish Negative from Neutral based on "Energy" alone. It defaults to guessing "Negative" (High Recall, Low Precision) or "Neutral" (High Precision, Low Recall) based on subtle noise.

### 4.2 Conclusion
Simple energy-based features (DE) are insufficient for separating Negative and Neutral emotions for certain subjects. The model requires:
1.  **Structural Features:** Connectivity patterns (GCN) that go beyond simple energy.
2.  **Class-Specific Weights:** Acknowledging that Gamma might be reliable for "Positive" but unreliable for "Negative".

---

## 5. Recommendations
*   **Multi-View Architecture:** Train separate branches weighted for specific class hypotheses (e.g., a "Positive Detector" that trusts Gamma, and a "Negative Detector" that trusts Alpha/Beta).
*   **Unsupervised Pre-training:** Use Autoencoders to find latent representations that separate Negative/Neutral better than raw DE.
