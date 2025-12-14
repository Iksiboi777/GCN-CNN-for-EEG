# GCN-DE 4s Experiment Report: Timeline & Findings

**Date:** December 12, 2025  
**Model:** Graph Convolutional Network (GCN)  
**Feature:** Differential Entropy (DE)  
**Window Size:** 4 Seconds  
**Split Mode:** Subject-Dependent  

---

## 1. Executive Summary
This report documents the systematic investigation into training a GCN on the SEED dataset using 4-second DE features. Over the course of 9+ attempts, we peeled back layers of "fake performance" (overfitting, leakage) to reveal the fundamental challenges of the dataset: **Session Drift** and **Stimulus-Specific Overfitting**.

The final "true" performance of a standard GCN on unseen video clips (cross-trial generalization) is approximately **56%**, significantly lower than the ~90% often reported in literature which typically uses random splits that leak temporal information.

---

## 2. Chronological Timeline of Experiments

### Phase 1: The Naive Baseline (Attempts 1-4)
*   **Goal:** Establish a baseline performance using a standard GCN architecture.
*   **Strategy:** **Session Holdout Split** (Train on Sessions 1 & 2, Test on Session 3).
*   **Configuration:** 2-Layer GCN, 64 Hidden Units, Dropout 0.5.
*   **Result:**
    *   **Training Accuracy:** > 90% (Model learned S1/S2 perfectly).
    *   **Test Accuracy:** ~50-60% (Peaked early, then degraded).
*   **Diagnosis:** **Overfitting to Source Domain.** The model memorized patterns specific to Sessions 1 & 2 that did not generalize to Session 3.

### Phase 2: The Regularization Push (Attempt 5)
*   **Hypothesis:** Stronger regularization will prevent memorization and force the learning of general emotional features.
*   **Action:** Increased Dropout to **0.6** and Weight Decay to **1e-3**.
*   **Result:**
    *   **Test Accuracy:** ~62% (Slight improvement).
    *   **Observation:** The gap between Train and Test narrowed, but performance plateaued.
*   **Conclusion:** Regularization helps stability but cannot bridge the fundamental distribution gap between sessions.

### Phase 3: Diagnostic Analysis (Evolution & Heatmaps)
*   **Action:** Developed `analyze_evolution.py` to visualize learning dynamics epoch-by-epoch.
*   **Key Discoveries:**
    1.  **Subject Bias:** Subjects 12, 13, and 14 were consistently "Hard" (Error > 70%) across all attempts.
    2.  **Negative Transfer:** Test accuracy often peaked at Epoch 5 (generic features) and dropped as the model learned S1/S2 specifics.
    3.  **Class Confusion:** High confusion between **Neutral** and **Negative** classes.

### Phase 4: The Normalization Fix (Attempt 6)
*   **Hypothesis:** The "Hard Subjects" and Session 3 failure are due to **Covariate Shift** (differences in signal amplitude/scale).
*   **Action:** Implemented **Subject-Specific & Session-Specific Z-Score Normalization**.
    *   Formula: $X_{norm} = (X - \mu_{sub,sess}) / \sigma_{sub,sess}$
*   **Result:**
    *   **Test Accuracy:** ~52% (Performance actually degraded).
    *   **Observation:** Normalization fixed the scales, making the **Pattern Mismatch** (Concept Drift) even more apparent.

---

## 3. Conclusion of 4s Experiments
The 4-second window experiments established the foundational understanding of the dataset's difficulty. The "Static GCN" approach proved insufficient for handling the severe inter-session drift. 

**Transition:** We moved to **1-second windows** and **Dynamic Graph Learning (DGCNN)** to attempt to solve these issues with higher temporal resolution and adaptive topology. See `DGCNN_Report.md` for the continuation of this work.
*   **Conclusion:** The problem is not just "volume" (amplitude); the "shape" of the brainwaves for specific emotions changes between sessions.

### Phase 5: The Drift Diagnosis
*   **Action:** Developed `analyze_session_drift.py` to quantify the statistical distance between sessions.
*   **Findings:**
    *   **S1 vs S2 Distance:** ~0.5 (Small shift).
    *   **S1 vs S3 Distance:** **> 2.0 (Huge shift)**.
*   **Implication:** Session 3 is statistically an "Alien World" compared to S1/S2. Training on S1/S2 and testing on S3 is a difficult **Domain Generalization** task, not a simple classification task.

### Phase 6: The Leakage Trap (Attempt 7)
*   **Hypothesis:** To handle drift, we must mix sessions during training.
*   **Action:** Switched to **Random Split** (Shuffle all 4s windows from all sessions).
*   **Result:**
    *   **Test Accuracy:** **100%**.
*   **Diagnosis:** **Temporal Data Leakage**.
    *   Adjacent 4s windows (e.g., Window 1 and Window 2 of the same trial) are nearly identical.
    *   The model memorized Window 1 (Train) and recognized Window 2 (Test).
    *   **Verdict:** This result is invalid.

### Phase 7: The True Baseline (Attempts 8-9)
*   **Hypothesis:** We must mix sessions to handle drift, but split by **Video Clip (Trial)** to prevent leakage.
*   **Action:** Implemented **Stratified Trial Split**.
    *   **Train:** 12 Trials per session (Videos A-L).
    *   **Test:** 3 Trials per session (Videos M-O, completely unseen).
*   **Result:**
    *   **Test Accuracy:** **~56%**.
*   **Conclusion:** This is the realistic performance of a GCN on this dataset without leakage. The drop from 85% (Train) to 56% (Test) indicates the model relies heavily on **Video-Specific Features** rather than generic **Emotion Features**.

### Phase 8: High Regularization (Attempt 10)
*   **Hypothesis:** The model is too complex and memorizing video signatures. Reducing capacity and increasing noise might force generalization.
*   **Action:**
    *   Reduced Hidden Dimension: **64 -> 32**.
    *   Increased Dropout: **0.5 -> 0.7**.
*   **Result:**
    *   Performance remained comparable to the baseline (~56%), indicating that simple regularization is insufficient to overcome the domain shift. The model struggles to find invariant features that persist across different video stimuli.

---

## 3. Key Technical Findings

### 1. Concept Drift is Severe
The brain response to the same emotion changes significantly over time (weeks). A model trained on data from Week 1 will fail on data from Week 2 unless specific **Domain Adaptation** techniques are used.
*   **Quantification:** Euclidean distance between class centroids shifts from ~0.5 (within-session) to >2.0 (cross-session).

### 2. Stimulus-Specific Overfitting
The model struggles to generalize across different video clips. It learns "What the 'Sad' video looks like" rather than "What 'Sadness' looks like." When shown a new 'Sad' video, it fails to recognize it.

### 3. Subject Bias Analysis
*   **Hard Subjects:** Subjects 12, 13, and 14 consistently showed high error rates (>70%).
*   **Easy Subjects:** Subjects 1, 6, and 9 often performed better.
*   **Implication:** Some subjects exhibit more stable neural patterns or less noise/drift than others. This suggests that a "one-size-fits-all" model is suboptimal.

### 4. Normalization is Necessary but Insufficient
Subject-Specific Normalization is critical for training stability, but it does not solve the geometric drift of the feature space.

---

## 4. Recommendations for Future Work

To improve beyond the 56% baseline, simple hyperparameter tuning is insufficient. Future attempts should focus on:

1.  **Invariant Learning:** Techniques to force the model to ignore "Trial ID" (Video signature) and focus on shared emotional patterns.
2.  **Domain Adversarial Training (DANN):** Using an adversary to penalize the model if it learns features specific to a single session or trial.
3.  **Data Augmentation:** Injecting noise or masking channels during training to prevent memorization of specific video patterns.
