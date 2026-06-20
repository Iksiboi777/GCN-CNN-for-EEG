# Data-Centric Approach Report: Forensic Analysis & Diagnostics

**Date:** January 3, 2026  
**Phase:** Data Pathology & Feature Engineering Validation  
**Objective:** To validate if "Smart Preprocessing" (Lateral Interpolation, Robust Scaling, Variance Features) could solve the "Hard" subjects (02, 12) and to diagnose the specific failure mode of the "Negative vs. Neutral" boundary.

---

## 1. Phase 1: The "Forensic" Implementation (Attempts 27 & 29)
Following the discovery of "Sinkholes" (dead midline channels) and "Screaming Channels" (high-variance noise), we implemented a targeted feature engineering pipeline in `train_de.py`.

### The Strategy
1.  **Lateral Interpolation:** Explicitly fixing `Cz`, `CPz`, `Fz` by averaging immediate left/right neighbors to rescue Subject 12.
2.  **Robust Scaling:** Using Median/IQR instead of Mean/Std to contain the massive noise spikes in Subject 02.
3.  **Variance Features:** Adding 5 channels of Rolling Variance to capture muscle artifacts (EMG) for Subject 15.
4.  **Band Weighting:** Penalizing Beta/Gamma for "Hard" subjects to force the model to look at Delta/Theta.

### The Results (Subject-Dependent, 3-Class)
*   **DGCNN (Attempt 27):** 64% Accuracy.
*   **GCN (Attempt 29):** 65% Accuracy.

### Key Findings
1.  **The Ceiling Remains:** The performance did not break the ~67% ceiling established by previous simpler models.
2.  **Lateral Interpolation Failed:** Subject 12's error rate remained near 100%. The "Sinkhole" was too deep; the neighbors (`C1`, `C2`) were likely also compromised or insufficient to reconstruct the vertex signal.
3.  **Robust Scaling Failed:** Subject 02's error rate remained high. The noise in `F7` was so dominant that even robust scaling left the signal-to-noise ratio too low for learning.
4.  **Variance Worked:** Subject 15 (The "Externalizer") showed low error rates, confirming that muscle tension is a valid predictor for this subject.
5.  **The "Neutral" Trap:** The model continued to struggle distinguishing "Negative" from "Neutral", often classifying Neutral samples as Negative due to aggressive class weighting.

---

## 2. Phase 2: The Binary Diagnostic (Negative vs. Neutral)
To isolate the problem, we stripped the pipeline down to a **Binary Classification** task (`train_binary_diag.py`), removing the "Easy" Positive class (which relies on the high-energy Gamma "Green Halo").

### Experiment A: Gamma Ablation (No Gamma)
*   **Setup:** Gamma Mean feature set to 0.0.
*   **Result:** **66% Accuracy**.
*   **Confusion Matrix:** High recall for Neutral (71%), lower for Negative (60%).
*   **Conclusion:** The model *can* distinguish Negative from Neutral better than random chance (50%) using only lower frequencies (Delta, Theta, Alpha, Beta) and Variance. However, the boundary is fuzzy.

### Experiment B: With Gamma
*   **Setup:** Gamma Mean feature included.
*   **Result:** **68% Accuracy**.
*   **Conclusion:** Adding Gamma only improved performance by **2%**.
    *   **Critical Insight:** This proves that for the "Hard" task (Neg vs. Neu), **Gamma is not the differentiator.** The high accuracy seen in 3-class models is almost entirely driven by the "Positive" class's Gamma signature. When that is removed, the model struggles.

### The "Fake Good" vs. "True Good"
*   **Subject 09:** Failed in the diagnostic (without Gamma), implying their previous success was Gamma-dependent (likely EMG noise).
*   **Subject 15:** Succeeded in the diagnostic, confirming their signals (Variance/Beta) are robust.

---

## 3. The "Rising Loss" Paradox
Across all experiments, we observed a concerning trend:
*   **Validation Accuracy:** Slowly rises or stays flat.
*   **Validation Loss:** **Explodes** (increases significantly).

### Diagnosis: Overconfidence & Memorization
This pattern indicates that the model is **memorizing subject identity** rather than learning emotion.
1.  The model learns to identify "Subject 12".
2.  It incorrectly guesses "Neutral" for a Negative trial.
3.  As training progresses, it becomes **99% confident** in that wrong guess because it recognizes the person, not the brain state.
4.  CrossEntropyLoss penalizes confident wrong answers heavily, driving Loss up, while Accuracy (the count of wrong answers) stays the same.

---

## 4. Conclusion & Next Steps
We have proven that **preprocessing alone cannot fix the "Hard" subjects** (02, 12) because the signal is likely absent, not just noisy. We also proved that the model's ability to distinguish Negative from Neutral is weak and barely aided by Gamma.

**The Critical Question:**
Is the model actually reacting to the **video stimulus** (the trial), or is it just classifying the **person**?

### Next Approach: Per-Trial Analysis
We must move from "Epoch-based" metrics to "Trial-based" metrics.
*   **Goal:** Verify if the model's predictions change when the video changes.
*   **Method:** Generate Heatmaps of Accuracy per Trial (Video Clip) for each subject.
*   **Hypothesis:**
    *   If a subject's row is solid Red/Blue, the model is ignoring the video (Pure Bias).
    *   If a subject's row is checkerboard, the model is reacting to the content.
