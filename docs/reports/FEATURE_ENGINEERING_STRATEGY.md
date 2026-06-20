# Feature Engineering Strategy & Dataset Pathology Analysis

**Date:** December 23, 2025  
**Project Phase:** Advanced Feature Analysis & Pipeline Refinement  
**Focus:** Solving the "Hard" Subjects (02, 12) without sacrificing the "Easy" Subjects (14, 15).

---

## 1. The Discovery: Three Subject Archetypes
Through deep-dive visual analysis of Channel Amplitude (Heatmaps), Band Power (Bar Charts), and Channel Variance (Line Plots), we discovered that the dataset is not homogeneous. Subjects fall into three distinct categories based on their physiological and artifact patterns.

### Type A: The "Internalizer" (Subject 14)
*   **Status:** Easy (High Accuracy).
*   **Signal Source:** Pure Brain Activity.
*   **Key Feature:** **"The Green Halo"**.
    *   In the **Gamma Band**, the Mean Amplitude for the **Positive** class (Green line) floats significantly higher than Negative/Neutral across the entire scalp.
*   **Variance:** Low and consistent. The signal is strong enough to overpower hardware noise.

### Type B: The "Externalizer" (Subject 15)
*   **Status:** Very Easy (High Accuracy).
*   **Signal Source:** Brain Activity + Muscle Artifacts (EMG).
*   **Key Feature:** **"The Frown Spike"**.
    *   In **Beta/Gamma Variance**, there is a massive, isolated spike at channel **FC5** (Frontal-Left) specifically for the **Negative** class.
    *   **Implication:** The subject physically frowns or tenses their jaw during negative clips. The model can use this artifact as a 100% predictor.

### Type C: "The Stone" (Subjects 02, 12)
*   **Status:** Hard (Low Accuracy / Random Guessing).
*   **Signal Source:** None (Brain signal is weak/buried).
*   **Key Feature:** **"Glued Lines"**.
    *   In Gamma/Beta, the Mean Amplitude lines for all three emotions are identical (overlapping).
    *   **The Savior:** The **Delta Band**.
    *   While Gamma is useless, the **Neutral** class shows distinct separation in the Delta band (lower amplitude).
    *   **Hypothesis:** The subject is physically still during Neutral clips (Low Delta) but fidgets randomly during emotional clips (High Delta/Noise).

---

## 2. Hardware Pathology: The "Bad" Channels
We identified systematic hardware failures that affect "Hard" subjects disproportionately.

### 1. The "Screaming" Channels (F7, T7)
*   **Observation:** Massive variance spikes in Frontal/Temporal regions.
*   **Subject 02:** The spike is random noise (loose contact). It destroys the scale of the data.
*   **Subject 15:** The spike is a signal (frowning).
*   **Solution:** We cannot simply drop these channels (or we lose Subject 15's advantage). We must use **Robust Scaling** to contain the outlier damage for Subject 02 while preserving the pattern for Subject 15.

### 2. The "Sinkhole" Channels (Cz, CPz)
*   **Observation:** In Subjects 02 and 12, the central channels **Cz** and **CPz** often drop to near-zero amplitude ("Dead Channels").
*   **Impact:** This breaks the GCN's spatial smoothing. A "zero" node pulls down the values of all its neighbors during graph convolution.
*   **Solution:** We must interpolate **Cz**, but we cannot use CPz to do it (since CPz is also dead).

---

## 3. The New Feature Engineering Strategy

Based on these findings, we are pivoting from a "One-Size-Fits-All" approach to a multi-view feature strategy.

### 3.1 Input Features: Mean + Variance
*   **Old Approach:** Input `[Batch, 62, 5]` (Mean DE only).
*   **Problem:** Throws away the "Frown Spike" (Variance) and makes Subject 15 look like Subject 02.
*   **New Approach:** Input `[Batch, 62, 10]` (Stacking Mean and Variance).
    *   **Mean DE:** Captures the "Green Halo" (Subject 14).
    *   **Variance DE:** Captures the "Frown" (Subject 15) and the "Fidgeting" (Subject 02).

### 3.2 Normalization: RobustScaler
*   **Old Approach:** `StandardScaler` (Mean/Std).
*   **Problem:** Subject 02's massive F7 noise spike squashes all other channels to 0.
*   **New Approach:** `RobustScaler` (Median/IQR).
    *   Ignores extreme outliers when calculating scale.
    *   Preserves the subtle "Green Halo" in Subject 14 even if F7 is noisy.

### 3.3 Channel Interpolation (The "Triangle" Fix)
*   **Target:** Fix **Cz** (Central Node).
*   **Method:** `Cz = (C1 + C2 + FCz) / 3`.
*   **Exclusion:** Explicitly **exclude CPz** from the calculation, as it is confirmed faulty in Hard subjects.

### 3.4 Frequency Bands
*   **Strategy:** Keep all 5 bands separate.
    *   **Gamma:** The primary feature for "Easy" subjects (Positive detection).
    *   **Delta:** The primary feature for "Hard" subjects (Neutral detection).
    *   **Beta:** The primary feature for "Expressive" subjects (Negative detection via artifacts).

---

## 4. Summary of Workflow
1.  **Visual Analysis:** Generated Line Plots and Heatmaps for Subjects 02, 12, 14, 15.
2.  **Pattern Recognition:** Identified that "Hard" subjects lack Gamma separation but possess Delta separation.
3.  **Artifact Diagnosis:** Correlated high variance at FC5/F7 with specific emotions for expressive subjects.
4.  **Pipeline Update:** Defined the need for Variance features and Robust Scaling to handle the dichotomy between "Signal Noise" (Subject 15) and "Garbage Noise" (Subject 02).
