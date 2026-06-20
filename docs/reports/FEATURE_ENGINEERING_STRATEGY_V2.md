# Feature Engineering Strategy & Dataset Pathology Analysis (v2)

**Date:** December 24, 2025  
**Project Phase:** Advanced Feature Analysis & Pipeline Refinement  
**Focus:** Solving the "Hard" Subjects (02, 12, 13, 07) and standardizing the pipeline for "Easy" Subjects (14, 15, 06).

---

## 1. The Discovery: Subject Archetypes & Global Patterns
Through deep-dive visual analysis of Channel Amplitude (Heatmaps), Band Power (Bar Charts), and Channel Variance (Line Plots), we discovered that the dataset is not homogeneous. Subjects fall into distinct categories, and there are **systemic hardware failures** affecting the entire dataset.

### Type A: The "Internalizer" (Subject 14)
*   **Status:** Easy (High Accuracy).
*   **Signal Source:** Pure Brain Activity.
*   **Key Feature:** **"The Green Halo"**.
    *   In the **Gamma Band**, the Mean Amplitude for the **Positive** class (Green line) floats significantly higher than Negative/Neutral across the entire scalp.

### Type B: The "Externalizer" (Subject 15)
*   **Status:** Very Easy (High Accuracy).
*   **Signal Source:** Brain Activity + Muscle Artifacts (EMG).
*   **Key Feature:** **"The Frown Spike"**.
    *   In **Beta/Gamma Variance**, there is a massive, isolated spike at channel **FC5** (Frontal-Left) specifically for the **Negative** class.

### Type C: "The Stone" (Subjects 02, 12)
*   **Status:** Hard (Low Accuracy / Random Guessing).
*   **Signal Source:** None (Brain signal is weak/buried).
*   **Key Feature:** **"Glued Lines"**.
    *   Gamma/Beta lines are identical.
    *   **The Savior:** The **Delta Band**. Neutral class shows distinct separation (lower amplitude/stillness).

### Type D: "The Hybrid" (Subject 06)
*   **Status:** Easy (~83%).
*   **Signal Source:** Strong Brain Signal + Specific Artifacts.
*   **Key Feature:** **"The Wink"**.
    *   Massive spike at **AF4** (Right Frontal) for Positive class (Winking/Squinting).
    *   Despite noisy channels (CF2, P1), the global signal is strong enough to compensate.

---

## 2. Hardware Pathology: The "Bad" Channels
We identified systematic hardware failures that affect the entire dataset, not just specific subjects.

### 1. The "Sinkholes" (Cz, CPz, CF2)
*   **Observation:** In the "All Subjects" average, there are massive V-shaped dips at **Cz** and **CF2**.
*   **Diagnosis:** These sensors failed systematically across the majority of subjects (likely poor contact at the vertex).
*   **Impact:** A "zero" node at Cz breaks the GCN's spatial smoothing, pulling down neighbors.
*   **Fix:** **Lateral Interpolation** (see Section 3.3).

### 2. The "Hidden Noise" (P1, PO6)
*   **Observation:** While Mean Amplitude looks normal, the **Variance** at **P1** and **PO6** is massive (10-12+) in the global average.
*   **Diagnosis:** "Sleeper Agents." These electrodes are loose and flapping, introducing random noise into the posterior spatial features.
*   **Fix:** Automated Variance Thresholding.

### 3. The "Screaming" Channels (F7, T7, FC5)
*   **Observation:** Massive variance spikes in Frontal/Temporal regions.
*   **Diagnosis:** Muscle Artifacts (EMG).
*   **Fix:** **RobustScaler** to contain the outlier damage while preserving the pattern for Type B subjects.

---

## 3. The New Feature Engineering Strategy

Based on these findings, we are pivoting to a "Smart Preprocessing" pipeline.

### 3.1 Input Features: Mean + Variance
*   **Old Approach:** Input `[Batch, 62, 5]` (Mean DE only).
*   **Problem:** Throws away the "Frown Spike" (Subject 15) and the "Wink" (Subject 06).
*   **New Approach:** Input `[Batch, 62, 10]` (Stacking Mean and Variance).
    *   **Mean DE:** Captures the "Green Halo" (Subject 14).
    *   **Variance DE:** Captures the "Frown" (Subject 15) and the "Fidgeting" (Subject 02).

### 3.2 Normalization: RobustScaler
*   **Old Approach:** `StandardScaler` (Mean/Std).
*   **Problem:** Subject 02's massive F7 noise spike squashes all other channels to 0.
*   **New Approach:** `RobustScaler` (Median/IQR).
    *   Ignores extreme outliers when calculating scale.
    *   Preserves the subtle "Green Halo" in Subject 14 even if F7 is noisy.

### 3.3 Smart Interpolation (The "Lateral Bridge")
*   **Problem:** We cannot use standard interpolation (using all neighbors) because the Front/Back neighbors of Cz (e.g., FCz, CPz) are often also broken.
*   **Solution:** **Lateral Interpolation.**
    *   **Fix Cz:** `Cz = (C1 + C2) / 2` (Use Left/Right neighbors only).
    *   **Fix CPz:** `CPz = (CP1 + CP2) / 2`.
    *   **Fix CF2:** `CF2 = (FC2 + C2 + CP2) / 3` (Triangulate).
*   **Automated Trigger:**
    *   If **Variance > 3 * Global_Band_Variance**: Mark as Bad.
    *   If **Mean < 0.1 * Global_Band_Mean**: Mark as Dead.

### 3.4 Frequency Bands
*   **Strategy:** Keep all 5 bands separate.
    *   **Gamma:** Primary feature for "Easy" subjects (Positive detection).
    *   **Delta:** Primary feature for "Hard" subjects (Neutral detection).
    *   **Beta:** Primary feature for "Expressive" subjects (Negative detection via artifacts).

---

## 4. Training Strategy Implications

### Subject Dependent Training
*   **Verdict:** The proposed pipeline (RobustScaler + Lateral Interpolation) is sufficient.
*   **Reasoning:** The model can learn to ignore a permanently broken channel if it's consistent within the subject.

### Subject Independent Training
*   **Verdict:** Requires **Standardized Topology**.
*   **Reasoning:** We cannot train on Subject 6 (Good Cz) and test on Subject 7 (Dead Cz). The domain shift is too large.
*   **Action:** We must force-interpolate Cz, CPz, and CF2 for **ALL** subjects in Subject-Independent mode to ensure the input graph structure is identical.

---

## 5. Implementation Checklist
1.  [ ] **Update Dataset Class:** Calculate `np.var` alongside `np.mean`.
2.  [ ] **Implement Lateral Interpolation:** Hard-code the "Bridge" logic for Cz/CPz.
3.  [ ] **Implement Automated Cleaning:** Add variance threshold check for P1/PO6.
4.  [ ] **Switch Scaler:** Replace `StandardScaler` with `RobustScaler`.
