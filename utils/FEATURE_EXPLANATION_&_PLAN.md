# Feature Explanation & Phase 2 Plan: Differential Entropy (DE)

**Date:** December 9, 2025  
**Context:** Transitioning from Raw Time-Series Data (Phase 1) to Pre-Extracted Features (Phase 2) for the SEED Dataset.

---

## 1. The "ExtractedFeatures" Dataset Structure

You are currently holding the **pre-processed, feature-engineered version** of the SEED dataset. This is the standard format used by almost all State-of-the-Art (SOTA) papers.

### 1.1 Why No Segmentation Script?
In Phase 1 (Raw Data), we had continuous EEG signals (e.g., 4 minutes of data at 200Hz). We had to manually cut them into 400-step windows.

**In Phase 2, this is already done for you.**
The files in `ExtractedFeatures_1s` contain data that has **already been segmented into 1-second non-overlapping windows**.

*   **Raw Data:** Continuous Waveform.
*   **Feature Data:** A sequence of 1-second "snapshots".

### 1.2 File Structure & Dimensions
Inside a file like `1_20131027.mat` (Subject 1, Session 1), you see keys like `de_LDS1`.

*   **Key Name:** `de_LDS1` (Differential Entropy, LDS Smoothed, Trial/Movie #1).
*   **Shape:** `(62, 235, 5)`
    *   **62 (Channels):** The physical electrodes on the cap.
    *   **235 (Time):** This is **NOT** raw time steps. This means the movie clip was 235 seconds long, and you have **235 samples** (segments).
    *   **5 (Bands):** The frequency bands (Delta, Theta, Alpha, Beta, Gamma).

**Crucial Takeaway:**
We do not need to "segment" this array. We simply treat the **235** dimension as our **Batch of Samples**.
*   Sample 1: The brain state at $t=0s$.
*   Sample 2: The brain state at $t=1s$.
*   ...

---

## 2. Feature Deep Dive

### 2.1 Differential Entropy (DE)
*   **Definition:** $h(X) = \frac{1}{2} \log(2\pi e \sigma^2)$.
*   **Intuition:** It is the **Logarithm of the Energy** in a specific frequency band.
*   **Why it works:**
    *   **Gamma Band Visibility:** In raw PSD (Power Spectral Density), low frequencies (Delta) are huge, and high frequencies (Gamma) are tiny. The neural network ignores Gamma. The `log` operation in DE scales them to be comparable.
    *   **Gaussianity:** DE features follow a distribution that is easier for classifiers to separate than raw power values.

### 2.2 Why `_LDS` (Linear Dynamic System)?
*   **The Problem:** EEG is noisy. A subject feeling "Happy" might have a split-second drop in Gamma waves due to a blink or noise.
*   **The Solution:** LDS is a smoothing algorithm. It looks at the timeline (Sample 1 -> Sample 2 -> Sample 3) and smooths out the "jitter".
*   **Impact:** Papers show LDS features yield **10-15% higher accuracy** than non-smoothed features.

### 2.3 Why Drop DASM/RASM?
*   **DASM:** $DE(Left) - DE(Right)$.
*   **RASM:** $DE(Left) / DE(Right)$.
*   **Reason for Exclusion:** These features were invented for old classifiers (SVMs) that couldn't understand geometry. A **GCN (Graph Convolutional Network)** is *designed* to learn spatial relationships. Feeding it DASM is redundant; the GCN will learn the asymmetry itself by comparing Node Left and Node Right.

---

## 3. The New Model Plan (Phase 2)

We are discarding the CNN feature extractor. The "Features" are already extracted.

### 3.1 Input Transformation
*   **Old Input:** `(Batch, 62, 400)` -> Raw Waves.
*   **New Input:** `(Batch, 62, 5)` -> 5 Numbers per channel.

### 3.2 Architecture: `GCN_DE_Model`
The model becomes much lighter and faster.

1.  **Input Layer:** Accepts 5 features per node.
2.  **GCN Layers:** 3 layers of Graph Convolution.
    *   *What it learns:* "If the Frontal Left node has high Gamma and the Parietal Right node has low Alpha, that means Positive Emotion."
3.  **Global Pooling:** Averages the nodes to get a graph-level vector.
4.  **Classifier:** Simple Linear Layer -> 3 Classes.

### 3.3 Training Strategy
*   **Data Loading:** We write a custom loader that opens `.mat` files, grabs `de_LDS*`, and stacks them into a big Tensor.
*   **Speed:** Since the input is 80x smaller than raw data, training will be lightning fast.
*   **Expectation:** We expect to break the ~37% ceiling and reach **70-90% accuracy**.

---

## 4. Summary of Changes

| Component | Phase 1 (Raw) | Phase 2 (DE Features) |
| :--- | :--- | :--- |
| **Input Data** | Raw Voltage (Time Series) | Differential Entropy (Frequency Bands) |
| **Segmentation** | Manual (Sliding Window) | **Pre-computed (1s windows)** |
| **Dimensions** | `(62, 400)` | `(62, 5)` |
| **Model** | CNN + GCN | **Pure GCN** |
| **Preprocessing** | Bandpass Filter, ICA | **LDS Smoothing (Done)** |
| **Target Accuracy** | ~36% (Failed) | **> 80% (SOTA)** |