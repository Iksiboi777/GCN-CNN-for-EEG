# Project Summary: GCN-CNN for EEG Emotion Recognition (SEED Dataset)

**Date:** December 6, 2025  
**Repository:** GCN-CNN-for-EEG  
**Current Status:** Functional Pipeline, Model Training, Domain Shift Identified.

---

## 1. Project Overview
The goal of this project is to classify emotions (Negative, Neutral, Positive) from EEG signals using the SEED dataset. We implemented a hybrid **Graph Convolutional Network (GCN) + Convolutional Neural Network (CNN)** architecture.

*   **Dataset:** SEED (15 Subjects, 3 Sessions per subject).
*   **Input Data:** Raw Time-Series EEG (62 Channels).
*   **Architecture:** 
    *   **Graph Construction:** Based on physical electrode distances (k-NN, k=5).
    *   **Feature Extractor:** 1D CNN (per node) to extract temporal features from raw waves.
    *   **Spatial Learning:** GCN layers to learn interactions between brain regions.
    *   **Classifier:** Global Mean Pooling + Fully Connected Layers.

---

## 2. Data Pipeline & Preprocessing
*   **Source:** `Cleaned_EEG_ICA` (.mat files).
*   **Segmentation:** Data was segmented into fixed-length windows.
    *   **Window Size:** 400 time steps (Raw signal).
    *   **Graph Nodes:** 62 (corresponding to EEG channels).
*   **Split Strategy:** **Subject Dependent** (Cross-Session).
    *   **Train:** Sessions 1 & 2 (All subjects mixed).
    *   **Test:** Session 3 (All subjects mixed).
    *   *Goal:* Test the model's ability to generalize to a new day (Session 3) after learning from previous days.

---

## 3. Development Log & Troubleshooting

### Phase 1: Initial Implementation & Memory Issues
*   **Issue:** `CUDA out of memory` on RTX 3060 (6GB).
*   **Cause:** Default batch size was 128, which was too large for the graph tensors.
*   **Fix:** Hardcoded `BATCH_SIZE = 32` in argument parsing.

### Phase 2: The "Silent GCN" Bug (Critical Logic Error)
*   **Issue:** The model ran but performed poorly.
*   **Diagnosis:** The GCN `edge_index` was defined for a single graph (62 nodes). When passing a batch (e.g., 32 * 62 nodes), PyTorch Geometric treated nodes 62-1984 as "isolated" with no edges.
*   **Fix:** Implemented **Batch Expansion logic** in `train.py` and `analyze_error.py`.
    ```python
    offsets = (torch.arange(curr_batch_size) * 62).view(-1, 1, 1)
    edge_index = (base_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    ```
    This replicates the graph structure for every sample in the batch.

### Phase 3: Model Collapse ("Learning Nothing")
*   **Issue:** Test Loss stuck at ~1.09 (Random Guessing). Model predicted only Class 0 and 1, completely ignoring Class 2 (Positive).
*   **Diagnosis:** 
    *   **Over-Regularization:** Weight Decay (`1e-3`) was too high, forcing weights to zero.
    *   **Learning Rate:** `0.005` was too aggressive.
*   **Fix:** 
    *   Reduced Learning Rate to `0.0005`.
    *   Reduced Weight Decay to `1e-4`.
    *   Added **Class Weights** `[1.0, 1.0, 1.2]` to `CrossEntropyLoss` to force the model to pay attention to the "Positive" class.

### Phase 4: The "Dummy Check" Scare
*   **Issue:** Debug print showed model output shape `[1, 3]` instead of `[2, 3]`.
*   **Diagnosis:** The dummy input check created a `batch_idx` of all zeros, causing Global Pooling to merge the entire batch into a single vector.
*   **Fix:** Corrected the dummy variable generation to properly assign batch indices.

---

## 4. Current Results (Raw Data)

After applying all fixes, the model began to learn:

*   **Training:**
    *   **Accuracy:** Steadily increasing (~36% -> ~42%).
    *   **Loss:** Steadily decreasing.
    *   **Behavior:** The model is successfully learning to classify the training data (Sessions 1 & 2).
*   **Validation (Session 3):**
    *   **Accuracy:** Stalled at **~35-36%**.
    *   **Loss:** Increasing (Overfitting).
    *   **Behavior:** The model fails to generalize to the unseen session.

---

## 5. The Frequency Band Experiment (Standard vs. Gamma)

To definitively rule out noise as the cause of poor performance, we split the raw data into two distinct frequency bands.

*   **Hypothesis:** The "Standard" band (1-49 Hz) might be too noisy, or the "Gamma" band (50-75 Hz) might contain the critical emotion signals that were previously being filtered out or drowned out.

### Pipeline Updates
1.  **ICA & Filtering (`ICA_for_SEED.py`):** Updated to generate two separate datasets:
    *   `Cleaned_EEG_ICA_1_49`: Bandpass 1-49 Hz.
    *   `Cleaned_EEG_ICA_50_75`: Bandpass 50-75 Hz.
2.  **Segmentation (`segmentation.py`):** Refactored to use **Memory Mapping (`np.memmap`)** to handle the increased data volume without crashing RAM (OOM errors).
3.  **Training (`train.py`):** Added `--band` argument to switch between datasets dynamically.

### Results

*   **Standard Band (1-49 Hz):**
    *   **Result:** Identical to previous "Broadband" results.
    *   **Accuracy:** ~36% (Validation).
    *   **Observation:** Overfitting to training sessions; fails to generalize.

*   **Gamma Band (50-75 Hz):**
    *   **Result:** **Mode Collapse.**
    *   **Accuracy:** ~37% (Validation).
    *   **Observation:** The model learned to game the class weights by predicting "Positive" almost exclusively (Recall > 0.90 for Positive). It found no robust features, just a statistical shortcut.

### Final Verdict on Raw Data
Splitting by frequency band did **not** solve the domain shift problem. This confirms that **Raw Time-Series Data** is insufficient for this architecture on the SEED dataset.

---

## 6. Conclusion & Diagnosis

### The Diagnosis: Domain Shift & Input Representation
The pipeline is **code-correct** and functional. The poor validation performance is **not a bug**, but a fundamental limitation of the current approach:

1.  **Non-Stationarity:** EEG signals change significantly between sessions (days). A raw waveform pattern for "Happy" on Day 1 looks different on Day 3.
2.  **Raw Data Limitation:** Shallow CNNs are notoriously bad at learning robust, invariant features from raw EEG data across sessions. They tend to overfit to session-specific noise.
3.  **Subject Interference:** Training on all 15 subjects simultaneously (Subject Dependent) with raw data likely creates a "muddy" average, as different subjects have different neural fingerprints.

### The Verdict
**Raw Time-Series Data is the bottleneck.** To achieve higher accuracy (>70-80%) on the SEED dataset, we must move away from raw data.

---

## 7. Next Steps: Phase 2 (Differential Entropy Features)

To align with State-of-the-Art (SOTA) results on SEED, we will pivot to **Feature-Based Input**.

### Action Plan:
1.  **Feature Extraction:**
    *   Instead of raw waves, we will extract **Differential Entropy (DE)** or **Power Spectral Density (PSD)** features.
    *   **Bands:** Delta, Theta, Alpha, Beta, Gamma.
    *   **New Input Shape:** `(Batch, 62, 5)` (62 Channels, 5 Frequency Bands).
2.  **Model Adjustment:**
    *   Remove the 1D CNN feature extractor.
    *   Feed the 5-channel DE features directly into the GCN.
3.  **Expected Outcome:**
    *   DE features are much more stable across sessions.
    *   The GCN will learn relationships like "High Gamma in Frontal Lobe = Positive", which holds true across days and subjects.
    *   Accuracy should jump significantly (aiming for >70%).
