# Adaptive Graph Input Layer (Learnable Input Normalization)

**Context:** EEG Emotion Recognition using GCNs.
**Component:** `AdaptiveGraphInputLayer` (often referred to as Learnable Attention or Adaptive Pooling in our discussions).

---

## 1. The Problem: Inter-Subject & Inter-Session Variability

EEG data is notoriously non-stationary and subject-dependent.
1.  **Impedance Mismatch:** In Session 1, the electrode contact might be perfect. In Session 2, the subject might have drier hair, leading to higher impedance and lower signal amplitude.
2.  **Individual Differences:** Subject A might have a naturally "loud" Gamma band (high amplitude), while Subject B has a "quiet" one.
3.  **Noise/Artifacts:** Frontal channels (Fp1, Fp2) are often corrupted by eye blinks. A standard GCN treats these noisy values as valid features, propagating errors to neighbors.

**Standard Normalization (Z-Score/RobustScaler)** helps, but it is **static**. It forces everything to a mean of 0 and std of 1, but it doesn't tell the model *which* channels are reliable or *which* bands are important.

---

## 2. The Solution: Adaptive Graph Input Layer

This layer acts as a **Learnable "Pre-Amp" or Equalizer** for the brain signals. It sits *before* the GCN layers.

### The Formula
$$ y_{i,f} = x_{i,f} \cdot \gamma_{i,f} + \beta_{i,f} $$

Where:
*   $x_{i,f}$: Input feature for Node $i$ (Channel) and Feature $f$ (Frequency Band).
*   $\gamma_{i,f}$ (**Gamma**): A learnable **Scaling Parameter**.
*   $\beta_{i,f}$ (**Beta**): A learnable **Shift Parameter**.

### How it Works (The "Magic")

1.  **Noise Suppression (The Mute Button):**
    *   If Channel `Fp1` is consistently noisy across the training set, the model learns to set $\gamma_{Fp1} \approx 0$.
    *   This effectively "mutes" the channel before it enters the graph convolution, preventing noise propagation.

2.  **Band Importance (The Bass Boost):**
    *   If the **Gamma Band** is crucial for detecting happiness, the model learns $\gamma_{Gamma} > 1.0$ to amplify it.
    *   If the **Theta Band** is irrelevant, it learns $\gamma_{Theta} < 0.5$ to suppress it.

3.  **Domain Adaptation:**
    *   By learning these parameters, the model finds a "Common Feature Space" where Subject A's loud Gamma and Subject B's quiet Gamma are scaled to be comparable.

---

## 3. "Aggressive" vs. "Passive" Adaptation

### Passive (Current Implementation)
*   **Global Parameters:** The $\gamma$ and $\beta$ are fixed parameters (shape `1x62x5`) learned over the entire training set.
*   **Behavior:** It learns the "Average Best Settings" for the dataset. It doesn't change per sample.

### Aggressive (Proposed: SE-Block / Attention)
*   **Dynamic Reweighting:** Instead of fixed parameters, we use a small neural network to predict $\gamma$ *from the input itself*.
*   **Mechanism:** `Input -> GlobalPool -> MLP -> Weights`.
*   **Behavior:**
    *   If a specific trial has a huge artifact in `Fp1`, the model detects it *in real-time* and sets the weight for `Fp1` to 0 for *that specific trial*.
    *   This is much more powerful for handling transient artifacts (like a sudden cough or blink).

---

## 4. Expected Results

*   **Improved Convergence:** The model doesn't have to fight against scaling differences.
*   **Robustness:** Better performance on "Hard" subjects (who usually have outlier signal properties).
*   **Interpretability:** We can inspect the learned $\gamma$ values to see which brain regions and bands the model actually used.
