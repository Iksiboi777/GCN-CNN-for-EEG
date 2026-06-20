# Dynamic Learning Analysis: Attempts 41-46

This document tracks the evolution of the model architecture from static GCNs to dynamic, attention-based systems. The primary goal during this phase was to solve inter-subject variability (specifically Subject 12's "black hole" and Subject 10's noise) without manual feature engineering.

---

## **Phase 1: Feature Expansion & Graph Basics**

### **Attempt 41: The Baseline (High Variance)**
*   **Architecture:** Standard GCN (3 Layers).
*   **Features:** Mean DE only (5 bands).
*   **Observation:** The model struggled to converge on emotional extremes. Neutral was over-predicted.
*   **Verdict:** Mean DE alone is insufficient for subjects where amplitude does not correlate with emotion (Subject 12).

### **Attempt 42 & 43: Incorporating Variance**
*   **Adjustment:** Added **Channel Variance** as 5 additional input features (Total Input: 10).
*   **Hypothesis:** If the mean amplitude is flat (Subject 12), the fluctuation (variance) might carry the signal.
*   **Outcome:** 
    *   Immediate improvement in distinguishing "active" emotional states from Neutral.
    *   **Problem:** Variance introduced massive instability. Artifacts (movements/blinks) create huge variance spikes that the GCN treats as "strong emotion."

---

## **Phase 2: The "CF2" Divergence**

### **Attempt 44: The Structural Fix (Targeting Subject 10)**
*   **Adjustment:** Manually set the adjacency weights for node `CF2` to 0.
*   **Why:** Subject 10 had a broken sensor at CF2 that was destroying neighbor updates.
*   **Outcome:** 
    *   Subject 10 improved massiveley.
    *   **Subject 12 deteriorated.** Hard-coding the graph structure prevented the model from using potentially valid signals in that region for other subjects.
    *   **Lesson:** "Hard" structural fixes hurt generalization. We need "Soft" learnable fixes.

---

## **Phase 3: Dynamic Architectures**

### **Attempt 45: Squeeze-and-Excitation (SE-Block)**
*   **Concept:** "Dynamic Feature Attention."
*   **Architecture:** Added an SE-Block before the GCN.
    *   *Squeeze:* Calculate Global Mean of each band.
    *   *Excitation:* Learn which bands (Gamma vs Delta) matter for *this specific 1-second clip*.
*   **Hypothesis:** The model should auto-detect that Subject 12 relies on Beta/Gamma variance and ignore the noisy Delta band.
*   **Result (The "Loud Noise" Failure):** 
    *   The classification bias remained.
    *   **Diagnosis:** The Global Mean Pooling in the SE-Block was corrupted by single-channel artifacts (Cz/CPz spikes). The "noise" was so loud that the SE-Block couldn't hear the "signal," leading it to boost the wrong features.

### **Attempt 46: Adaptive Inputs & Sparsity (The "Silencer")**
*   **Concept:** "Learnable Sensor Calibration."
*   **Architecture:** `AdaptiveGraphInputLayer` (Learnable $\gamma$ and $\beta$ per node).
*   **Optimization:** Added **Strong L2/L1 Regularization** specifically to the $\gamma$ parameters.
*   **Why:** To force the model to drive the weights of noisy sensors (Cz, CPz) to zero automatically.
*   **Result:** 
    *   **Pros:** Better separation of Negative class in later epochs.
    *   **Cons:** Confusion Matrix instability (drastic shifts at epoch 31).
    *   **Critical Flaw:** The presence of `LayerNorm` after the Adaptive Layer essentially "undid" the silencing. If $\gamma$ reduced a noise spike to 0.01, Normalization expanded it back to 1.0.

---

## **Summary of Trajectory & Results**

| Attempt | Architecture | Features | Avg Accuracy | Subj 10 (Noisy) | Subj 12 (Disengaged) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **41** | Baseline GCN | Mean DE (5) | ~76% | Poor (<65%) | Poor (<35%) | **Baseline.** Fails to capture signal quality or low-arousal states. |
| **43** | **Variance GCN** | Mean + Var (10) | **~81%** | **Degraded** | **Improved** | Variance distinguishes "Sad" from "Neutral" but amplifies artifacts. |
| **44** | Manual Graph | Mean + Var | ~80% | Improved | Degraded | Hard-coding filters helps one subject but hurts generalization. |
| **45** | SE-Block | Mean + Var | ~79% | Failed | Poor | Global Mean Pooling was corrupted by high-amplitude noise spikes. |
| **46** | Adaptive Inputs | Mean + Var | ~82% | Good | Unstable | LayerNorm negated the learned sparsity, preventing true noise suppression. |
| **47** | **Global Attn** | Mean + Var | **~83%** | **Fixed (>80%)** | **Capped (~36%)** | **Final Model.** Successfully filters noise (Subj 10) but proves Subj 12 has no valid signal. |

## **Next Proposed Step (Attempt 48)**
**"Forensic Validation"**
1.  **Correlation Check:** Verify if the "Dead" subjects are actually dead.
2.  **Finding:** Subject 12 is NOT dead. The sensors are active and correlated ($\rho \approx 0.61$), but the signal is **"Coherent Noise"** (disengaged alpha blocking).



# ...existing code...
### Attempt 47: Global Attention + Sparsity + Label Smoothing (Current)
*   **Concept:** "The Dictator Mechanism" (Attention) with Sparsity penalties.
*   **Status:** **Success (Functionally), Failure (Outcome).**
*   **Evidence from Deep Dive (Subj 12):**
    *   **Model's Focus:** `T8, C6, T7, C5, TP8` (Temporal/Lateral). These are valid emotion-processing regions.
    *   **Identified Noise:** `Cz, CPz, F7, C3` (Highest output variance).
    *   **The Mechanism:** The model's "Favorite" list does **not** contain the "Noisiest" list. This proves the Attention mechanism successfully learned to **decouple Variance from Importance**. It correctly ignores the massive noise at Cz/CPz.
*   **Why Subj 12 still fails:**
    *   The model is looking at the "Correct" channels (T7/T8).
    *   The accuracy is still 0% on specific trials.
    *   **Conclusion:** If the model looks at the right place and finds nothing, the data in the *signal* channels (T7/T8) is likely corrupted or missing for Subject 12 during those trials. Attention cannot fix a signal that doesn't exist.

## **Conclusion on Dynamic Architecture**
We have reached the limit of what architecture can fix. The model is effectively:
1.  Ignoring Noise (Cz).
2.  Focusing on Theory (Temporal Lobes).
3.  Smoothing Labels (to avoid infinite loss).

The remaining errors are likely **Irreducible Data Errors**.
# ...existing code...