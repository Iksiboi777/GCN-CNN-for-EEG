# Reproducibility, Trial Dynamics, and The "Subject Identity" Trap

**Date:** January 5, 2026
**Context:** Investigation into reproducing high-performance runs (Attempt 18) and analyzing model failure modes via Trial Heatmaps.

---

## 1. Understanding "Trials" in SEED-IV

In the context of this dataset and our analysis, a **"Trial"** refers to a single discrete event of emotion elicitation.

*   **Structure:** Each **Session** (1, 2, or 3) consists of **15 Trials**.
*   **Stimulus:** During a trial, the subject watches a specific movie clip designed to elicit one of three emotions: **Negative**, **Neutral**, or **Positive**.
*   **Duration:** Trials vary in length but typically last between 60 seconds to several minutes.
*   **Data Shape:** A single trial results in a tensor of shape `(62 Channels, Time_Samples, 5 Bands)`.
*   **The Goal:** The model must predict the emotion label (0, 1, 2) for the *entire trial* (or windows within it).

### The "Trial ID" Variable
In our heatmaps, "Trial ID" (1-15) represents the chronological order of the video clips shown to the subject.
*   **Trial 1:** The first video the subject saw that day.
*   **Trial 15:** The last video.
*   **Significance:** This chronological order is crucial because of **Subject Fatigue**. By Trial 15, the subject has been sitting in an EEG cap for ~45+ minutes. Their brain state is fundamentally different (tired, bored, drowsy) compared to Trial 1, regardless of the emotion being elicited.

---

## 2. The "Attempt 18" Investigation (Reproducibility Crisis)

We attempted to reproduce the configuration of **Attempt 18**, which achieved a remarkable **76% Accuracy** on the Session Holdout task.

### Configuration Reconstructed (Attempt 35)
Based on file history and forensic analysis, we believed Attempt 18 used a "Vanilla" configuration:
*   **Model:** Standard GCN (No Feature Attention, No Adaptive Layers).
*   **Features:** Raw Differential Entropy (5 Bands: Delta, Theta, Alpha, Beta, Gamma). **No Variance features.**
*   **Normalization:** Manual Z-Score Normalization (Subject-Specific + Session-Specific). **No RobustScaler.**
*   **Preprocessing:** Manual Fix for Channel `CF2` (Triangulation). **No automated bad channel detection.**
*   **Band Weights:** Disabled (All 1.0).

### The Result (Attempt 35)
*   **Best Validation Accuracy:** **67.26%** (Epoch 2).
*   **Behavior:** The model quickly overfitted. Training accuracy climbed to 83%, while Validation accuracy stagnated and Loss increased.
*   **Conclusion:** We failed to reproduce the 76% result. This implies **Attempt 18 had a "Secret Ingredient"** we missed.
    *   *Hypothesis A:* Attempt 18 might have actually used **Variance Features** (10 inputs), and our assumption that it didn't was wrong.
    *   *Hypothesis B:* The "Manual Normalization" in Attempt 18 might have been slightly different (e.g., global scaling vs. per-band).
    *   *Hypothesis C:* Random Seed luck (unlikely to account for 9% difference).

---

## 3. Deep Dive: The "Subject Identity" Trap

Our analysis of the **Trial Heatmaps** (Accuracy vs. Trial ID per Subject) revealed the fundamental pathology of our models.

### The Pathology
The models are often **classifying the Person, not the Movie.**

*   **Visual Evidence:** In the Bias Heatmaps, we see solid horizontal bars of color.
    *   **Subject 4:** Often a solid purple bar (Predicted "Negative" for *every* trial).
    *   **Subject 10 & 14:** Often solid yellow bars (Predicted "Positive" for *every* trial).
*   **The Mechanism:**
    *   Subject 4 likely has a "Low Energy" resting brain state (low Gamma/Beta). The model learns: "Low Energy = Negative." Since Subject 4 is *always* Low Energy (compared to others), the model labels them "Negative" permanently.
    *   Subject 10 has a "High Energy" resting state (high Gamma/EMG noise). The model learns: "High Energy = Positive." Subject 10 is permanently labeled "Positive."

### The "Rising Loss" Paradox
We observed that Validation Loss increases while Accuracy stays flat or improves slightly. The Heatmaps explain why:
1.  **Epoch 10:** Model is unsure. It guesses "Positive" for Subject 10 with 50% confidence.
2.  **Epoch 50:** Model is **arrogant**. It recognizes Subject 10's "fingerprint" and predicts "Positive" with **99.9% confidence**.
3.  **The Penalty:** When Subject 10 watches a *Negative* video, the model is now "Confidently Wrong." The CrossEntropyLoss for that sample explodes.

---

## 4. The "Fatigue Effect" (Trial Dynamics)

A critical finding from the Attempt 18 analysis was its performance drop-off in **Trials 12-15**.

*   **Observation:** Accuracy degrades significantly in the final 4 videos.
*   **Confusion:** The model confuses **Negative** videos with **Neutral** ones.
*   **Physiological Explanation:**
    *   **Tiredness** increases **Alpha waves** (8-12 Hz) and decreases Beta/Gamma.
    *   **"Neutral" emotion** is also characterized by dominant Alpha waves and low Beta/Gamma.
    *   **The Conflict:** A subject watching a Negative video (Trial 14) while tired looks like a subject watching a Neutral video while alert. The "Tiredness Signal" overpowers the "Emotion Signal."

---

## 5. Future Strategy: Breaking the Identity Trap

To fix the "Subject Identity Trap" and the "Fatigue Effect," we propose the following:

### A. Subject-Specific Centering (The "Relative" Approach)
Instead of feeding raw DE values, we should feed **Deviations from Baseline**.
*   *Formula:* $X_{input} = X_{raw} - \text{Median}(X_{subject})$
*   *Effect:* If Subject 10 has naturally high Gamma, subtracting their median brings them to 0. The model will only see when their Gamma goes *even higher* (Positive) or drops (Negative/Neutral).

### B. Trial-Aware Normalization
To combat fatigue, we could normalize **within the trial** or include `Trial_ID` as a feature.
*   *Idea:* If we normalize data *per trial*, the "Global Fatigue" (baseline shift over 45 mins) is removed, leaving only the relative dynamics of that specific video.

### C. The "Hard" Subject Protocol
Subjects 02, 12, 07, and 13 are fundamentally different (likely poor contact or thick skulls).
*   **Action:** They might need a separate model or a dedicated "Calibration" phase where we explicitly teach the model their specific baseline.
