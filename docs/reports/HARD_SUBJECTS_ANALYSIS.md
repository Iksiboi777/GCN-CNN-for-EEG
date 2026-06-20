# Hard Subjects Analysis: The "Stones" & "Broken"

**Date:** December 23, 2025  
**Focus:** Subjects with Low Accuracy (<60%)  
**Archetypes:** Type C (No Signal / Mechanical Failure)

---

## 1. Master Comparison Table: General Diagnostics

| Feature | **Subject 02 (The Distracted)** | **Subject 12 (The Stone)** | **Subject 13 (The Noisy)** | **Subject 07 (The Twin Sinkholes)** |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **~47%** | **~36%** (Random) | **~54%** | **~40-45%** (Est.) |
| **Primary Failure** | **F7 Noise Explosion.**<br>Frontal-Left muscle noise destroys scaling. | **Cz/CPz Sinkholes.**<br>Midline channels drop to zero power. | **POz Spike + CPz Sinkhole.**<br>Loose back sensor + Dead midline. | **Cz + CF2 Sinkholes.**<br>Two distinct dead channels with high variance. |
| **Signal Separation** | **Weak.**<br>F7 noise squashes other channels. | **None.**<br>Lines are glued together. | **Good (Hidden).**<br>Gamma separates well if noise is removed. | **Poor.**<br>Lines are glued; noise dominates. |
| **Variance Profile** | **Global Noise.**<br>High variance everywhere. | **Flatline.**<br>Low variance (dead signal). | **Correlated Noise.**<br>POz variance is high for Positive. | **Localized Chaos.**<br>Massive spikes at Cz and CF2. |
| **Mechanical Diagnosis** | **Muscle Tension.**<br>Subject is moving/clenching jaw (F7). | **Poor Fit (Midline).**<br>Top strip of headset is loose. | **Poor Fit (Back/Mid).**<br>Back sensor loose, Top sensor dead. | **Poor Fit (Central).**<br>Central sensors are not touching scalp. |
| **The Fix** | **RobustScaler** | **Lateral Interpolation** | **Lateral Interp + Drop POz** | **Lateral Interp (Cz, CF2)** |

---

## 2. Master Table: Band-Specific Problems

| Subject | **Gamma Band** (Excitement) | **Beta Band** (Cognitive) | **Alpha Band** (Relaxation) | **Theta/Delta Bands** (Rest) | **The Fix Strategy** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subj 02** | **Masked.**<br>F7 noise spike hides the signal. | **Noisy.**<br>Global variance is too high. | **Weak.**<br>No clear alpha peak. | **Chaotic.**<br>Drifting baseline artifacts. | **RobustScaler** (Squash F7). |
| **Subj 07** | **Glued.**<br>Lines overlap; AF4 muscle noise. | **Broken.**<br>Cz & CF2 sinkholes appear. | **Broken.**<br>Cz & CF2 variance explodes. | **Unstable.**<br>Massive variance at Cz/CF2. | **Double Interpolation** (Cz & CF2). |
| **Subj 12** | **Dead.**<br>Lines are glued tight. | **Depressed.**<br>Midline (Cz/CPz) power is low. | **Flat.**<br>No alpha blocking visible. | **Flat.**<br>Low energy everywhere. | **Lateral Interpolation** (Cz/CPz). |
| **Subj 13** | **Contaminated.**<br>High separation on T7/T8 due to **Jaw EMG** (muscle tension). | **Noisy.**<br>T7/T8 EMG noise + POz spike. | **Distorted.**<br>CPz sinkhole + POz spike. | **Noisy.**<br>POz variance dominates. | **Interp CPz** + **Drop POz** + **RobustScaler**. |

---

## 3. Key Findings

### The "Midline Failure" (Cz/CPz)
Subjects 12, 13, and 07 all suffer from **Dead Channels** along the center line of the scalp.
*   **Cause:** The headset strip along the top of the head was likely loose or not making contact.
*   **Impact:** These channels drop to near-zero amplitude ("Sinkholes"). This breaks the GCN's spatial convolution, as a zero-value node pulls down the values of all its neighbors.
*   **Solution:** **Lateral Interpolation.** We must rebuild Cz/CPz using their Left/Right neighbors (C1/C2, CP1/CP2) rather than Front/Back neighbors (which are also often broken).

### The "Muscle Masking" (F7/T7/T8)
Subjects 02 and 13 suffer from massive **EMG Artifacts**.
*   **Subject 02:** F7 (Frontal Left) has random, massive spikes that destroy the data scale.
*   **Subject 13:** T7/T8 (Temporal) have spikes correlated with Positive emotion (Smiling/Jaw Tension).
*   **Solution:** **RobustScaler.** Standard normalization (Mean/Std) fails here because the spikes are so large they squash the brain signal to zero. RobustScaler (Median/IQR) will preserve the brain signal while keeping the spikes as outliers.

### The "Delta Savior"
For subjects with "Glued Lines" in Gamma (Subject 02, 12), the **Delta Band** is the only reliable feature.
*   **Pattern:** The **Neutral** class often shows lower Delta power (Physical Stillness) compared to Emotional classes (Fidgeting).
*   **Strategy:** We must include Delta as a distinct input feature (not averaged) to allow the model to learn this "Stillness vs. Movement" distinction.
