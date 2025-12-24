# Easy Subjects Analysis: The "Internalizers" & "Externalizers"

**Date:** December 23, 2025  
**Focus:** Subjects with High Accuracy (>80%)  
**Archetypes:** Type A (Pure Brain Signal) and Type B (Artifact-Assisted Signal)

---

## 1. Master Comparison Table: Easy Subjects

| Feature | **Subject 14 (The Internalizer)** | **Subject 15 (The Externalizer)** | **Subject 06 (The Hybrid)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **~82%** | **~94%** | **~83%** |
| **Archetype** | **Type A (Pure Signal)** | **Type B (Artifact-Assisted)** | **Type A/B Mix** |
| **Gamma Mean** | **"The Green Halo"**<br>Positive line floats significantly higher than others across the entire scalp. | **High Separation.**<br>Positive > others globally. | **High Separation.**<br>Positive > others globally. |
| **Variance Profile** | **Low / Random.**<br>Variance is background noise. Signal strength overpowers it. | **Discriminative (Frown).**<br>Massive spike at **FC5** specifically for Negative class. | **Discriminative (Wink).**<br>Massive spike at **AF4** specifically for Positive class. |
| **Physiology** | **Brain Only.**<br>Strong neural response to stimuli. | **Brain + Body.**<br>Neural response + Facial Muscle Tension (Frowning). | **Brain + Body.**<br>Neural response + Facial Muscle Tension (Winking/Squinting). |
| **Bad Channels** | None significant. | None significant. | **CF2, P1** (Loose/Noisy).<br>Ignored by model due to strong global signal. |
| **Model Strategy** | Relies on **Mean Amplitude** (Gamma). | Relies on **Variance** (FC5) + **Mean Amplitude**. | Relies on **Mean** (Global) + **Variance** (AF4). |

---

## 2. Key Findings

### The "Green Halo" (Gamma Separation)
For all "Easy" subjects, the **Gamma Band** is the primary driver of performance.
*   **Pattern:** The Mean Differential Entropy (DE) for the **Positive** class is consistently higher (1-2 units) than Negative or Neutral across almost all channels.
*   **Implication:** These subjects have a genuine, high-energy physiological reaction to positive stimuli.

### The "Artifact Advantage"
Subjects 15 and 06 perform exceptionally well because they provide **two** signals:
1.  **Neural:** The standard Gamma increase.
2.  **Muscular:** Specific facial expressions that correlate 100% with the label.
    *   **Subject 15:** Frowns during Negative clips (FC5 spike).
    *   **Subject 06:** Tenses right eye during Positive clips (AF4 spike).

### Conclusion
"Easy" subjects are easy because their signal-to-noise ratio is high. Even when they have noisy channels (like Subject 6's CF2), the global "Green Halo" pattern is so strong that the model can easily ignore the bad data.
