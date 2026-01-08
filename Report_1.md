# Final Research Report: Adaptive Graph Convolutional Networks for EEG Emotion Recognition (SEED Dataset)

**Date:** January 9, 2026  
**Subject:** Cross-Session Generalization, Forensic Diagnostics, and Adaptive Neural Modeling  
**Key Result:** 83% Average Accuracy via Adaptive GCN & Feature Integration  

---

## Abstract
This report documents the evolution of a Graph Neural Network (GNN) framework for emotion recognition. We address the primary bottleneck of EEG research: non-stationarity across sessions. Through a "Forensic" approach, we identify hardware pathologies and subject archetypes, ultimately developing an Adaptive DGCNN that achieves 83% accuracy by learning to ignore noise and amplify subtle neural markers.

---

## Chapter 1: Introduction (The Feature Pivot)

The goal of this project was to classify emotions (Negative, Neutral, Positive) using the SEED dataset across three separate recording sessions. The primary challenge identified was **Session Drift**: a "Sad" brain signal from Session 1 is statistically distinct from a "Sad" signal in Session 3.

Initially, we utilized raw time-series EEG data processed through a hybrid GCN-CNN. This approach reached a hard performance ceiling below 40%. The "Grand Realization" of this phase was that raw EEG waveforms are too non-stationary; models tend to overfit to session-specific noise (like the physical pulse of the cap) rather than emotional content. We pivoted to **Differential Entropy (DE)** features across five frequency bands (Delta, Theta, Alpha, Beta, Gamma), which provided a stable, logarithmic representation of neural energy.

### SPECULATIONS & DISCUSSION
* **The Gamma Signature:** We speculate the model may be acting as a high-frequency Gamma energy detector. Since Positive clips elicit higher engagement, the model might be learning "Arousal" rather than "Emotion."
* **Feature Information Loss:** By discarding raw waves for DE, we speculate that we lose "Timing" (phase-locking) information, which may prevent reaching 90%+ accuracy.
* **The Stimulus Mirage:** At early stages, the model may overfit to the visual properties of specific videos rather than the underlying emotional state.

---

## Chapter 2: Part I - The Diagnostic Phase (The 67% Ceiling)

Transitioning to DE features in 1-second windows provided ~50,000 samples, stabilizing training and raising accuracy to **67%**. 

### 2.1 The Energy Hypothesis
Visual analysis of feature distributions revealed the "Energy Hypothesis." The **Positive** class is easily separable because it manifests as high-intensity Gamma energy. However, **Negative vs. Neutral** classes showed significant overlap. For many subjects, the energy levels for "Boredom" and "Sadness" were mathematically identical, making the boundary invisible to a standard GCN.



### 2.2 The Static Topology Constraint
Early GCNs used a fixed adjacency matrix based on physical electrode distance ($k=5$). This assumed brain regions only communicate with immediate neighbors. We found this invalid for "Hard" subjects, as a dead sensor (e.g., Cz) would act as a spatial barrier, corrupting its neighbors during graph convolution.

### SPECULATIONS & DISCUSSION
* **The Split-Brain Problem:** We speculate that dead midline sensors create a functional disconnect, preventing the model from integrating global emotional states between the left and right hemispheres.
* **Class-Specific Trust:** We speculate that the model should dynamically trust Gamma for Positive detection but perhaps rely on Delta for low-arousal states.

---

## Chapter 3: Part II - The Forensic Phase (Archetypes & Pathologies)

We performed "Forensic Biopsies" to categorize subjects into distinct archetypes based on signal-to-noise ratios.

### 3.1 Subject Archetypes
* **The Internalizers (e.g., Subj 14):** Accuracy ~82%. Driven by the **"Green Halo"**—a stable Gamma band separation with minimal noise.
* **The Externalizers (e.g., Subj 15):** Accuracy ~94%. Driven by **The Artifact Advantage**. The model's "Attention" locked onto channel **FC5** (muscle tension/frowning), using facial expressions as a shortcut.
* **The Stones (e.g., Subj 12):** Accuracy ~36%. Exhibits "Neural Silence" due to systemic hardware failure (midline sinkholes).

### 3.2 Hardware Pathologies
* **Midline Sinkholes (Cz, CPz, Fz):** Systemic failure where sensors dropped to near-zero amplitude, breaking spatial convolutions.
* **Screaming Channels (F7, T7, T8):** High-variance noise that "squashed" valid brain signals during standard normalization.

### SPECULATIONS & DISCUSSION
* **The Validity of Artifacts:** If a subject always frowns when sad, is it "cheating" to use that? We speculate that while useful for real-world wearables, it complicates the search for a purely neural emotion model.
* **Neural Silence:** For "The Stones," we speculate the signal is **physically missing**, not just noisy—likely due to high scalp impedance or poor contact.

---

## Chapter 4: Part III - The Architectural Evolution (The 83% Breakthrough)

Attempts 41–47 introduced the **Adaptive Graph Input Layer** and **Dynamic Learning**, breaking the 70% barrier.

### 4.1 The Adaptive Layer & Variance Injection
We implemented a learnable "Pre-Amp": $y = x \cdot \gamma + \beta$. This allowed the model to scale quiet signals and dampen loud noise. By injecting **Rolling Variance** as a secondary feature, we successfully resolved the confusion between "Neutral" and "Sad" states.

### 4.2 Master Results Table
| Model / Attempt | Accuracy | Primary Driver |
| :--- | :--- | :--- |
| GCN-CNN (Raw) | < 40% | Overfitting |
| GCN v2 (1s) | 67% | DE Baseline |
| Attempt 42/43 | 71% | Variance Features |
| Attempt 47 | 78% | Adaptive Layer |
| **Final DGCNN** | **83%** | **SE-Blocks (Attention)** |

### SPECULATIONS & DISCUSSION
* **The Dictator Mechanism:** The Squeeze-and-Excitation (SE) Blocks allow the model to selectively shut down irrelevant nodes. We speculate this may make the model less generalizable to new subjects with different "bad" sensors.
* **The Variance Trap:** Variance is sensitive to movement. We speculate that sudden head movement might trigger a false positive.

---

## Chapter 5: Part IV - The Grand Realization (Identity & Fatigue)

### 5.1 The Subject Identity Trap
The model found it easier to learn "Who the person is" rather than "How they feel." This led to **Confidence Bloat**, where the model was 99% confident in a wrong guess because it recognized the subject's unique noise signature.

### 5.2 The Fatigue Effect (Trial 15 Phenomenon)
Accuracy drops toward the end of sessions. We speculate that **Subject Fatigue** mutes neural responses to movie clips by the 15th trial.

### SPECULATIONS & DISCUSSION
* **Personalized Artifacts:** We speculate that a "Subject ID" signature might actually be a feature, not a bug, for personalized healthcare models.
* **Multi-Modal Future:** Reaching 90%+ likely requires secondary inputs like Heart Rate or Eye Tracking to confirm the brain's "silent" signals.

---