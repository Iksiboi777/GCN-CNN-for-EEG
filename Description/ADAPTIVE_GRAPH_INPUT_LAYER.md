# Evaluation of Adaptive Graph Learning Integration (AGLI) and Attention Mechanisms in EEG Classification

## 1. Theoretical Foundations: AGLI and ML Theory

### Defining AGLI (Adaptive Graph Learning Integration)
Adaptive Graph Learning Integration (AGLI) is a learnable input-normalization paradigm designed to address the inherent non-stationarity of EEG signals across different subjects. Unlike static graph approaches that assume a uniform signal-to-noise ratio across all sensors, AGLI introduces an affine transformation layer applied per electrode and per feature band:
$$y = x \cdot \gamma + \beta$$
By making $\gamma$ (gain) and $\beta$ (bias) learnable parameters, the network effectively functions as a "Pre-Amplifier." This allows the model to mathematically suppress noise-heavy sensors (where $\gamma \to 0$) or re-scale low-impedance channels before they reach the graph convolution stages.

### Dynamic Adjacency and Topological Learning
Beyond fixed physical graphs, this research explored **Dynamic Graph CNN (DGCNN)** principles. In these models, the graph is not predefined but learned on-the-fly via a self-attention mechanism:
$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)$$
This enables the discovery of functional connectivity—synchronization between distant brain regions that are emotionally correlated—providing a data-driven alternative to anatomical priors.



---

## 2. Motivation & Implementation

### Inter-subject Variability and Dead Channels
EEG data is plagued by inter-subject variability (differences in skull thickness/impedance) and systemic artifacts (dead or hyperactive channels). In the SEED dataset, **Subjects 2, 12, and 13** consistently exhibited such "sinkholes."
* **The Failure of Static Graphs:** Fixed graphs propagate noisy sensor data to neighbors, "poisoning" the local topology.
* **The AGLI Solution:** By learning to "mute" specific sensors, AGLI prevents artifact leakage, effectively "cooling" the subject-specific error rates (the "Wall of Fire") seen in earlier heatmaps.

### Testing Rolling Variance Computation
A pivotal part of the feature engineering phase involved determining the optimal window for computing **Rolling Variance (Var DE)**.
* **The Theory:** While Mean DE captures the average power, Variance captures the "texture" or volatility of the signal—a key differentiator between the erratic energy of Negative emotions and the flat signal of Neutral states.
* **The Test (3s vs. 9s):** We tested short-term (3s) vs. long-term (9s) variance. 
    * **9s Variance:** Proved to be a "hallucination" in training; it over-smoothed the data, diluting the emotional transitions.
    * **3s Variance:** Provided the "sweet spot" of temporal context, allowing the SE-Block to identify frequency-specific volatility without losing the signal's sharpness.

---

## 3. Master Table: 4s Window Model Evolution

The 4s era focused on establishing the baseline for GCN and the initial fragility of dynamic graph approaches.

| Attempt | Model | Features | Logic | Acc | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **28** | GCN | 5M + 5V | AGLI Only | 77% | Baseline; significantly limited by Subjects 2 & 12. |
| **29** | GCN | 5M + 5V | AGLI Only | 76% | Proved 9s rolling variance is too smoothed for the signal. |
| **30** | GCN | 5M + 5V | AGLI + SE | 81% | **Breakthrough:** SE-Block acts as an "Equalizer" for noise. |
| **31** | GCN | 5M + 5V | AGLI + SE | **82%** | **4s Champion:** Optimized balance of spatial and spectral weights. |
| **34** | GCN | 5 Mean | AGLI + SE | 80% | Proved Variance is needed to separate Neutral/Negative. |
| **36** | DGCNN| 5M + 5V | AGLI + SE | 74% | DGCNN struggles with low data density in 4s windows. |
| **37** | DGCNN| 5 Mean | AGLI + SE | 62% | Total topology collapse without Variance features. |
| **38** | DGCNN| 5 Mean | AGLI + SE (LR 0.05)| 62% | Confirmed Mean-only is the bottleneck, not the learning rate. |
| **39** | DGCNN| 5M + 5V | AGLI + SE (LR 0.05)| 73% | Variance helps DGCNN attention, but still inferior to GCN. |

---

## 4. Master Table: 1s Window Model Evolution

The transition to 1s windows provided the data density required to maximize both AGLI and SE-Block performance.

| Attempt | Model | Features | Logic | Acc | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **50** | DGCNN | 5M + 5V | AGLI Only | 79% | High density allows DGCNN to find patterns without SE. |
| **51** | DGCNN | 5 Mean | AGLI Only | 67% | Even with more data, Mean-only features fail the DGCNN. |
| **52** | DGCNN | 5 Mean | AGLI + SE | 69% | SE-Block acts as a "safety net" but can't replace Variance. |
| **53** | DGCNN | 5M + 5V | AGLI + SE | 79% | SE-Block is redundant for DGCNN at high data densities. |
| **54** | GCN | 5M + 5V | AGLI + SE | **84%** | **Overall Champion:** Perfectly balanced class recall. |
| **55** | GCN | 5M + 5V | AGLI Only | 82% | Proved SS & SS Norm provides a massive baseline boost. |
| **56** | GCN | 5 Mean | AGLI Only | 79% | Baseline GCN performance at 1s resolution. |
| **57** | GCN | 5 Mean | AGLI + SE | **84%** | **Efficiency King:** Hit 83.6% accuracy in just 2 epochs. |

---

## 5. Comparative Analysis: The "Window Size Paradox"

A critical finding was why **AdaptiveDGCNN** performs significantly worse on 4s windows compared to 1s windows, while **AdaptiveGCN** remains stable:
1.  **Dependency Noise:** In 4s windows, the "average" signal is too coarse for dynamic attention to learn specific topological connections.
2.  **Feature Density:** DGCNN is "data-hungry." The 50k+ samples in the 1s window provide the statistical mass needed for the $QK^T$ attention to stabilize, whereas the GCN relies on the physical prior, making it robust even with fewer samples.



---

## 6. Final Synthesis & Recommendations

### The Best Combination
The **Adaptive GCN + SE-Block + 1s Window** (Attempts 54/57) is the definitive architecture. It utilizes **SS & SS Normalization** to center the subject baseline, **AGLI** to fix spatial sensor noise, and **SE-Blocks** to filter spectral band energy.

### Future Path: Subject-Specific Early Stopping
Current logs show that while the global accuracy peaks late, "Hard Subjects" reach their generalization peak much earlier (e.g., Attempt 57 reaching 83.6% by Epoch 2). 
* **Recommendation:** Implement a training loop that tracks per-subject validation accuracy and saves the "Hard-Subject Optimal" model to prevent the network from overfitting to "Easy" subjects at the expense of the outliers.