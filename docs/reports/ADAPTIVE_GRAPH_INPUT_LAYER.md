# Evaluation of Adaptive Graph Learning Integration (AGLI) and Attention Mechanisms in EEG Classification

## 1. Theoretical Foundations: AGLI and ML Theory

### Defining AGLI (Adaptive Graph Learning Integration)
Adaptive Graph Learning Integration (AGLI) is a learnable input-normalization paradigm designed to address the inherent non-stationarity of EEG signals across different subjects. Unlike static graph approaches that assume a uniform signal-to-noise ratio across all sensors, AGLI introduces an affine transformation layer applied per electrode and per feature band:
$$y = x \cdot \gamma + \beta$$
By making $\gamma$ (gain) and $\beta$ (bias) learnable parameters, the network effectively functions as a "Pre-Amplifier." This allows the model to mathematically suppress noise-heavy sensors (where $\gamma \to 0$) or re-scale low-impedance channels before they reach the graph convolution stages.

### Dynamic Adjacency and Topological Learning
Beyond fixed physical graphs, this research explored **Adaptive DGCNN** principles. In these models, the graph is not predefined but learned on-the-fly via a self-attention mechanism:
$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)$$
This enables the discovery of functional connectivity—synchronization between distant brain regions that are emotionally correlated—providing a data-driven alternative to anatomical priors.


---

## 2. Motivation & Implementation

### Inter-subject Variability and Dead Channels
EEG data is plagued by inter-subject variability (differences in skull thickness/impedance) and systemic artifacts (dead or hyperactive channels). In the SEED dataset, **Subjects 2, 12, and 13** consistently exhibited such "sinkholes."
* **The Failure of Static Graphs:** Fixed graphs propagate noisy sensor data to neighbors, "poisoning" the local topology.
* **The AGLI Solution:** By learning to "mute" specific sensors via learnable weights, AGLI prevents artifact leakage, effectively "cooling" the subject-specific error rates (mitigating the "Wall of Fire" effect).

### Feature Engineering: The Rolling Variance Test
A pivotal discovery was the impact of temporal context on feature extraction. We tested different windows for computing **Rolling Variance (Var DE)**:
* **9s Variance (Attempt 29):** Proved to be a "hallucination"; the window was too long, over-smoothing the data and diluting emotional transitions.
* **3s Variance (Attempt 31/54):** Provided the "Sweet Spot." It captured the signal's "texture"—the micro-fluctuations in power that distinguish the erratic energy of Negative emotions from the flat signal of Neutral states.

---

## 3. Master Table: 4s Window Model Evolution
The 4s window established the baseline for GCN stability and demonstrated the initial fragility of dynamic graph approaches.

| Attempt | Model | Features | Logic | Accuracy | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **28** | GCN | 5M + 5V (3s) | AGLI Only | 77% | Baseline; limited by "Hard Subject" artifacts. |
| **29** | GCN | 5M + 5V (9s) | AGLI Only | 76% | Long-term variance dilutes the emotional signal. |
| **30** | GCN | 5M + 5V (3s) | AGLI + SE | 81% | **Breakthrough:** SE-Block acts as an spectral equalizer. |
| **31** | GCN | 5M + 5V (3s) | AGLI + SE | **82%** | **4s Champion:** Optimized spatial/spectral weights. |
| **34** | GCN | 5 Mean | AGLI + SE | 80% | Proved Variance is required to separate Neutral/Negative. |
| **36** | DGCNN| 5M + 5V | AGLI + SE | 74% | DGCNN struggles with low data density in 4s windows. |
| **37** | DGCNN| 5 Mean | AGLI + SE | 62% | Total topology collapse without Variance features. |
| **38** | DGCNN| 5 Mean | AGLI+SE (LR .0005) | 62% | Confirmed feature starvation is the bottleneck. |
| **39** | DGCNN| 5M + 5V | AGLI+SE (LR .0005) | 73% | Variance anchors the DGCNN attention mechanism. |

---

## 4. Master Table: 1s Window Model Evolution
The 1s window provided the data density required to maximize both AGLI and SE-Block performance.

| Attempt | Model | Features | Logic | Accuracy | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **50** | DGCNN | 5M + 5V | AGLI Only | 79% | High density allows DGCNN to find patterns without SE. |
| **51** | DGCNN | 5 Mean | AGLI Only | 67% | Mean-only features fail even with 1s density. |
| **52** | DGCNN | 5 Mean | AGLI + SE | 69% | SE-Block acts as a "safety net" but can't replace Variance. |
| **53** | DGCNN | 5M + 5V | AGLI + SE | 79% | SE-Block redundant for DGCNN at high density. |
| **54** | GCN | 5M + 5V | AGLI + SE | **84%** | **Overall Champion:** Perfectly balanced class recall. |
| **55** | GCN | 5M + 5V | AGLI Only | 82% | Proved SS & SS Norm provides massive baseline boost. |
| **56** | GCN | 5 Mean | AGLI Only | 79% | Baseline GCN performance at 1s resolution. |
| **57** | GCN | 5 Mean | AGLI + SE | **84%** | **Efficiency King:** Peaks at Epoch 2; fastest learning curve. |

---

## 5. Comparative Analysis

### Architecture: Adaptive GCN vs. Adaptive DGCNN
* **Sensitivity to Variance:** **Adaptive DGCNN** is critically dependent on Variance features. Because it learns its graph from scratch, it requires "texture" to define topology. Without Variance, it "hallucinates" connections, leading to a performance drop of ~15%.
* **Sensitivity to SE-Block:** **Adaptive GCN** is more impacted by the **SE-Block**. Since GCN has a fixed spatial prior, its main weakness is spectral noise. The SE-Block acts as the "Spectral Gatekeeper," ensuring the physical graph only propagates clean frequency data.

### Temporal: 1s vs. 4s Windows
* **The DGCNN Paradox:** DGCNN performs poorly on 4s windows (~74%) but succeeds on 1s windows (~79%). This is because the dynamic attention mechanism ($QK^T$) needs high data density (50k+ samples) to stabilize its learned topology.
* **The GCN Stability:** GCN remains the superior choice for SEED. Its reliance on physical distance provides a "natural regularizer" that prevents it from over-tuning to artifacts, allowing it to reach 84% accuracy.

---

## 6. Final Synthesis & Conclusions

### What Mattered Most?
1. **AGLI:** Changed everything by allowing the model to learn subject-specific sensor reliability on-the-fly, essentially "erasing" the dead channels of hard subjects.
2. **SE-Block:** Critical for GCN models to balance the recall between Negative and Neutral classes.
3. **Variance:** The "texture" anchor that prevents model hallucination and enables class separation.

### The Winning Combination
The **Adaptive GCN + SE-Block + 1s Window + 10 Features (Mean/Var)** is the definitive architecture for this project.

### Recommendations for Future Review
* **Subject-Specific Early Stopping:** Models reach generalization peaks for "Hard Subjects" much earlier than the global average. 
* **Hybrid Adjacency:** Initializing DGCNN with a GCN physical prior to combine structural stability with dynamic flexibility.
