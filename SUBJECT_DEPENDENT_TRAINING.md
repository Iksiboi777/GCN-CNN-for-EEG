# Subject-Dependent Training: Comprehensive Summary

**Project Phase:** Subject-Dependent Analysis (Session Holdout)  
**Dataset:** SEED (EEG Emotion Recognition)  
**Models:** GCN, DGCNN  
**Date:** December 14, 2025  

---

## 1. Introduction
This document summarizes the entire "Subject-Dependent" phase of the project. In this phase, we trained models on data from the *same* set of subjects (Sessions 1 & 2) and tested on their future data (Session 3). 

**Objective:** To determine if a Graph Neural Network can learn a personalized emotion model that generalizes across time (different days).

---

## 2. Methodology Evolution

### 2.1 The Static Approach (GCN)
*   **Hypothesis:** EEG electrodes have a fixed physical relationship. A graph based on physical distance (KNN) should capture spatial dependencies.
*   **Outcome:** The model learned well for "Standard" subjects but failed for "Outliers".
*   **Key Finding:** Physical distance $\neq$ Functional connectivity. Just because two electrodes are close doesn't mean they are correlated during an emotion.

### 2.2 The Dynamic Approach (DGCNN)
*   **Hypothesis:** A model that *learns* the graph structure can adapt to individual differences and ignore artifacts (e.g., disconnecting noisy frontal channels).
*   **Implementation:** We implemented an Input-Dependent DGCNN that computes the Adjacency Matrix $A$ dynamically using Self-Attention on the input features.
*   **Outcome:** Performance remained capped at ~67%. The model failed to "unlock" the outlier subjects.

### 2.3 The Spectral Approach (Band Weighting)
*   **Hypothesis:** Different subjects express emotion in different frequency bands.
*   **Implementation:**
    1.  **Manual:** Fisher Scores (ANOVA) to calculate F-values per band.
    2.  **Automatic:** Learnable Band Attention (SE-Block).
*   **Outcome:** Confirmed that **Gamma (50-75Hz)** is the primary carrier of emotional information. However, weighting did not solve the Session Drift problem.

---

## 3. The "Negative Transfer" Phenomenon
The most critical finding of this phase is the existence of **Negative Transfer**.

*   **Definition:** When training on a source domain (Sessions 1+2) actually *hurts* performance on a target domain (Session 3) because the distributions are different.
*   **Evidence:**
    1.  **Early Peaking:** Validation accuracy peaks early (Epoch 5-10) and then degrades as the model overfits to the specific "style" of Sessions 1+2.
    2.  **Random Split Failure:** Mixing sessions resulted in chance-level accuracy (~53%), proving that the data distributions are fundamentally incompatible without alignment.
    3.  **Subject Lock:** The model consistently sacrificed Subjects 2, 4, and 12 to maximize performance on the easier subjects (14, 15).

---

## 4. Related Work & Context
Our findings align with recent literature on the SEED dataset, which emphasizes that **Cross-Session Generalization** is the hardest challenge.
*   **Standard GCNs:** Typically achieve ~80-90% only when using *Random Splits* (which we proved leak information).
*   **DGCNN Papers:** Often report high accuracy, but careful reading reveals they often use *Subject-Dependent Cross-Validation* (mixing sessions) or extensive *Domain Adaptation*.
*   **Our Contribution:** We rigorously demonstrated that without Domain Adaptation, even advanced architectures (DGCNN + Attention) cannot overcome the non-stationarity of EEG signals over time.

---

## 5. Conclusion & Next Steps
The "Subject-Dependent" phase is concluded. We have established a **True Baseline** of ~67% for cross-session generalization.

**Decision:** We will now move to **Subject-Independent (Leave-One-Subject-Out)** analysis.
*   **Why?** This is the standard benchmark for real-world applicability.
*   **Goal:** To explicitly measure the "Generalization Gap" between subjects and apply Domain Adaptation techniques (e.g., DAN, DANN) to bridge it.
