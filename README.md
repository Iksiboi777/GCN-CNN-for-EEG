# Graph Neural Networks for EEG Emotion Recognition

**A controlled comparison of three graph-neural-network paradigms for cross-subject affect decoding on the SEED dataset.**

> Master's thesis, University of Zagreb (FER) — *Klasifikacija emocija iz EEG signala korištenjem neuronskih mreža temeljenih na grafovima* — Ivan Kušeta, mentor doc. dr. sc. Nikolina Frid.

This repository treats the EEG electrode montage as a **graph** (electrodes = nodes, functional/spatial relationships = edges) and asks a focused scientific question: **how does the strength of a model's structural prior trade off against its ability to generalize across people?** It implements and rigorously compares three architectures along that spectrum, under two evaluation protocols, with a controlled ablation of the feature set and attention modules.

---

## 1. Overview

- **Task.** 3-class emotion classification (Negative / Neutral / Positive) from Differential-Entropy (DE) EEG features on **SEED** (62 channels, 15 subjects, 3 sessions).
- **Question.** Static physical prior vs. learned topology vs. local aggregation — which generalizes, and at what cost?
- **Contribution.** Not a single "best model," but a **comparison** that surfaces *non-uniform* interactions: the attention (SE) block helps one architecture and hurts another; the variance feature is decisive for two models and irrelevant to a third; shorter windows buy stability but not always accuracy.

The study's first phase (an end-to-end CNN-GCN on the *raw* signal) reached only **35–36%** (≈ chance), which motivated the pivot to spectral DE features — itself a reported result.

## 2. Headline results

Mean accuracy (%), 1-second windows, best configuration per model:

| Architecture | Structural prior | Session-Holdout | LOSO | LOSO σ | SH→LOSO drop |
|---|---|---:|---:|---:|---:|
| **GCN\_DE** | static physical k-NN graph | 84.0 | **82.3** | 5.43 | **1.69** |
| **Adaptive\_DGCNN** | learned + static (hybrid) | 81.0 | 77.5 | **4.63** | 3.51 |
| **GraphSAGE** | local inductive aggregation | 83.0 | 76.9 | 6.02 | 6.13 |

**LOSO** = Leave-One-Subject-Out (train on 14 subjects, test on the unseen 15th). The **SH→LOSO drop** is the cross-subject generalization gap — the quantity that matters most for transfer.

**What the comparison shows** (see [§6](#6-results--interpretation)): the *static physical prior* is hard to beat and transfers best; learning the graph (DGCNN) buys **stability**, not peak accuracy; purely local aggregation (SAGE) transfers worst. Positive-class F1 is ~0.92 across all models, while **neutral↔negative separation** (F1 as low as 0.62) is the universal failure mode.

## 3. Architecture

![Pipeline: raw EEG → DE features → AGLI → SE block → GNN → Subject Bias → prediction](images/The%20architecture.png)

Each model shares a calibration/attention front-end and differs in how it models inter-channel structure:

| Block | Role |
|---|---|
| **AGLI** (Adaptive Graph Input Layer) | Learnable per-(channel, band) affine `x·γ + β`; a strong L2 penalty on `γ` lets the model **silence unreliable sensors** (`γ → 0`). |
| **SE** (Squeeze-and-Excitation) | Attention-pooled recalibration of frequency bands — a *regulator* that helps when data/dimensionality is sufficient and can hurt when it is not. |
| **Subject Bias** | A per-subject embedding added to the logits, absorbing each recording's baseline offset to find a common decision boundary. |
| **Attentional aggregation** | Gated read-out (vs. mean pooling) that down-weights artifact electrodes. |

The three architectures:

- **`GCN_DE`** — spectral `GCNConv` over a *fixed* k-NN graph built from the 10–20 montage geometry. Strongest prior; least dependent on feature engineering.
- **`Adaptive_DGCNN`** — learns a *dynamic* adjacency via query/key attention and blends it with the static prior through a learnable `α`. Highest LOSO stability.
- **`GraphSAGE`** — inductive local neighbourhood aggregation (max aggregator); no spectral convolution.

![SE block](images/SEBlock.png) ![Adaptive subject bias](images/AdaptiveSubjectBias.png)

## 4. Repository structure

```
src/eeg_gnn/            Installable package
├── config.py           Hyperparameters (TrainConfig), paths, run bookkeeping
├── data/               SEED loader, DE features, group-wise z-score, feature engineering
├── graph/              k-NN channel-graph construction (+ bundled electrode positions)
├── models/             gcn_de · adaptive_dgcnn · graphsage · build_model() registry
├── training/           Training engine (LOSO/SH-safe, checkpointing) + FocalLoss
└── train.py            CLI orchestration (eeg-gnn-train)
scripts/
├── train.py            Thin launcher
├── cloud/              Modal cloud-execution scripts
├── analysis/           Result aggregation & visualization
└── prep/               Raw → DE feature-building pipeline (ICA, segmentation, …)
tests/                  Pytest suite (numpy unit tests + torch-gated model tests)
docs/                   Thesis PDF, analysis reports, architecture map, figures
```

## 5. Getting started

```bash
# 1. Install (editable) — pulls torch / torch-geometric / mne / scikit-learn …
pip install -e .

# 2. Place the SEED ExtractedFeatures under Data/ (not redistributed here):
#    Data/ExtractedFeatures_1s/{1_*.mat, …, label.mat}
#    Data/ExtractedFeatures_4s/...

# 3. Train. Leave-One-Subject-Out (cross-subject generalization):
eeg-gnn-train --model_type GCN --window_size 1s --mode sub_indep
#    or Session-Holdout (within-subject, across sessions):
eeg-gnn-train --model_type Adaptive_DGCNN --window_size 1s --mode sub_dep --no_se

# 4. Tests
pytest                      # numpy unit tests run anywhere; model tests need torch+PyG
```

Models: `GCN`, `ADAPTIVE_DGCNN`, `GraphSAGE`. Features: `--in_features {5,10}` (DE, or DE + rolling variance). The DE feature set is built from raw SEED via the scripts under [`scripts/prep/`](scripts/prep/).

## 6. Results & interpretation

The findings resolve onto **three orthogonal axes**, not a ranking:

1. **Structural prior vs. accuracy.** The static physical graph (GCN) is hard to beat. *Learning* the topology (DGCNN) did **not** raise mean accuracy — it lowered variance. The 10–20 montage encodes real functional organization that free-form attention struggles to recover from limited data.
2. **Feature dependence.** GCN is feature-independent (5 DE ≈ 10 DE); SAGE/DGCNN collapse without the variance channel, reading voltage fluctuation as affect. The stronger the prior, the less feature engineering it needs.
3. **The generalization gap.** SH→LOSO drop: GCN **1.7%** (excellent transfer), DGCNN 3.5%, SAGE 6.1% (local aggregation overfits training-subject topology). This gap — **inter-subject variability** — is the real frontier.

**Block ablation.** Subject Bias is the highest-leverage stabiliser for cross-session/cross-subject shift; SE is a double-edged regulator (helps DGCNN at 10 features, *hurts* GCN at 4 s / 5 features); AGLI provides per-sensor calibration; attentional read-out adds artifact robustness. Subject 12 is the hardest fold across every model (atypical neural patterns; see [`docs/reports/HARD_SUBJECTS_ANALYSIS.md`](docs/reports/HARD_SUBJECTS_ANALYSIS.md)).

Full per-configuration tables and learning-curve analyses are in [`docs/reports/`](docs/reports/) and the [thesis](docs/Master%20Thesis.pdf).

## 7. Future directions & research alignment

The two open problems above — **neutral↔negative separation** and the **cross-subject generalization gap** — point to concrete next steps, each grounded in current literature:

- **Functional-connectivity graphs.** Replace/augment the geometric k-NN graph with phase-locking-value or coherence graphs; regularized graph networks were built for exactly SEED cross-subject decoding ([Zhong et al., 2020](https://ieeexplore.ieee.org/document/9091308); [Progressive GCN, 2021](https://arxiv.org/abs/2112.09069)).
- **Domain adaptation.** The single biggest lever on the LOSO gap: domain-adversarial / multi-source methods reach **90–93%** cross-subject on SEED vs. 82% here ([multi-source adversarial DA](https://www.sciencedirect.com/science/article/abs/pii/S1746809425005270); [DANN + pseudo-labels](https://www.sciencedirect.com/science/article/pii/S0950705125004150)). *Subject Bias is a primitive version of this idea.*
- **Self-supervised / foundation pre-training.** This repo already contains **embryonic foundation-model machinery** — Subject Bias = subject/site conditioning, AGLI = channel harmonization. Scaling these to a graph-aware EEG foundation model pre-trained across datasets and montages ([LaBraM](https://arxiv.org/abs/2405.18765); [BIOT](https://arxiv.org/abs/2305.10351); [EEG-FM-Bench](https://arxiv.org/abs/2508.17742)) is the path to a **data-agnostic** pipeline that handles heterogeneous channel counts (the 14-ch DREAMER ↔ 62-ch SEED problem).

### From EEG subjects to ICU hospitals

This work's central lesson — **explicit per-source conditioning + a strong structural prior closes a distribution gap that free-form learning cannot** — transfers directly to **multicentre clinical foundation models**. The experimental philosophy is identical in shape:

| This thesis | Multicentre ICU foundation models |
|---|---|
| Leave-one-**subject**-out | Leave-one-**hospital**-out |
| Subject Bias embedding | Site / centre conditioning |
| "Multi-subject training helps" | "Multi-centre training mitigates the AUROC drop" ([Rockenschaub et al., *Crit Care Med* 2024](https://journals.lww.com/ccmjournal/fulltext/2024/11000/the_impact_of_multi_institution_datasets_on_the.5.aspx)) |
| AGLI channel silencing | Robustness to missing / unreliable measurements |
| "Variance mattered as much as architecture" | "Preprocessing choices matter more than model class" ([YAIB, *ICLR* 2024](https://arxiv.org/abs/2306.05109)) |

Graph machinery ports too: patient-similarity graphs use the same node-graph + attention design for ICU risk prediction ([dynamic graph-attention](https://pubmed.ncbi.nlm.nih.gov/37679182/); [EHR similarity GNNs](https://arxiv.org/abs/2101.06800)), and multicentre EHR foundation models need [&lt;1% local data after pre-training](https://www.nature.com/articles/s41746-024-01166-w) to match locally-trained models.

## 8. Citation

```bibtex
@thesis{kuseta2026eeggnn,
  author = {Ivan Kušeta},
  title  = {Klasifikacija emocija iz EEG signala korištenjem neuronskih mreža temeljenih na grafovima},
  school = {University of Zagreb, Faculty of Electrical Engineering and Computing},
  year   = {2026},
  note   = {Diploma thesis no. 1188}
}
```

Dataset: SEED — [Zheng & Lu, 2015](https://ieeexplore.ieee.org/document/7104132) (SJTU BCMI Lab).

## 9. License

Released under the MIT License — see [`LICENSE`](LICENSE).
