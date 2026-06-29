# Architecture Map

Visual map of the `eeg_gnn` pipeline (post-refactor `src/` package). Renders natively
on GitHub. For *why* the architecture looks like this, see [`RESEARCH_HISTORY.md`](../RESEARCH_HISTORY.md);
for *how to run it*, see the [README](../README.md).

> The older `docs/architecture_map.txt` describes the **pre-refactor** layout
> (`Models/var_B.py`, `train_de.py`, "no package structure") and is now stale — this
> file supersedes it.

---

## 1 · Data & training pipeline

```mermaid
flowchart TD
    RAW["SEED .mat files<br/>Data/ExtractedFeatures_1s/<br/>de_LDS1..15 + label.mat"]

    subgraph DATA["Data layer · eeg_gnn/data"]
        LOAD["seed.py · load_seed_de()<br/>resolve axis order<br/>+ optional rolling variance (features.py)<br/>= 5 or 10 features/node"]
        NORM["normalization.py · groupwise_zscore()<br/>per-(subject, session) z-score"]
        BUNDLE["SeedBundle<br/>X (N,62,F) · y · sessions · subjects · trials"]
        LOAD --> NORM --> BUNDLE
    end

    subgraph GRAPH["Graph · eeg_gnn/graph"]
        KNN["construction.py · get_knn_adjacency_matrix(k=5)<br/>from channel_62_pos.locs (10-20 montage)"]
        EDGE["base_edge_index<br/>shared static graph"]
        KNN --> EDGE
    end

    CFG["config.py · TrainConfig<br/>model_type · mode · in_features=10 · use_se<br/>epochs=60 · batch=1024 · lr=5e-4 · knn_k=5"]

    subgraph MODELS["Model zoo · eeg_gnn/models/registry.build_model()"]
        M1["gcn_de.py<br/>GCN_DE — best accuracy"]
        M2["adaptive_dgcnn.py<br/>Adaptive_DGCNN — best stability"]
        M3["graphsage.py<br/>GraphSAGE — inductive baseline"]
    end

    subgraph TRAIN["Training · eeg_gnn/training"]
        ENGINE["engine.py · train_model_with_interrupt()<br/>Adam + OneCycleLR<br/>heavier L2 on AGLI gamma"]
        LOSS["losses.py · FocalLoss (LOSO)<br/>or CrossEntropy + label smoothing (SessHold)"]
    end

    MODE{"train.py · main()<br/>dispatch on --mode"}
    LOSO["LOSO · sub_indep<br/>15 leave-one-subject folds<br/>torch.multiprocessing · per-fold GPU<br/>aggregate from disk"]
    SH["Session-Holdout · sub_dep<br/>train sessions 1+2 → test session 3"]
    OUT["Outputs<br/>Results/ · Params/ · Errors/<br/>LOSO_Global_Summary.txt + plots"]

    RAW --> LOAD
    BUNDLE --> ENGINE
    EDGE --> MODELS
    CFG --> MODELS
    CFG --> MODE
    MODELS --> ENGINE
    LOSS --> ENGINE
    ENGINE --> MODE
    MODE -->|"sub_indep"| LOSO
    MODE -->|"sub_dep"| SH
    LOSO --> OUT
    SH --> OUT
```

---

## 2 · The three model forward passes

All three share the author's signature blocks — **AGLI** (learnable per-channel/per-band
input gate), an **SE block** (frequency-band recalibration), and a **Subject Bias**
embedding added to the logits — but differ in how they build and use the graph.

```mermaid
flowchart TB
    IN["Input per sample<br/>x: 62 channels x F features<br/>(F=10: DE mean + rolling variance)"]

    subgraph GCN["GCN_DE  (gcn_de.py) — static physical graph"]
        direction TB
        A1["AGLI — NO LayerNorm<br/>gamma to 0 fully silences noisy channels"]
        A2["SE block — attention-pooled squeeze"]
        A3["2x GCNConv + BatchNorm + ReLU"]
        A4["AttentionalAggregation pool"]
        A5["Linear + Subject Bias = 3 logits"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph DG["Adaptive_DGCNN  (adaptive_dgcnn.py) — learned + static hybrid"]
        direction TB
        B1["AGLI — WITH LayerNorm (gamma init 1.0)"]
        B2["SE block — mean-pooled squeeze"]
        B3["Dynamic adjacency A_dyn = ReLU(Q . K^T)<br/>A = (1-alpha)*static + alpha*A_dyn<br/>alpha = sigmoid(learnable)"]
        B4["2x DenseGCNConv + BatchNorm + ReLU"]
        B5["tanh-gated attention pool"]
        B6["Linear + Subject Bias = 3 logits"]
        B1 --> B2 --> B3 --> B4 --> B5 --> B6
    end

    subgraph SG["GraphSAGE  (graphsage.py) — inductive baseline"]
        direction TB
        C1["AGLI"]
        C2["SE (per-node) then LayerNorm"]
        C3["2x SAGEConv (aggr=max) + BatchNorm + ReLU"]
        C4["AttentionalAggregation pool"]
        C5["MLP classifier + Subject Bias = 3 logits"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    IN --> A1
    IN --> B1
    IN --> C1
```

---

## 3 · One-line component index

| Layer | File | Responsibility |
| :-- | :-- | :-- |
| Entry point | `train.py` | CLI → load data → build graph → dispatch on `--mode` |
| Config | `config.py` | `TrainConfig` dataclass: all hyperparameters + paths |
| Data | `data/seed.py` | Load SEED DE features → `SeedBundle` |
| Data | `data/features.py` | Rolling-variance channel (5 → 10 features) |
| Data | `data/normalization.py` | Per-(subject, session) z-score |
| Graph | `graph/construction.py` | k-NN electrode graph from the 10-20 montage |
| Models | `models/registry.py` | `build_model()` factory dispatched by `--model_type` |
| Models | `models/{gcn_de,adaptive_dgcnn,graphsage}.py` | The three GNN paradigms |
| Training | `training/engine.py` | Train/eval loop, checkpointing, interrupt-safe save |
| Training | `training/losses.py` | `FocalLoss` (LOSO) |
