# `eeg_gnn.models` — model zoo

**Flow.** `train.py` calls `registry.build_model()`, the single factory that dispatches
`--model_type` to one of three architectures. Each model carries its **own** copy of the
author's signature blocks — an AGLI input gate, an SE frequency-band recalibrator, and a
per-subject Subject-Bias head — added to the logits. The duplication is intentional: the
thesis compares subtly different block variants, so they are deliberately *not* merged.

| Script | What it does |
|---|---|
| `registry.py` | `build_model()` factory; centralises the per-architecture constructor differences (notably that `Adaptive_DGCNN` needs a dense static adjacency derived from `edge_index`) in one place. |
| `gcn_de.py` | `GCN_DE_Model`: spectral `GCNConv` over a fixed k-NN graph; its AGLI **drops LayerNorm** so the gain can silence noisy channels. Best accuracy / best transfer. |
| `adaptive_dgcnn.py` | `Adaptive_DGCNN`: learns a dynamic adjacency (`Q·Kᵀ`) and blends it with the static graph via a learnable `α`. Best LOSO stability. |
| `graphsage.py` | `GraphSAGE_EEG_Model`: inductive local neighbourhood aggregation (`max`); no spectral convolution. The inductive baseline. |
| `__init__.py` | Exports the three models + factory, and documents why each keeps its own AGLI/SEBlock variant. |
