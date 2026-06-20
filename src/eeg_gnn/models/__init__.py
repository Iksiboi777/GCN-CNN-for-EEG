"""GNN architectures and the model factory.

Each architecture keeps its *own* ``AdaptiveGraphInputLayer`` (AGLI) and
``SEBlock``. This duplication is intentional and scientifically meaningful: the
thesis compares subtly different block variants (e.g. ``GCN_DE`` drops LayerNorm
and initialises the AGLI gain at 0.5 to enable channel silencing, whereas
``Adaptive_DGCNN`` keeps LayerNorm and a unit-gain init). They are therefore
*not* merged into a shared module.
"""
from __future__ import annotations

from eeg_gnn.models.adaptive_dgcnn import Adaptive_DGCNN
from eeg_gnn.models.gcn_de import GCN_DE_Model
from eeg_gnn.models.graphsage import GraphSAGE_EEG_Model
from eeg_gnn.models.registry import MODEL_TYPES, build_model

__all__ = [
    "GCN_DE_Model",
    "Adaptive_DGCNN",
    "GraphSAGE_EEG_Model",
    "build_model",
    "MODEL_TYPES",
]
