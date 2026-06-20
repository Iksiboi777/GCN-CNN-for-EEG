"""Model registry — one factory for all three architectures.

The original ``train_de.py`` repeated the same ``if model_type == 'GCN': ...
elif ...`` construction block in two places (LOSO and Session-Holdout). This
module centralises that logic behind a single :func:`build_model` call so the
constructor differences (notably that ``Adaptive_DGCNN`` needs a dense static
adjacency derived from ``edge_index``) live in exactly one place.
"""
from __future__ import annotations

import torch
from torch_geometric.utils import to_dense_adj

from eeg_gnn.models.adaptive_dgcnn import Adaptive_DGCNN
from eeg_gnn.models.gcn_de import GCN_DE_Model
from eeg_gnn.models.graphsage import GraphSAGE_EEG_Model

#: Canonical model-type identifiers accepted by :func:`build_model`.
MODEL_TYPES = ("GCN", "ADAPTIVE_DGCNN", "GraphSAGE")


def build_model(
    model_type: str,
    in_features: int,
    base_edge_index: torch.Tensor,
    device: torch.device,
    *,
    num_nodes: int = 62,
    num_classes: int = 3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    use_se: bool = True,
    use_doubling: bool = False,
    num_subjects: int = 15,
) -> torch.nn.Module:
    """Construct a GNN by name and move it onto ``device``.

    Parameters
    ----------
    model_type:
        One of ``MODEL_TYPES`` (case-insensitive). ``"DGCNN"`` is accepted as an
        alias for ``"ADAPTIVE_DGCNN"``.
    in_features:
        Per-node input dimension (5 for DE bands, 10 with rolling variance).
    base_edge_index:
        The shared channel-graph connectivity in PyG ``edge_index`` form. For
        ``Adaptive_DGCNN`` it is densified into the static-prior adjacency.
    device:
        Target torch device.

    Returns
    -------
    torch.nn.Module
        The instantiated model on ``device``.

    Raises
    ------
    ValueError
        If ``model_type`` is not recognised.
    """
    key = model_type.strip().lower()

    if key == "gcn":
        model: torch.nn.Module = GCN_DE_Model(
            num_nodes=num_nodes,
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=0.5,
            num_layers=num_layers,
            num_subjects=num_subjects,
            use_doubling=use_doubling,
            use_se=use_se,
        )
    elif key in ("adaptive_dgcnn", "dgcnn"):
        static_adj = to_dense_adj(base_edge_index, max_num_nodes=num_nodes)[0].to(device)
        model = Adaptive_DGCNN(
            static_adj=static_adj,
            num_nodes=num_nodes,
            in_features=in_features,
            num_classes=num_classes,
            num_subjects=num_subjects,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_se=use_se,
            use_doubling=use_doubling,
        )
    elif key == "graphsage":
        model = GraphSAGE_EEG_Model(
            num_nodes=num_nodes,
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            aggregator="max",
            use_se=use_se,
            use_doubling=use_doubling,
            dropout_rate=0.5,
            num_subjects=num_subjects,
        )
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of {MODEL_TYPES}."
        )

    return model.to(device)
