"""Channel-graph construction utilities."""
from __future__ import annotations

from eeg_gnn.graph.construction import STANDARD_CH_NAMES, get_knn_adjacency_matrix

__all__ = ["get_knn_adjacency_matrix", "STANDARD_CH_NAMES"]
