"""Tests for channel-graph construction (requires torch + torch_geometric + mne)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("mne")

from eeg_gnn.config import LOCS_FILE
from eeg_gnn.graph import STANDARD_CH_NAMES, get_knn_adjacency_matrix


def test_edge_index_is_2xE_long():
    ei = get_knn_adjacency_matrix(str(LOCS_FILE), k=5)
    assert ei.shape[0] == 2
    assert ei.dtype == torch.long


def test_self_loops_present_for_all_nodes():
    ei = get_knn_adjacency_matrix(str(LOCS_FILE), k=5)
    self_loops = (ei[0] == ei[1]).sum().item()
    assert self_loops >= len(STANDARD_CH_NAMES) == 62
