"""Tests for the model registry and forward passes (requires torch + PyG)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("mne")

from eeg_gnn.config import LOCS_FILE
from eeg_gnn.graph import get_knn_adjacency_matrix
from eeg_gnn.models import MODEL_TYPES, build_model

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def edge_index():
    return get_knn_adjacency_matrix(str(LOCS_FILE), k=5)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("in_features", [5, 10])
def test_build_model_has_parameters(model_type, in_features, edge_index):
    model = build_model(model_type, in_features, edge_index, DEVICE)
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_build_model_rejects_unknown(edge_index):
    with pytest.raises(ValueError):
        build_model("not_a_model", 10, edge_index, DEVICE)


def test_dense_model_forward_shape(edge_index):
    """Adaptive_DGCNN consumes dense ``(batch, nodes, features)`` input."""
    model = build_model("ADAPTIVE_DGCNN", 10, edge_index, DEVICE).eval()
    x = torch.randn(4, 62, 10)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 3)


def test_sparse_model_forward_shape(edge_index):
    """GCN_DE consumes flattened nodes with a batched ``edge_index``."""
    model = build_model("GCN", 10, edge_index, DEVICE).eval()
    B, N, F = 4, 62, 10
    x = torch.randn(B * N, F)
    batch_index = torch.arange(B).repeat_interleave(N)
    offsets = (torch.arange(B) * N).view(-1, 1, 1)
    batched_ei = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    with torch.no_grad():
        out = model(x, batched_ei, batch_index)
    assert out.shape == (B, 3)
