"""End-to-end smoke test: drive the *real* training entry points on synthetic
SEED-shaped data (CPU, no ``.mat`` files needed).

This exercises the full glue that the unit tests do not — synthetic
``SeedBundle`` -> k-NN graph -> ``build_model`` -> optimizer/scheduler/loss ->
train/eval loop -> checkpoint + history written to disk — for both evaluation
protocols:

* ``run_session_holdout`` (sub_dep) for all three architectures
* ``run_single_subject_fold`` (one LOSO fold; sparse model + FocalLoss)

Stdout is routed through a writer that fails exactly like a Windows ``cp1250``
console, so the runs also guard against re-introducing non-ASCII characters in
printed training output (a U+2605 "best epoch" marker once crashed real runs).

Requires torch + PyG + mne; skips automatically otherwise, matching the rest of
the suite. All artifacts are written under pytest's ``tmp_path``.
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("mne")

from eeg_gnn.config import LOCS_FILE, TrainConfig
from eeg_gnn.data.seed import SeedBundle
from eeg_gnn.graph import get_knn_adjacency_matrix
from eeg_gnn.models import MODEL_TYPES
from eeg_gnn.train import _aggregate_loso, run_session_holdout, run_single_subject_fold

NODES, FEATS, CLASSES, SUBJECTS = 62, 10, 3, 3
PER_GROUP = 12  # samples per (subject, session); cycles through all 3 classes


def _make_bundle(seed: int = 0) -> SeedBundle:
    """Tiny synthetic bundle: 3 subjects x 3 sessions, every class in every group."""
    rng = np.random.default_rng(seed)
    X, y, sess, sub, trial = [], [], [], [], []
    for s in range(1, SUBJECTS + 1):
        for ses in (1, 2, 3):
            X.append(rng.standard_normal((PER_GROUP, NODES, FEATS)).astype(np.float32))
            y.append(np.arange(PER_GROUP) % CLASSES)  # all 3 classes present
            sess.append(np.full(PER_GROUP, ses))
            sub.append(np.full(PER_GROUP, s))
            trial.append(np.full(PER_GROUP, s * 1000 + ses * 100))
    return SeedBundle(
        X=np.concatenate(X), y=np.concatenate(y),
        sessions=np.concatenate(sess), subjects=np.concatenate(sub),
        trials=np.concatenate(trial),
    )


def _tiny_cfg(model_type: str) -> TrainConfig:
    """Smallest config that still exercises the full loop (2 epochs == OneCycle steps)."""
    return TrainConfig(
        model_type=model_type, mode="sub_dep",
        in_features=FEATS, num_nodes=NODES, num_classes=CLASSES,
        num_subjects=SUBJECTS, hidden_dim=32, epochs=2, batch_size=64,
    )


class _CP1250Stdout(io.TextIOBase):
    """A stdout stand-in that fails like a non-UTF-8 Windows console (cp1250)."""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        s.encode("cp1250")  # raises UnicodeEncodeError on e.g. U+2605
        return len(s)


@pytest.fixture
def strict_console(monkeypatch):
    """Make printed training output crash on any non-cp1250-encodable character."""
    monkeypatch.setattr(sys, "stdout", _CP1250Stdout())


@pytest.fixture(scope="module")
def edge_index():
    return get_knn_adjacency_matrix(str(LOCS_FILE), k=5)


@pytest.fixture(scope="module")
def bundle():
    return _make_bundle()


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_session_holdout_runs(model_type, bundle, edge_index, tmp_path, monkeypatch, strict_console):
    """sub_dep path trains, evaluates, and checkpoints for each architecture."""
    monkeypatch.chdir(tmp_path)
    run_session_holdout(_tiny_cfg(model_type), bundle, edge_index,
                        run_id=1, model_name=f"{model_type}_smoke")

    params, results = tmp_path / "Params", tmp_path / "Results"
    assert params.is_dir() and results.is_dir()
    assert any(p.suffix == ".pth" for p in params.rglob("*"))  # checkpoint saved
    assert any(p.suffix == ".npy" for p in results.rglob("*"))  # history saved


def test_single_loso_fold_runs(bundle, edge_index, tmp_path, monkeypatch, strict_console):
    """sub_indep path: one leave-one-subject-out fold (sparse model + FocalLoss)."""
    monkeypatch.chdir(tmp_path)
    X = torch.tensor(bundle.X, dtype=torch.float32)
    y = torch.tensor(bundle.y, dtype=torch.long)
    sub = torch.tensor(bundle.subjects, dtype=torch.long)

    run_single_subject_fold(
        subject_id=1, cfg=_tiny_cfg("GCN"),
        X_full=X, y_full=y, sub_full=sub,
        base_edge_index=edge_index, run_id=1, model_name="GCN_smoke_loso",
    )

    params = tmp_path / "Params"
    assert params.is_dir()
    assert any(p.suffix == ".pth" for p in params.rglob("*"))  # fold checkpoint saved


def _run_fold(subject_id, bundle, edge_index, model_name):
    """Drive one real LOSO fold on the synthetic bundle."""
    run_single_subject_fold(
        subject_id=subject_id, cfg=_tiny_cfg("GCN"),
        X_full=torch.tensor(bundle.X, dtype=torch.float32),
        y_full=torch.tensor(bundle.y, dtype=torch.long),
        sub_full=torch.tensor(bundle.subjects, dtype=torch.long),
        base_edge_index=edge_index, run_id=1, model_name=model_name,
    )


def test_loso_fold_writes_the_file_the_aggregator_reads(
    bundle, edge_index, tmp_path, monkeypatch, strict_console
):
    """Regression guard: the fold must write ``final_test_preds_sub<k>.npy``.

    ``_aggregate_loso`` reads exactly this path. When the engine did not write it,
    every fold counted as missing and the global summary reported 0.00% -- while
    the old assertions (any ``.npy`` exists) still passed.
    """
    monkeypatch.chdir(tmp_path)
    _run_fold(1, bundle, edge_index, "GCN_aggr")

    res = (tmp_path / "Results" / "GCN_aggr" / "Attempt_1_LOSO_Parallel"
           / "Subject_1" / "final_test_preds_sub1.npy")
    assert res.exists(), "fold did not write the predictions file the aggregator reads"

    data = np.load(res, allow_pickle=True).item()
    assert {"preds", "true", "acc"} <= set(data)
    assert len(data["preds"]) == len(data["true"]) > 0


def test_best_epoch_is_not_selected_on_the_test_split(
    bundle, edge_index, tmp_path, monkeypatch, strict_console
):
    """The fold must select its checkpoint on a validation split, not the test fold."""
    monkeypatch.chdir(tmp_path)
    _run_fold(1, bundle, edge_index, "GCN_noleak")

    data = np.load(
        tmp_path / "Results" / "GCN_noleak" / "Attempt_1_LOSO_Parallel"
        / "Subject_1" / "final_test_preds_sub1.npy", allow_pickle=True).item()
    assert data["leaky_selection"] is False
    assert data["selection_split"] == "Val"


def test_loso_aggregation_produces_a_real_score(
    bundle, edge_index, tmp_path, monkeypatch, strict_console
):
    """All folds -> a global summary with a non-zero mean and every subject present."""
    monkeypatch.chdir(tmp_path)
    for sid in range(1, SUBJECTS + 1):
        _run_fold(sid, bundle, edge_index, "GCN_full")

    _aggregate_loso(_tiny_cfg("GCN"), list(range(1, SUBJECTS + 1)), 1, "GCN_full")

    summary = (tmp_path / "Results" / "GCN_full" / "Attempt_1_LOSO_Parallel"
               / "LOSO_Global_Summary.txt")
    assert summary.exists()
    text = summary.read_text()
    assert "Global LOSO Average: 0.00%" not in text
    assert f"Folds aggregated: {SUBJECTS}/{SUBJECTS}" in text
    assert "MISSING FOLDS" not in text
    assert "LEAKY SELECTION" not in text
    for sid in range(1, SUBJECTS + 1):
        assert f"Subject {sid}:" in text
