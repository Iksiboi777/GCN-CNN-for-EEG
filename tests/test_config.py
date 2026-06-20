"""Tests for configuration and run bookkeeping (no heavy dependencies)."""
from __future__ import annotations

from eeg_gnn.config import LOCS_FILE, TrainConfig, next_run_id


def test_defaults_reproduce_primary_setup():
    cfg = TrainConfig()
    assert cfg.model_type == "GCN"
    assert cfg.window_size == "1s"
    assert cfg.in_features == 10
    assert cfg.num_nodes == 62 and cfg.num_classes == 3


def test_to_dict_roundtrip():
    cfg = TrainConfig(model_type="GraphSAGE", in_features=5)
    d = cfg.to_dict()
    assert d["model_type"] == "GraphSAGE" and d["in_features"] == 5


def test_bundled_locs_file_present():
    assert LOCS_FILE.exists(), "channel position fallback must ship with the package"


def test_next_run_id_increments(tmp_path):
    cfg_file = tmp_path / "run_config.json"
    assert next_run_id("1s", cfg_file) == 1
    assert next_run_id("1s", cfg_file) == 2
    assert next_run_id("4s", cfg_file) == 1  # independent counter per window
