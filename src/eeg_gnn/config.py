"""Central configuration: paths, hyperparameters, device, and run bookkeeping.

This module replaces the scattered module-level globals and the ``get_args()``
calls that the original ``train_de.py`` made (including one inside the data
loop). All tunable knobs now live on a single :class:`TrainConfig` dataclass.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"
ERRORS_DIR = PROJECT_ROOT / "Errors"
PARAMS_DIR = PROJECT_ROOT / "Params"
RUN_CONFIG_FILE = PROJECT_ROOT / "run_config.json"

#: Bundled standard-10-20 channel-position fallback for graph construction.
LOCS_FILE = Path(__file__).resolve().parent / "graph" / "channel_62_pos.locs"

ModelType = Literal["GCN", "ADAPTIVE_DGCNN", "GraphSAGE"]
Mode = Literal["sub_indep", "sub_dep"]
WindowSize = Literal["1s", "2s", "4s"]


def get_device():
    """Return the best available torch device (CUDA if present, else CPU).

    Imported lazily so that :mod:`eeg_gnn.config` stays importable without torch
    (e.g. in lightweight unit tests).
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    """All hyperparameters for a single training run.

    Attributes mirror the original CLI flags and module globals so behaviour is
    unchanged; defaults reproduce the thesis's primary configuration.
    """

    # Experiment selection
    model_type: ModelType = "GCN"
    mode: Mode = "sub_indep"
    window_size: WindowSize = "1s"
    in_features: int = 10  # 5 DE bands, or 10 (DE + rolling variance)

    # Architecture
    num_nodes: int = 62
    num_classes: int = 3
    hidden_dim: int = 128
    num_layers: int = 2
    use_se: bool = True
    use_doubling: bool = False
    use_overlap_logic: bool = False
    num_subjects: int = 15

    # Optimisation
    batch_size: int = 1024
    epochs: int = 60
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    l1_lambda: float = 1e-4
    patience: int = 20

    # Graph
    knn_k: int = 5

    # Feature engineering
    rolling_var_window: int = 3

    # Parallelism (LOSO)
    max_parallel: int = 3

    def to_dict(self) -> dict:
        """Return a JSON-serialisable view of the configuration."""
        return asdict(self)


def next_run_id(window_size: str, config_file: Path = RUN_CONFIG_FILE) -> int:
    """Read and atomically increment the per-window run counter.

    Mirrors the original ``run_config.json`` bookkeeping used to name runs
    ``Attempt_<N>``. Returns the new run id.
    """
    config: dict = {}
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            config = {}

    key = f"run_counter_{window_size}"
    next_id = int(config.get(key, 0)) + 1
    config[key] = next_id
    config_file.write_text(json.dumps(config, indent=4))
    return next_id
