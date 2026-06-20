"""Training engine and loss functions."""
from __future__ import annotations

from eeg_gnn.training.engine import (
    TrainingManager,
    evaluate,
    train_epoch,
    train_model_with_interrupt,
)
from eeg_gnn.training.losses import FocalLoss

__all__ = [
    "train_model_with_interrupt",
    "train_epoch",
    "evaluate",
    "TrainingManager",
    "FocalLoss",
]
