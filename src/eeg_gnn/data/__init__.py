"""Data loading, feature engineering, and normalization.

The SEED loader (:func:`load_seed_de`) is exposed lazily so that numpy-only
consumers — and lightweight unit tests — do not pay the cost of importing
``scipy`` until they actually load a dataset.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from eeg_gnn.data.features import (
    BANDS,
    compute_rolling_variance,
    differential_entropy,
    get_de_features,
)
from eeg_gnn.data.normalization import groupwise_zscore

if TYPE_CHECKING:  # pragma: no cover
    from eeg_gnn.data.seed import SeedBundle, load_seed_de

__all__ = [
    "load_seed_de",
    "SeedBundle",
    "compute_rolling_variance",
    "differential_entropy",
    "get_de_features",
    "groupwise_zscore",
    "BANDS",
]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name in ("load_seed_de", "SeedBundle"):
        from eeg_gnn.data import seed

        return getattr(seed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
