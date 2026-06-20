"""Tests for DE feature engineering (numpy-only; run without torch)."""
from __future__ import annotations

import numpy as np

from eeg_gnn.data.features import compute_rolling_variance, differential_entropy


def test_rolling_variance_preserves_shape():
    data = np.random.randn(62, 40, 5)
    out = compute_rolling_variance(data, window_size=3)
    assert out.shape == data.shape


def test_rolling_variance_window_one_is_zero():
    # A length-1 window has zero variance everywhere.
    data = np.random.randn(8, 10, 5)
    out = compute_rolling_variance(data, window_size=1)
    assert np.allclose(out, 0.0)


def test_rolling_variance_nonnegative():
    out = compute_rolling_variance(np.random.randn(8, 10, 5), window_size=5)
    assert (out >= 0).all()


def test_differential_entropy_zero_point():
    # DE = 0.5*ln(2*pi*e*var) == 0 exactly when var = 1/(2*pi*e).
    var = 1.0 / (2 * np.pi * np.e)
    assert np.isclose(differential_entropy(var), 0.0)


def test_differential_entropy_monotonic():
    assert differential_entropy(2.0) > differential_entropy(1.0)
