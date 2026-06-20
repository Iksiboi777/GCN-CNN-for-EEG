"""Tests for group-wise normalization (numpy-only; run without torch)."""
from __future__ import annotations

import numpy as np

from eeg_gnn.data.normalization import groupwise_zscore


def test_shape_preserved():
    X = np.random.randn(30, 62, 10)
    subs = np.repeat([1, 2, 3], 10)
    sess = np.ones(30, dtype=int)
    assert groupwise_zscore(X, subs, sess).shape == X.shape


def test_single_group_is_standardised():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(500, 4, 2))
    subs = np.ones(500, dtype=int)
    sess = np.ones(500, dtype=int)
    out = groupwise_zscore(X, subs, sess)
    # Per (node, feature) position, the group should be ~zero-mean, unit-std.
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-2)


def test_constant_channel_does_not_nan():
    # A constant channel has zero std and must not produce NaNs/Infs.
    X = np.ones((20, 3, 2))
    subs = np.ones(20, dtype=int)
    sess = np.ones(20, dtype=int)
    out = groupwise_zscore(X, subs, sess)
    assert np.isfinite(out).all()
    assert np.allclose(out, 0.0)


def test_groups_normalised_independently():
    # Two subjects with very different offsets should both end up centred.
    X = np.concatenate([np.full((10, 2, 2), 100.0), np.full((10, 2, 2), -100.0)])
    X += np.random.randn(20, 2, 2) * 0.1
    subs = np.repeat([1, 2], 10)
    sess = np.ones(20, dtype=int)
    out = groupwise_zscore(X, subs, sess)
    assert abs(out[:10].mean()) < 1e-6
    assert abs(out[10:].mean()) < 1e-6
