"""Group-wise normalization for cross-session / cross-subject EEG features."""
from __future__ import annotations

import numpy as np


def groupwise_zscore(
    X: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Z-score features independently within each ``(subject, session)`` group.

    This neutralises per-recording baseline shifts — a recording's electrode
    impedance and cap placement drift between sessions — which is one of the
    main drivers of the cross-session/cross-subject distribution shift studied
    in this project. The implementation is fully vectorised via
    :func:`numpy.add.at` (no Python-level loop over groups).

    Parameters
    ----------
    X:
        Feature array shaped ``(N, num_nodes, num_features)``.
    subjects, sessions:
        Integer label arrays of length ``N`` identifying each sample's source.
    eps:
        Standard deviations below this are treated as 1.0 to avoid division by
        zero on (near-)constant channels.

    Returns
    -------
    numpy.ndarray
        The normalised feature array (same shape and dtype as ``X``).
    """
    group_ids = subjects * 1000 + sessions
    _, group_indices = np.unique(group_ids, return_inverse=True)
    n_groups = int(group_indices.max()) + 1

    group_counts = np.bincount(group_indices)[:, None, None]

    group_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    np.add.at(group_sums, group_indices, X)
    group_means = group_sums / group_counts

    X_centered = X - group_means[group_indices]

    group_sq_sums = np.zeros((n_groups, *X.shape[1:]), dtype=X.dtype)
    np.add.at(group_sq_sums, group_indices, X_centered**2)
    group_stds = np.sqrt(group_sq_sums / group_counts)
    group_stds[group_stds < eps] = 1.0

    return X_centered / group_stds[group_indices]
