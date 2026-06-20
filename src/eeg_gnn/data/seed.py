"""Loader for the SEED differential-entropy (DE) feature set.

Reads the pre-computed, LDS-smoothed DE ``.mat`` files distributed with SEED
(``de_LDS1`` .. ``de_LDS15`` per session file), optionally augments them with a
rolling-variance channel, and applies per-``(subject, session)`` z-score
normalization. Returns a flat :class:`SeedBundle` of aligned arrays.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

from eeg_gnn.data.features import compute_rolling_variance
from eeg_gnn.data.normalization import groupwise_zscore

logger = logging.getLogger(__name__)

#: SEED stores labels as {-1, 0, 1}; map to contiguous class ids {0, 1, 2}.
LABEL_MAP = {-1: 0, 0: 1, 1: 2}


@dataclass
class SeedBundle:
    """Flat, sample-aligned arrays produced by :func:`load_seed_de`.

    All arrays share a leading sample axis of length ``N``.
    """

    X: np.ndarray  # (N, num_nodes, in_features)
    y: np.ndarray  # (N,) class ids in {0, 1, 2}
    sessions: np.ndarray  # (N,) session id in {1, 2, 3}
    subjects: np.ndarray  # (N,) subject id in {1..15}
    trials: np.ndarray  # (N,) unique trial id

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _canonicalise_de_shape(
    data: np.ndarray, num_nodes: int = 62, num_bands: int = 5
) -> np.ndarray:
    """Reorder a raw ``de_LDS`` array to canonical ``(nodes, samples, bands)``.

    SEED files are not stored with a consistent axis order; this resolves the
    layout from the known channel count and band count.
    """
    shape = data.shape
    if shape[0] == num_nodes:
        if shape[2] == num_bands:
            return data
        if shape[1] == num_bands:
            return np.transpose(data, (0, 2, 1))
    elif shape[1] == num_nodes:
        if shape[2] == num_bands:
            return np.transpose(data, (1, 0, 2))
        if shape[0] == num_bands:
            return np.transpose(data, (1, 2, 0))
    elif shape[2] == num_nodes:
        if shape[1] == num_bands:
            return np.transpose(data, (2, 0, 1))
        if shape[0] == num_bands:
            return np.transpose(data, (2, 1, 0))
    return data


def load_seed_de(
    data_dir: Path,
    in_features: int = 10,
    *,
    rolling_var_window: int = 3,
    num_nodes: int = 62,
    num_bands: int = 5,
    normalize: bool = True,
) -> SeedBundle:
    """Load and assemble the SEED DE feature set from ``data_dir``.

    Parameters
    ----------
    data_dir:
        Directory containing the per-subject session ``.mat`` files and
        ``label.mat`` (e.g. ``Data/ExtractedFeatures_1s``).
    in_features:
        ``5`` for raw DE bands, ``10`` to append the rolling-variance channel.
    rolling_var_window:
        Window length for :func:`compute_rolling_variance` when ``in_features==10``.
    num_nodes, num_bands:
        Expected channel and band counts (used to resolve array layout).
    normalize:
        Whether to apply per-``(subject, session)`` z-scoring.

    Returns
    -------
    SeedBundle
    """
    data_dir = Path(data_dir)
    label_file = data_dir / "label.mat"
    logger.info("Loading SEED DE features from %s", data_dir)

    trial_labels = scipy.io.loadmat(label_file)["label"][0]
    mapped_labels = [LABEL_MAP[int(v)] for v in trial_labels]

    # Group session files by subject id (the leading token of the filename).
    files = [f for f in data_dir.iterdir() if f.suffix == ".mat" and f.name != "label.mat"]
    subject_files: dict[int, list[Path]] = {}
    for f in files:
        try:
            subj_id = int(f.name.split("_")[0])
        except ValueError:
            continue
        subject_files.setdefault(subj_id, []).append(f)

    X_list, y_list, session_list, subject_list, trial_list = [], [], [], [], []

    for subj_id in sorted(subject_files):
        session_files = sorted(subject_files[subj_id], key=lambda p: p.name.split("_")[1])
        for sess_idx, fpath in enumerate(session_files):
            session_id = sess_idx + 1
            mat = scipy.io.loadmat(fpath)
            for trial_i in range(1, 16):
                key = f"de_LDS{trial_i}"
                if key not in mat:
                    continue

                data = _canonicalise_de_shape(mat[key], num_nodes, num_bands)

                if in_features == 10:
                    data_var = compute_rolling_variance(data, window_size=rolling_var_window)
                    data = np.concatenate([data, data_var], axis=2)
                # -> (samples, nodes, features)
                data_final = np.transpose(data, (1, 0, 2))

                n = data_final.shape[0]
                X_list.append(data_final)
                y_list.append(np.full(n, mapped_labels[trial_i - 1]))
                session_list.append(np.full(n, session_id))
                subject_list.append(np.full(n, subj_id))
                trial_list.append(np.full(n, subj_id * 1000 + session_id * 100 + trial_i))

    if not X_list:
        raise FileNotFoundError(f"No SEED DE feature files found under {data_dir}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    sessions = np.concatenate(session_list, axis=0)
    subjects = np.concatenate(subject_list, axis=0)
    trials = np.concatenate(trial_list, axis=0)

    if normalize:
        logger.info("Applying per-(subject, session) z-score normalization")
        X = groupwise_zscore(X, subjects, sessions)

    logger.info("Loaded %d samples (in_features=%d)", X.shape[0], in_features)
    return SeedBundle(X=X, y=y, sessions=sessions, subjects=subjects, trials=trials)
