"""Feature engineering for differential-entropy (DE) EEG representations.

The SEED pipeline consumes *pre-computed* LDS-smoothed DE features and augments
them at load time with a rolling-variance channel (:func:`compute_rolling_variance`).
The raw-signal DE extractor (:func:`differential_entropy`, :func:`get_de_features`)
is retained for dataset-agnostic use (e.g. DREAMER) and is written to work for
an arbitrary channel count.
"""
from __future__ import annotations

import numpy as np

# Canonical EEG frequency bands (Hz).
BANDS: dict[str, tuple[float, float]] = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 50),
}


def differential_entropy(variance: np.ndarray | float) -> np.ndarray | float:
    """Differential entropy of a Gaussian, ``0.5 * ln(2 * pi * e * var)``.

    For a band-limited EEG signal assumed Gaussian, DE reduces to a closed form
    in the signal variance and is monotonic in log-power — a psychophysically
    motivated, session-stable feature.
    """
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def compute_rolling_variance(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Rolling temporal variance of a DE tensor (edge-padded, centred window).

    Parameters
    ----------
    data:
        Array shaped ``(channels, samples, bands)``.
    window_size:
        Odd window length over the sample (time) axis.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``data`` giving the per-window variance —
        an indicator of the temporal stability of band power.
    """
    pad_width = window_size // 2
    padded = np.pad(data, ((0, 0), (pad_width, pad_width), (0, 0)), mode="edge")

    vars_list = [
        np.var(padded[:, i : i + window_size, :], axis=1)
        for i in range(data.shape[1])
    ]
    return np.stack(vars_list, axis=1)


def get_de_features(segment: np.ndarray, fs: float) -> np.ndarray:
    """Extract per-band DE features from a raw EEG segment.

    Parameters
    ----------
    segment:
        Raw signal shaped ``(time, channels)``.
    fs:
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        DE features shaped ``(channels, len(BANDS))``.
    """
    from scipy.signal import butter, filtfilt

    n_channels = segment.shape[1]
    nyquist = 0.5 * fs
    de = np.zeros((n_channels, len(BANDS)))

    for band_idx, (low, high) in enumerate(BANDS.values()):
        b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
        for ch in range(n_channels):
            filtered = filtfilt(b, a, segment[:, ch])
            de[ch, band_idx] = differential_entropy(np.var(filtered))
    return de
