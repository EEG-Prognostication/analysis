"""Shared EEG microstate primitives — template fitting, back-fitting, and
per-epoch feature extraction.

Used by:
    lib/resting.py                 — whole-session descriptive microstate summary
    explore_command_classifiers.py — per-sub-epoch 'micro' feature set

See lib/resting.py for the full methodology note (Lehmann 1987; Michel & Koenig
2018) — polarity-aligned k-means is a documented approximation of the modified
k-means algorithm, adequate for descriptive/classification features but not a
substitute for canonical A/B/C/D template matching.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


def fit_microstate_templates(data: np.ndarray, k: int = 4, max_peaks: int = 3000,
                              seed: int = 45) -> tuple[np.ndarray, int] | None:
    """Fit k polarity-aligned microstate templates via k-means on GFP-peak topographies.

    data: (n_channels, n_times). Returns (centers, n_peaks_used), centers
    being (k, n_channels) and L2-normalized, or None if there are too few GFP
    peaks for stable clustering (< k * 20).
    """
    gfp = data.std(axis=0)
    peaks, _ = find_peaks(gfp)
    if len(peaks) < k * 20:
        return None

    rng = np.random.default_rng(seed)
    if len(peaks) > max_peaks:
        peaks = np.sort(rng.choice(peaks, size=max_peaks, replace=False))

    maps = data[:, peaks].T                                        # (n_peaks, n_ch)
    maps = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-30)

    # Polarity alignment: flip each map so its correlation with the first is >= 0
    # (a microstate and its polarity inversion are the same class).
    signs = np.sign(maps @ maps[0])
    signs[signs == 0] = 1.0
    maps_aligned = maps * signs[:, None]

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(maps_aligned)
    centers = km.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-30)
    return centers, int(len(peaks))


def backfit_microstates(data: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-fit every sample in `data` to its best-correlating template.

    data: (n_channels, n_times). Returns (assign, corr2): assign is (n_times,)
    int (sign-invariant — a microstate and its polarity inversion are the same
    class), corr2 is (n_times,) the squared spatial correlation with the
    assigned template (per-sample explained variance).
    """
    data_norm = data / (np.linalg.norm(data, axis=0, keepdims=True) + 1e-30)
    corr = centers @ data_norm                                     # (k, n_times)
    assign = np.argmax(np.abs(corr), axis=0)
    corr2 = np.max(corr ** 2, axis=0)
    return assign, corr2


def microstate_epoch_features(data: np.ndarray, centers: np.ndarray, sfreq: float) -> np.ndarray:
    """Per-epoch microstate feature vector for a single epoch.

    data: (n_channels, n_times) for one epoch. Returns (k + 2,): per-class
    coverage fractions, mean GEV (spatial correlation^2 with the assigned
    template), and transition rate (state changes per second).
    """
    assign, corr2 = backfit_microstates(data, centers)
    k = centers.shape[0]
    coverage = np.array([(assign == state).mean() for state in range(k)])
    gev = float(corr2.mean())
    transitions = float(np.sum(np.diff(assign) != 0) / (data.shape[1] / sfreq))
    return np.concatenate([coverage, [gev, transitions]])
