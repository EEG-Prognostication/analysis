"""Shared preprocessing helpers for analysis notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import mne


def load_filtered_eeg(
    raw: mne.io.BaseRaw,
    eeg_channels: Sequence[str],
    l_freq: float,
    h_freq: float,
    notch_freqs: Sequence[float] = (60.0,),
    verbose: bool | str = False,
) -> mne.io.BaseRaw:
    """Load selected EEG channels into memory and apply standard filtering."""
    filtered = raw.copy().pick(list(eeg_channels)).load_data(verbose=verbose)
    filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)
    if notch_freqs:
        filtered.notch_filter(freqs=list(notch_freqs), verbose=verbose)
    return filtered
