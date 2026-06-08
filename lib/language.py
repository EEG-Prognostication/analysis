"""Language tracking ITPC analysis — batch helper.

Exports:
    run_language   — full language ITPC pipeline
    STATS_SENTINEL — filename used as run-complete sentinel
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.fft import fft

from .io import DEFAULT_EEG_CHANNELS, _crop_to_paradigm
from .preprocessing import load_filtered_eeg

DEFAULT_N_PERMS = 1000
STATS_SENTINEL  = 'results.json'


def run_language(subject_id: str, raw, sfreq: float, available_eeg: list,
                 df: pd.DataFrame, out_dir: Path, *, force: bool = False):
    # Crop to language trials only. post_s=20 covers the ~15.36s epoch window.
    raw = _crop_to_paradigm(raw, df, df['stim_type'] == 'language',
                            pre_s=2.0, post_s=20.0)

    raw_lang = load_filtered_eeg(raw, available_eeg, l_freq=0.1, h_freq=25, verbose=False)
    raw_lang.notch_filter(freqs=[60, 120], verbose=False)

    lang_df = df[df['stim_type'] == 'language'].copy()
    print(f'  [language] {len(lang_df)} trials')

    dur = lang_df['edf_end'] - lang_df['edf_start']
    EPOCH_TMAX = min(15.36, float(dur.min()))

    lang_events = np.column_stack([
        lang_df['start_sample'].values,
        np.zeros(len(lang_df), dtype=int),
        np.ones(len(lang_df), dtype=int),
    ])
    epochs = mne.Epochs(
        raw_lang, events=lang_events, event_id={'language': 1},
        tmin=0, tmax=EPOCH_TMAX, baseline=None, preload=True, verbose=False,
    )
    epochs.resample(256, verbose=False)
    print(f'  [language] epochs shape: {epochs.get_data().shape}')

    def compute_itpc(epochs_data, fs):
        data    = np.transpose(epochs_data, (2, 1, 0))  # (n_samples, n_ch, n_trials)
        freqs   = np.fft.fftfreq(data.shape[0], 1 / fs)
        spectra = fft(data, axis=0)  # complex spectra (n_freq, n_ch, n_trials)
        itpc    = np.abs(np.exp(1j * np.angle(spectra)).mean(axis=2))
        return itpc, freqs, spectra

    itpc, freqs, spectra = compute_itpc(epochs.get_data(), fs=epochs.info['sfreq'])

    N_PERMS      = 1000
    TARGET_FREQS = [0.78, 1.56, 3.125]
    epochs_data  = epochs.get_data()
    fs           = epochs.info['sfreq']
    rng          = np.random.default_rng(42)

    n_trials  = epochs_data.shape[0]
    n_samples = epochs_data.shape[2]
    target_bin_indices = [np.argmin(np.abs(freqs - f)) for f in TARGET_FREQS]

    observed = {f: itpc[bin_idx, :].mean() for f, bin_idx in zip(TARGET_FREQS, target_bin_indices)}
    null = {f: [] for f in TARGET_FREQS}
    for _ in range(N_PERMS):
        shifts = rng.integers(1, n_samples, size=n_trials)
        for f, bin_idx in zip(TARGET_FREQS, target_bin_indices):
            phase_shifts = np.exp(2j * np.pi * bin_idx * shifts / n_samples)
            perm_spec    = spectra[bin_idx, :, :] * phase_shifts[None, :]
            null[f].append(np.abs(np.exp(1j * np.angle(perm_spec)).mean(axis=1)).mean())

    results = {}
    for f in TARGET_FREQS:
        obs      = observed[f]
        null_arr = np.array(null[f])
        p        = np.mean(null_arr >= obs)
        results[f] = {'observed': obs, 'null_mean': null_arr.mean(), 'p_value': p}
        sig = '✓' if p < 0.05 else ''
        print(f'  [language] {f:.3f} Hz  ITPC={obs:.4f}  p={p:.4f}  {sig}')

    fmin, fmax  = 0.5, 4.0
    pos_idx     = (freqs >= fmin) & (freqs <= fmax)
    plot_freqs  = freqs[pos_idx]
    avg_itpc    = itpc[pos_idx, :].mean(axis=1)
    TARGETS     = [(0.78, 'teal', '0.78 Hz'), (1.56, 'darkorchid', '1.56 Hz'), (3.125, 'firebrick', '3.125 Hz')]

    # Average ITPC spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_freqs, avg_itpc, color='steelblue', lw=1.5, label='Observed ITPC')
    for f, c, lbl in TARGETS:
        p       = results[f]['p_value']
        marker  = ' *' if p < 0.05 else ''
        null_95 = np.percentile(null[f], 95)
        ax.axvspan(f - 0.04, f + 0.04, color=c, alpha=0.2, label=f'{lbl}{marker} (p={p:.3f})')
        ax.axhline(null_95, color=c, lw=0.8, ls=':', alpha=0.6)
    ax.set(xlabel='Frequency (Hz)', ylabel='ITPC',
           title=f'{subject_id}: Language ITPC (avg across {len(available_eeg)} channels)',
           xlim=(fmin, fmax))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_lang_itpc_avg.png', dpi=150)
    plt.close(fig)

    # Per-channel ITPC
    n_ch  = len(epochs.ch_names)
    ncols = 4
    nrows = (n_ch + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows), sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for i, ch_name in enumerate(epochs.ch_names):
        ax_flat[i].plot(plot_freqs, itpc[pos_idx, i], color='steelblue', lw=1)
        for f, c, _ in TARGETS:
            ax_flat[i].axvspan(f - 0.04, f + 0.04, color=c, alpha=0.2)
        ax_flat[i].set_title(ch_name, fontsize=9)
        ax_flat[i].grid(True, alpha=0.2)
    for j in range(i + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)
    fig.suptitle(f'{subject_id}: Language ITPC per channel', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_lang_itpc_channels.png', dpi=120)
    plt.close(fig)

    # Topomap
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage, match_case=False, on_missing='warn')
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4))
    for ax, (f, c, lbl) in zip(axes, TARGETS):
        idx      = np.argmin(np.abs(freqs - f))
        ch_itpc  = itpc[idx, :]
        im, _    = mne.viz.plot_topomap(ch_itpc, epochs.info, axes=ax, show=False,
                                         vlim=(0, ch_itpc.max()), cmap='hot_r')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        p = results[f]['p_value']
        ax.set_title(f'{lbl}  (p={p:.3f})', fontsize=10)
    fig.suptitle(f'{subject_id}: Language ITPC topomap', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_lang_itpc_topomap.png', dpi=150)
    plt.close(fig)

    # Write results sentinel
    with open(out_dir / 'results.json', 'w') as _f:
        json.dump({'itpc_results': {str(k): v for k, v in results.items()}}, _f, indent=2)

    print(f'  [language] Saved 3 figures to {out_dir}')
