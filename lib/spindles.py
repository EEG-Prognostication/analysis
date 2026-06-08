"""Sleep spindle detection — batch helper.

Exports:
    run_spindles   — passive spindle detection on resting EEG
    STATS_SENTINEL — filename used as run-complete sentinel
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert as _sp_hilbert

from .io import DEFAULT_EEG_CHANNELS, first_paradigm_onset
from .preprocessing import load_filtered_eeg

DEFAULT_N_PERMS = 1000
STATS_SENTINEL  = 'metadata.json'


def run_spindles(subject_id: str, raw, sfreq: float, available_eeg: list,
                 df: pd.DataFrame, out_dir: Path, *, force: bool = False):
    """Passive spindle detection on resting EEG before the first stimulus.

    Not run by default. Invoke with: python run_all.py --analyses spindles

    Algorithm: bandpass 12-15 Hz, Hilbert envelope, threshold at mean + 2 SD,
    retain events with duration 0.5-2 s. Output: spindle rate (events/min)
    and mean amplitude. Both predict CMD positivity (Claassen NatMed 2025).
    """
    first_stim = first_paradigm_onset(df)
    rest_end = min(first_stim, 300.0) if first_stim else 300.0
    if rest_end < 30.0:
        print(f'  [spindles] Resting period < 30 s — skipping.')
        return

    raw_rest = load_filtered_eeg(raw, available_eeg, l_freq=12, h_freq=15, verbose=False)
    raw_rest.set_eeg_reference('average', projection=False, verbose=False)
    rest_data = raw_rest.copy().crop(tmin=0, tmax=rest_end).get_data(picks=available_eeg)
    rest_dur_min = rest_end / 60.0

    envelope  = np.abs(_sp_hilbert(rest_data, axis=1))
    avg_env   = envelope.mean(axis=0)
    threshold = avg_env.mean() + 2.0 * avg_env.std()

    above = avg_env > threshold
    transitions = np.diff(above.astype(int))
    starts = np.where(transitions == 1)[0]
    ends   = np.where(transitions == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        spindle_rate, spindle_amp, spindles = 0.0, float('nan'), []
    else:
        if ends[0] < starts[0]:
            ends = ends[1:]
        n_pairs = min(len(starts), len(ends))
        starts, ends = starts[:n_pairs], ends[:n_pairs]
        durations = (ends - starts) / sfreq
        valid     = (durations >= 0.5) & (durations <= 2.0)
        spindles  = [{'start_s': float(starts[i] / sfreq),
                      'end_s':   float(ends[i]   / sfreq),
                      'amp_uv':  float(avg_env[starts[i]:ends[i]].max() * 1e6)}
                     for i in range(n_pairs) if valid[i]]
        spindle_rate = len(spindles) / rest_dur_min
        spindle_amp  = float(np.mean([s['amp_uv'] for s in spindles])) if spindles else float('nan')

    print(f'  [spindles] {len(spindles)} spindles in {rest_dur_min:.1f} min '
          f'({spindle_rate:.2f}/min), mean amp={spindle_amp:.2f} uV')

    t_rest = np.arange(len(avg_env)) / sfreq
    fig_sp, ax_sp = plt.subplots(figsize=(14, 4))
    ax_sp.plot(t_rest, avg_env * 1e6, color='steelblue', lw=0.8, alpha=0.8,
               label='Mean 12-15 Hz envelope')
    ax_sp.axhline(threshold * 1e6, color='firebrick', lw=1.2, ls='--',
                  label=f'Threshold (mean + 2 SD = {threshold*1e6:.2f} uV)')
    for sp in spindles:
        ax_sp.axvspan(sp['start_s'], sp['end_s'], color='gold', alpha=0.4)
    ax_sp.set(xlabel='Time (s)', ylabel='Envelope (uV)',
              title=f'{subject_id}: Sleep Spindle Detection '
                    f'({len(spindles)} spindles, {spindle_rate:.2f}/min)')
    ax_sp.legend(fontsize=9)
    ax_sp.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_sp.savefig(out_dir / f'{subject_id}_spindles_trace.png', dpi=150)
    plt.close(fig_sp)

    fig_sm, axes_sm = plt.subplots(1, 2, figsize=(8, 4))
    axes_sm[0].bar(['Spindle rate'], [spindle_rate], color='steelblue', alpha=0.8)
    axes_sm[0].set_ylabel('Events per minute')
    axes_sm[0].set_title('Spindle Rate')
    axes_sm[0].axhline(0, color='k', lw=0.5)
    axes_sm[0].grid(True, alpha=0.3, axis='y')
    axes_sm[1].bar(['Mean amplitude'], [spindle_amp if not np.isnan(spindle_amp) else 0],
                   color='darkorange', alpha=0.8)
    axes_sm[1].set_ylabel('uV')
    axes_sm[1].set_title('Mean Spindle Amplitude')
    axes_sm[1].grid(True, alpha=0.3, axis='y')
    fig_sm.suptitle(f'{subject_id}: Spindle Summary  '
                    f'(Claassen NatMed 2025 — predicts CMD and recovery)',
                    fontsize=10)
    plt.tight_layout()
    fig_sm.savefig(out_dir / f'{subject_id}_spindles_summary.png', dpi=150)
    plt.close(fig_sm)

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump({
            'n_spindles':            len(spindles),
            'rest_duration_min':     round(rest_dur_min, 2),
            'spindle_rate_per_min':  round(spindle_rate, 3),
            'mean_amplitude_uv':     round(spindle_amp, 3) if not np.isnan(spindle_amp) else None,
            'threshold_uv':          round(threshold * 1e6, 3),
        }, f, indent=2)
    print(f'  [spindles] Saved figures to {out_dir}')
