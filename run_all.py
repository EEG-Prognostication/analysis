#!/usr/bin/env python3
"""
Batch runner: run EEG analyses for all patients missing results.

Usage:
    python run_all.py                              # run all missing analyses
    python run_all.py --force                      # re-run even if outputs exist
    python run_all.py --patients CON011 CON012     # specific patient(s)
    python run_all.py --analyses oddball language  # specific analyses
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless — no display needed
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from mne.decoding import LinearModel, get_coef
from mne.time_frequency import psd_array_multitaper

try:
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from lib.io import DEFAULT_EEG_CHANNELS, align_stimulus_csv, load_raw_eeg_metadata
from lib.preprocessing import load_filtered_eeg

mne.set_log_level('WARNING')

REPO_ROOT   = ANALYSIS_ROOT.parent
CSV_DIR     = REPO_ROOT / 'stimulus_software' / 'patient_data' / 'results'
EDF_DIR     = REPO_ROOT / 'stimulus_software' / 'patient_data' / 'edfs'
RESULTS_DIR = ANALYSIS_ROOT / 'results'

ALL_ANALYSES = ['oddball', 'language', 'command']


# ── Session discovery ──────────────────────────────────────────────────────────

def discover_sessions() -> list[dict]:
    import re
    edf_by_patient = {
        f.name.replace('_clipped.EDF', '').replace('_clipped.edf', ''): f
        for f in sorted(EDF_DIR.glob('*.[Ee][Dd][Ff]'))
        if '_clipped' in f.name
    }
    sessions = []
    for f in sorted(CSV_DIR.glob('*_stimulus_results.csv')):
        m = re.search(r'(\d{4}-\d{2}-\d{2})_stimulus_results', f.stem)
        if not m:
            continue
        date = m.group(1)
        pid  = f.stem[:f.stem.index('_' + date)]
        edf  = edf_by_patient.get(pid)
        if edf:
            sessions.append({'patient_id': pid, 'date': date, 'csv': f, 'edf': edf})
    return sessions


def has_results(patient_id: str, analysis: str) -> bool:
    out_dir = RESULTS_DIR / patient_id / analysis
    return out_dir.exists() and any(out_dir.glob('*.png'))


def has_paradigm(df: pd.DataFrame, analysis: str) -> bool:
    st = df['stim_type']
    if analysis == 'oddball':
        return st.str.startswith('oddball').any()
    if analysis == 'language':
        return (st == 'language').any()
    if analysis == 'command':
        has_pairs = st.str.match(r'(right|left)_(keep|stop)', na=False).any()
        has_runs  = st.str.contains('command', na=False).any()
        return has_pairs or has_runs
    return False


# ── Shared EEG load ────────────────────────────────────────────────────────────

def load_session(edf_path: Path, csv_path: Path):
    raw, sfreq, available_eeg = load_raw_eeg_metadata(
        edf_path, eeg_channels=DEFAULT_EEG_CHANNELS, bad_channels=[], preload=False, verbose=False
    )
    df, _ = align_stimulus_csv(csv_path, sfreq=sfreq, n_times=raw.n_times)
    return raw, sfreq, available_eeg, df


# ── Oddball P300 + Johnsen ─────────────────────────────────────────────────────

ODDBALL_REJECT_UV = 200


def _build_oddball_evoked(raw, available_eeg, df, tag='oddball'):
    """Filter, epoch, and average oddball EEG. Returns (epochs, evoked_rare, evoked_std,
    diff_evoked, counts) where counts=(n_rare_pre, n_std_pre, n_rare_post, n_std_post).
    Returns None on missing data."""
    raw_p300 = load_filtered_eeg(raw, available_eeg, l_freq=0.1, h_freq=30, verbose=False)
    raw_p300.set_eeg_reference('average', projection=False, verbose=False)

    odd_df  = df[df['stim_type'].str.startswith('oddball')].copy()
    rare_df = odd_df[odd_df['notes'] == 'rare_tone']
    std_df  = odd_df[odd_df['notes'] == 'standard_tone']

    if rare_df.empty:
        return None

    n_rare_pre = len(rare_df)
    n_std_pre  = len(std_df)

    rare_events = np.column_stack([
        rare_df['start_sample'].values,
        np.zeros(len(rare_df), dtype=int),
        np.full(len(rare_df), 2, dtype=int),
    ])
    std_events = np.column_stack([
        std_df['start_sample'].values,
        np.zeros(len(std_df), dtype=int),
        np.full(len(std_df), 1, dtype=int),
    ])
    all_events = np.vstack([rare_events, std_events])
    all_events = all_events[all_events[:, 0].argsort()]

    epochs = mne.Epochs(
        raw_p300, events=all_events,
        event_id={'standard': 1, 'rare': 2},
        tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
        reject=dict(eeg=ODDBALL_REJECT_UV * 1e-6),
        preload=True, verbose=False,
    )

    n_rare_post = len(epochs['rare'])
    n_std_post  = len(epochs['standard'])
    n_rare_rej  = n_rare_pre - n_rare_post
    n_std_rej   = n_std_pre  - n_std_post
    print(f'  [{tag}] {len(epochs)} epochs ({n_rare_post} rare, {n_std_post} std)'
          f'  --  rejected {n_rare_rej} rare, {n_std_rej} std (>{ODDBALL_REJECT_UV} uV)')

    evoked_rare = epochs['rare'].average()
    evoked_std  = epochs['standard'].average()
    diff_evoked = mne.combine_evoked([evoked_rare, evoked_std], weights=[1, -1])

    counts = (n_rare_pre, n_std_pre, n_rare_post, n_std_post)
    return epochs, evoked_rare, evoked_std, diff_evoked, counts


def run_oddball(subject_id: str, raw, sfreq: float, available_eeg: list, df: pd.DataFrame,
                plots_only: bool = False):
    out_dir = RESULTS_DIR / subject_id / 'oddball'
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _build_oddball_evoked(raw, available_eeg, df)
    if result is None:
        print(f'  [oddball] No per-beep rows -- skipping.')
        return
    epochs, evoked_rare, evoked_std, diff_evoked, (n_rare_pre, n_std_pre, n_rare_post, n_std_post) = result

    COMPONENTS = {
        # N1 uses source='standard': sign-flip test on standard-tone evoked amplitude.
        # N1 is obligatory (both tones drive it equally), so rare-standard difference
        # cancels out — the meaningful test is whether the standard average is negative.
        'N1':  {'win': (0.050, 0.100), 'ch': 'Cz', 'sign': -1, 'label': 'Primary auditory response', 'source': 'standard'},
        'MMN': {'win': (0.100, 0.200), 'ch': 'Fz', 'sign': -1, 'label': 'Automatic mismatch',        'source': 'diff'},
        'P3a': {'win': (0.200, 0.300), 'ch': 'Cz', 'sign': +1, 'label': 'Automatic orienting',       'source': 'diff'},
        'P3b': {'win': (0.300, 0.600), 'ch': 'Pz', 'sign': +1, 'label': 'Conscious updating (P300)', 'source': 'diff'},
    }

    epochs_data = epochs.get_data()
    labels      = epochs.events[:, 2]
    t_ep        = epochs.times

    _null_cache = out_dir / 'null_arrays.npz'

    if plots_only:
        # Load obs/p from metadata.json — no permutation or SVM runs needed.
        # null arrays loaded from cache if present; histograms skipped otherwise.
        _meta_path = out_dir / 'metadata.json'
        if not _meta_path.exists():
            print('  [oddball] --plots-only: no metadata.json — run without --plots-only first.')
            return
        with open(_meta_path) as _f:
            _meta = json.load(_f)
        _arrays = np.load(_null_cache) if _null_cache.exists() else {}
        perm_results = {}
        for _name, _comp in COMPONENTS.items():
            _cm = _meta.get('components', {}).get(_name, {})
            if _cm:
                _nk = f'null_{_name}'
                perm_results[_name] = {
                    'obs':  _cm['observed_uv'],
                    'p':    _cm['p_value'],
                    'ch':   _comp['ch'],
                    'comp': _comp,
                    'null': _arrays[_nk] if _nk in _arrays else None,
                }
        _fn = _meta.get('fn_result')
        if _fn:
            perm_results['FN'] = {
                'obs':  _fn['observed_uv'],
                'p':    _fn['p_value'],
                'ch':   _fn['channels'],
                'comp': {'sign': +1, 'win': (0.300, 0.600),
                         'label': 'P3b dipole index (parietal minus frontal)', 'source': 'diff'},
                'null': _arrays['null_FN'] if 'null_FN' in _arrays else None,
            }
        _svm = _meta.get('svm_result')
        svm_result = (
            {'acc': _svm['accuracy'], 'p': _svm['p_value'],
             'null': _arrays['null_svm'] if 'null_svm' in _arrays else None}
            if _svm else None
        )
        fischer_score = _meta.get('fischer_score', 0)
        n_components  = _meta.get('n_components', 4)
        par_chs = [ch for ch in ['P3', 'Pz', 'P4'] if ch in epochs.ch_names]
        fro_chs = [ch for ch in ['F7', 'F3', 'Fz', 'F4', 'F8'] if ch in epochs.ch_names]
        svm_mask  = (t_ep >= 0.0) & (t_ep <= 0.600)
        svm_data  = epochs_data[:, :, svm_mask]
        ds_factor = max(1, int(sfreq // 32))
        svm_data  = svm_data[:, :, ::ds_factor]
        X_svm = svm_data.reshape(len(labels), -1)
        y_svm = (labels == 2).astype(int)
        _has_null = bool(_arrays)
        print(f'  [oddball] plots-only: loaded stats from metadata.json'
              f'{" + null cache" if _has_null else " (null histograms skipped — no cache)"}')


    else:
        # Full computation path
        N_PERMS         = 1000
        COMPONENT_SEEDS = {'N1': 39, 'MMN': 40, 'P3a': 41, 'P3b': 42}

        perm_results = {}
        for name, comp in COMPONENTS.items():
            ch = comp['ch']
            if ch not in epochs.ch_names:
                continue
            ch_idx   = epochs.ch_names.index(ch)
            win_mask = (t_ep >= comp['win'][0]) & (t_ep <= comp['win'][1])
            sign     = comp['sign']
            source   = comp.get('source', 'diff')
            rng      = np.random.default_rng(COMPONENT_SEEDS.get(name, 42))

            if source == 'standard':
                per_epoch = epochs['standard'].get_data()[:, ch_idx][:, win_mask].mean(axis=1) * 1e6
                obs  = float(per_epoch.mean())
                null = np.array([(rng.choice([-1, 1], size=len(per_epoch)) * per_epoch).mean()
                                 for _ in range(N_PERMS)])
            else:
                def _amp(data, labs, _ch=ch_idx, _win=win_mask):
                    return (data[labs == 2, _ch][:, _win].mean()
                            - data[labs == 1, _ch][:, _win].mean()) * 1e6
                obs  = _amp(epochs_data, labels)
                null = np.array([_amp(epochs_data, rng.permutation(labels)) for _ in range(N_PERMS)])

            p = np.mean(null <= obs) if sign < 0 else np.mean(null >= obs)
            perm_results[name] = {'obs': obs, 'null': null, 'p': p, 'ch': ch, 'comp': comp}

        _fischer_names = ('N1', 'MMN', 'P3a', 'P3b')
        fischer_score = sum(1 for name, res in perm_results.items()
                            if name in _fischer_names and res['p'] < 0.05)
        n_components  = sum(1 for name in _fischer_names if name in perm_results)

        par_chs = [ch for ch in ['P3', 'Pz', 'P4'] if ch in epochs.ch_names]
        fro_chs = [ch for ch in ['F7', 'F3', 'Fz', 'F4', 'F8'] if ch in epochs.ch_names]
        if par_chs and fro_chs:
            par_idx = [epochs.ch_names.index(ch) for ch in par_chs]
            fro_idx = [epochs.ch_names.index(ch) for ch in fro_chs]
            di_mask = (t_ep >= 0.300) & (t_ep <= 0.600)
            rng_fn  = np.random.default_rng(43)

            def _di_amp(data, labs, _par=par_idx, _fro=fro_idx, _mask=di_mask):
                par_mean = data[:, _par, :][:, :, _mask].mean(axis=(1, 2)) * 1e6
                fro_mean = data[:, _fro, :][:, :, _mask].mean(axis=(1, 2)) * 1e6
                contrast = par_mean - fro_mean
                return contrast[labs == 2].mean() - contrast[labs == 1].mean()

            fn_obs  = _di_amp(epochs_data, labels)
            fn_null = np.array([_di_amp(epochs_data, rng_fn.permutation(labels))
                                for _ in range(N_PERMS)])
            fn_p    = float(np.mean(fn_null >= fn_obs))
            perm_results['FN'] = {
                'obs': fn_obs, 'null': fn_null, 'p': fn_p, 'ch': 'parietal-frontal contrast',
                'comp': {'sign': +1, 'win': (0.300, 0.600),
                         'label': 'P3b dipole index (parietal minus frontal)', 'source': 'diff'},
            }

        svm_result = None
        svm_mask   = (t_ep >= 0.0) & (t_ep <= 0.600)
        svm_data   = epochs_data[:, :, svm_mask]
        ds_factor  = max(1, int(sfreq // 32))
        svm_data   = svm_data[:, :, ::ds_factor]
        X_svm = svm_data.reshape(len(labels), -1)
        y_svm = (labels == 2).astype(int)
        loo   = LeaveOneOut()

        preds = []
        for train_idx, test_idx in loo.split(X_svm):
            sc  = StandardScaler()
            clf = LinearSVC(max_iter=1000, random_state=0, class_weight='balanced')
            clf.fit(sc.fit_transform(X_svm[train_idx]), y_svm[train_idx])
            preds.append(clf.predict(sc.transform(X_svm[test_idx]))[0])
        svm_acc = float(np.mean(np.array(preds) == y_svm))

        rng_svm     = np.random.default_rng(44)
        N_SVM_PERMS = 500
        svm_null    = []
        for _ in range(N_SVM_PERMS):
            y_perm = rng_svm.permutation(y_svm)
            p_perm = []
            for train_idx, test_idx in loo.split(X_svm):
                sc  = StandardScaler()
                clf = LinearSVC(max_iter=1000, random_state=0, class_weight='balanced')
                clf.fit(sc.fit_transform(X_svm[train_idx]), y_perm[train_idx])
                p_perm.append(clf.predict(sc.transform(X_svm[test_idx]))[0])
            svm_null.append(float(np.mean(np.array(p_perm) == y_perm)))
        svm_null   = np.array(svm_null)
        svm_p      = float(np.mean(svm_null >= svm_acc))
        svm_result = {'acc': svm_acc, 'null': svm_null, 'p': svm_p}
        print(f'  [oddball] SVM LOO accuracy={svm_acc:.3f}  p={svm_p:.3f}')

        # Cache null arrays so --plots-only reruns skip this entire block
        _save = {f'null_{n}': r['null'] for n, r in perm_results.items()}
        if svm_result is not None:
            _save['null_svm'] = svm_result['null']
        np.savez(_null_cache, **_save)

    # Shao 2025 clinical thresholds: |MMN Fz| >= 2.044 µV AND P3b Pz >= 1.095 µV
    mmn_obs = perm_results.get('MMN', {}).get('obs', float('nan'))
    p3b_obs = perm_results.get('P3b', {}).get('obs', float('nan'))
    shao_mmn_pos = bool(abs(mmn_obs) >= 2.044) if not np.isnan(mmn_obs) else False
    shao_p3b_pos = bool(p3b_obs >= 1.095)      if not np.isnan(p3b_obs) else False

    for name, res in perm_results.items():
        print(f'  [oddball] {name} {res["ch"]} amp={res["obs"]:+.3f} µV  p={res["p"]:.4f}')
    print(f'  [oddball] Fischer hierarchy score: {fischer_score}/{n_components}')
    print(f'  [oddball] Shao thresholds: MMN {"✓" if shao_mmn_pos else "✗"} ({abs(mmn_obs):.2f} µV, thresh 2.044)  '
          f'P3b {"✓" if shao_p3b_pos else "✗"} ({p3b_obs:.2f} µV, thresh 1.095)')

    # Per-component ERP figures — one file per component, only the relevant electrodes shown
    times_ms = evoked_rare.times * 1000

    # Electrodes to show per component:
    #   N1  → Cz (primary) + T3/T4 (bilateral auditory cortex)
    #   MMN → Fz (primary) + Cz
    #   P3a → Cz (primary) + Fz
    #   P3b → Pz (primary, positive) + Fz (expected negative — the dipole key)
    COMP_PLOT = {
        'N1':  {'electrodes': ['Cz', 'T3', 'T4'], 'color': '#b0a0e0', 'alpha': 0.35},
        'MMN': {'electrodes': ['Fz', 'Cz'],        'color': '#4da6e8', 'alpha': 0.30},
        'P3a': {'electrodes': ['Cz', 'Fz'],        'color': '#4dc44d', 'alpha': 0.30},
        'P3b': {'electrodes': ['Pz', 'Fz'],        'color': '#f0b800', 'alpha': 0.35},
    }

    # Delete legacy combined file so it does not appear in report
    legacy_erp = out_dir / f'{subject_id}_oddball_p300.png'
    if legacy_erp.exists():
        legacy_erp.unlink()

    for name, cfg in COMP_PLOT.items():
        _win   = COMPONENTS[name]['win'] if name in COMPONENTS else cfg['win']
        win_ms = (_win[0] * 1000, _win[1] * 1000)
        res    = perm_results.get(name, {})
        obs    = res.get('obs', float('nan'))
        p_val  = res.get('p',   float('nan'))

        avail = [ch for ch in cfg['electrodes'] if ch in diff_evoked.ch_names]
        if not avail:
            continue

        if len(avail) > 3:
            ncols = 2
            nrows = (len(avail) + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharex=True)
            axes = axes.flatten()
            for ax in axes[len(avail):]:
                ax.set_visible(False)
        else:
            fig, axes = plt.subplots(len(avail), 1, figsize=(10, 3.5 * len(avail)), sharex=True)
            if len(avail) == 1:
                axes = [axes]

        for ax, ch_name in zip(axes, avail):
            idx     = evoked_rare.ch_names.index(ch_name)
            rare_uv = evoked_rare.data[idx] * 1e6
            std_uv  = evoked_std.data[idx]  * 1e6
            diff_uv = rare_uv - std_uv
            ax.plot(times_ms, std_uv,  color='steelblue', lw=1.5, label='Standard', alpha=0.85)
            ax.plot(times_ms, rare_uv, color='firebrick',  lw=1.5, label='Rare',     alpha=0.85)
            ax.plot(times_ms, diff_uv, color='darkgreen',  lw=1.5, ls='--', label='Rare − Std')
            ax.axvspan(win_ms[0], win_ms[1], color=cfg['color'], alpha=cfg['alpha'],
                       label=f'{name} ({int(win_ms[0])}–{int(win_ms[1])} ms)')
            ax.axvline(0, color='k', lw=0.8, ls=':')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel('µV')
            ax.set_title(ch_name, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (ms)')
        p_str = f'p = {p_val:.3f}' if not np.isnan(p_val) else ''
        fig.suptitle(
            f'{subject_id}: {name} ({p_str}, obs = {obs:+.2f} µV)',
            fontsize=11
        )
        plt.tight_layout()
        fig.savefig(out_dir / f'{subject_id}_oddball_erp_{name.lower()}.png', dpi=150)
        plt.close(fig)

    # P3b Dipole Index ERP — parietal vs frontal averages on one panel
    if 'FN' in perm_results and par_chs and fro_chs:
        par_avail = [ch for ch in par_chs if ch in diff_evoked.ch_names]
        fro_avail = [ch for ch in fro_chs if ch in diff_evoked.ch_names]
        if par_avail and fro_avail:
            par_diff = np.mean(
                [diff_evoked.data[diff_evoked.ch_names.index(ch)] for ch in par_avail], axis=0
            ) * 1e6
            fro_diff = np.mean(
                [diff_evoked.data[diff_evoked.ch_names.index(ch)] for ch in fro_avail], axis=0
            ) * 1e6
            fn_obs_di = perm_results['FN']['obs']
            fn_p_di   = perm_results['FN']['p']
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(times_ms, par_diff, color='#e84040', lw=2.0,
                    label=f'Parietal avg ({"+".join(par_avail)})')
            ax.plot(times_ms, fro_diff, color='#5b9bd5', lw=2.0,
                    label=f'Frontal avg ({"+".join(fro_avail)})')
            ax.axvspan(300, 600, color='#f0b800', alpha=0.25, label='300-600 ms')
            ax.axvline(0, color='k', lw=0.8, ls=':')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Rare minus standard (µV)')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.suptitle(
                f'{subject_id}: P3b Dipole Index (p = {fn_p_di:.3f}, contrast = {fn_obs_di:+.2f} µV)',
                fontsize=11
            )
            plt.tight_layout()
            fig.savefig(out_dir / f'{subject_id}_oddball_erp_fn.png', dpi=150)
            plt.close(fig)

    # Butterfly plot — all channels overlaid, coloured by scalp region
    REGION_COLOR = {
        'Fp1': '#5b9bd5', 'Fp2': '#5b9bd5',
        'F7':  '#5b9bd5', 'F3':  '#5b9bd5', 'Fz': '#5b9bd5', 'F4': '#5b9bd5', 'F8': '#5b9bd5',
        'T3':  '#ed7d31', 'C3':  '#70ad47', 'Cz': '#70ad47', 'C4': '#70ad47', 'T4': '#ed7d31',
        'T5':  '#ed7d31', 'P3':  '#e84040', 'Pz': '#e84040', 'P4': '#e84040', 'T6': '#ed7d31',
        'O1':  '#9b59b6', 'O2':  '#9b59b6',
    }
    REGION_LABEL = {
        '#5b9bd5': 'Frontal', '#70ad47': 'Central',
        '#ed7d31': 'Temporal', '#e84040': 'Parietal', '#9b59b6': 'Occipital',
    }
    HIGHLIGHT = {'Fz': '#1a6bb5', 'Cz': '#2e7d32', 'Pz': '#b71c1c'}

    # Compute per-window topographic summary from the difference wave
    topo_summary = {}
    for name, comp in COMPONENTS.items():
        win     = comp['win']
        mask    = (diff_evoked.times >= win[0]) & (diff_evoked.times <= win[1])
        ch_data = diff_evoked.data[:, mask].mean(axis=1) * 1e6  # (n_ch,)
        eeg_chs = [c for c in diff_evoked.ch_names]
        ranked  = sorted(zip(ch_data, eeg_chs))
        top_pos = [(ch, amp) for amp, ch in ranked[-3:][::-1]]
        top_neg = [(ch, amp) for amp, ch in ranked[:3]]
        topo_summary[name] = {'pos': top_pos, 'neg': top_neg,
                               'sign': comp['sign'], 'win': win}

    # Build butterfly plot — single axis, coloured by region
    fig, ax_wave = plt.subplots(figsize=(18, 8))

    seen_region_labels = set()
    for ch_name in diff_evoked.ch_names:
        idx   = diff_evoked.ch_names.index(ch_name)
        y     = diff_evoked.data[idx] * 1e6
        color = REGION_COLOR.get(ch_name, '#aaaaaa')
        if ch_name in HIGHLIGHT:
            ax_wave.plot(times_ms, y, color=HIGHLIGHT[ch_name], lw=3.0, zorder=4, label=ch_name)
        else:
            rlabel = REGION_LABEL.get(color)
            lbl    = rlabel if (rlabel and rlabel not in seen_region_labels) else '_'
            if rlabel:
                seen_region_labels.add(rlabel)
            ax_wave.plot(times_ms, y, color=color, lw=1.5, zorder=2, alpha=0.85, label=lbl)

    # Component labels: dotted vertical boundary lines + bold label at top edge.
    # get_xaxis_transform() gives x in data coords (ms), y in axes fraction — correct for both.
    COMP_BRACKET_COLOR = {'N1': '#6A0DAD', 'MMN': '#1565C0', 'P3a': '#2E7D32', 'P3b': '#E65100'}
    xform = ax_wave.get_xaxis_transform()
    for name, comp in COMPONENTS.items():
        lo_ms  = comp['win'][0] * 1000
        hi_ms  = comp['win'][1] * 1000
        color  = COMP_BRACKET_COLOR.get(name, '#555555')
        mid_ms = (lo_ms + hi_ms) / 2
        ax_wave.axvline(lo_ms, color=color, lw=1.0, ls=':', alpha=0.55, zorder=1)
        ax_wave.axvline(hi_ms, color=color, lw=1.0, ls=':', alpha=0.55, zorder=1)
        ax_wave.text(mid_ms, 0.97, name, ha='center', va='top', fontsize=11,
                     color=color, fontweight='bold', transform=xform)

    ax_wave.axvline(0, color='k', lw=0.8, ls=':')
    ax_wave.axhline(0, color='k', lw=0.5)
    ax_wave.set_title(f'{subject_id}: Rare minus Standard, all electrodes', fontsize=13)
    ax_wave.set(xlabel='Time (ms)', ylabel='Rare − Standard (µV)')
    ax_wave.legend(loc='lower right', fontsize=10, ncol=2)
    ax_wave.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_oddball_butterfly.png', dpi=150)
    plt.close(fig)

    # Build plain-English topo summary strings (saved to metadata; used in report caption)
    def _p3b_interp(pos, neg):
        pos_str = ', '.join(f'{ch} ({amp:+.1f})' for ch, amp in pos[:3])
        neg_str = ', '.join(f'{ch} ({amp:+.1f})' for ch, amp in neg[:3] if amp < 0)
        frontal_neg  = any(ch.startswith(('F', 'Fp')) for ch, amp in neg[:3] if amp < 0)
        parietal_max = pos[0][0] in ('Pz', 'P3', 'P4', 'O1', 'O2')
        if parietal_max and frontal_neg:
            interp = 'Parietal-positive + frontal-negative pattern: topographic signature of genuine P3b.'
        elif parietal_max:
            interp = 'Parietal maximum present; frontal negativity absent — P3b likely but P3a cannot be fully excluded.'
        else:
            interp = 'No clear parietal maximum — P3b topographic interpretation is uncertain.'
        return f'Most positive: {pos_str}. {("Most negative: " + neg_str + ". ") if neg_str else ""}{interp}'

    interp_map = {
        'N1':  lambda p, n: (f'Most negative: {n[0][0]} ({n[0][1]:+.1f} µV) — '
                              'auditory pathway intact (expected central negativity).'),
        'MMN': lambda p, n: (f'Most negative: {n[0][0]} ({n[0][1]:+.1f} µV); '
                              f'most positive: {p[0][0]} ({p[0][1]:+.1f} µV) — '
                              'frontocentral negativity = automatic mismatch detection.'),
        'P3a': lambda p, n: (f'Most positive: {p[0][0]} ({p[0][1]:+.1f} µV), '
                              f'{p[1][0]} ({p[1][1]:+.1f} µV) — '
                              'central positivity consistent with automatic orienting.'),
        'P3b': lambda p, n: _p3b_interp(p, n),
    }

    topo_summary_text = {}
    for name, ts in topo_summary.items():
        try:
            topo_summary_text[name] = interp_map[name](ts['pos'], ts['neg'])
        except Exception:
            topo_summary_text[name] = ''

    # Per-component null distributions — one full-size figure each
    legacy_null = out_dir / f'{subject_id}_p300_null.png'
    if legacy_null.exists():
        legacy_null.unlink()

    bonferroni = 0.05 / 4  # four Fischer components: N1, MMN, P3a, P3b
    for name, res in perm_results.items():
        null_arr  = res['null']
        if null_arr is None:
            continue  # no null cache — skip histogram
        sign      = res['comp']['sign']
        obs       = res['obs']
        p         = res['p']
        ch        = res['ch']
        pct_val   = np.percentile(null_arr, 5 if sign < 0 else 95)
        pct_label = '5th percentile (p=0.05)' if sign < 0 else '95th percentile (p=0.05)'
        if name == 'FN':
            xlabel = 'Parietal minus frontal contrast (µV)'
        elif res['comp'].get('source') == 'standard':
            xlabel = 'Standard-evoked amplitude (µV)'
        else:
            xlabel = 'Rare − Standard amplitude (µV)'
        thresh_str = ('supporting test, p < 0.05' if name == 'FN'
                      else f'Bonferroni p < {bonferroni:.4f}')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(null_arr, bins=50, color='steelblue', alpha=0.7,
                label='Null distribution (1,000 shuffles)')
        ax.axvline(obs,     color='firebrick', lw=2,
                   label=f'Observed: {obs:+.2f} µV  (p = {p:.3f})')
        ax.axvline(pct_val, color='k',         lw=1.5, ls='--', label=pct_label)
        ax.set(xlabel=xlabel, ylabel='Count')
        ax.set_title(
            f'{subject_id}: {name} at {ch}  ({thresh_str})',
            fontsize=11
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f'{subject_id}_oddball_null_{name.lower()}.png', dpi=150)
        plt.close(fig)

    # SVM null histogram
    if svm_result is not None and svm_result.get('null') is not None:
        pct95 = np.percentile(svm_result['null'], 95)
        # Accuracy is discrete (multiples of 1/n_epochs); use ~20 bins so bar
        # width >= discrete step, avoiding the narrow-bar comb pattern.
        null_arr   = svm_result['null']
        n_bins_svm = min(20, max(5, len(np.unique(np.round(null_arr, 4)))))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(null_arr, bins=n_bins_svm, color='steelblue', alpha=0.7,
                label='Null distribution (500 label shuffles)')
        ax.axvline(svm_result['acc'], color='firebrick', lw=2,
                   label=f'Observed: {svm_result["acc"]:.3f}  (p = {svm_result["p"]:.3f})')
        ax.axvline(pct95, color='k', lw=1.5, ls='--', label='95th percentile (p=0.05)')
        ax.set(xlabel='LOO classification accuracy', ylabel='Count')
        ax.set_title(f'{subject_id}: Single-trial SVM accuracy (supporting test, p < 0.05)',
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / f'{subject_id}_oddball_svm_null.png', dpi=150)
        plt.close(fig)

    # Topomap — 2x3 grid for larger, clearer heads
    montage = mne.channels.make_standard_montage('standard_1020')
    diff_evoked.set_montage(montage, match_case=False, on_missing='warn')
    times_topo = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    data_topo  = diff_evoked.data * 1e6
    vmax_topo  = np.percentile(np.abs(data_topo), 99)
    fig_topo, axes_topo = plt.subplots(2, 3, figsize=(12, 7))
    axes_topo = axes_topo.flatten()
    im_topo = None
    for ax_t, t in zip(axes_topo, times_topo):
        idx_t = np.argmin(np.abs(diff_evoked.times - t))
        result = mne.viz.plot_topomap(
            data_topo[:, idx_t], diff_evoked.info, axes=ax_t, show=False,
            cmap='RdBu_r', vlim=(-vmax_topo, vmax_topo), extrapolate='head',
        )
        im_topo = result[0] if isinstance(result, (tuple, list)) else result
        ax_t.set_xlabel(f'{int(t * 1000)} ms', fontsize=11)
    fig_topo.suptitle(f'{subject_id}: Rare minus Standard topomap', fontsize=12)
    fig_topo.subplots_adjust(top=0.91, bottom=0.08, left=0.02, right=0.87, hspace=0.35, wspace=0.3)
    if im_topo is not None:
        cbar_ax = fig_topo.add_axes([0.90, 0.15, 0.02, 0.65])
        fig_topo.colorbar(im_topo, cax=cbar_ax, label='µV')
    fig_topo.savefig(out_dir / f'{subject_id}_p300_topomap.png', dpi=150, bbox_inches='tight')
    plt.close(fig_topo)

    # SVM Haufe spatial patterns — train once on all data to extract stable weights
    # diff_evoked.info already has montage positions set from the block above.
    if svm_result is not None:
        sc_all  = StandardScaler()
        X_all   = sc_all.fit_transform(X_svm)
        clf_all = LinearSVC(max_iter=1000, random_state=0, class_weight='balanced')
        clf_all.fit(X_all, y_svm)
        w = clf_all.coef_[0]  # (n_features,)

        # Haufe transform: A = Cov(X) @ w / (w^T @ Cov(X) @ w)
        # Converts SVM weight vector to interpretable spatial pattern (Haufe et al. 2014)
        cov_X = np.cov(X_all.T)    # (n_feat, n_feat)
        Cw    = cov_X @ w
        denom = float(w @ Cw)
        A     = Cw / denom if abs(denom) > 1e-30 else Cw

        n_chs_svm  = svm_data.shape[1]
        n_ds_times = svm_data.shape[2]
        A_map      = A.reshape(n_chs_svm, n_ds_times)  # (n_ch, n_ds_times)
        t_svm      = t_ep[svm_mask][::ds_factor]

        haufe_wins = [
            ('N1',  0.050, 0.100),
            ('MMN', 0.100, 0.200),
            ('P3a', 0.200, 0.300),
            ('P3b', 0.300, 0.600),
        ]
        window_patterns = [
            A_map[:, (t_svm >= lo) & (t_svm <= hi)].mean(axis=1)
            for _, lo, hi in haufe_wins
            if ((t_svm >= lo) & (t_svm <= hi)).sum() > 0
        ]
        vmax_h = max(np.abs(p).max() for p in window_patterns) if window_patterns else 1.0

        fig_h, axes_h = plt.subplots(1, 4, figsize=(14, 5))
        for ax_h, (win_name, t_lo, t_hi) in zip(axes_h, haufe_wins):
            mask_w = (t_svm >= t_lo) & (t_svm <= t_hi)
            if mask_w.sum() == 0:
                ax_h.axis('off')
                ax_h.set_xlabel(f'{win_name}\n({int(t_lo*1000)}–{int(t_hi*1000)} ms)', fontsize=10)
                continue
            pattern = A_map[:, mask_w].mean(axis=1)
            mne.viz.plot_topomap(
                pattern, diff_evoked.info, axes=ax_h, show=False,
                cmap='RdBu_r', vlim=(-vmax_h, vmax_h), extrapolate='head',
            )
            ax_h.set_xlabel(f'{win_name}\n({int(t_lo*1000)}–{int(t_hi*1000)} ms)', fontsize=10)
        fig_h.suptitle(
            f'{subject_id}: SVM Haufe Spatial Patterns\n'
            f'LOO acc = {svm_result["acc"]:.3f}   p = {svm_result["p"]:.3f}',
            fontsize=11, y=0.98,
        )
        fig_h.subplots_adjust(top=0.82, bottom=0.12, left=0.02, right=0.98, wspace=0.35)
        fig_h.savefig(out_dir / f'{subject_id}_oddball_svm_haufe.png', dpi=150, bbox_inches='tight')
        plt.close(fig_h)

    # Johnsen band-power reactivity
    FREQ_BANDS = {'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 30)}
    ref_epochs_j = mne.Epochs(
        raw_p300, events=all_events, event_id={'standard': 1, 'rare': 2},
        tmin=-2.0, tmax=-0.05, baseline=None, preload=True, verbose=False,
    )
    act_epochs_j = mne.Epochs(
        raw_p300, events=all_events, event_id={'standard': 1, 'rare': 2},
        tmin=0.0, tmax=2.0, baseline=None, preload=True, verbose=False,
    )
    n_fft_j = min(int(sfreq * 2), ref_epochs_j.get_data().shape[-1])
    ref_psd_j = ref_epochs_j.compute_psd(method='welch', n_fft=n_fft_j, n_overlap=n_fft_j // 2,
                                          fmin=1, fmax=30, verbose=False)
    act_psd_j = act_epochs_j.compute_psd(method='welch', n_fft=n_fft_j, n_overlap=n_fft_j // 2,
                                          fmin=1, fmax=30, verbose=False)

    def log_band_power(psd_obj, bands):
        data, freqs = psd_obj.get_data(return_freqs=True)
        result = {}
        for band, (flo, fhi) in bands.items():
            idx = np.where((freqs >= flo) & (freqs <= fhi))[0]
            result[band] = np.log(data[:, :, idx].mean(axis=(1, 2)) + 1e-30)
        return result

    ref_log_j = log_band_power(ref_psd_j, FREQ_BANDS)
    act_log_j = log_band_power(act_psd_j, FREQ_BANDS)
    z_scores_j = {
        band: (act_log_j[band] - ref_log_j[band].mean()) / (ref_log_j[band].std() + 1e-30)
        for band in FREQ_BANDS
    }

    colors = ['steelblue', 'darkorange', 'forestgreen', 'mediumpurple']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    for ax, (band, zs), color in zip(axes, z_scores_j.items(), colors):
        flo, fhi = FREQ_BANDS[band]
        t_idx = np.arange(len(zs))
        ax.plot(t_idx, zs, color=color, marker='o', ms=4, lw=1.2)
        ax.axhline( 1.96, color='green', ls='--', lw=0.9, label='+1.96 (p≈0.05)')
        ax.axhline(-1.96, color='red',   ls='--', lw=0.9, label='-1.96 (p≈0.05)')
        ax.axhline(0, color='k', lw=0.5)
        sig_up   = zs >  1.96
        sig_down = zs < -1.96
        if sig_up.any():   ax.scatter(t_idx[sig_up],   zs[sig_up],   color='green', zorder=5, s=50)
        if sig_down.any(): ax.scatter(t_idx[sig_down], zs[sig_down], color='red',   zorder=5, s=50)
        ax.set(title=f'{band.capitalize()}  ({flo}–{fhi} Hz)',
               xlabel='Epoch index', ylabel='Z-score')
        ax.grid(True, alpha=0.3)
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f'{subject_id}: Band-Power Reactivity (post vs pre-stimulus)', fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_dir / f'{subject_id}_oddball_johnsen_reactivity.png', dpi=150)
    plt.close(fig)

    # Write per-patient metadata sidecar for the report generator
    metadata = {
        'n_rare_pre_rejection':  n_rare_pre,
        'n_std_pre_rejection':   n_std_pre,
        'n_rare_post_rejection': n_rare_post,
        'n_std_post_rejection':  n_std_post,
        'n_rare_rejected':       n_rare_pre - n_rare_post,
        'n_std_rejected':        n_std_pre  - n_std_post,
        'rejection_threshold_uv': ODDBALL_REJECT_UV,
        'highpass_hz':           0.1,
        'lowpass_hz':            30,
        'reference':             'average',
        'fischer_score':         fischer_score,
        'n_components':          n_components,
        'shao_mmn_positive':     shao_mmn_pos,
        'shao_p3b_positive':     shao_p3b_pos,
        'components': {
            name: {'observed_uv': float(res['obs']), 'p_value': float(res['p'])}
            for name, res in perm_results.items()
            if name in ('N1', 'MMN', 'P3a', 'P3b')
        },
        'fn_result': {
            'observed_uv': float(perm_results['FN']['obs']),
            'p_value':     float(perm_results['FN']['p']),
            'channels':    perm_results['FN']['ch'],
        } if 'FN' in perm_results else None,
        'topo_summary': topo_summary_text,
        'svm_result': {
            'accuracy': svm_result['acc'],
            'p_value':  svm_result['p'],
        } if svm_result is not None else None,
    }
    if not plots_only:
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f'  [oddball] Saved figures to {out_dir}')


# ── Oddball video ───────────────────────────────────────────────────────────────

def run_oddball_video(subject_id: str, raw, sfreq: float, available_eeg: list, df: pd.DataFrame):
    out_dir = RESULTS_DIR / subject_id / 'oddball'
    out_dir.mkdir(parents=True, exist_ok=True)

    result = _build_oddball_evoked(raw, available_eeg, df, tag='oddball-video')
    if result is None:
        print(f'  [oddball-video] No per-beep rows -- skipping.')
        return

    _, _, _, diff_evoked, _ = result

    montage = mne.channels.make_standard_montage('standard_1020')
    diff_evoked.set_montage(montage, match_case=False, on_missing='warn')

    video_path = out_dir / f'{subject_id}_oddball_topomap.mp4'
    print(f'  [oddball-video] Generating topomap animation (20 fps) ...')
    try:
        fig, anim = diff_evoked.animate_topomap(
            times=diff_evoked.times, ch_type='eeg', frame_rate=20,
            time_unit='ms', show=False, blit=False,
        )
        anim.save(str(video_path), writer='ffmpeg', dpi=100)
        plt.close(fig)
        print(f'  [oddball-video] Saved: {video_path}')
    except Exception as e:
        print(f'  [oddball-video] Failed: {e}')
        plt.close('all')


# ── Language ITPC ──────────────────────────────────────────────────────────────

def run_language(subject_id: str, raw, sfreq: float, available_eeg: list, df: pd.DataFrame):
    out_dir = RESULTS_DIR / subject_id / 'language'
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f'  [language] Saved 3 figures to {out_dir}')


# ── Command ERD + SVM ──────────────────────────────────────────────────────────

def _screen_background_eeg(raw, sfreq: float, available_eeg: list) -> dict:
    """Automated pathological background screen on the first 2 minutes of EEG.

    Based on Claassen group medRxiv 2025: burst suppression, voltage suppression,
    and large inter-hemispheric asymmetry predict zero CMD yield regardless of
    how clean the active-task recording looks.

    Returns a dict with scalar features and a 'flags' list (empty = pass).
    """
    screen_dur_s = min(120.0, raw.times[-1])
    data = raw.copy().crop(tmax=screen_dur_s).get_data(picks=available_eeg)

    win_n   = int(0.5 * sfreq)
    n_wins  = data.shape[1] // win_n
    win_rms = np.zeros(n_wins)
    n_supp  = 0

    for w in range(n_wins):
        seg       = data[:, w * win_n:(w + 1) * win_n]
        ch_rms    = np.sqrt((seg ** 2).mean(axis=1))
        win_rms[w] = ch_rms.mean()
        if ch_rms.max() < 5e-6:  # all channels below 5 µV
            n_supp += 1

    suppression_frac = n_supp / max(n_wins, 1)
    bs_score         = win_rms.std() / (win_rms.mean() + 1e-30)  # CV — high when bimodal

    left_chs  = [ch for ch in ['F3', 'C3', 'T3', 'P3'] if ch in available_eeg]
    right_chs = [ch for ch in ['F4', 'C4', 'T4', 'P4'] if ch in available_eeg]
    asym = float('nan')
    if left_chs and right_chs:
        li = [available_eeg.index(c) for c in left_chs]
        ri = [available_eeg.index(c) for c in right_chs]
        rms_l = np.sqrt((data[li] ** 2).mean())
        rms_r = np.sqrt((data[ri] ** 2).mean())
        asym  = abs(rms_l - rms_r) / (rms_l + rms_r + 1e-30)

    flags = []
    if suppression_frac > 0.30:
        flags.append(f'voltage suppression ({100 * suppression_frac:.0f}% of windows <5uV)')
    if bs_score > 1.5:
        flags.append(f'possible burst-suppression (window-RMS CV={bs_score:.2f})')
    if not np.isnan(asym) and asym > 0.40:
        flags.append(f'inter-hemispheric asymmetry ({100 * asym:.0f}%)')

    return {
        'suppression_frac': suppression_frac,
        'bs_score':         bs_score,
        'asymmetry':        asym,
        'flags':            flags,
        'pass':             len(flags) == 0,
    }


_CMD_AUDIO_FALLBACK = {
    'right_keep': 2.712, 'right_stop': 2.904,
    'left_keep':  2.760, 'left_stop':  2.928,
    'prompt':     3.809,
}

def _measure_command_audio_durations() -> dict:
    """Return command audio file durations in seconds via ffprobe.
    Falls back to pre-measured defaults if files are missing or ffprobe fails."""
    import subprocess
    audio_root = REPO_ROOT / 'stimulus_software' / 'audio_data'
    file_map = {
        'right_keep': audio_root / 'static'  / 'right_keep.mp3',
        'right_stop': audio_root / 'static'  / 'right_stop.mp3',
        'left_keep':  audio_root / 'static'  / 'left_keep.mp3',
        'left_stop':  audio_root / 'static'  / 'left_stop.mp3',
        'prompt':     audio_root / 'prompts' / 'motorcommandprompt.wav',
    }
    durations = {}
    for name, path in file_map.items():
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', str(path)],
                capture_output=True, text=True, timeout=10,
            )
            durations[name] = float(result.stdout.strip())
        except Exception:
            durations[name] = _CMD_AUDIO_FALLBACK[name]
            if not path.exists():
                print(f'  [command] WARNING: audio file not found: {path.name}, using fallback {_CMD_AUDIO_FALLBACK[name]:.3f}s')
    return durations


def run_command(subject_id: str, raw, sfreq: float, available_eeg: list, df: pd.DataFrame):
    out_dir = RESULTS_DIR / subject_id / 'command'
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = _screen_background_eeg(raw, sfreq, available_eeg)
    if bg['flags']:
        for flag in bg['flags']:
            print(f'  [command] BACKGROUND WARNING: {flag}')
        print(f'  [command] Pathological background predicts zero CMD yield '
              f'(Claassen 2025 medRxiv) — proceeding but interpret with caution')
    else:
        asym_str = f'{100 * bg["asymmetry"]:.0f}%' if not np.isnan(bg['asymmetry']) else 'n/a'
        print(f'  [command] Background screen: pass  '
              f'(suppression={100 * bg["suppression_frac"]:.0f}%  '
              f'BS-CV={bg["bs_score"]:.2f}  asym={asym_str})')

    raw_erd = load_filtered_eeg(raw, available_eeg, l_freq=1, h_freq=40, verbose=False)

    has_pairs = df['stim_type'].str.match(r'(right|left)_(keep|stop)', na=False).any()
    has_runs  = df['stim_type'].str.contains('command', na=False).any()
    SCHEMA = 'pairs' if has_pairs else 'runs' if has_runs else None
    if SCHEMA is None:
        print(f'  [command] No command rows — skipping.')
        return
    print(f'  [command] Schema: {SCHEMA}')

    # Audio durations needed by both schemas: pairs to offset audio onset → imagery onset,
    # runs to reconstruct individual cycle onsets from the single run-level timestamp.
    audio_dur = _measure_command_audio_durations()

    if SCHEMA == 'pairs':
        keep_df = df[df['stim_type'].str.match(r'(right|left)_keep', na=False)].copy()
        stop_df = df[df['stim_type'].str.match(r'(right|left)_stop', na=False)].copy()
        keep_df['side'] = keep_df['stim_type'].str.extract(r'(right|left)')
        stop_df['side'] = stop_df['stim_type'].str.extract(r'(right|left)')

        # start_sample is the audio onset; imagery window begins after the audio ends.
        keep_audio_offset = np.where(
            keep_df['side'].values == 'right',
            int(audio_dur['right_keep'] * sfreq),
            int(audio_dur['left_keep']  * sfreq),
        )
        stop_audio_offset = np.where(
            stop_df['side'].values == 'right',
            int(audio_dur['right_stop'] * sfreq),
            int(audio_dur['left_stop']  * sfreq),
        )
        keep_events = np.column_stack([
            keep_df['start_sample'].values + keep_audio_offset,
            np.zeros(len(keep_df), dtype=int),
            np.ones(len(keep_df), dtype=int),
        ])
        stop_events = np.column_stack([
            stop_df['start_sample'].values + stop_audio_offset,
            np.zeros(len(stop_df), dtype=int),
            np.full(len(stop_df), 2, dtype=int),
        ])
        keep_meta_df = keep_df[['side']].reset_index(drop=True)
        stop_meta_df = stop_df[['side']].reset_index(drop=True)

    else:
        cmd_df = df[df['stim_type'].str.contains('command', na=False)].copy()
        cmd_df['side']       = cmd_df['stim_type'].str.extract(r'(right|left)')
        cmd_df['has_prompt'] = cmd_df['stim_type'].str.contains(r'\+p', na=False)

        KEEP_PAUSE_S   = 10.0
        STOP_PAUSE_S   = 10.0
        TOTAL_CYCLES   = 8
        PROMPT_DELAY_S = 2.0  # CommandStimParams.PROMPT_DELAY_MS / 1000

        prompt_total_s = audio_dur['prompt'] + PROMPT_DELAY_S
        has_prompt     = cmd_df['has_prompt'].values

        keep_ev_list, stop_ev_list = [], []
        keep_meta, stop_meta = [], []

        for i, (_, run) in enumerate(cmd_df.iterrows()):
            side     = run['side']
            keep_dur = audio_dur[f'{side}_keep']
            stop_dur = audio_dur[f'{side}_stop']

            t = run['edf_start'] + (prompt_total_s if has_prompt[i] else 0)
            for cycle in range(TOTAL_CYCLES):
                # Epoch onset is AFTER the command audio ends (paper: 10s window follows command)
                keep_ev_list.append([int((t + keep_dur) * sfreq), 0, 1])
                keep_meta.append({'side': side, 'cycle': cycle, 'run': i})
                stop_t = t + keep_dur + KEEP_PAUSE_S
                stop_ev_list.append([int((stop_t + stop_dur) * sfreq), 0, 2])
                stop_meta.append({'side': side, 'cycle': cycle, 'run': i})
                t = stop_t + stop_dur + STOP_PAUSE_S

        keep_events  = np.array(keep_ev_list, dtype=int)
        stop_events  = np.array(stop_ev_list, dtype=int)
        keep_meta_df = pd.DataFrame(keep_meta)
        stop_meta_df = pd.DataFrame(stop_meta)

        # Warn if any stimulus_paused row falls inside a command run window.
        # Pauses shift the real cycle onsets but the reconstruction above ignores them.
        pause_df = df[df['stim_type'] == 'stimulus_paused']
        if not pause_df.empty:
            run_dur_est = TOTAL_CYCLES * (KEEP_PAUSE_S + STOP_PAUSE_S) + 30
            for _, run_row in cmd_df.iterrows():
                run_start = run_row['edf_start']
                hits = pause_df[
                    (pause_df['edf_start'] >= run_start) &
                    (pause_df['edf_start'] <= run_start + run_dur_est)
                ]
                if not hits.empty:
                    print(f'  [command] WARNING: {len(hits)} pause(s) detected '
                          f'during {run_row["stim_type"]} run at t={run_start:.1f}s — '
                          f'reconstructed cycle onsets after the pause may be invalid')

    print(f'  [command] keep events: {len(keep_events)}  stop events: {len(stop_events)}')

    MOTOR_CHANNELS = [ch for ch in ['C3', 'Cz', 'C4'] if ch in available_eeg]

    all_cmd_events = np.vstack([keep_events, stop_events])
    all_cmd_events = all_cmd_events[all_cmd_events[:, 0].argsort()]

    epochs_cmd = mne.Epochs(
        raw_erd, events=all_cmd_events,
        event_id={'keep': 1, 'stop': 2},
        tmin=0, tmax=9.9, baseline=None,
        preload=True, verbose=False,
    )
    print(f'  [command] cmd epochs: {len(epochs_cmd)} ({len(epochs_cmd["keep"])} keep, {len(epochs_cmd["stop"])} stop)')

    # ERD PSD — overview averaged across all sides and motor channels (summary visualization)
    keep_ep = epochs_cmd['keep'].copy().pick(MOTOR_CHANNELS)
    stop_ep = epochs_cmd['stop'].copy().pick(MOTOR_CHANNELS)
    psd_k   = keep_ep.compute_psd(method='welch', fmin=1, fmax=40, verbose=False)
    psd_s   = stop_ep.compute_psd(method='welch', fmin=1, fmax=40, verbose=False)
    data_k, freqs_psd = psd_k.get_data(return_freqs=True)
    data_s, _         = psd_s.get_data(return_freqs=True)

    keep_avg = data_k.mean(axis=(0, 1))
    stop_avg = data_s.mean(axis=(0, 1))
    erd_db   = 10 * np.log10(keep_avg / (stop_avg + 1e-30))

    MU_BAND       = (8,  12)
    BETA_BAND     = (14, 30)
    CONTRALATERAL = {'right': 'C3', 'left': 'C4'}

    def band_power_per_ch(data, freqs, fmin, fmax):
        """Band power per epoch per channel. Returns shape (n_epochs, n_ch)."""
        mask = (freqs >= fmin) & (freqs <= fmax)
        return data[:, :, mask].mean(axis=2)

    def cohens_d_paired(a, b):
        diff = a - b
        return diff.mean() / (diff.std() + 1e-30)

    # Per-side, per-channel ERD statistics — correct lateralization test.
    # Right commands -> expected ERD (power decrease) at C3 (contralateral). Left -> C4.
    erd_stats     = {}
    sides_present = [s for s in ['right', 'left']
                     if (keep_meta_df['side'] == s).any() and (stop_meta_df['side'] == s).any()]

    for side in sides_present:
        contra_ch = CONTRALATERAL.get(side)
        ipsi_ch   = 'C4' if side == 'right' else 'C3'
        k_mask    = keep_meta_df['side'].values == side
        s_mask    = stop_meta_df['side'].values == side

        ep_k_s = mne.Epochs(raw_erd, keep_events[k_mask], event_id={'keep': 1},
                              tmin=0, tmax=9.9, baseline=None, preload=True, verbose=False)
        ep_s_s = mne.Epochs(raw_erd, stop_events[s_mask], event_id={'stop': 2},
                              tmin=0, tmax=9.9, baseline=None, preload=True, verbose=False)
        ep_k_s.pick(MOTOR_CHANNELS)
        ep_s_s.pick(MOTOR_CHANNELS)

        dk_s, fk = ep_k_s.compute_psd(method='welch', fmin=1, fmax=40,
                                        verbose=False).get_data(return_freqs=True)
        ds_s, _  = ep_s_s.compute_psd(method='welch', fmin=1, fmax=40,
                                        verbose=False).get_data(return_freqs=True)

        mu_k_ch   = band_power_per_ch(dk_s, fk, *MU_BAND)    # (n_epochs, n_ch)
        mu_s_ch   = band_power_per_ch(ds_s, fk, *MU_BAND)
        beta_k_ch = band_power_per_ch(dk_s, fk, *BETA_BAND)
        beta_s_ch = band_power_per_ch(ds_s, fk, *BETA_BAND)

        erd_stats[side] = {}
        for ci, ch in enumerate(ep_k_s.ch_names):
            t_mu,   p2_mu   = stats.ttest_rel(mu_k_ch[:, ci],   mu_s_ch[:, ci])
            t_beta, p2_beta = stats.ttest_rel(beta_k_ch[:, ci], beta_s_ch[:, ci])
            p_mu   = p2_mu   / 2 if t_mu   < 0 else 1 - p2_mu   / 2
            p_beta = p2_beta / 2 if t_beta < 0 else 1 - p2_beta / 2
            erd_stats[side][ch] = dict(
                mu_p=p_mu,   mu_d=cohens_d_paired(mu_k_ch[:, ci],   mu_s_ch[:, ci]),
                beta_p=p_beta, beta_d=cohens_d_paired(beta_k_ch[:, ci], beta_s_ch[:, ci]),
            )
            tag = ' <- contra' if ch == contra_ch else ''
            st  = erd_stats[side][ch]
            print(f'  [command] {side} {ch}{tag}: '
                  f'Mu p={st["mu_p"]:.4f} d={st["mu_d"]:.3f}  '
                  f'Beta p={st["beta_p"]:.4f} d={st["beta_d"]:.3f}')

        # Lateralization Index: (contra - ipsi) / (|contra| + |ipsi|)
        # +1 = all suppression at contralateral channel (expected for true motor imagery)
        # -1 = all suppression at ipsilateral channel (unexpected / artifact)
        li_mu = li_beta = float('nan')
        if contra_ch in ep_k_s.ch_names and ipsi_ch in ep_k_s.ch_names:
            ci_c = ep_k_s.ch_names.index(contra_ch)
            ci_i = ep_k_s.ch_names.index(ipsi_ch)
            # ERD defined as stop - keep (positive when keep < stop = power suppressed)
            erd_c_mu   = (mu_s_ch[:, ci_c]   - mu_k_ch[:, ci_c]).mean()
            erd_i_mu   = (mu_s_ch[:, ci_i]   - mu_k_ch[:, ci_i]).mean()
            erd_c_beta = (beta_s_ch[:, ci_c] - beta_k_ch[:, ci_c]).mean()
            erd_i_beta = (beta_s_ch[:, ci_i] - beta_k_ch[:, ci_i]).mean()
            li_mu   = (erd_c_mu   - erd_i_mu)   / (abs(erd_c_mu)   + abs(erd_i_mu)   + 1e-30)
            li_beta = (erd_c_beta - erd_i_beta) / (abs(erd_c_beta) + abs(erd_i_beta) + 1e-30)
            erd_stats[side]['LI_mu']   = li_mu
            erd_stats[side]['LI_beta'] = li_beta
            print(f'  [command] {side} LI: Mu={li_mu:+.3f}  Beta={li_beta:+.3f}  '
                  f'({contra_ch} vs {ipsi_ch}; +1=contra dominant)')

        # Lateralization plot: per-channel ERD curves with p-value annotations.
        # Reuses the PSD data already computed above — no redundant epoch creation.
        fig, axes_lat = plt.subplots(1, len(MOTOR_CHANNELS), figsize=(4 * len(MOTOR_CHANNELS), 4))
        if len(MOTOR_CHANNELS) == 1:
            axes_lat = [axes_lat]
        for ax, ch in zip(axes_lat, MOTOR_CHANNELS):
            ci_lat   = ep_k_s.ch_names.index(ch)
            avg_k_ch = dk_s[:, ci_lat, :].mean(axis=0)
            avg_s_ch = ds_s[:, ci_lat, :].mean(axis=0)
            erd_ch   = 10 * np.log10(avg_k_ch / (avg_s_ch + 1e-30))
            ax.plot(fk, erd_ch, color='purple', lw=1.5)
            ax.axhline(0, color='k', lw=0.8, ls='--')
            for f0, f1, c in [(8, 12, 'gold'), (14, 30, 'lightgreen')]:
                ax.axvspan(f0, f1, color=c, alpha=0.2)
            st     = erd_stats[side].get(ch, {})
            ch_tag = ' (contra)' if ch == contra_ch else (' (ipsi)' if ch == ipsi_ch else '')
            ax.set(title=(f'{ch}{ch_tag}\n'
                          f'Mu p={st.get("mu_p", float("nan")):.3f}  '
                          f'Beta p={st.get("beta_p", float("nan")):.3f}'),
                   xlabel='Hz', ylabel='Keep−Stop (dB)')
            ax.grid(True, alpha=0.3)
        li_str = (f'  |  LI Mu={li_mu:+.2f}  Beta={li_beta:+.2f}'
                  if not np.isnan(li_mu) else '')
        fig.suptitle(f'{subject_id}: {side.capitalize()} command ERD by channel{li_str}', fontsize=11)
        plt.tight_layout()
        fig.savefig(out_dir / f'{subject_id}_command_{side}_lateralization.png', dpi=150)
        plt.close(fig)

        # Time-frequency ERD — reveals WHEN and at WHAT frequency suppression emerges.
        # Uses Morlet wavelets on the same side-specific epochs already computed.
        freqs_tfr = np.arange(4, 35, 1).astype(float)
        n_cyc_tfr = np.maximum(freqs_tfr / 2.0, 3.0)
        tfr_k = mne.time_frequency.tfr_morlet(
            ep_k_s, freqs=freqs_tfr, n_cycles=n_cyc_tfr,
            return_itc=False, average=True, verbose=False,
        )
        tfr_s = mne.time_frequency.tfr_morlet(
            ep_s_s, freqs=freqs_tfr, n_cycles=n_cyc_tfr,
            return_itc=False, average=True, verbose=False,
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            erd_tfr = 10 * np.log10(tfr_k.data / (tfr_s.data + 1e-30))  # (n_ch, n_freq, n_t)

        fig_tfr, axes_tfr = plt.subplots(
            1, len(MOTOR_CHANNELS), figsize=(5 * len(MOTOR_CHANNELS), 4),
        )
        if len(MOTOR_CHANNELS) == 1:
            axes_tfr = [axes_tfr]
        for ax_t, ch in zip(axes_tfr, tfr_k.ch_names):
            ci_t = tfr_k.ch_names.index(ch)
            img  = ax_t.imshow(
                erd_tfr[ci_t], aspect='auto', origin='lower',
                extent=[tfr_k.times[0], tfr_k.times[-1], freqs_tfr[0], freqs_tfr[-1]],
                cmap='RdBu_r', vmin=-3, vmax=3,
            )
            for fline, col in [(8, 'gold'), (12, 'gold'), (14, 'lightgreen'), (30, 'lightgreen')]:
                ax_t.axhline(fline, color=col, lw=0.8, ls='--', alpha=0.7)
            ch_tag = ' (contra)' if ch == contra_ch else (' (ipsi)' if ch == ipsi_ch else '')
            ax_t.set(xlabel='Time (s)', ylabel='Frequency (Hz)', title=f'{ch}{ch_tag}')
            plt.colorbar(img, ax=ax_t, label='ERD (dB)')
        fig_tfr.suptitle(
            f'{subject_id}: {side.capitalize()} command Time-Frequency ERD  '
            f'(blue=suppression, red=enhancement)',
            fontsize=10,
        )
        plt.tight_layout()
        fig_tfr.savefig(out_dir / f'{subject_id}_command_{side}_tfr.png', dpi=150)
        plt.close(fig_tfr)

    # ERD overview spectrum (all sides and channels averaged — see lateralization plots for stats)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].semilogy(freqs_psd, keep_avg * 1e12, color='steelblue', lw=1.5, label='Keep (motor imagery)')
    axes[0].semilogy(freqs_psd, stop_avg * 1e12, color='firebrick',  lw=1.5, label='Stop (rest)')
    for f0, f1, c, lbl in [(8, 12, 'gold', 'Mu'), (14, 30, 'lightgreen', 'Beta')]:
        axes[0].axvspan(f0, f1, color=c, alpha=0.2, label=lbl)
    axes[0].set(xlabel='Hz', ylabel='PSD (pV²/Hz)',
                title=f'{subject_id}: Motor channels ({MOTOR_CHANNELS}), Keep vs Stop PSD (all sides averaged)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freqs_psd, erd_db, color='purple', lw=1.5)
    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    for f0, f1, c, lbl in [(8, 12, 'gold', 'Mu'), (14, 30, 'lightgreen', 'Beta')]:
        axes[1].axvspan(f0, f1, color=c, alpha=0.2, label=lbl)
    axes[1].set(xlabel='Hz', ylabel='Keep − Stop (dB)',
                title='ERD overview — see lateralization plots for per-channel stats')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_command_erd.png', dpi=150)
    plt.close(fig)

    # Claassen SVM
    SUB_EPOCH_DUR  = 2.0
    N_SUB          = 5
    sub_events_list, sub_labels, sub_groups = [], [], []
    n_pairs = min(len(keep_events), len(stop_events))
    if len(keep_events) != len(stop_events):
        print(f'  [command] WARNING: keep/stop count mismatch '
              f'({len(keep_events)} keep vs {len(stop_events)} stop) — using {n_pairs} pairs')

    for pair_idx in range(n_pairs):
        k_sample = keep_events[pair_idx, 0]
        s_sample = stop_events[pair_idx, 0]
        for sub in range(N_SUB):
            offset = int(sub * SUB_EPOCH_DUR * sfreq)
            sub_events_list.append([k_sample + offset, 0, 1])
            sub_labels.append(1)
            sub_groups.append(pair_idx)
            sub_events_list.append([s_sample + offset, 0, 2])
            sub_labels.append(0)
            sub_groups.append(pair_idx)

    sub_events_arr = np.array(sub_events_list, dtype=int)
    sub_labels     = np.array(sub_labels)
    sub_groups     = np.array(sub_groups)

    sub_epochs = mne.Epochs(
        raw_erd, events=sub_events_arr,
        event_id={'keep': 1, 'stop': 2},
        tmin=0, tmax=SUB_EPOCH_DUR - 1 / sfreq,
        baseline=None, preload=True, verbose=False,
    ).pick(available_eeg)

    if len(sub_epochs) != len(sub_events_arr):
        n_oob = len(sub_events_arr) - len(sub_epochs)
        print(f'  [command] WARNING: {n_oob} sub-epochs out of bounds — '
              f'realigning sub_labels/sub_groups to selection')
        sub_labels = sub_labels[sub_epochs.selection]
        sub_groups = sub_groups[sub_epochs.selection]

    SVM_BANDS       = [(1, 3), (4, 7), (8, 13), (14, 30)]
    SVM_BAND_LABELS = ['delta', 'theta', 'alpha', 'beta']

    data_sub        = sub_epochs.get_data()
    n_sub_ep, n_ch_svm, _ = data_sub.shape
    psds_sub, psd_freqs_svm = psd_array_multitaper(
        data_sub, sfreq=sfreq, fmin=1, fmax=30, verbose=False
    )

    X_svm = np.zeros((n_sub_ep, n_ch_svm * len(SVM_BANDS)))
    for bi, (flo, fhi) in enumerate(SVM_BANDS):
        freq_idx = np.where((psd_freqs_svm >= flo) & (psd_freqs_svm <= fhi))[0]
        X_svm[:, bi * n_ch_svm:(bi + 1) * n_ch_svm] = psds_sub[:, :, freq_idx].mean(axis=2)

    logo    = LeaveOneGroupOut()
    clf_svm = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, dual='auto'))

    # LOO decision function values → AUC + per-epoch probabilities for time-course plot
    decision_vals = cross_val_predict(
        clf_svm, X_svm, sub_labels,
        method='decision_function', cv=logo, groups=sub_groups,
    )
    mean_auc_svm = roc_auc_score(sub_labels, decision_vals)
    # Sigmoid maps decision function to [0,1] for display
    prob_keep = 1.0 / (1.0 + np.exp(-decision_vals))

    N_PERMS_SVM     = 500
    perm_scores_svm = []
    rng_svm = np.random.default_rng(42)
    for _ in range(N_PERMS_SVM):
        perm_auc = roc_auc_score(
            rng_svm.permutation(sub_labels), decision_vals
        )
        perm_scores_svm.append(perm_auc)

    perm_scores_svm = np.array(perm_scores_svm)
    p_svm = (np.sum(perm_scores_svm >= mean_auc_svm) + 1) / (N_PERMS_SVM + 1)
    print(f'  [command] SVM AUC={mean_auc_svm:.3f}  p={p_svm:.4f}')

    # SVM null distribution plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(perm_scores_svm, bins=30, color='steelblue', alpha=0.7, label='Permutation AUC')
    ax.axvline(mean_auc_svm, color='firebrick', lw=2,
               label=f'Observed AUC: {mean_auc_svm:.3f}  (p={p_svm:.3f})')
    ax.axvline(0.5, color='k', lw=1, ls='--', label='Chance (0.5)')
    ax.set(xlabel='AUC', ylabel='Count',
           title=f'{subject_id}: SVM permutation test')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_command_svm_null.png', dpi=150)
    plt.close(fig)

    # Decoding probability time-course (Claassen Fig. 3 equivalent)
    t_min = np.arange(len(prob_keep)) * SUB_EPOCH_DUR / 60
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_min, prob_keep, color='steelblue', lw=1.0)
    ax.axhline(0.5, color='k', ls='--', lw=0.8, label='Chance (0.5)')
    ax.set(ylim=(0, 1), xlabel='Time (min)',
           ylabel='P("keep moving")',
           title=f'{subject_id}: SVM decoding probability  AUC={mean_auc_svm:.3f}  p={p_svm:.3f}')
    for i in range(n_pairs):
        keep_idx = np.where((sub_groups == i) & (sub_labels == 1))[0]
        stop_idx  = np.where((sub_groups == i) & (sub_labels == 0))[0]
        if len(keep_idx):
            ax.axvline(keep_idx[0] * SUB_EPOCH_DUR / 60, color='green', lw=1.0, alpha=0.7,
                       label='Keep onset' if i == 0 else None)
        if len(stop_idx):
            ax.axvline(stop_idx[0] * SUB_EPOCH_DUR / 60, color='red', lw=1.0, alpha=0.7,
                       label='Stop onset' if i == 0 else None)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_command_decoding.png', dpi=150)
    plt.close(fig)

    # SVM spatial patterns
    clf_patterns = make_pipeline(
        StandardScaler(), LinearModel(LinearSVC(max_iter=10000, dual='auto'))
    )
    clf_patterns.fit(X_svm, sub_labels)
    patterns         = get_coef(clf_patterns, 'patterns_', inverse_transform=True)
    spatial_patterns = patterns.reshape(n_ch_svm, len(SVM_BANDS))

    montage_svm = mne.channels.make_standard_montage('standard_1020')
    sub_epochs.set_montage(montage_svm, match_case=False, on_missing='warn')
    fig, axes = plt.subplots(1, len(SVM_BANDS), figsize=(12, 3))
    for ax, (flo, fhi), lbl, sp_band in zip(axes, SVM_BANDS, SVM_BAND_LABELS, spatial_patterns.T):
        scale = np.percentile(np.abs(sp_band), 99) or 1.0
        im, _ = mne.viz.plot_topomap(sp_band, sub_epochs.info,
                                      vlim=(-scale, scale), cmap='RdBu_r', axes=ax, show=False)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'{lbl}\n{flo}–{fhi} Hz')
    fig.suptitle(f'{subject_id}: SVM Spatial Patterns (Keep vs Stop)', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_command_svm_patterns.png', dpi=150)
    plt.close(fig)

    # Riemannian MDM classifier — operates on covariance matrices (SPD manifold).
    # Consistently outperforms LinearSVC on PSD features on benchmark motor imagery datasets.
    if HAS_PYRIEMANN:
        print('  [command] Running Riemannian MDM classifier...')
        cov_sub   = Covariances(estimator='lwf').fit_transform(data_sub)
        riem_prob = cross_val_predict(
            MDM(metric='riemann'), cov_sub, sub_labels,
            cv=logo, groups=sub_groups, method='predict_proba',
        )[:, 1]  # P(keep)
        mean_auc_riem = roc_auc_score(sub_labels, riem_prob)

        perm_riem = []
        rng_riem  = np.random.default_rng(43)
        for _ in range(N_PERMS_SVM):
            perm_riem.append(roc_auc_score(rng_riem.permutation(sub_labels), riem_prob))
        perm_riem = np.array(perm_riem)
        p_riem    = (np.sum(perm_riem >= mean_auc_riem) + 1) / (N_PERMS_SVM + 1)
        print(f'  [command] Riemannian MDM AUC={mean_auc_riem:.3f}  p={p_riem:.4f}  '
              f'(SVM: {mean_auc_svm:.3f}  p={p_svm:.4f})')

        fig_riem, ax_riem = plt.subplots(figsize=(8, 3))
        ax_riem.hist(perm_riem, bins=30, color='steelblue', alpha=0.7, label='Permutation AUC')
        ax_riem.axvline(mean_auc_riem, color='firebrick', lw=2,
                        label=f'MDM observed: {mean_auc_riem:.3f}  (p={p_riem:.3f})')
        ax_riem.axvline(0.5, color='k', lw=1, ls='--', label='Chance (0.5)')
        ax_riem.set(xlabel='AUC', ylabel='Count',
                    title=f'{subject_id}: Riemannian MDM permutation test')
        ax_riem.legend(fontsize=9)
        ax_riem.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_riem.savefig(out_dir / f'{subject_id}_command_riemannian_null.png', dpi=150)
        plt.close(fig_riem)
    else:
        print('  [command] pyriemann not installed — skipping Riemannian MDM '
              '(pip install pyriemann)')

    print(f'  [command] Saved figures to {out_dir}')


# ── Main ───────────────────────────────────────────────────────────────────────

RUNNERS = {
    'oddball':  run_oddball,
    'language': run_language,
    'command':  run_command,
}


def main():
    parser = argparse.ArgumentParser(description='Batch EEG analysis runner')
    parser.add_argument('--force', action='store_true', help='Re-run even if outputs exist')
    parser.add_argument('--plots-only', action='store_true',
                        help='Regenerate figures only -- skip permutations and SVM (oddball). '
                             'Requires a previous full run to have cached null_arrays.npz.')
    parser.add_argument('--video', action='store_true',
                        help='Generate oddball topographic MP4 animation (20 fps, requires ffmpeg). '
                             'Skips all other analyses.')
    parser.add_argument('--patients', nargs='+', metavar='ID', help='Limit to specific patient IDs')
    parser.add_argument('--analyses', nargs='+', choices=ALL_ANALYSES, metavar='NAME',
                        help='Limit to specific analyses')
    args = parser.parse_args()

    sessions = discover_sessions()
    if args.patients:
        sessions = [s for s in sessions if s['patient_id'] in args.patients]
    sessions = [s for s in sessions if not s['patient_id'].lower().startswith(('jo', 'test'))]

    print(f'Sessions to consider: {[s["patient_id"] for s in sessions]}')

    if args.video:
        for session in sessions:
            pid  = session['patient_id']
            date = session['date']
            video_path = RESULTS_DIR / pid / 'oddball' / f'{pid}_oddball_topomap.mp4'
            if not args.force and video_path.exists():
                print(f'\n{pid}: video already exists -- skipping (use --force to regenerate).')
                continue
            print(f'\n{"="*60}')
            print(f'{pid} ({date}): generating topomap video')
            print(f'{"="*60}')
            try:
                raw, sfreq, available_eeg, df = load_session(session['edf'], session['csv'])
            except Exception as e:
                print(f'  ERROR loading session: {e}')
                continue
            if not has_paradigm(df, 'oddball'):
                print(f'  [oddball-video] No oddball rows in CSV -- skipping.')
                continue
            try:
                run_oddball_video(pid, raw, sfreq, available_eeg, df)
            except Exception as e:
                import traceback
                print(f'  ERROR generating video: {e}')
                traceback.print_exc()
        print('\nDone.')
        return

    target_analyses = args.analyses or ALL_ANALYSES

    for session in sessions:
        pid  = session['patient_id']
        date = session['date']

        work = [a for a in target_analyses
                if (args.force or not has_results(pid, a))]
        if not work:
            print(f'\n{pid} ({date}): all analyses present -- skipping.')
            continue

        print(f'\n{"="*60}')
        print(f'{pid} ({date}): running {work}')
        print(f'{"="*60}')

        try:
            raw, sfreq, available_eeg, df = load_session(session['edf'], session['csv'])
        except Exception as e:
            print(f'  ERROR loading session: {e}')
            continue

        for analysis in work:
            if not has_paradigm(df, analysis):
                print(f'  [{analysis}] Not in CSV -- skipping.')
                continue
            print(f'  Running {analysis}...')
            try:
                if analysis == 'oddball':
                    RUNNERS[analysis](pid, raw, sfreq, available_eeg, df,
                                      plots_only=args.plots_only)
                else:
                    RUNNERS[analysis](pid, raw, sfreq, available_eeg, df)
            except Exception as e:
                import traceback
                print(f'  ERROR in {analysis}: {e}')
                traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
