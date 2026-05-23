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
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

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

def run_oddball(subject_id: str, raw, sfreq: float, available_eeg: list, df: pd.DataFrame,
                plots_only: bool = False):
    out_dir = RESULTS_DIR / subject_id / 'oddball'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0.1 Hz highpass (not 1 Hz) — preserves slow ERP components like MMN (100-200 ms)
    # Average reference — removes bias from the recording reference electrode
    raw_p300 = load_filtered_eeg(raw, available_eeg, l_freq=0.1, h_freq=30, verbose=False)
    raw_p300.set_eeg_reference('average', projection=False, verbose=False)

    odd_df  = df[df['stim_type'].str.startswith('oddball')].copy()
    rare_df = odd_df[odd_df['notes'] == 'rare_tone']
    std_df  = odd_df[odd_df['notes'] == 'standard_tone']

    if rare_df.empty:
        print(f'  [oddball] No per-beep rows — skipping.')
        return

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

    REJECT_THRESHOLD_UV = 200
    epochs = mne.Epochs(
        raw_p300, events=all_events,
        event_id={'standard': 1, 'rare': 2},
        tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
        reject=dict(eeg=REJECT_THRESHOLD_UV * 1e-6),
        preload=True, verbose=False,
    )

    n_rare_post = len(epochs['rare'])
    n_std_post  = len(epochs['standard'])
    n_rare_rej  = n_rare_pre - n_rare_post
    n_std_rej   = n_std_pre  - n_std_post
    print(f'  [oddball] {len(epochs)} epochs ({n_rare_post} rare, {n_std_post} std)'
          f'  —  rejected {n_rare_rej} rare, {n_std_rej} std (>{REJECT_THRESHOLD_UV} µV)')

    evoked_rare = epochs['rare'].average()
    evoked_std  = epochs['standard'].average()
    diff_evoked = mne.combine_evoked([evoked_rare, evoked_std], weights=[1, -1])

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

    # Component labels inside the plot: dotted vertical boundary lines + bold text at the top edge.
    # All in axes/data coords — no extra top margin needed.
    COMP_BRACKET_COLOR = {'N1': '#6A0DAD', 'MMN': '#1565C0', 'P3a': '#2E7D32', 'P3b': '#E65100'}
    for name, comp in COMPONENTS.items():
        lo_ms  = comp['win'][0] * 1000
        hi_ms  = comp['win'][1] * 1000
        color  = COMP_BRACKET_COLOR.get(name, '#555555')
        mid_ms = (lo_ms + hi_ms) / 2
        ax_wave.axvline(lo_ms, color=color, lw=1.0, ls=':', alpha=0.55, zorder=1)
        ax_wave.axvline(hi_ms, color=color, lw=1.0, ls=':', alpha=0.55, zorder=1)
        ax_wave.text(mid_ms, 0.97, name, ha='center', va='top', fontsize=11,
                     color=color, fontweight='bold', transform=ax_wave.transAxes)

    ax_wave.axvline(0, color='k', lw=0.8, ls=':')
    ax_wave.axhline(0, color='k', lw=0.5)
    ax_wave.set_title(f'{subject_id}: Rare minus Standard, all electrodes', fontsize=13)
    ax_wave.set(xlabel='Time (ms)', ylabel='Rare − Standard (µV)')
    ax_wave.legend(loc='lower right', fontsize=10, ncol=2)
    ax_wave.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_oddball_butterfly.png', dpi=150, bbox_inches='tight')
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
        'n_rare_rejected':       n_rare_rej,
        'n_std_rejected':        n_std_rej,
        'rejection_threshold_uv': REJECT_THRESHOLD_UV,
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

def run_command(subject_id: str, raw, sfreq: float, available_eeg: list, df: pd.DataFrame):
    out_dir = RESULTS_DIR / subject_id / 'command'
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_erd = load_filtered_eeg(raw, available_eeg, l_freq=1, h_freq=40, verbose=False)

    has_pairs = df['stim_type'].str.match(r'(right|left)_(keep|stop)', na=False).any()
    has_runs  = df['stim_type'].str.contains('command', na=False).any()
    SCHEMA = 'pairs' if has_pairs else 'runs' if has_runs else None
    if SCHEMA is None:
        print(f'  [command] No command rows — skipping.')
        return
    print(f'  [command] Schema: {SCHEMA}')

    if SCHEMA == 'pairs':
        keep_df = df[df['stim_type'].str.match(r'(right|left)_keep', na=False)].copy()
        stop_df = df[df['stim_type'].str.match(r'(right|left)_stop', na=False)].copy()
        keep_df['side'] = keep_df['stim_type'].str.extract(r'(right|left)')
        stop_df['side'] = stop_df['stim_type'].str.extract(r'(right|left)')

        keep_events = np.column_stack([
            keep_df['start_sample'].values,
            np.zeros(len(keep_df), dtype=int),
            np.ones(len(keep_df), dtype=int),
        ])
        stop_events = np.column_stack([
            stop_df['start_sample'].values,
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
        PROMPT_DUR_EST = 4.0

        run_durs   = (cmd_df['edf_end'] - cmd_df['edf_start']).values
        has_prompt = cmd_df['has_prompt'].values

        keep_ev_list, stop_ev_list = [], []
        keep_meta, stop_meta = [], []

        for i, (_, run) in enumerate(cmd_df.iterrows()):
            effective_dur   = run_durs[i] - (PROMPT_DUR_EST if has_prompt[i] else 0)
            audio_per_cycle = max((effective_dur / TOTAL_CYCLES) - KEEP_PAUSE_S - STOP_PAUSE_S, 1.5)
            keep_audio_est  = audio_per_cycle / 2
            side            = run['side']

            t = run['edf_start'] + (PROMPT_DUR_EST if has_prompt[i] else 0)
            for cycle in range(TOTAL_CYCLES):
                keep_ev_list.append([int(t * sfreq), 0, 1])
                keep_meta.append({'side': side, 'cycle': cycle, 'run': i})
                stop_t = t + keep_audio_est + KEEP_PAUSE_S
                stop_ev_list.append([int(stop_t * sfreq), 0, 2])
                stop_meta.append({'side': side, 'cycle': cycle, 'run': i})
                t = stop_t + keep_audio_est + STOP_PAUSE_S

        keep_events  = np.array(keep_ev_list, dtype=int)
        stop_events  = np.array(stop_ev_list, dtype=int)
        keep_meta_df = pd.DataFrame(keep_meta)
        stop_meta_df = pd.DataFrame(stop_meta)

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

    # ERD PSD
    keep_ep = epochs_cmd['keep'].copy().pick(MOTOR_CHANNELS)
    stop_ep = epochs_cmd['stop'].copy().pick(MOTOR_CHANNELS)
    psd_k   = keep_ep.compute_psd(method='welch', fmin=1, fmax=40, verbose=False)
    psd_s   = stop_ep.compute_psd(method='welch', fmin=1, fmax=40, verbose=False)
    data_k, freqs_psd = psd_k.get_data(return_freqs=True)
    data_s, _          = psd_s.get_data(return_freqs=True)

    keep_avg = data_k.mean(axis=(0, 1))
    stop_avg = data_s.mean(axis=(0, 1))
    erd_db   = 10 * np.log10(keep_avg / (stop_avg + 1e-30))

    MU_BAND   = (8,  12)
    BETA_BAND = (14, 30)

    def band_power(data, freqs, fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        return data[:, :, mask].mean(axis=(1, 2))

    mu_k   = band_power(data_k, freqs_psd, *MU_BAND)
    mu_s   = band_power(data_s, freqs_psd, *MU_BAND)
    beta_k = band_power(data_k, freqs_psd, *BETA_BAND)
    beta_s = band_power(data_s, freqs_psd, *BETA_BAND)

    t_mu,   p_mu_two   = stats.ttest_rel(mu_k,   mu_s)
    t_beta, p_beta_two = stats.ttest_rel(beta_k, beta_s)
    p_mu   = p_mu_two   / 2 if t_mu   < 0 else 1 - p_mu_two   / 2
    p_beta = p_beta_two / 2 if t_beta < 0 else 1 - p_beta_two / 2

    def cohens_d_paired(a, b):
        diff = a - b
        return diff.mean() / (diff.std() + 1e-30)

    d_mu   = cohens_d_paired(mu_k, mu_s)
    d_beta = cohens_d_paired(beta_k, beta_s)
    print(f'  [command] Mu p={p_mu:.4f} d={d_mu:.3f}  Beta p={p_beta:.4f} d={d_beta:.3f}')

    # ERD spectrum + difference plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].semilogy(freqs_psd, keep_avg * 1e12, color='steelblue', lw=1.5, label='Keep (motor imagery)')
    axes[0].semilogy(freqs_psd, stop_avg * 1e12, color='firebrick',  lw=1.5, label='Stop (rest)')
    for f0, f1, c, lbl in [(8, 12, 'gold', 'Mu'), (14, 30, 'lightgreen', 'Beta')]:
        axes[0].axvspan(f0, f1, color=c, alpha=0.2, label=lbl)
    axes[0].set(xlabel='Hz', ylabel='PSD (pV²/Hz)',
                title=f'{subject_id}: Motor channels ({MOTOR_CHANNELS}), Keep vs Stop PSD')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freqs_psd, erd_db, color='purple', lw=1.5)
    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    for f0, f1, c, lbl in [
        (8,  12, 'gold',       f'Mu  p={p_mu:.3f}{"✓" if p_mu < 0.05 else ""}'),
        (14, 30, 'lightgreen', f'Beta  p={p_beta:.3f}{"✓" if p_beta < 0.05 else ""}'),
    ]:
        axes[1].axvspan(f0, f1, color=c, alpha=0.2, label=lbl)
    axes[1].set(xlabel='Hz', ylabel='Keep − Stop (dB)',
                title='ERD: negative = power decrease during Keep (expected for motor imagery)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_command_erd.png', dpi=150)
    plt.close(fig)

    # Lateralization plots
    if 'C3' in available_eeg and 'C4' in available_eeg:
        for side in ['right', 'left']:
            side_keep = keep_meta_df['side'] == side
            side_stop = stop_meta_df['side'] == side
            if side_keep.sum() == 0:
                continue
            ep_k = mne.Epochs(raw_erd, keep_events[side_keep.values], event_id={'keep': 1},
                               tmin=0, tmax=9.9, baseline=None, preload=True, verbose=False)
            ep_s = mne.Epochs(raw_erd, stop_events[side_stop.values], event_id={'stop': 2},
                               tmin=0, tmax=9.9, baseline=None, preload=True, verbose=False)
            contra = 'C3' if side == 'right' else 'C4'
            ipsi   = 'C4' if side == 'right' else 'C3'
            fig, axes = plt.subplots(1, len(MOTOR_CHANNELS), figsize=(4 * len(MOTOR_CHANNELS), 4))
            if len(MOTOR_CHANNELS) == 1:
                axes = [axes]
            for ax, ch in zip(axes, MOTOR_CHANNELS):
                pk   = ep_k.copy().pick([ch]).compute_psd(method='welch', fmin=1, fmax=40, verbose=False)
                ps   = ep_s.copy().pick([ch]).compute_psd(method='welch', fmin=1, fmax=40, verbose=False)
                dk, f = pk.get_data(return_freqs=True)
                ds, _ = ps.get_data(return_freqs=True)
                erd_ch = 10 * np.log10(dk.mean(axis=(0, 1)) / (ds.mean(axis=(0, 1)) + 1e-30))
                ax.plot(f, erd_ch, color='purple', lw=1.5)
                ax.axhline(0, color='k', lw=0.8, ls='--')
                for f0, f1, c in [(8, 12, 'gold'), (14, 30, 'lightgreen')]:
                    ax.axvspan(f0, f1, color=c, alpha=0.2)
                label = ch
                if ch == contra: label += ' (contra ← expected ERD)'
                elif ch == ipsi: label += ' (ipsi)'
                ax.set(title=label, xlabel='Hz', ylabel='Keep−Stop (dB)')
                ax.grid(True, alpha=0.3)
            fig.suptitle(f'{subject_id}: {side.capitalize()} hand command ERD', fontsize=11)
            plt.tight_layout()
            fig.savefig(out_dir / f'{subject_id}_command_{side}_lateralization.png', dpi=150)
            plt.close(fig)

    # Claassen SVM
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
    from mne.time_frequency import psd_array_multitaper
    from mne.decoding import LinearModel, get_coef

    SUB_EPOCH_DUR  = 2.0
    N_SUB          = 5
    sub_events_list, sub_labels, sub_groups = [], [], []
    n_pairs = min(len(keep_events), len(stop_events))

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

    SVM_BANDS       = [(1, 3), (4, 7), (8, 13), (14, 30)]
    SVM_BAND_LABELS = ['delta', 'theta', 'alpha', 'beta']

    data_sub        = sub_epochs.get_data()
    n_sub_ep, n_ch_svm, _ = data_sub.shape
    psds_list = []
    for ep_data in data_sub:
        psds, psd_freqs_svm = psd_array_multitaper(ep_data, sfreq=sfreq, fmin=1, fmax=30, verbose=False)
        psds_list.append(psds)
    psds_sub = np.array(psds_list)

    X_svm = np.zeros((n_sub_ep, n_ch_svm * len(SVM_BANDS)))
    for bi, (flo, fhi) in enumerate(SVM_BANDS):
        freq_idx = np.where((psd_freqs_svm >= flo) & (psd_freqs_svm <= fhi))[0]
        X_svm[:, bi * n_ch_svm:(bi + 1) * n_ch_svm] = psds_sub[:, :, freq_idx].mean(axis=2)

    logo      = LeaveOneGroupOut()
    clf_svm   = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, dual=True))
    auc_scores = cross_val_score(
        clf_svm, X_svm, sub_labels,
        scoring='roc_auc', cv=logo, groups=sub_groups,
    )
    mean_auc_svm = auc_scores.mean()

    N_PERMS_SVM    = 200
    perm_scores_svm = []
    rng_svm = np.random.default_rng(42)
    for _ in range(N_PERMS_SVM):
        perm_auc = cross_val_score(
            clf_svm, X_svm, rng_svm.permutation(sub_labels),
            scoring='roc_auc', cv=logo, groups=sub_groups,
        ).mean()
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

    # SVM spatial patterns
    clf_patterns = make_pipeline(
        StandardScaler(), LinearModel(LinearSVC(max_iter=10000, dual=True))
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
                        help='Regenerate figures only — skip permutations and SVM (oddball). '
                             'Requires a previous full run to have cached null_arrays.npz.')
    parser.add_argument('--patients', nargs='+', metavar='ID', help='Limit to specific patient IDs')
    parser.add_argument('--analyses', nargs='+', choices=ALL_ANALYSES, metavar='NAME',
                        help='Limit to specific analyses')
    args = parser.parse_args()

    target_analyses  = args.analyses or ALL_ANALYSES
    sessions         = discover_sessions()

    if args.patients:
        sessions = [s for s in sessions if s['patient_id'] in args.patients]
    # Exclude test sessions
    sessions = [s for s in sessions if not s['patient_id'].lower().startswith(('jo', 'test'))]

    print(f'Sessions to consider: {[s["patient_id"] for s in sessions]}')

    for session in sessions:
        pid  = session['patient_id']
        date = session['date']

        work = [a for a in target_analyses
                if (args.force or not has_results(pid, a))]
        if not work:
            print(f'\n{pid} ({date}): all analyses present — skipping.')
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
                print(f'  [{analysis}] Not in CSV — skipping.')
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
