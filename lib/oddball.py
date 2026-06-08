"""Oddball P300 ERP analysis — batch helper.

Exports:
    run_oddball        — full oddball pipeline (ERP, permutations, SVM, plots)
    run_oddball_video  — topomap MP4 animation only
    _build_oddball_evoked — shared epoch-build helper
    STATS_SENTINEL     — filename used as run-complete sentinel
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
from scipy import stats
from scipy.signal import hilbert as _hilbert
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from mne.time_frequency import psd_array_multitaper

try:
    from pyriemann.classification import MDM
    from pyriemann.estimation import XdawnCovariances
    from sklearn.pipeline import Pipeline as _Pipeline
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

from autoreject import AutoReject
from antropy import lziv_complexity as _lziv
import joblib

from .io import DEFAULT_EEG_CHANNELS, _crop_to_paradigm
from .preprocessing import load_filtered_eeg

DEFAULT_N_PERMS = 1000
STATS_SENTINEL  = 'metadata.json'


def _build_oddball_evoked(raw, available_eeg, df, out_dir=None,
                          force_rejection=False, tag='oddball'):
    """Filter, epoch, and average oddball EEG. Returns
    (epochs, evoked_rare, evoked_std, diff_evoked, counts, rejection_info) or None.
    counts = (n_rare_pre, n_std_pre, n_rare_post, n_std_post).
    rejection_info = dict describing method, per-channel interpolation counts, etc.

    If out_dir is given: cache the fitted AutoReject model (autoreject_model.pkl) and
    save cleaned epochs (epochs-epo.fif) for the pooled analysis to re-use."""
    raw_p300 = load_filtered_eeg(raw, available_eeg, l_freq=0.1, h_freq=30, verbose=False)
    raw_p300.set_eeg_reference('average', projection=False, verbose=False)

    odd_df  = df[df['stim_type'].str.startswith('oddball')].copy()
    rare_df = odd_df[odd_df['notes'] == 'rare_tone']
    std_df  = odd_df[odd_df['notes'] == 'standard_tone']

    if rare_df.empty:
        return None

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

    # Create epochs without rejection — autoreject or threshold applied below
    epochs_raw = mne.Epochs(
        raw_p300, events=all_events,
        event_id={'standard': 1, 'rare': 2},
        tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
        preload=True, verbose=False,
    )

    n_rare_pre = len(epochs_raw['rare'])
    n_std_pre  = len(epochs_raw['standard'])

    # Montage needed for spherical spline interpolation of bad channels
    _montage = mne.channels.make_standard_montage('standard_1020')
    epochs_raw.set_montage(_montage, match_case=False, on_missing='warn')

    ar_cache = (out_dir / 'autoreject_model.pkl') if out_dir else None
    if ar_cache and ar_cache.exists() and not force_rejection:
        ar = joblib.load(ar_cache)
        epochs, reject_log = ar.transform(epochs_raw, return_log=True)
        print(f'  [{tag}] autoreject: using cached model')
    else:
        ar = AutoReject(n_interpolate=np.array([1, 2, 3]), random_state=42, verbose=False)
        epochs, reject_log = ar.fit_transform(epochs_raw, return_log=True)
        if ar_cache:
            joblib.dump(ar, ar_cache)

    # labels: 0 = good, 1 = channel interpolated but epoch kept, 2 = epoch fully rejected
    labels          = reject_log.labels      # (n_orig_epochs, n_channels)
    bad_epochs_mask = reject_log.bad_epochs  # (n_orig_epochs,) bool

    orig_event_ids  = epochs_raw.events[:, 2]
    n_rare_rejected = int(bad_epochs_mask[orig_event_ids == 2].sum())
    n_std_rejected  = int(bad_epochs_mask[orig_event_ids == 1].sum())

    # Per-channel count of how many times it was interpolated (label == 1) in a kept epoch
    kept_mask = ~bad_epochs_mask
    interp_counts = {
        ch: int((labels[kept_mask, ci] == 1).sum())
        for ci, ch in enumerate(epochs_raw.ch_names)
        if (labels[kept_mask, ci] == 1).sum() > 0
    }
    n_interp_epochs = int(((labels == 1).any(axis=1) & kept_mask).sum())

    rejection_info = {
        'method': 'autoreject',
        'threshold_uv': None,
        'n_interpolated_epochs': n_interp_epochs,
        'n_rare_rejected': n_rare_rejected,
        'n_std_rejected': n_std_rejected,
        'interpolated_channels': interp_counts,
    }
    print(f'  [{tag}] autoreject: kept {len(epochs["rare"])} rare / {len(epochs["standard"])} std  '
          f'({n_rare_rejected} rare / {n_std_rejected} std fully rejected; '
          f'{n_interp_epochs} epochs had channels interpolated)')
    if interp_counts:
        top_ch = sorted(interp_counts.items(), key=lambda x: -x[1])[:5]
        print(f'  [{tag}] most interpolated: {", ".join(f"{c} ({n}x)" for c, n in top_ch)}')

    n_rare_post = len(epochs['rare'])
    n_std_post  = len(epochs['standard'])

    evoked_rare = epochs['rare'].average()
    evoked_std  = epochs['standard'].average()
    diff_evoked = mne.combine_evoked([evoked_rare, evoked_std], weights=[1, -1])

    # Save cleaned epochs for pooled analysis
    if out_dir:
        epochs.save(out_dir / 'epochs-epo.fif', overwrite=True, verbose=False)

    counts = (n_rare_pre, n_std_pre, n_rare_post, n_std_post)
    return epochs, evoked_rare, evoked_std, diff_evoked, counts, rejection_info


def _save_n1_bandwidth_comparison(raw, available_eeg, df, out_dir, subject_id: str):
    """Save a simple N1 bandwidth comparison figure for the oddball session."""
    odd_df = df[df['stim_type'].str.startswith('oddball')].copy()
    rare_df = odd_df[odd_df['notes'] == 'rare_tone']
    std_df = odd_df[odd_df['notes'] == 'standard_tone']
    if rare_df.empty or std_df.empty:
        return
    if 'Cz' not in available_eeg:
        return

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

    bandwidths = [
        ('0.1-30 Hz', 30),
        ('0.1-100 Hz', 100),
        ('0.1-200 Hz', 200),
    ]
    fig, axes = plt.subplots(len(bandwidths), 1, figsize=(10, 12), sharex=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, (band_label, h_freq) in zip(axes, bandwidths):
        filtered = load_filtered_eeg(raw, available_eeg, l_freq=0.1, h_freq=h_freq, verbose=False)
        filtered.set_eeg_reference('average', projection=False, verbose=False)
        epochs_band = mne.Epochs(
            filtered, events=all_events,
            event_id={'standard': 1, 'rare': 2},
            tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
            preload=True, verbose=False,
        )
        evoked_rare = epochs_band['rare'].average()
        evoked_std = epochs_band['standard'].average()
        diff_evoked = mne.combine_evoked([evoked_rare, evoked_std], weights=[1, -1])
        cz_idx = evoked_rare.ch_names.index('Cz')
        rare_uv = evoked_rare.data[cz_idx] * 1e6
        std_uv = evoked_std.data[cz_idx] * 1e6
        diff_uv = diff_evoked.data[cz_idx] * 1e6

        ax.plot(evoked_rare.times * 1000, std_uv, color='steelblue', lw=1.5, label='Standard')
        ax.plot(evoked_rare.times * 1000, rare_uv, color='firebrick', lw=1.5, label='Rare')
        ax.plot(evoked_rare.times * 1000, diff_uv, color='darkgreen', lw=1.5, ls='--', label='Rare - Std')
        ax.axvspan(50, 100, color='#f0b800', alpha=0.25, label='N1 window')
        ax.axvline(0, color='k', lw=0.8, ls=':')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('µV')
        ax.set_title(f'Cz [{band_label}]', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle(
        f'{subject_id}: N1 at Cz — Rare minus Standard across bandwidths',
        fontsize=13
    )
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_oddball_n1_bandwidth.png', dpi=150)
    plt.close(fig)


def _save_p3b_bandwidth_comparison(raw, available_eeg, df, out_dir, subject_id: str):
    """Save a P3b bandwidth comparison figure — Pz across 30/15/10 Hz low-pass settings.

    Validates that COMP_DISPLAY_LP['P3b'] = 10 Hz does not distort the 300-600 ms component.
    """
    odd_df = df[df['stim_type'].str.startswith('oddball')].copy()
    rare_df = odd_df[odd_df['notes'] == 'rare_tone']
    std_df  = odd_df[odd_df['notes'] == 'standard_tone']
    if rare_df.empty or std_df.empty:
        return
    ch = next((c for c in ['Pz', 'P3', 'P4'] if c in available_eeg), None)
    if ch is None:
        return

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

    bandwidths = [
        ('0.1-30 Hz', 30),
        ('0.1-15 Hz', 15),
        ('0.1-10 Hz', 10),
    ]
    fig, axes = plt.subplots(len(bandwidths), 1, figsize=(10, 12), sharex=True)

    for ax, (band_label, h_freq) in zip(axes, bandwidths):
        filtered = load_filtered_eeg(raw, available_eeg, l_freq=0.1, h_freq=h_freq, verbose=False)
        filtered.set_eeg_reference('average', projection=False, verbose=False)
        epochs_band = mne.Epochs(
            filtered, events=all_events,
            event_id={'standard': 1, 'rare': 2},
            tmin=-0.2, tmax=0.8, baseline=(-0.2, 0),
            preload=True, verbose=False,
        )
        evoked_rare = epochs_band['rare'].average()
        evoked_std  = epochs_band['standard'].average()
        ch_idx  = evoked_rare.ch_names.index(ch)
        rare_uv = evoked_rare.data[ch_idx] * 1e6
        std_uv  = evoked_std.data[ch_idx]  * 1e6
        diff_uv = rare_uv - std_uv

        ax.plot(evoked_rare.times * 1000, std_uv,  color='steelblue', lw=1.5, label='Standard')
        ax.plot(evoked_rare.times * 1000, rare_uv, color='firebrick',  lw=1.5, label='Rare')
        ax.plot(evoked_rare.times * 1000, diff_uv, color='darkgreen',  lw=1.5, ls='--', label='Rare - Std')
        ax.axvspan(300, 600, color='#f0b800', alpha=0.25, label='P3b window')
        ax.axvline(0, color='k', lw=0.8, ls=':')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel('µV')
        ax.set_title(f'{ch} [{band_label}]', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle(
        f'{subject_id}: P3b at {ch} — Rare vs Standard across filter bandwidths',
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out_dir / f'{subject_id}_oddball_p3b_bandwidth.png', dpi=150)
    plt.close(fig)


def run_oddball(subject_id: str, raw, sfreq: float, available_eeg: list, df: pd.DataFrame,
                out_dir: Path, *, force: bool = False, plots_only: bool = False,
                n_perms: int = DEFAULT_N_PERMS, force_rejection: bool = False):
    # Crop to oddball events only before filtering — 5-10x faster than full session.
    # pre_s=5 covers Johnsen -2s pre-stimulus; post_s=5 covers +2s Johnsen and +0.8s P300.
    raw = _crop_to_paradigm(raw, df, df['stim_type'].str.startswith('oddball'),
                            pre_s=5.0, post_s=5.0)

    result = _build_oddball_evoked(raw, available_eeg, df,
                                   out_dir=out_dir, force_rejection=force_rejection)
    if result is None:
        print(f'  [oddball] No per-beep rows -- skipping.')
        return
    epochs, evoked_rare, evoked_std, diff_evoked, (n_rare_pre, n_std_pre, n_rare_post, n_std_post), rejection_info = result

    if force or not (out_dir / f'{subject_id}_oddball_n1_bandwidth.png').exists():
        _save_n1_bandwidth_comparison(raw, available_eeg, df, out_dir, subject_id)
    if force or not (out_dir / f'{subject_id}_oddball_p3b_bandwidth.png').exists():
        _save_p3b_bandwidth_comparison(raw, available_eeg, df, out_dir, subject_id)

    COMPONENTS = {
        # N1 uses source='standard': sign-flip test on standard-tone evoked amplitude.
        # N1 is obligatory (both tones drive it equally), so rare-standard difference
        # cancels out — the meaningful test is whether the standard average is negative.
        'N1':  {'win': (0.050, 0.100), 'ch': 'Cz', 'sign': -1, 'label': 'Primary auditory response', 'source': 'standard'},
        'MMN': {'win': (0.100, 0.200), 'ch': 'Fz', 'sign': -1, 'label': 'Automatic mismatch',        'source': 'diff'},
        'P3a': {'win': (0.200, 0.300), 'ch': 'Cz', 'sign': +1, 'label': 'Automatic orienting',       'source': 'diff'},
        'P3b': {'win': (0.300, 0.600), 'ch': 'Pz', 'sign': +1, 'label': 'Conscious updating (P300)', 'source': 'diff'},
    }

    # Per-component bandpass for BOTH statistics and display.
    # The slower the component the wider the gap to alpha (~10 Hz), so the more
    # aggressively we can filter without distorting the peak. This removes noise
    # that is irrelevant to each component and improves amplitude estimate quality.
    # SVM and XDAWN+MDM stay broadband — they use alpha/beta as discriminative features.
    COMP_DISPLAY_LP = {
        'N1':  30,   # ~12 Hz content; alpha is in-band — full bandwidth
        'MMN': 20,   # ~8 Hz content; alpha still passes (both overlap)
        'P3a': 15,   # ~4 Hz content; remove alpha and above
        'P3b': 10,   # ~2.5 Hz content; completely eliminate alpha ripple
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
        N_PERMS         = n_perms
        COMPONENT_SEEDS = {'N1': 39, 'MMN': 40, 'P3a': 41, 'P3b': 42}

        # Pre-filter epoch data to each component's bandwidth.
        # Filtering is linear so filter(average) = average(filter) — stats and
        # display plots are fully consistent. Cache by lowpass value to avoid
        # redundant computation when components share a cutoff.
        from mne.filter import filter_data as _fd
        _filt_cache: dict = {}
        def _comp_epochs(lp):
            if lp not in _filt_cache:
                _filt_cache[lp] = (
                    epochs_data if lp >= 30
                    else _fd(epochs_data, sfreq, l_freq=0.1, h_freq=lp, verbose=False)
                )
            return _filt_cache[lp]

        _std_raw = epochs['standard'].get_data()
        _std_cache: dict = {}
        def _std_epochs(lp):
            if lp not in _std_cache:
                _std_cache[lp] = (
                    _std_raw if lp >= 30
                    else _fd(_std_raw, sfreq, l_freq=0.1, h_freq=lp, verbose=False)
                )
            return _std_cache[lp]

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
            lp       = COMP_DISPLAY_LP[name]

            if source == 'standard':
                std_data  = _std_epochs(lp)
                per_epoch = std_data[:, ch_idx][:, win_mask].mean(axis=1) * 1e6
                obs  = float(per_epoch.mean())
                null = np.array([(rng.choice([-1, 1], size=len(per_epoch)) * per_epoch).mean()
                                 for _ in range(N_PERMS)])
            else:
                cd = _comp_epochs(lp)
                def _amp(data, labs, _ch=ch_idx, _win=win_mask):
                    return (data[labs == 2, _ch][:, _win].mean()
                            - data[labs == 1, _ch][:, _win].mean()) * 1e6
                obs  = _amp(cd, labels)
                null = np.array([_amp(cd, rng.permutation(labels)) for _ in range(N_PERMS)])

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

            # Dipole Index is a P3b-window measure — use P3b's filter (0.1-10 Hz)
            di_data = _comp_epochs(COMP_DISPLAY_LP['P3b'])

            def _di_amp(data, labs, _par=par_idx, _fro=fro_idx, _mask=di_mask):
                par_mean = data[:, _par, :][:, :, _mask].mean(axis=(1, 2)) * 1e6
                fro_mean = data[:, _fro, :][:, :, _mask].mean(axis=(1, 2)) * 1e6
                contrast = par_mean - fro_mean
                return contrast[labs == 2].mean() - contrast[labs == 1].mean()

            fn_obs  = _di_amp(di_data, labels)
            fn_null = np.array([_di_amp(di_data, rng_fn.permutation(labels))
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
        N_SVM_PERMS = max(50, n_perms // 2)
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

        # Apply component-specific display filter to evoked copies only
        lp = COMP_DISPLAY_LP.get(name, 30)
        ev_rare_d = evoked_rare.copy().filter(l_freq=0.1, h_freq=lp, verbose=False)
        ev_std_d  = evoked_std.copy().filter(l_freq=0.1,  h_freq=lp, verbose=False)
        diff_d    = mne.combine_evoked([ev_rare_d, ev_std_d], weights=[1, -1])

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
            idx     = ev_rare_d.ch_names.index(ch_name)
            rare_uv = ev_rare_d.data[idx] * 1e6
            std_uv  = ev_std_d.data[idx]  * 1e6
            diff_uv = rare_uv - std_uv
            ax.plot(times_ms, std_uv,  color='steelblue', lw=1.5, label='Standard', alpha=0.85)
            ax.plot(times_ms, rare_uv, color='firebrick',  lw=1.5, label='Rare',     alpha=0.85)
            ax.plot(times_ms, diff_uv, color='darkgreen',  lw=1.5, ls='--', label='Rare - Std')
            ax.axvspan(win_ms[0], win_ms[1], color=cfg['color'], alpha=cfg['alpha'],
                       label=f'{name} ({int(win_ms[0])}-{int(win_ms[1])} ms)')
            ax.axvline(0, color='k', lw=0.8, ls=':')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel('µV')
            ax.set_title(ch_name, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (ms)')
        p_str = f'p = {p_val:.3f}' if not np.isnan(p_val) else ''
        fig.suptitle(
            f'{subject_id}: {name} ({p_str}, obs = {obs:+.2f} µV)  [0.1–{lp} Hz]',
            fontsize=11
        )
        plt.tight_layout()
        fig.savefig(out_dir / f'{subject_id}_oddball_erp_{name.lower()}.png', dpi=150)
        plt.close(fig)

    # P3b Dipole Index ERP — parietal vs frontal averages on one panel
    # Uses 0.1-10 Hz display filter (same as P3b component — dipole is in the P3b window)
    if 'FN' in perm_results and par_chs and fro_chs:
        par_avail = [ch for ch in par_chs if ch in diff_evoked.ch_names]
        fro_avail = [ch for ch in fro_chs if ch in diff_evoked.ch_names]
        if par_avail and fro_avail:
            diff_fn_d = diff_evoked.copy().filter(l_freq=0.1, h_freq=10, verbose=False)
            par_diff = np.mean(
                [diff_fn_d.data[diff_fn_d.ch_names.index(ch)] for ch in par_avail], axis=0
            ) * 1e6
            fro_diff = np.mean(
                [diff_fn_d.data[diff_fn_d.ch_names.index(ch)] for ch in fro_avail], axis=0
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
                f'{subject_id}: P3b Dipole Index (p = {fn_p_di:.3f}, contrast = {fn_obs_di:+.2f} µV)'
                '  [0.1–10 Hz]',
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

    # All-electrode SG reference — one subplot per channel, SG-15 Hz smoothed.
    # Use this to inspect spatial distribution and identify per-electrode signal quality.
    # This is the only figure that uses SG smoothing; all other plots use component-specific
    # bandpass filters (see COMP_DISPLAY_LP above).
    diff_sg = diff_evoked.copy().savgol_filter(15)
    _n_chs  = len(diff_sg.ch_names)
    _ncols  = 5
    _nrows  = (_n_chs + _ncols - 1) // _ncols
    fig_sg, axes_sg = plt.subplots(_nrows, _ncols,
                                   figsize=(_ncols * 3, _nrows * 2.5),
                                   sharex=True, sharey=True)
    axes_sg_flat = axes_sg.flatten()
    for _i, _ch in enumerate(diff_sg.ch_names):
        _ax = axes_sg_flat[_i]
        _ax.plot(times_ms, diff_sg.data[_i] * 1e6, color='darkgreen', lw=1.2)
        _ax.axvline(0, color='k', lw=0.7, ls=':')
        _ax.axhline(0, color='k', lw=0.5)
        for _cn, _comp in COMPONENTS.items():
            _c = COMP_BRACKET_COLOR.get(_cn, '#888')
            _ax.axvspan(_comp['win'][0]*1000, _comp['win'][1]*1000, color=_c, alpha=0.07)
        _ax.set_title(_ch, fontsize=8, pad=2)
        _ax.grid(True, alpha=0.2)
    for _j in range(_n_chs, len(axes_sg_flat)):
        axes_sg_flat[_j].set_visible(False)
    fig_sg.suptitle(
        f'{subject_id}: Rare minus Standard — all electrodes  [SG-15 Hz reference]',
        fontsize=11
    )
    plt.tight_layout()
    fig_sg.savefig(out_dir / f'{subject_id}_oddball_allch_sg.png', dpi=150)
    plt.close(fig_sg)

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

    # ── Induced gamma power (30–80 Hz, 200–600 ms) ───────────────────────────────
    # Non-phase-locked signal invisible to ERP averaging and to Johnsen (stops at 30 Hz).
    # Global ignition theory predicts a gamma burst accompanying conscious P3b.
    gamma_result = None
    try:
        raw_gamma = load_filtered_eeg(raw, available_eeg, l_freq=30, h_freq=80, verbose=False)
        raw_gamma.set_eeg_reference('average', projection=False, verbose=False)
        gamma_epochs = mne.Epochs(
            raw_gamma, events=epochs.events, event_id={'standard': 1, 'rare': 2},
            tmin=-0.2, tmax=0.8, baseline=None, preload=True, verbose=False,
        )
        gamma_data   = gamma_epochs.get_data()             # (n_ep, n_ch, n_t)
        t_gam        = gamma_epochs.times
        gam_win      = (t_gam >= 0.2) & (t_gam <= 0.6)
        envelope     = np.abs(_hilbert(gamma_data, axis=2))
        # Mean envelope per epoch across all channels (scalar per epoch)
        gam_power    = envelope[:, :, gam_win].mean(axis=(1, 2))  # (n_ep,)
        gam_labels   = gamma_epochs.events[:, 2]
        gam_rare     = gam_power[gam_labels == 2]
        gam_std      = gam_power[gam_labels == 1]
        obs_gam      = float(gam_rare.mean() - gam_std.mean())
        rng_gam      = np.random.default_rng(50)
        pool         = np.concatenate([gam_rare, gam_std])
        nr           = len(gam_rare)
        null_gam     = np.array([
            rng_gam.permutation(pool)[:nr].mean() - rng_gam.permutation(pool)[nr:].mean()
            for _ in range(N_PERMS if not plots_only else 1)
        ]) if not plots_only else np.array([0.0])
        p_gam        = float(np.mean(null_gam >= obs_gam))
        gamma_result = {'obs_uv': obs_gam * 1e6, 'p_value': p_gam,
                        'n_rare': int(nr), 'n_std': int(len(gam_std))}
        print(f'  [oddball] Induced gamma: obs={obs_gam*1e6:+.3f} µV·envelope  p={p_gam:.3f}')

        # Figure: box plot + permutation null
        fig_gam, (ax_box, ax_null) = plt.subplots(1, 2, figsize=(10, 4))
        ax_box.boxplot([gam_rare * 1e6, gam_std * 1e6],
                       tick_labels=['Rare', 'Standard'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#f0b800', alpha=0.7),
                       medianprops=dict(color='k', lw=2), showfliers=False)
        ax_box.set_ylabel('Mean envelope (µV)', fontsize=9)
        ax_box.set_title('Gamma power by condition', fontsize=10)
        ax_box.grid(True, alpha=0.3, axis='y')
        pct95_gam = np.percentile(null_gam, 95)
        ax_null.hist(null_gam * 1e6, bins=40, color='steelblue', alpha=0.7,
                     label=f'Null ({n_perms} shuffles)')
        ax_null.axvline(obs_gam * 1e6, color='firebrick', lw=2,
                        label=f'Obs: {obs_gam*1e6:+.3f} µV  p={p_gam:.3f}')
        ax_null.axvline(pct95_gam * 1e6, color='k', lw=1.5, ls='--',
                        label='95th pct (p=0.05)')
        ax_null.set(xlabel='Rare minus standard (µV)', ylabel='Count')
        ax_null.legend(fontsize=8)
        ax_null.grid(True, alpha=0.3)
        fig_gam.suptitle(
            f'{subject_id}: Induced Gamma Power (30–80 Hz, 200–600 ms post-stimulus)',
            fontsize=11)
        plt.tight_layout()
        fig_gam.savefig(out_dir / f'{subject_id}_oddball_gamma.png', dpi=150)
        plt.close(fig_gam)
    except Exception as _e:
        print(f'  [oddball] Induced gamma skipped: {_e}')

    # ── Pre-stimulus alpha correlation ────────────────────────────────────────────
    # Per rare-tone epoch: alpha power (8–12 Hz, –200 to 0 ms) correlated with
    # P3b amplitude at Pz (300–600 ms). Present in MCS, absent in UWS.
    alpha_corr_result = None
    try:
        rare_ep_data = epochs['rare'].get_data()  # (n_rare, n_ch, n_t)
        t_ep2        = epochs.times
        bl_mask_a    = (t_ep2 >= -0.2) & (t_ep2 < 0.0)
        p3b_mask_a   = (t_ep2 >= 0.3)  & (t_ep2 <= 0.6)
        cz_ch = 'Cz' if 'Cz' in epochs.ch_names else None
        pz_ch = 'Pz' if 'Pz' in epochs.ch_names else None
        if cz_ch and pz_ch and len(rare_ep_data) >= 4:
            cz_idx = epochs.ch_names.index(cz_ch)
            pz_idx = epochs.ch_names.index(pz_ch)
            # Compute alpha power via multitaper PSD on the baseline window.
            # Avoids filter-length issues with short (~100-sample) baseline segments.
            bl_data = rare_ep_data[:, cz_idx, :][:, bl_mask_a]  # (n_rare, n_bl)
            psd_bl, _ = psd_array_multitaper(
                bl_data, sfreq=sfreq, fmin=8, fmax=12, verbose=False,
            )
            pre_alpha = psd_bl.mean(axis=1)   # (n_rare,) mean alpha power per epoch
            p3b_amp      = rare_ep_data[:, pz_idx, :][:, p3b_mask_a].mean(axis=1) * 1e6
            from scipy.stats import pearsonr as _pearsonr
            r_a, p_a = _pearsonr(pre_alpha, p3b_amp)
            alpha_corr_result = {'r': float(r_a), 'p_value': float(p_a),
                                  'n': int(len(pre_alpha))}
            print(f'  [oddball] Pre-stim alpha–P3b corr: r={r_a:+.3f}  p={p_a:.3f}'
                  f'  n={len(pre_alpha)}')

            # Figure: scatter + trend line
            fig_ac, ax_ac = plt.subplots(figsize=(6, 5))
            ax_ac.scatter(pre_alpha * 1e6, p3b_amp, color='#5b9bd5', alpha=0.7, s=60, zorder=3)
            if len(pre_alpha) >= 3:
                _m, _b = np.polyfit(pre_alpha * 1e6, p3b_amp, 1)
                _x = np.linspace((pre_alpha * 1e6).min(), (pre_alpha * 1e6).max(), 100)
                ax_ac.plot(_x, _m * _x + _b, color='firebrick', lw=2, zorder=4)
            ax_ac.set(xlabel='Pre-stimulus alpha envelope at Cz (µV)',
                      ylabel='P3b amplitude at Pz (µV)')
            ax_ac.set_title(
                f'{subject_id}: Pre-stimulus Alpha vs P3b  '
                f'(r={r_a:+.3f}, p={p_a:.3f}, n={len(pre_alpha)})',
                fontsize=10)
            ax_ac.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_ac.savefig(out_dir / f'{subject_id}_oddball_alpha_corr.png', dpi=150)
            plt.close(fig_ac)
    except Exception as _e:
        print(f'  [oddball] Alpha correlation skipped: {_e}')

    # ── XDAWN + Riemannian MDM ────────────────────────────────────────────────────
    # P300-specific spatial filtering + Riemannian covariance classifier.
    # Designed for imbalanced rare/standard ratios; more stable than SVM at small N.
    xdawn_result = None
    if HAS_PYRIEMANN and not plots_only:
        try:
            N_XDAWN_PERMS = max(20, n_perms // 10)
            xdawn_pipe = _Pipeline([
                ('xdawn', XdawnCovariances(nfilter=3, estimator='lwf')),
                ('mdm',   MDM(metric='riemann')),
            ])
            loo_xd  = LeaveOneOut()
            xdawn_preds = cross_val_predict(
                xdawn_pipe, epochs_data, (labels == 2).astype(int), cv=loo_xd,
            )
            xdawn_acc = float(np.mean(xdawn_preds == (labels == 2).astype(int)))

            rng_xd  = np.random.default_rng(51)
            y_xd    = (labels == 2).astype(int)
            null_xd = []
            for _ in range(N_XDAWN_PERMS):
                y_p = rng_xd.permutation(y_xd)
                null_xd.append(float(np.mean(
                    cross_val_predict(xdawn_pipe, epochs_data, y_p, cv=loo_xd) == y_p
                )))
            null_xd  = np.array(null_xd)
            p_xd     = float((np.sum(null_xd >= xdawn_acc) + 1) / (N_XDAWN_PERMS + 1))
            xdawn_result = {'accuracy': xdawn_acc, 'p_value': p_xd}
            print(f'  [oddball] XDAWN+MDM LOO acc={xdawn_acc:.3f}  p={p_xd:.3f}  '
                  f'(SVM: {svm_result["acc"]:.3f})')

            # Figure: null distribution
            fig_xd, ax_xd = plt.subplots(figsize=(10, 5))
            ax_xd.hist(null_xd, bins=max(5, N_XDAWN_PERMS // 5),
                       color='steelblue', alpha=0.7,
                       label=f'Null ({N_XDAWN_PERMS} shuffles)')
            ax_xd.axvline(xdawn_acc, color='firebrick', lw=2,
                          label=f'Observed: {xdawn_acc:.3f}  (p={p_xd:.3f})')
            ax_xd.axvline(np.percentile(null_xd, 95), color='k', lw=1.5, ls='--',
                          label='95th pct (p=0.05)')
            ax_xd.set(xlabel='LOO accuracy', ylabel='Count')
            ax_xd.set_title(
                f'{subject_id}: XDAWN + Riemannian MDM  '
                f'(supporting test, p < 0.05)',
                fontsize=11)
            ax_xd.legend(fontsize=9)
            ax_xd.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_xd.savefig(out_dir / f'{subject_id}_oddball_xdawn_null.png', dpi=150)
            plt.close(fig_xd)
        except Exception as _e:
            print(f'  [oddball] XDAWN+MDM skipped: {_e}')

    # ── Delta-gamma phase-amplitude coupling (PAC) ───────────────────────────────
    # Tort et al. (2010) KL-based Modulation Index: does delta (1-4 Hz) phase at Pz
    # organize gamma (30-80 Hz) amplitude at Pz during 200-600 ms post-stimulus?
    # MCS patients show significantly higher coupling than UWS (delta-gamma PAC
    # literature on standard 2-tone oddball data). Permutation test on rare vs
    # standard MI difference, mirroring the gamma/alpha-corr blocks above.
    # Skipped during --plots-only: Hilbert-filtering the continuous recording is
    # not cheap and this is a supporting test, like XDAWN+MDM above.
    pac_result = None
    if not plots_only:
        try:
            pz_pac = 'Pz' if 'Pz' in epochs.ch_names else None
            if pz_pac and len(epochs['rare']) >= 4 and len(epochs['standard']) >= 4:
                # Hilbert transform on the continuous (unepoched) signal — avoids
                # the too-few-cycles problem of filtering 400 ms epochs at 1-4 Hz.
                raw_delta = load_filtered_eeg(raw, available_eeg, l_freq=1, h_freq=4, verbose=False)
                raw_delta.set_eeg_reference('average', projection=False, verbose=False)
                raw_gamma_pac = load_filtered_eeg(raw, available_eeg, l_freq=30, h_freq=80, verbose=False)
                raw_gamma_pac.set_eeg_reference('average', projection=False, verbose=False)

                phase_full = np.angle(_hilbert(raw_delta.get_data(picks=[pz_pac])[0]))
                ampl_full  = np.abs(_hilbert(raw_gamma_pac.get_data(picks=[pz_pac])[0]))
                n_samples_pac = len(phase_full)

                # epochs.events[:, 0] holds ABSOLUTE sample numbers (relative to the
                # original, uncropped recording — the MNE convention so events still
                # line up after raw.crop()). raw_delta/raw_gamma_pac are derived from
                # the cropped raw, so subtract first_samp to index into their arrays.
                first_samp_pac = raw_delta.first_samp
                win_lo_pac, win_hi_pac = int(round(0.2 * sfreq)), int(round(0.6 * sfreq))
                N_BINS_PAC = 18
                bin_edges_pac = np.linspace(-np.pi, np.pi, N_BINS_PAC + 1)

                def _modulation_index(samples):
                    ph_chunks, am_chunks = [], []
                    for samp in samples:
                        rel = int(samp) - first_samp_pac
                        lo, hi = rel + win_lo_pac, rel + win_hi_pac
                        if 0 <= lo and hi <= n_samples_pac:
                            ph_chunks.append(phase_full[lo:hi])
                            am_chunks.append(ampl_full[lo:hi])
                    if not ph_chunks:
                        return float('nan')
                    ph = np.concatenate(ph_chunks)
                    am = np.concatenate(am_chunks)
                    bin_idx  = np.clip(np.digitize(ph, bin_edges_pac) - 1, 0, N_BINS_PAC - 1)
                    sums     = np.bincount(bin_idx, weights=am, minlength=N_BINS_PAC)
                    counts   = np.bincount(bin_idx, minlength=N_BINS_PAC)
                    mean_amp = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
                    p_dist   = np.clip(mean_amp / mean_amp.sum(), 1e-12, None)
                    h        = -np.sum(p_dist * np.log(p_dist))
                    return float((np.log(N_BINS_PAC) - h) / np.log(N_BINS_PAC))

                rare_samples_pac = epochs['rare'].events[:, 0]
                std_samples_pac  = epochs['standard'].events[:, 0]
                mi_rare = _modulation_index(rare_samples_pac)
                mi_std  = _modulation_index(std_samples_pac)
                obs_pac = mi_rare - mi_std

                rng_pac      = np.random.default_rng(52)
                all_samp_pac = np.concatenate([rare_samples_pac, std_samples_pac])
                n_rare_pac   = len(rare_samples_pac)
                null_pac = []
                for _ in range(N_PERMS):
                    perm = rng_pac.permutation(all_samp_pac)
                    null_pac.append(_modulation_index(perm[:n_rare_pac])
                                    - _modulation_index(perm[n_rare_pac:]))
                null_pac = np.array(null_pac)
                p_pac    = float(np.mean(null_pac >= obs_pac))
                pac_result = {'mi_rare': mi_rare, 'mi_standard': mi_std,
                              'obs_diff': obs_pac, 'p_value': p_pac,
                              'n_bins': N_BINS_PAC, 'channel': pz_pac}
                print(f'  [oddball] Delta-gamma PAC ({pz_pac}): MI rare={mi_rare:.4f}  '
                      f'std={mi_std:.4f}  diff={obs_pac:+.4f}  p={p_pac:.3f}')

                fig_pac, (ax_bar, ax_null) = plt.subplots(1, 2, figsize=(10, 4))
                ax_bar.bar(['Rare', 'Standard'], [mi_rare, mi_std],
                           color=['#c0392b', '#2980b9'], alpha=0.8)
                ax_bar.set_ylabel('Modulation index (Tort 2010)', fontsize=9)
                ax_bar.set_title(f'Delta→gamma PAC at {pz_pac}\n(200–600 ms post-stimulus)', fontsize=10)
                ax_bar.grid(True, alpha=0.3, axis='y')
                pct95_pac = np.percentile(null_pac, 95)
                ax_null.hist(null_pac, bins=40, color='steelblue', alpha=0.7,
                             label=f'Null ({n_perms} shuffles)')
                ax_null.axvline(obs_pac, color='firebrick', lw=2,
                                label=f'Obs: {obs_pac:+.4f}  p={p_pac:.3f}')
                ax_null.axvline(pct95_pac, color='k', lw=1.5, ls='--', label='95th pct (p=0.05)')
                ax_null.set(xlabel='MI(rare) minus MI(standard)', ylabel='Count')
                ax_null.legend(fontsize=8)
                ax_null.grid(True, alpha=0.3)
                fig_pac.suptitle(
                    f'{subject_id}: Delta (1–4 Hz) → Gamma (30–80 Hz) Phase-Amplitude Coupling',
                    fontsize=11)
                plt.tight_layout()
                fig_pac.savefig(out_dir / f'{subject_id}_oddball_pac.png', dpi=150)
                plt.close(fig_pac)
        except Exception as _e:
            print(f'  [oddball] PAC skipped: {_e}')

    # ── Lempel-Ziv complexity (LZc) ───────────────────────────────────────────────
    # Captures non-phase-locked dynamics that ERP averaging destroys — a genuinely
    # additional measure, not a replacement for the amplitude-based component tests.
    # Per epoch: binarize each channel by its own median (0-600 ms window), compute
    # normalized Lempel-Ziv complexity (Zhang et al. 2009) per channel, average
    # across channels. Compare rare vs standard distributions by permutation test.
    lzc_result = None
    if not plots_only:
        try:
            if len(epochs['rare']) >= 4 and len(epochs['standard']) >= 4:
                lzc_win  = (t_ep >= 0.0) & (t_ep <= 0.6)
                lzc_data = epochs_data[:, :, lzc_win]   # (n_ep, n_ch, n_t)

                def _epoch_lzc(ep_2d):
                    return float(np.mean([
                        _lziv((row >= np.median(row)).astype(np.uint8), normalize=True)
                        for row in ep_2d
                    ]))

                lzc_per_epoch = np.array([_epoch_lzc(ep) for ep in lzc_data])
                lzc_rare = lzc_per_epoch[labels == 2]
                lzc_std  = lzc_per_epoch[labels == 1]
                obs_lzc  = float(lzc_rare.mean() - lzc_std.mean())

                rng_lzc  = np.random.default_rng(53)
                null_lzc = []
                for _ in range(N_PERMS):
                    perm = rng_lzc.permutation(labels)
                    null_lzc.append(lzc_per_epoch[perm == 2].mean()
                                    - lzc_per_epoch[perm == 1].mean())
                null_lzc = np.array(null_lzc)
                p_lzc    = float(np.mean(null_lzc >= obs_lzc))
                lzc_result = {'lzc_rare': float(lzc_rare.mean()), 'lzc_standard': float(lzc_std.mean()),
                              'obs_diff': obs_lzc, 'p_value': p_lzc,
                              'n_rare': int(len(lzc_rare)), 'n_std': int(len(lzc_std))}
                print(f'  [oddball] Lempel-Ziv complexity: rare={lzc_rare.mean():.4f}  '
                      f'std={lzc_std.mean():.4f}  diff={obs_lzc:+.4f}  p={p_lzc:.3f}')

                fig_lzc, (ax_box, ax_null) = plt.subplots(1, 2, figsize=(10, 4))
                ax_box.boxplot([lzc_rare, lzc_std], tick_labels=['Rare', 'Standard'],
                               patch_artist=True,
                               boxprops=dict(facecolor='#8e44ad', alpha=0.7),
                               medianprops=dict(color='k', lw=2), showfliers=False)
                ax_box.set_ylabel('Normalized LZ complexity', fontsize=9)
                ax_box.set_title('LZc by condition\n(0–600 ms, mean across channels)', fontsize=10)
                ax_box.grid(True, alpha=0.3, axis='y')
                pct95_lzc = np.percentile(null_lzc, 95)
                ax_null.hist(null_lzc, bins=40, color='steelblue', alpha=0.7,
                             label=f'Null ({n_perms} shuffles)')
                ax_null.axvline(obs_lzc, color='firebrick', lw=2,
                                label=f'Obs: {obs_lzc:+.4f}  p={p_lzc:.3f}')
                ax_null.axvline(pct95_lzc, color='k', lw=1.5, ls='--', label='95th pct (p=0.05)')
                ax_null.set(xlabel='Rare minus standard LZc', ylabel='Count')
                ax_null.legend(fontsize=8)
                ax_null.grid(True, alpha=0.3)
                fig_lzc.suptitle(
                    f'{subject_id}: Lempel-Ziv Complexity (0–600 ms post-stimulus)',
                    fontsize=11)
                plt.tight_layout()
                fig_lzc.savefig(out_dir / f'{subject_id}_oddball_lzc.png', dpi=150)
                plt.close(fig_lzc)
        except Exception as _e:
            print(f'  [oddball] LZc skipped: {_e}')

    # Johnsen band-power reactivity
    # raw_p300 and all_events are local to _build_oddball_evoked; reconstruct here.
    # Filtering is fast; epochs.events carries the same event array already sorted.
    raw_p300_j = load_filtered_eeg(raw, available_eeg, l_freq=0.1, h_freq=30, verbose=False)
    raw_p300_j.set_eeg_reference('average', projection=False, verbose=False)
    all_events_j = epochs.events

    FREQ_BANDS = {'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 13), 'beta': (14, 30)}
    ref_epochs_j = mne.Epochs(
        raw_p300_j, events=all_events_j, event_id={'standard': 1, 'rare': 2},
        tmin=-2.0, tmax=-0.05, baseline=None, preload=True, verbose=False,
    )
    act_epochs_j = mne.Epochs(
        raw_p300_j, events=all_events_j, event_id={'standard': 1, 'rare': 2},
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
        'n_rare_pre_rejection':        n_rare_pre,
        'n_std_pre_rejection':         n_std_pre,
        'n_rare_post_rejection':       n_rare_post,
        'n_std_post_rejection':        n_std_post,
        'n_rare_rejected':             n_rare_pre - n_rare_post,
        'n_std_rejected':              n_std_pre  - n_std_post,
        'rejection_method':            rejection_info['method'],
        'rejection_threshold_uv':      rejection_info['threshold_uv'],
        'n_epochs_with_interpolation': rejection_info['n_interpolated_epochs'],
        'interpolated_channels':       rejection_info['interpolated_channels'],
        'highpass_hz':                 0.1,
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
        'xdawn_result':       xdawn_result,
        'gamma_result':       gamma_result,
        'alpha_corr_result':  alpha_corr_result,
        'pac_result':         pac_result,
        'lzc_result':         lzc_result,
    }
    if not plots_only:
        with open(out_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f'  [oddball] Saved figures to {out_dir}')


def run_oddball_video(subject_id: str, raw, sfreq: float, available_eeg: list,
                      df: pd.DataFrame, out_dir: Path, *, force: bool = False):
    result = _build_oddball_evoked(raw, available_eeg, df, tag='oddball-video')
    if result is None:
        print(f'  [oddball-video] No per-beep rows -- skipping.')
        return

    _, _, _, diff_evoked, _, _ = result

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
