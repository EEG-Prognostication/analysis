"""Resting-state EEG analysis suite — batch helper.

Optional, complementary to the task-evoked paradigms (oddball / language /
command). Runs entirely on the passive pre-paradigm resting segment — no
task, no patient cooperation required. Not run by default; invoke with:

    python run_all.py --analyses resting

Five measures, all on the same resting window (mirrors lib/spindles.py):
    1. Permutation entropy        — antropy.perm_entropy, per channel, averaged
    2. Lempel-Ziv complexity      — antropy.lziv_complexity, per channel, averaged
    3. EEG microstates            — canonical 4-class k-means on GFP-peak topographies
    4. wSMI                       — weighted symbolic mutual information (King 2013)
    5. Background EEG quality screen — continuity / asymmetry / burst-suppression

Exports:
    run_resting    — full resting-state suite
    STATS_SENTINEL — filename used as run-complete sentinel
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from antropy import lziv_complexity as _lziv, perm_entropy as _perm_entropy

from .io import first_paradigm_onset
from .microstates import backfit_microstates, fit_microstate_templates
from .preprocessing import load_filtered_eeg

STATS_SENTINEL = 'metadata.json'

# Minimum resting duration for a meaningful run. Spindles can work on shorter
# segments (single-channel envelope); microstates and wSMI need enough GFP
# peaks / symbol windows to be stable, hence the higher floor here.
MIN_REST_S = 60.0


# ── 1+2: Permutation entropy and Lempel-Ziv complexity ───────────────────────
# Both are single-channel measures of signal unpredictability/compressibility.
# Della Bella et al. (2025) showed weighted entropy measures decrease
# healthy > MCS > UWS > acute and predict recovery — these are the same family
# of complexity measures applied to resting rather than task-evoked data.

def _complexity_measures(rest_data: np.ndarray) -> tuple[float, float, list[float], list[float]]:
    """Per-channel permutation entropy and normalized LZc, averaged across channels."""
    pe_per_ch  = [float(_perm_entropy(ch, order=3, normalize=True)) for ch in rest_data]
    lzc_per_ch = [float(_lziv((ch >= np.median(ch)).astype(np.uint8), normalize=True))
                  for ch in rest_data]
    return float(np.mean(pe_per_ch)), float(np.mean(lzc_per_ch)), pe_per_ch, lzc_per_ch


# ── 3: EEG microstates ────────────────────────────────────────────────────────
# Canonical 4-class model (Lehmann et al. 1987; Michel & Koenig 2018): cluster
# the scalp topographies at GFP (global field power) local maxima — these are
# the moments of highest signal-to-noise — into k=4 classes via polarity-
# invariant k-means, then "back-fit" every time sample to its best-correlating
# class. Reports per-class coverage, mean duration, and occurrence rate.
#
# Simplification vs. the full modified-k-means (Pascual-Marqui et al. 1995):
# rather than re-deriving a polarity-invariant distance metric, we align map
# polarity to a running reference before a standard k-means pass. This is a
# common, well-documented approximation (e.g. used as the pycrostates default
# initialization) — adequate for a descriptive batch summary, not a substitute
# for interactive microstate analysis with canonical A/B/C/D template matching
# (which requires a normative map library we do not have).

def _microstates(rest_data: np.ndarray, sfreq: float, k: int = 4,
                 max_peaks: int = 3000, seed: int = 45) -> dict | None:
    fit = fit_microstate_templates(rest_data, k=k, max_peaks=max_peaks, seed=seed)
    if fit is None:
        return None
    centers, n_peaks_used = fit

    # Back-fit every sample to the class with the largest |spatial correlation|
    # (sign-invariant — a microstate and its polarity inversion are the same class).
    assign, corr2 = backfit_microstates(rest_data, centers)
    gev = float(corr2.mean())                                     # global explained variance

    stats = []
    rest_dur_s = rest_data.shape[1] / sfreq
    for state in range(k):
        mask = (assign == state)
        coverage = float(mask.mean())
        # Run-length encode contiguous True segments
        d = np.diff(mask.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends   = np.where(d == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(mask)]
        durations_ms = (ends - starts) / sfreq * 1000.0
        stats.append({
            'class':              int(state),
            'coverage_pct':       round(coverage * 100, 2),
            'mean_duration_ms':   round(float(durations_ms.mean()), 2) if len(durations_ms) else 0.0,
            'occurrence_per_min': round(len(durations_ms) / (rest_dur_s / 60.0), 2),
        })

    return {'centers': centers, 'assign': assign, 'gev': gev, 'stats': stats,
            'n_peaks_used': n_peaks_used}


# ── 4: weighted Symbolic Mutual Information (wSMI) ───────────────────────────
# King et al. (2013): symbolize each channel's signal into ordinal patterns of
# length k=3 (six possible permutations), then compute mutual information
# between every channel pair, down-weighting symbol-pairs that are identical or
# time-reversed — these are the patterns most likely produced by a single
# source under volume conduction, so down-weighting them suppresses spurious
# "connectivity" between nearby electrodes picking up the same generator.
# wSMI is reduced in UWS relative to MCS and healthy controls and is one of
# the most replicated long-range-connectivity markers in the DoC literature.

_K_SYM = 3
_PERMS3 = list(permutations(range(_K_SYM)))                       # 6 orderings of (0,1,2)
_SYM_LUT = np.full(_K_SYM ** _K_SYM, -1, dtype=int)
for _i, _p in enumerate(_PERMS3):
    _code = sum(v * (_K_SYM ** (_K_SYM - 1 - j)) for j, v in enumerate(_p))
    _SYM_LUT[_code] = _i

_WSMI_W = np.ones((len(_PERMS3), len(_PERMS3)))
for _i, _p in enumerate(_PERMS3):
    _WSMI_W[_i, _i] = 0.0                                          # identical patterns
    _rev = _PERMS3.index(_p[::-1])
    _WSMI_W[_i, _rev] = 0.0                                        # time-reversed patterns


def _symbolize(x: np.ndarray, tau: int) -> np.ndarray:
    """Ordinal-pattern symbolization with embedding dimension 3, lag `tau`."""
    n = len(x) - 2 * tau
    a, b, c = x[:n], x[tau:tau + n], x[2 * tau:2 * tau + n]
    order = np.argsort(np.vstack([a, b, c]), axis=0)               # (3, n)
    codes = order[0] * _K_SYM ** 2 + order[1] * _K_SYM + order[2]
    return _SYM_LUT[codes]


def _wsmi_pair(sym_i: np.ndarray, sym_j: np.ndarray) -> float:
    n_sym = len(_PERMS3)
    joint = np.zeros((n_sym, n_sym))
    np.add.at(joint, (sym_i, sym_j), 1.0)
    joint /= joint.sum()
    p_i = joint.sum(axis=1, keepdims=True)
    p_j = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(joint > 0, joint / (p_i * p_j), 1.0)
        terms = np.where(joint > 0, _WSMI_W * joint * np.log(ratio), 0.0)
    return float(terms.sum())


def _wsmi(rest_data: np.ndarray, sfreq: float, ch_names: list,
          target_hz: float = 10.0) -> dict | None:
    # tau ~ 1 / (4 * f): probes coupling near `target_hz` (alpha, ~8-12 Hz —
    # the band most consistently reported in the wSMI/DoC literature)
    tau = max(1, int(round(sfreq / (4 * target_hz))))
    n_ch = rest_data.shape[0]
    symbols = [_symbolize(rest_data[c], tau) for c in range(n_ch)]

    mat = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            v = _wsmi_pair(symbols[i], symbols[j])
            mat[i, j] = mat[j, i] = v
    iu = np.triu_indices(n_ch, k=1)
    mean_wsmi = float(mat[iu].mean())
    return {'matrix': mat, 'mean': mean_wsmi, 'tau_samples': tau,
            'tau_ms': round(tau / sfreq * 1000, 2), 'target_hz': target_hz,
            'channels': list(ch_names)}


# ── 5: Background EEG quality screen ─────────────────────────────────────────
# Claassen group (medRxiv 2025): a pathological background (burst suppression,
# voltage suppression, periodic discharges, BIRDs, seizures) predicts zero CMD
# yield regardless of recording quality — a normal background is a necessary
# but not sufficient condition. This is a coarse three-feature heuristic
# screen, NOT a clinical EEG read: it flags sessions worth a closer look, it
# does not diagnose pathological patterns. Thresholds below are reasoned
# defaults from the CLAUDE.md spec, not validated cutoffs — treat flags as
# "warrants review", not "abnormal".

_SUPPRESSION_UV = 5.0          # below this peak-to-peak in ALL channels = suppressed window
_SUPPRESSION_FRAC_FLAG = 0.10  # >10% suppressed windows -> flag
_ASYMMETRY_RATIO_FLAG = 1.5    # hemispheric RMS ratio outside [1/1.5, 1.5] -> flag
_BURST_SD_FLAG = 1.5           # SD of log window-power outside this -> flag (bimodal-ish)


def _quality_screen(rest_data: np.ndarray, sfreq: float, ch_names: list) -> dict:
    win = int(round(0.5 * sfreq))
    n_win = rest_data.shape[1] // win
    windows = rest_data[:, :n_win * win].reshape(rest_data.shape[0], n_win, win)

    ptp_uv = (windows.max(axis=2) - windows.min(axis=2)) * 1e6     # (n_ch, n_win)
    suppressed = ptp_uv.max(axis=0) < _SUPPRESSION_UV              # ALL channels flat
    suppressed_frac = float(suppressed.mean())
    continuity = 1.0 - suppressed_frac

    right = [ch_names.index(c) for c in ('C4', 'T4') if c in ch_names]
    left  = [ch_names.index(c) for c in ('C3', 'T3') if c in ch_names]
    if right and left:
        rms_r = float(np.sqrt(np.mean(rest_data[right] ** 2)))
        rms_l = float(np.sqrt(np.mean(rest_data[left] ** 2)))
        asymmetry = rms_r / rms_l if rms_l > 0 else float('nan')
    else:
        asymmetry = None

    window_power = (windows ** 2).mean(axis=2).mean(axis=0)        # (n_win,)
    log_power = np.log(window_power + 1e-30)
    burst_sd = float(log_power.std())

    flags = {
        'suppression':      suppressed_frac > _SUPPRESSION_FRAC_FLAG,
        'asymmetry':        (asymmetry is not None
                             and not np.isnan(asymmetry)
                             and (asymmetry > _ASYMMETRY_RATIO_FLAG
                                  or asymmetry < 1.0 / _ASYMMETRY_RATIO_FLAG)),
        'burst_suppression': burst_sd > _BURST_SD_FLAG,
    }
    return {
        'continuity':            round(continuity, 4),
        'suppressed_window_pct': round(suppressed_frac * 100, 2),
        'asymmetry_ratio':       round(asymmetry, 4) if asymmetry is not None and not np.isnan(asymmetry) else None,
        'burst_suppression_sd':  round(burst_sd, 4),
        'flags':                 flags,
        'any_flag':              any(flags.values()),
        'window_power_log':      log_power,
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_resting(subject_id: str, raw, sfreq: float, available_eeg: list,
                df: pd.DataFrame, out_dir: Path, *, force: bool = False):
    """Resting-state complexity / connectivity / quality suite.

    Not run by default. Invoke with: python run_all.py --analyses resting
    """
    first_stim = first_paradigm_onset(df)
    rest_end = min(first_stim, 300.0) if first_stim else 300.0
    if rest_end < MIN_REST_S:
        print(f'  [resting] Resting period {rest_end:.1f}s < {MIN_REST_S:.0f}s — skipping.')
        return

    raw_rest = load_filtered_eeg(raw, available_eeg, l_freq=1, h_freq=40, verbose=False)
    raw_rest.set_eeg_reference('average', projection=False, verbose=False)
    rest_data = raw_rest.copy().crop(tmin=0, tmax=rest_end).get_data(picks=available_eeg)
    rest_dur_min = rest_end / 60.0
    ch_names = list(available_eeg)
    print(f'  [resting] Using {rest_dur_min:.1f} min of pre-paradigm resting EEG '
          f'({rest_data.shape[0]} channels)')

    # 1+2: complexity
    pe_mean, lzc_mean, pe_per_ch, lzc_per_ch = _complexity_measures(rest_data)
    print(f'  [resting] Permutation entropy={pe_mean:.4f}  LZc={lzc_mean:.4f} (mean over channels)')

    # 3: microstates
    ms = _microstates(rest_data, sfreq)
    if ms is not None:
        cov_str = ', '.join(f"{s['class']}: {s['coverage_pct']:.1f}%" for s in ms['stats'])
        print(f'  [resting] Microstates: GEV={ms["gev"]:.3f}  coverage [{cov_str}]')
    else:
        print('  [resting] Microstates skipped — too few GFP peaks for stable clustering.')

    # 4: wSMI
    wsmi = _wsmi(rest_data, sfreq, ch_names)
    if wsmi is not None:
        print(f'  [resting] wSMI: mean={wsmi["mean"]:.5f}  '
              f'(tau={wsmi["tau_ms"]:.1f} ms, targeting ~{wsmi["target_hz"]:.0f} Hz)')

    # 5: background quality screen
    quality = _quality_screen(rest_data, sfreq, ch_names)
    flag_str = ', '.join(k for k, v in quality['flags'].items() if v) or 'none'
    print(f'  [resting] Background screen: continuity={quality["continuity"]:.3f}  '
          f'asymmetry={quality["asymmetry_ratio"]}  '
          f'burst_sd={quality["burst_suppression_sd"]:.3f}  flags=[{flag_str}]')

    # ── Figure 1: complexity summary ──────────────────────────────────────────
    fig_c, axes_c = plt.subplots(1, 2, figsize=(10, 4))
    axes_c[0].bar(ch_names, pe_per_ch, color='teal', alpha=0.8)
    axes_c[0].axhline(pe_mean, color='firebrick', lw=1.5, ls='--', label=f'Mean = {pe_mean:.3f}')
    axes_c[0].set(title='Permutation entropy (order 3)', ylabel='Normalized PE')
    axes_c[0].tick_params(axis='x', rotation=90, labelsize=7)
    axes_c[0].legend(fontsize=8)
    axes_c[0].grid(True, alpha=0.3, axis='y')
    axes_c[1].bar(ch_names, lzc_per_ch, color='darkorange', alpha=0.8)
    axes_c[1].axhline(lzc_mean, color='firebrick', lw=1.5, ls='--', label=f'Mean = {lzc_mean:.3f}')
    axes_c[1].set(title='Lempel-Ziv complexity (normalized)', ylabel='Normalized LZc')
    axes_c[1].tick_params(axis='x', rotation=90, labelsize=7)
    axes_c[1].legend(fontsize=8)
    axes_c[1].grid(True, alpha=0.3, axis='y')
    fig_c.suptitle(f'{subject_id}: Resting-State Complexity '
                   f'({rest_dur_min:.1f} min, per-channel then averaged)', fontsize=11)
    plt.tight_layout()
    fig_c.savefig(out_dir / f'{subject_id}_resting_complexity.png', dpi=150)
    plt.close(fig_c)

    # ── Figure 2: microstates ─────────────────────────────────────────────────
    if ms is not None:
        montage = mne.channels.make_standard_montage('standard_1020')
        info_ms = mne.create_info(ch_names, sfreq, ch_types='eeg')
        info_ms.set_montage(montage, match_case=False, on_missing='warn')

        k = ms['centers'].shape[0]
        fig_ms, axes_ms = plt.subplots(2, k, figsize=(3.2 * k, 6.5),
                                       gridspec_kw={'height_ratios': [2, 1]})
        vmax_ms = np.abs(ms['centers']).max()
        for state in range(k):
            mne.viz.plot_topomap(ms['centers'][state], info_ms, axes=axes_ms[0, state],
                                 show=False, cmap='RdBu_r', vlim=(-vmax_ms, vmax_ms),
                                 extrapolate='head')
            axes_ms[0, state].set_title(f'Class {state}\n'
                                        f'{ms["stats"][state]["coverage_pct"]:.1f}% coverage',
                                        fontsize=10)
        labels_dur = [f'Class {s["class"]}' for s in ms['stats']]
        for state in range(k):
            axes_ms[1, state].remove()
        ax_bar = fig_ms.add_subplot(2, 1, 2)
        x = np.arange(k)
        w = 0.35
        ax_bar.bar(x - w / 2, [s['mean_duration_ms'] for s in ms['stats']], w,
                   label='Mean duration (ms)', color='steelblue', alpha=0.85)
        ax_bar2 = ax_bar.twinx()
        ax_bar2.bar(x + w / 2, [s['occurrence_per_min'] for s in ms['stats']], w,
                    label='Occurrence (per min)', color='darkorange', alpha=0.85)
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels_dur)
        ax_bar.set_ylabel('Mean duration (ms)', color='steelblue')
        ax_bar2.set_ylabel('Occurrence (per min)', color='darkorange')
        ax_bar.grid(True, alpha=0.3, axis='y')
        h1, l1 = ax_bar.get_legend_handles_labels()
        h2, l2 = ax_bar2.get_legend_handles_labels()
        ax_bar.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
        fig_ms.suptitle(
            f'{subject_id}: EEG Microstates (k=4, polarity-aligned k-means on GFP peaks, '
            f'GEV={ms["gev"]:.3f}, n={ms["n_peaks_used"]} peaks)\n'
            f'Classes are data-driven clusters, not canonical A/B/C/D '
            f'(no normative template matching performed)', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_ms.savefig(out_dir / f'{subject_id}_resting_microstates.png', dpi=150)
        plt.close(fig_ms)

    # ── Figure 3: wSMI connectivity matrix ────────────────────────────────────
    if wsmi is not None:
        fig_w, ax_w = plt.subplots(figsize=(7, 6))
        im = ax_w.imshow(wsmi['matrix'], cmap='viridis', vmin=0)
        ax_w.set_xticks(range(len(ch_names))); ax_w.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax_w.set_yticks(range(len(ch_names))); ax_w.set_yticklabels(ch_names, fontsize=7)
        fig_w.colorbar(im, ax=ax_w, label='wSMI', shrink=0.8)
        ax_w.set_title(
            f'{subject_id}: weighted Symbolic Mutual Information\n'
            f'(tau={wsmi["tau_ms"]:.1f} ms ~ {wsmi["target_hz"]:.0f} Hz, '
            f'mean={wsmi["mean"]:.5f})', fontsize=10)
        plt.tight_layout()
        fig_w.savefig(out_dir / f'{subject_id}_resting_wsmi.png', dpi=150)
        plt.close(fig_w)

    # ── Figure 4: background quality screen ───────────────────────────────────
    fig_q, axes_q = plt.subplots(1, 3, figsize=(13, 4))
    c0 = 'firebrick' if quality['flags']['suppression'] else 'forestgreen'
    axes_q[0].bar(['Continuity'], [quality['continuity']], color=c0, alpha=0.8)
    axes_q[0].axhline(1 - _SUPPRESSION_FRAC_FLAG, color='k', lw=1, ls='--', label='Flag threshold')
    axes_q[0].set(ylim=(0, 1.05), ylabel='Fraction of windows non-suppressed (>5 µV)',
                  title=f'Continuity\n({"FLAG" if quality["flags"]["suppression"] else "pass"})')
    axes_q[0].legend(fontsize=8)
    axes_q[0].grid(True, alpha=0.3, axis='y')

    if quality['asymmetry_ratio'] is not None:
        c1 = 'firebrick' if quality['flags']['asymmetry'] else 'forestgreen'
        axes_q[1].bar(['R / L RMS'], [quality['asymmetry_ratio']], color=c1, alpha=0.8)
        axes_q[1].axhline(1.0, color='k', lw=1)
        axes_q[1].axhline(_ASYMMETRY_RATIO_FLAG, color='k', lw=1, ls='--', label='Flag thresholds')
        axes_q[1].axhline(1 / _ASYMMETRY_RATIO_FLAG, color='k', lw=1, ls='--')
        axes_q[1].set(title=f'Inter-hemispheric asymmetry\n'
                            f'({"FLAG" if quality["flags"]["asymmetry"] else "pass"})',
                      ylabel='RMS ratio (C4/T4 over C3/T3)')
        axes_q[1].legend(fontsize=8)
    else:
        axes_q[1].axis('off')
        axes_q[1].text(0.5, 0.5, 'C3/C4/T3/T4\nnot available', ha='center', va='center')
    axes_q[1].grid(True, alpha=0.3, axis='y')

    c2 = 'firebrick' if quality['flags']['burst_suppression'] else 'forestgreen'
    axes_q[2].hist(quality['window_power_log'], bins=30, color=c2, alpha=0.8)
    axes_q[2].set(title=f'Window log-power distribution\n'
                        f'(SD={quality["burst_suppression_sd"]:.2f}, '
                        f'{"FLAG" if quality["flags"]["burst_suppression"] else "pass"})',
                  xlabel='log(mean power)', ylabel='Count')
    axes_q[2].grid(True, alpha=0.3, axis='y')
    fig_q.suptitle(
        f'{subject_id}: Background EEG Quality Screen — heuristic pre-flight check '
        f'(Claassen medRxiv 2025); not a clinical EEG read', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_q.savefig(out_dir / f'{subject_id}_resting_quality.png', dpi=150)
    plt.close(fig_q)

    metadata = {
        'rest_duration_min': round(rest_dur_min, 2),
        'permutation_entropy': {'mean': round(pe_mean, 5),
                                'per_channel': {ch: round(v, 5) for ch, v in zip(ch_names, pe_per_ch)}},
        'lzc': {'mean': round(lzc_mean, 5),
                'per_channel': {ch: round(v, 5) for ch, v in zip(ch_names, lzc_per_ch)}},
        'microstates': ({
            'gev': round(ms['gev'], 4),
            'n_peaks_used': ms['n_peaks_used'],
            'classes': ms['stats'],
        } if ms is not None else None),
        'wsmi': ({
            'mean': round(wsmi['mean'], 6),
            'tau_samples': wsmi['tau_samples'],
            'tau_ms': wsmi['tau_ms'],
            'target_hz': wsmi['target_hz'],
        } if wsmi is not None else None),
        'quality_screen': {
            'continuity':            quality['continuity'],
            'suppressed_window_pct': quality['suppressed_window_pct'],
            'asymmetry_ratio':       quality['asymmetry_ratio'],
            'burst_suppression_sd':  quality['burst_suppression_sd'],
            'flags':                 quality['flags'],
            'any_flag':              quality['any_flag'],
        },
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  [resting] Saved figures and metadata to {out_dir}')
