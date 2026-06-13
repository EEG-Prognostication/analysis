"""Command following ERD + SVM analysis — batch helper.

Exports:
    run_command    — full command pipeline (ERD, lateralization, TFR, SVM, plots)
    STATS_SENTINEL — filename used as run-complete sentinel
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from mne.decoding import LinearModel, get_coef
from mne.time_frequency import psd_array_multitaper

try:
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

from .io import DEFAULT_EEG_CHANNELS, _crop_to_paradigm
from .preprocessing import load_filtered_eeg

DEFAULT_N_PERMS = 1000
STATS_SENTINEL  = 'results.json'

_COMMAND_LIB_DIR = Path(__file__).resolve().parent
_REPO_ROOT       = _COMMAND_LIB_DIR.parent.parent   # analysis/lib/../.. = repo root

_CMD_AUDIO_FALLBACK = {
    'right_keep': 2.712, 'right_stop': 2.904,
    'left_keep':  2.760, 'left_stop':  2.928,
    'prompt':     3.809,
}


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


def _measure_command_audio_durations() -> dict:
    """Return command audio file durations in seconds via ffprobe.
    Falls back to pre-measured defaults if files are missing or ffprobe fails."""
    import subprocess
    audio_root = _REPO_ROOT / 'stimulus_software' / 'audio_data'
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


@dataclass
class CommandSubEpochs:
    """Return value of build_command_subepochs() — see that function's docstring."""
    raw_erd: mne.io.BaseRaw
    SCHEMA: str
    keep_events: np.ndarray
    stop_events: np.ndarray
    keep_meta_df: pd.DataFrame
    stop_meta_df: pd.DataFrame
    MOTOR_CHANNELS: list
    epochs_cmd: mne.Epochs
    sub_epochs: mne.Epochs
    data_sub: np.ndarray
    sub_labels: np.ndarray
    sub_groups: np.ndarray
    X_svm: np.ndarray
    psds_sub: np.ndarray
    psd_freqs_svm: np.ndarray
    SVM_BANDS: list
    SVM_BAND_LABELS: list


def build_command_subepochs(raw, sfreq: float, available_eeg: list,
                             df: pd.DataFrame) -> CommandSubEpochs | None:
    """Crop/filter raw, detect command schema, reconstruct keep/stop events, and
    build the 2s sub-epochs used for the SVM/Riemannian classifiers.

    Shared by run_command (production pipeline) and exploratory feature/classifier
    comparisons (explore_command_classifiers.py) — both need the identical
    schema-detection and event-reconstruction logic.

    Returns None if no command rows are present in df.
    """
    # Crop to command events only, before filtering.
    # post_s=15 covers the 9.9s epoch window plus a buffer.
    cmd_mask = (df['stim_type'].str.match(r'(right|left)_(keep|stop)', na=False) |
                df['stim_type'].str.contains('command', na=False))
    raw = _crop_to_paradigm(raw, df, cmd_mask, pre_s=5.0, post_s=15.0)
    raw_erd = load_filtered_eeg(raw, available_eeg, l_freq=1, h_freq=40, verbose=False)

    has_pairs = df['stim_type'].str.match(r'(right|left)_(keep|stop)', na=False).any()
    has_runs  = df['stim_type'].str.contains('command', na=False).any()
    SCHEMA = 'pairs' if has_pairs else 'runs' if has_runs else None
    if SCHEMA is None:
        return None
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

    # Claassen SVM sub-epochs
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

    # Band power is approximately log-normal; log-transform compresses multiplicative
    # outliers (e.g. a transient electrode artifact 1000x normal power becomes a +3
    # log-unit outlier instead of a +1000 raw-unit one) before RobustScaler — see
    # CLAUDE.md "Log-transform PSD features before scaling" (2026-06-10).
    X_svm = np.log10(X_svm)

    return CommandSubEpochs(
        raw_erd=raw_erd, SCHEMA=SCHEMA,
        keep_events=keep_events, stop_events=stop_events,
        keep_meta_df=keep_meta_df, stop_meta_df=stop_meta_df,
        MOTOR_CHANNELS=MOTOR_CHANNELS, epochs_cmd=epochs_cmd,
        sub_epochs=sub_epochs, data_sub=data_sub,
        sub_labels=sub_labels, sub_groups=sub_groups,
        X_svm=X_svm, psds_sub=psds_sub, psd_freqs_svm=psd_freqs_svm,
        SVM_BANDS=SVM_BANDS, SVM_BAND_LABELS=SVM_BAND_LABELS,
    )


def plot_svm_patterns(X_svm, sub_labels, sub_epochs, SVM_BANDS, SVM_BAND_LABELS,
                       subject_id: str, out_dir: Path) -> None:
    """Haufe et al. (2014) spatial patterns for the keep-vs-stop SVM, one topomap
    per frequency band.

    inverse_transform=False: get_coef's inverse_transform applies
    RobustScaler.inverse_transform (x * scale_ + center_) to the pattern vector,
    but center_ is an additive per-feature offset (~the median log10 band power,
    ~-9 to -7) that swamps the much smaller (~0.03-0.3) pattern values themselves
    -- every band/channel ends up dominated by center_ and comes out uniformly
    negative, rendering as all-blue topomaps regardless of the true pattern.
    Patterns in scaled-feature space are sign/relative-magnitude correct, which is
    all the per-band percentile-scaled topomap needs.

    n_ch_svm * len(SVM_BANDS) features are laid out band-major in X_svm (band 0's
    channels, then band 1's, ...), so the flat pattern vector must be reshaped as
    (len(SVM_BANDS), n_ch_svm) and transposed -- reshaping directly to
    (n_ch_svm, len(SVM_BANDS)) interleaves channels and bands.
    """
    n_ch_svm = len(sub_epochs.ch_names)
    clf_patterns = make_pipeline(
        RobustScaler(), LinearModel(LinearSVC(max_iter=10000, dual='auto'))
    )
    clf_patterns.fit(X_svm, sub_labels)
    patterns         = get_coef(clf_patterns, 'patterns_', inverse_transform=False)
    spatial_patterns = patterns.reshape(len(SVM_BANDS), n_ch_svm).T

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


def _plot_decoding(subject_id: str, out_dir: Path, sub_groups, sub_labels, prob_keep,
                    auc: float, p_val: float, classifier_label: str, suffix: str) -> None:
    """Decoding prediction per trial — Claassen 2019 Figure 3 style.

    Averages the 5 sub-epochs within each keep or stop period so each trial
    contributes one keep value and one stop value. Plots as scatter + smoothed
    trend with a box-plot summary panel showing the overall separation.
    """
    from scipy.ndimage import uniform_filter1d as _uf1d
    _pk_map: dict = {}; _ps_map: dict = {}
    for _i, (_pi, _lbl) in enumerate(zip(sub_groups, sub_labels)):
        (_pk_map if _lbl == 1 else _ps_map).setdefault(_pi, []).append(prob_keep[_i])
    _pairs_both = sorted(set(_pk_map) & set(_ps_map))
    _pk = np.array([np.mean(_pk_map[_p]) for _p in _pairs_both])
    _ps = np.array([np.mean(_ps_map[_p])  for _p in _pairs_both])
    _xp = np.arange(1, len(_pairs_both) + 1)

    # Zoom the y-axis to where the predictions (plus the chance line) actually
    # fall — with a weak classifier the values cluster tightly around 0.5 and
    # a fixed 0-1 axis makes that separation invisible.
    _all_probs = np.concatenate([_pk, _ps, [0.5]])
    _pad = max(0.03, 0.1 * float(np.ptp(_all_probs)))
    _ylo = max(0.0, float(_all_probs.min()) - _pad)
    _yhi = min(1.0, float(_all_probs.max()) + _pad)

    fig_dec, (ax_sc, ax_bx) = plt.subplots(
        1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [3, 1]},
    )

    # Scatter panel — one dot per trial per condition
    ax_sc.axhline(0.5, color='k', ls='--', lw=0.8, zorder=1, label='Chance (0.5)')
    ax_sc.scatter(_xp, _pk, color='#f5a623', s=18, alpha=0.65, zorder=3, label='Keep (move)')
    ax_sc.scatter(_xp, _ps, color='#4a90d9', s=18, alpha=0.65, zorder=3, label='Stop (rest)')
    _sm = max(5, len(_xp) // 8)  # smoothing window ~ 1/8 of trials
    if len(_pk) >= _sm:
        ax_sc.plot(_xp, _uf1d(_pk, size=_sm), color='#c87800', lw=2.0, zorder=4)
        ax_sc.plot(_xp, _uf1d(_ps, size=_sm), color='#1a5ba0', lw=2.0, zorder=4)
    ax_sc.set_ylim(_ylo, _yhi)
    ax_sc.set_xlim(0.5, len(_pairs_both) + 0.5)
    ax_sc.set_xlabel('Trial (keep+stop pair)', fontsize=10)
    ax_sc.set_ylabel('Decoding Prediction', fontsize=10)
    ax_sc.text(0.01, 0.97, 'Move', transform=ax_sc.transAxes,
               fontsize=9, ha='left', va='top', color='#555')
    ax_sc.text(0.01, 0.03, 'Rest', transform=ax_sc.transAxes,
               fontsize=9, ha='left', va='bottom', color='#555')
    ax_sc.legend(fontsize=9, loc='upper right')
    ax_sc.grid(True, alpha=0.2)

    # Box panel — overall keep vs stop distribution
    _bplt = ax_bx.boxplot(
        [_pk, _ps], tick_labels=['Keep\n(move)', 'Stop\n(rest)'],
        patch_artist=True, widths=0.5,
        medianprops=dict(color='k', lw=2),
        whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
        showfliers=False,
    )
    _bplt['boxes'][0].set_facecolor('#f5a623'); _bplt['boxes'][0].set_alpha(0.7)
    _bplt['boxes'][1].set_facecolor('#4a90d9'); _bplt['boxes'][1].set_alpha(0.7)
    ax_bx.axhline(0.5, color='k', ls='--', lw=0.8)
    ax_bx.set_ylim(_ylo, _yhi)
    ax_bx.set_ylabel('Prediction', fontsize=10)
    ax_bx.grid(True, alpha=0.25, axis='y')

    fig_dec.suptitle(
        f'{subject_id}: {classifier_label} — Keep vs Stop  '
        f'AUC={auc:.3f}  p={p_val:.3f}',
        fontsize=11,
    )
    plt.tight_layout()
    fig_dec.savefig(out_dir / f'{subject_id}{suffix}', dpi=150)
    plt.close(fig_dec)


def run_command(subject_id: str, raw, sfreq: float, available_eeg: list,
                df: pd.DataFrame, out_dir: Path, *, force: bool = False):
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

    built = build_command_subepochs(raw, sfreq, available_eeg, df)
    if built is None:
        print(f'  [command] No command rows — skipping.')
        return

    raw_erd         = built.raw_erd
    SCHEMA          = built.SCHEMA
    keep_events     = built.keep_events
    stop_events     = built.stop_events
    keep_meta_df    = built.keep_meta_df
    stop_meta_df    = built.stop_meta_df
    MOTOR_CHANNELS  = built.MOTOR_CHANNELS
    epochs_cmd      = built.epochs_cmd
    sub_epochs      = built.sub_epochs
    data_sub        = built.data_sub
    sub_labels      = built.sub_labels
    sub_groups      = built.sub_groups
    X_svm           = built.X_svm
    psds_sub        = built.psds_sub
    psd_freqs_svm   = built.psd_freqs_svm
    SVM_BANDS       = built.SVM_BANDS
    SVM_BAND_LABELS = built.SVM_BAND_LABELS

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

        erd_stats[side] = {'contra_channel': contra_ch, 'ipsi_channel': ipsi_ch}
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

        from scipy.ndimage import gaussian_filter as _gf
        erd_tfr = _gf(erd_tfr, sigma=(0, 0.8, 1.5))  # smooth time + freq, not channels

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

    # Claassen SVM — feature matrix X_svm and sub-epoch arrays already built by
    # build_command_subepochs() above.
    logo    = LeaveOneGroupOut()
    clf_svm = make_pipeline(RobustScaler(), LinearSVC(max_iter=10000, dual='auto'))

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

    # Logistic Regression on the same band-power features — second decoding view
    # alongside the production LinearSVC, predict_proba gives P(keep) directly.
    clf_lr = make_pipeline(RobustScaler(), LogisticRegression(max_iter=10000, class_weight='balanced'))
    prob_keep_lr = cross_val_predict(
        clf_lr, X_svm, sub_labels,
        method='predict_proba', cv=logo, groups=sub_groups,
    )[:, 1]
    mean_auc_lr = roc_auc_score(sub_labels, prob_keep_lr)

    rng_lr = np.random.default_rng(45)
    perm_scores_lr = np.array([
        roc_auc_score(rng_lr.permutation(sub_labels), prob_keep_lr)
        for _ in range(N_PERMS_SVM)
    ])
    p_lr = (np.sum(perm_scores_lr >= mean_auc_lr) + 1) / (N_PERMS_SVM + 1)
    print(f'  [command] LogReg AUC={mean_auc_lr:.3f}  p={p_lr:.4f}')

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

    # Band power comparison at motor channels — keep vs stop, per frequency band.
    # Shows what the SVM is distinguishing: distribution of band power under each condition.
    _motor_plot_chs = [ch for ch in ['C3', 'Cz', 'C4'] if ch in sub_epochs.ch_names]
    fig_bp, axes_bp = plt.subplots(1, len(SVM_BANDS), figsize=(14, 4.5), sharey=False)
    for _bi, (_ax_bp, (_flo, _fhi), _lbl) in enumerate(zip(axes_bp, SVM_BANDS, SVM_BAND_LABELS)):
        _fidx = np.where((psd_freqs_svm >= _flo) & (psd_freqs_svm <= _fhi))[0]
        _band_pow = psds_sub[:, :, _fidx].mean(axis=2) * 1e12  # (n_sub_ep, n_ch), pV²/Hz
        _pos = 1.0
        _all_pos, _all_data, _all_colors, _xtick_pos, _xtick_lbl = [], [], [], [], []
        for _ch in _motor_plot_chs:
            _ci = sub_epochs.ch_names.index(_ch)
            _kp = _band_pow[sub_labels == 1, _ci]
            _sp = _band_pow[sub_labels == 0, _ci]
            _bp_obj = _ax_bp.boxplot(
                [_kp, _sp], positions=[_pos, _pos + 0.55],
                patch_artist=True, widths=0.4,
                medianprops=dict(color='k', lw=2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                showfliers=False,
            )
            _bp_obj['boxes'][0].set_facecolor('#f5a623'); _bp_obj['boxes'][0].set_alpha(0.7)
            _bp_obj['boxes'][1].set_facecolor('#4a90d9'); _bp_obj['boxes'][1].set_alpha(0.7)
            _xtick_pos.append(_pos + 0.275)
            _xtick_lbl.append(_ch)
            _pos += 1.7
        _ax_bp.set_xticks(_xtick_pos)
        _ax_bp.set_xticklabels(_xtick_lbl, fontsize=10)
        _ax_bp.set_title(f'{_lbl}  ({_flo}–{_fhi} Hz)', fontsize=10)
        if _bi == 0:
            _ax_bp.set_ylabel('Band power (pV²/Hz)', fontsize=9)
        _ax_bp.grid(True, alpha=0.25, axis='y')
        _ax_bp.set_xlim(0.4, _pos - 0.9)
    # Shared legend
    from matplotlib.patches import Patch as _BP
    axes_bp[-1].legend(
        handles=[_BP(facecolor='#f5a623', alpha=0.7, label='Keep (move)'),
                 _BP(facecolor='#4a90d9', alpha=0.7, label='Stop (rest)')],
        fontsize=9, loc='upper right',
    )
    fig_bp.suptitle(
        f'{subject_id}: Band Power at Motor Channels — Keep vs Stop\n'
        f'(each box = distribution across all sub-epochs; outliers hidden)',
        fontsize=10,
    )
    plt.tight_layout()
    fig_bp.savefig(out_dir / f'{subject_id}_command_psd_features.png', dpi=150)
    plt.close(fig_bp)

    # Decoding prediction per trial — Claassen 2019 Figure 3 style. Plotted once
    # per classifier (Linear SVM and Logistic Regression, both on band-power
    # features) via _plot_decoding below.
    _plot_decoding(subject_id, out_dir, sub_groups, sub_labels, prob_keep,
                    mean_auc_svm, p_svm, 'Linear SVM Decoding (Band Power Features)',
                    '_command_decoding.png')
    _plot_decoding(subject_id, out_dir, sub_groups, sub_labels, prob_keep_lr,
                    mean_auc_lr, p_lr, 'Logistic Regression Decoding (Band Power Features)',
                    '_command_decoding_lr.png')

    # SVM spatial patterns
    plot_svm_patterns(X_svm, sub_labels, sub_epochs, SVM_BANDS, SVM_BAND_LABELS,
                      subject_id, out_dir)

    # Riemannian MDM classifier — operates on covariance matrices (SPD manifold).
    # Consistently outperforms LinearSVC on PSD features on benchmark motor imagery datasets.
    riemannian_result = None
    if HAS_PYRIEMANN:
        print('  [command] Running Riemannian MDM classifier...')
        # Clip to +/-200 uV (matches the oddball fallback artifact threshold) before
        # covariance estimation — an unclipped electrode-pop transient inflates that
        # channel's variance by orders of magnitude and dominates the SPD covariance
        # matrix. See CLAUDE.md "Clip raw amplitude before MDM covariance" (2026-06-10).
        CLIP_V  = 200e-6
        cov_sub = Covariances(estimator='lwf').fit_transform(np.clip(data_sub, -CLIP_V, CLIP_V))
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
        riemannian_result = {'auc': float(mean_auc_riem), 'p_value': float(p_riem)}

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

    # Write results sentinel + summary stats (consumed by generate_reports.py for the
    # patient-divider "Bottom Line" synthesis paragraph).
    asym = bg['asymmetry']
    results_summary = {
        'completed':          True,
        'schema':             SCHEMA,
        'n_keep_epochs':      len(epochs_cmd['keep']),
        'n_stop_epochs':      len(epochs_cmd['stop']),
        'background_screen': {
            'pass':             bg['pass'],
            'suppression_frac': float(bg['suppression_frac']),
            'bs_score':         float(bg['bs_score']),
            'asymmetry':        None if np.isnan(asym) else float(asym),
            'flags':            bg['flags'],
        },
        'erd':                erd_stats,
        'svm_result':         {'auc': float(mean_auc_svm), 'p_value': float(p_svm)},
        'lr_result':          {'auc': float(mean_auc_lr), 'p_value': float(p_lr)},
        'riemannian_result':  riemannian_result,
    }
    with open(out_dir / 'results.json', 'w') as _f:
        json.dump(results_summary, _f, indent=2)

    print(f'  [command] Saved figures to {out_dir}')
