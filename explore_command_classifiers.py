#!/usr/bin/env python3
"""
Exploratory comparison of feature-extraction methods x classifiers for the
command-following keep-vs-stop sub-epoch decoding task.

Standalone script — does not modify run_all.py, results.json, or PDF reports.
Reuses build_command_subepochs() from lib/command.py for the exact same
event reconstruction and 2s sub-epochs used by the production SVM/MDM.

Feature sets:
    psd       — multitaper band power, all channels (production baseline)
    psd_motor — multitaper band power, C3/Cz/C4 only
    lat       — per-band contra-minus-ipsilateral log power at C3/C4
                (side-aware: right command -> contra=C3, left -> contra=C4)
    ratio     — per motor channel, log10(mu power) - log10(beta power)
    csp       — Common Spatial Patterns log-variance
    ts        — Riemannian tangent-space vector (covariance -> Euclidean vector)
    tfr       — Morlet wavelet power, motor channels, time-resolved

Classifiers: LinearSVC, LogisticRegression, and shrinkage LDA (all
RobustScaler-scaled) and RandomForestClassifier, plus the existing
covariance + MDM classifier for reference.

Usage:
    python explore_command_classifiers.py
    python explore_command_classifiers.py --patients CON012 CON014
    python explore_command_classifiers.py --n-perms 50 --n-boot 200   # quick test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Larger default text across every figure this script produces.
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
})
import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from mne.time_frequency import tfr_array_morlet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

try:
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from lib.command import build_command_subepochs
from run_all import RESULTS_DIR, discover_sessions, has_paradigm, load_session

mne.set_log_level('WARNING')

N_TREES        = 200
N_PERMS        = 200
N_BOOT         = 2000
CSP_COMPONENTS = 4
CLIP_V         = 200e-6
TFR_FREQS      = np.arange(8, 31, 2.0)   # mu+beta, 8-30 Hz
TFR_BINS       = 4                        # time bins within each 2s sub-epoch

COMBOS = [
    ('psd', 'svm'), ('psd', 'rf'), ('psd', 'lr'), ('psd', 'lda'),
    ('psd_motor', 'svm'), ('psd_motor', 'rf'), ('psd_motor', 'lr'), ('psd_motor', 'lda'),
    ('lat', 'svm'), ('lat', 'rf'), ('lat', 'lr'), ('lat', 'lda'),
    ('ratio', 'svm'), ('ratio', 'rf'), ('ratio', 'lr'), ('ratio', 'lda'),
    ('csp', 'svm'), ('csp', 'rf'), ('csp', 'lr'), ('csp', 'lda'),
    ('ts',  'svm'), ('ts',  'rf'), ('ts',  'lr'), ('ts',  'lda'),
    ('tfr', 'svm'), ('tfr', 'rf'), ('tfr', 'lr'), ('tfr', 'lda'),
    ('cov', 'mdm'),
]
CLASSIFIER_COLOR = {
    'svm': '#4a90d9', 'rf': '#7cb342', 'lr': '#e67e22', 'lda': '#c0392b', 'mdm': '#9b59b6',
}

# Human-readable labels for plot axes -- (feature, classifier) -> two-line / compact name
COMBO_LABEL = {
    ('psd', 'svm'): 'Band Power\nLinear SVM',
    ('psd', 'rf'):  'Band Power\nRandom Forest',
    ('psd', 'lr'):  'Band Power\nLogistic Regression',
    ('psd', 'lda'): 'Band Power\nShrinkage LDA',
    ('psd_motor', 'svm'): 'Motor Band Power\nLinear SVM',
    ('psd_motor', 'rf'):  'Motor Band Power\nRandom Forest',
    ('psd_motor', 'lr'):  'Motor Band Power\nLogistic Regression',
    ('psd_motor', 'lda'): 'Motor Band Power\nShrinkage LDA',
    ('lat', 'svm'): 'C3-C4 Laterality\nLinear SVM',
    ('lat', 'rf'):  'C3-C4 Laterality\nRandom Forest',
    ('lat', 'lr'):  'C3-C4 Laterality\nLogistic Regression',
    ('lat', 'lda'): 'C3-C4 Laterality\nShrinkage LDA',
    ('ratio', 'svm'): 'Mu/Beta Ratio\nLinear SVM',
    ('ratio', 'rf'):  'Mu/Beta Ratio\nRandom Forest',
    ('ratio', 'lr'):  'Mu/Beta Ratio\nLogistic Regression',
    ('ratio', 'lda'): 'Mu/Beta Ratio\nShrinkage LDA',
    ('csp', 'svm'): 'CSP\nLinear SVM',
    ('csp', 'rf'):  'CSP\nRandom Forest',
    ('csp', 'lr'):  'CSP\nLogistic Regression',
    ('csp', 'lda'): 'CSP\nShrinkage LDA',
    ('ts',  'svm'): 'Tangent Space\nLinear SVM',
    ('ts',  'rf'):  'Tangent Space\nRandom Forest',
    ('ts',  'lr'):  'Tangent Space\nLogistic Regression',
    ('ts',  'lda'): 'Tangent Space\nShrinkage LDA',
    ('tfr', 'svm'): 'Wavelet/TFR\nLinear SVM',
    ('tfr', 'rf'):  'Wavelet/TFR\nRandom Forest',
    ('tfr', 'lr'):  'Wavelet/TFR\nLogistic Regression',
    ('tfr', 'lda'): 'Wavelet/TFR\nShrinkage LDA',
    ('cov', 'mdm'): 'Covariance\nRiemannian MDM',
}
COMBO_LABEL_COMPACT = {k: v.replace('\n', ' - ') for k, v in COMBO_LABEL.items()}
N_TRIAL_PAIRS = 48  # all current command sessions use the "runs" schema (48 keep+stop pairs)


def _extract_tfr_features(data_sub, sfreq, ch_names, motor_channels):
    """Per-sub-epoch Morlet wavelet power at motor channels, mu/beta bands,
    split into TFR_BINS time bins across the 2s sub-epoch."""
    motor_idx = [ch_names.index(ch) for ch in motor_channels]
    decim     = max(1, int(sfreq // 32))
    n_cycles  = np.maximum(TFR_FREQS / 2.0, 3.0)
    power = tfr_array_morlet(
        data_sub[:, motor_idx, :], sfreq=sfreq, freqs=TFR_FREQS,
        n_cycles=n_cycles, output='power', decim=decim, n_jobs=1, verbose=False,
    )  # (n_epochs, n_motor_ch, n_freqs, n_times_decim)

    n_t = power.shape[-1]
    bin_edges = np.linspace(0, n_t, TFR_BINS + 1).astype(int)
    mu_idx   = np.where((TFR_FREQS >= 8)  & (TFR_FREQS <= 12))[0]
    beta_idx = np.where((TFR_FREQS >= 14) & (TFR_FREQS <= 30))[0]

    feats = []
    for b in range(TFR_BINS):
        seg = power[:, :, :, bin_edges[b]:bin_edges[b + 1]]
        feats.append(seg[:, :, mu_idx, :].mean(axis=(2, 3)))
        feats.append(seg[:, :, beta_idx, :].mean(axis=(2, 3)))
    X_tfr = np.concatenate(feats, axis=1)  # (n_epochs, n_motor_ch * 2 bands * TFR_BINS)
    return np.log10(X_tfr)


def _extract_motor_psd_features(X_psd, ch_names, motor_channels, n_bands):
    """Subset of the production log-band-power vector restricted to motor
    channels (e.g. C3/Cz/C4) -- tests whether the other ~16 channels are
    adding signal or just noise dimensions for the classifier."""
    n_ch = len(ch_names)
    motor_idx = [ch_names.index(ch) for ch in motor_channels]
    cols = [bi * n_ch + ci for bi in range(n_bands) for ci in motor_idx]
    return X_psd[:, cols]


def _extract_laterality_features(X_psd, ch_names, sub_sides, n_bands):
    """Per-band contralateral-minus-ipsilateral log power at C3/C4, signed by
    command side (right command -> contra=C3, left command -> contra=C4).
    Genuine ERD during 'keep' should make this more negative."""
    n_ch = len(ch_names)
    c3, c4 = ch_names.index('C3'), ch_names.index('C4')
    is_right = (sub_sides == 'right')
    contra_idx = np.where(is_right, c3, c4)
    ipsi_idx   = np.where(is_right, c4, c3)
    n = X_psd.shape[0]
    rows = np.arange(n)
    X_lat = np.stack([
        X_psd[rows, bi * n_ch + contra_idx] - X_psd[rows, bi * n_ch + ipsi_idx]
        for bi in range(n_bands)
    ], axis=1)
    return X_lat


def _extract_mu_beta_ratio_features(X_psd, ch_names, motor_channels, SVM_BAND_LABELS):
    """Per motor channel, log10(mu power) - log10(beta power) = log10(mu/beta) --
    a within-channel ERD ratio that cancels broadband gain differences between
    sub-epochs (e.g. movement artifacts that inflate all bands together)."""
    n_ch = len(ch_names)
    mu_idx   = SVM_BAND_LABELS.index('alpha')
    beta_idx = SVM_BAND_LABELS.index('beta')
    motor_idx = [ch_names.index(ch) for ch in motor_channels]
    mu   = X_psd[:, [mu_idx   * n_ch + ci for ci in motor_idx]]
    beta = X_psd[:, [beta_idx * n_ch + ci for ci in motor_idx]]
    return mu - beta


def bootstrap_auc_ci(y, scores, groups, n_boot=N_BOOT, seed=0, ci=95):
    """Bootstrap CI for AUC by resampling trial-pairs (groups) with replacement.

    Quantifies how much the LOGO-CV AUC would vary if a different sample of
    trial-pairs had been collected, given the same per-sub-epoch predictions.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_idx[g] for g in sampled])
        boots[b] = roc_auc_score(y[idx], scores[idx])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi), boots


def permutation_p(y, scores, observed_auc, n_perms=N_PERMS, seed=0):
    rng = np.random.default_rng(seed)
    perm_aucs = np.array([
        roc_auc_score(rng.permutation(y), scores) for _ in range(n_perms)
    ])
    p = (np.sum(perm_aucs >= observed_auc) + 1) / (n_perms + 1)
    return float(p), perm_aucs


def _fit_classifiers(oof, name, Xtr, Xte, test_idx, y_tr):
    scaler = RobustScaler().fit(Xtr)
    Xtr_scaled, Xte_scaled = scaler.transform(Xtr), scaler.transform(Xte)

    svm = LinearSVC(max_iter=10000, dual='auto').fit(Xtr_scaled, y_tr)
    oof[f'{name}_svm'][test_idx] = svm.decision_function(Xte_scaled)

    rf = RandomForestClassifier(
        n_estimators=N_TREES, class_weight='balanced', random_state=0, n_jobs=1,
    ).fit(Xtr, y_tr)
    oof[f'{name}_rf'][test_idx] = rf.predict_proba(Xte)[:, 1]

    lr = LogisticRegression(max_iter=10000, class_weight='balanced').fit(Xtr_scaled, y_tr)
    oof[f'{name}_lr'][test_idx] = lr.predict_proba(Xte_scaled)[:, 1]

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Xtr_scaled, y_tr)
    oof[f'{name}_lda'][test_idx] = lda.predict_proba(Xte_scaled)[:, 1]


def run_comparison_for_patient(pid, raw, sfreq, available_eeg, df, out_dir,
                                n_perms=N_PERMS, n_boot=N_BOOT):
    built = build_command_subepochs(raw, sfreq, available_eeg, df)
    if built is None:
        print(f'{pid}: no command rows — skipping')
        return None

    y        = built.sub_labels
    groups   = built.sub_groups
    data_sub = built.data_sub
    X_psd    = built.X_svm
    ch_names = built.sub_epochs.ch_names
    n        = len(y)
    n_groups = len(np.unique(groups))

    print(f'  [explore] {pid}: {n} sub-epochs, {n_groups} LOGO groups (trial-pairs)')

    X_tfr = _extract_tfr_features(data_sub, sfreq, ch_names, built.MOTOR_CHANNELS)
    n_bands = len(built.SVM_BANDS)
    X_psd_motor = _extract_motor_psd_features(X_psd, ch_names, built.MOTOR_CHANNELS, n_bands)
    X_ratio = _extract_mu_beta_ratio_features(X_psd, ch_names, built.MOTOR_CHANNELS, built.SVM_BAND_LABELS)

    has_lat = 'C3' in ch_names and 'C4' in ch_names
    if has_lat:
        sub_sides = built.keep_meta_df['side'].values[groups]
        X_lat = _extract_laterality_features(X_psd, ch_names, sub_sides, n_bands)

    cov_sub = None
    if HAS_PYRIEMANN:
        cov_sub = Covariances(estimator='lwf').fit_transform(np.clip(data_sub, -CLIP_V, CLIP_V))

    oof = {f'{feat}_{clf}': np.full(n, np.nan) for feat, clf in COMBOS}

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(np.zeros(n), y, groups=groups):
        y_tr = y[train_idx]

        _fit_classifiers(oof, 'psd', X_psd[train_idx], X_psd[test_idx], test_idx, y_tr)
        _fit_classifiers(oof, 'tfr', X_tfr[train_idx], X_tfr[test_idx], test_idx, y_tr)
        _fit_classifiers(oof, 'psd_motor', X_psd_motor[train_idx], X_psd_motor[test_idx], test_idx, y_tr)
        _fit_classifiers(oof, 'ratio', X_ratio[train_idx], X_ratio[test_idx], test_idx, y_tr)
        if has_lat:
            _fit_classifiers(oof, 'lat', X_lat[train_idx], X_lat[test_idx], test_idx, y_tr)

        csp = CSP(n_components=CSP_COMPONENTS, reg='ledoit_wolf', log=True, norm_trace=False)
        Xtr_csp = csp.fit_transform(data_sub[train_idx], y_tr)
        Xte_csp = csp.transform(data_sub[test_idx])
        _fit_classifiers(oof, 'csp', Xtr_csp, Xte_csp, test_idx, y_tr)

        if HAS_PYRIEMANN:
            ts = TangentSpace(metric='riemann')
            Xtr_ts = ts.fit_transform(cov_sub[train_idx])
            Xte_ts = ts.transform(cov_sub[test_idx])
            _fit_classifiers(oof, 'ts', Xtr_ts, Xte_ts, test_idx, y_tr)

            mdm = MDM(metric='riemann').fit(cov_sub[train_idx], y_tr)
            oof['cov_mdm'][test_idx] = mdm.predict_proba(cov_sub[test_idx])[:, 1]

    n_features = {
        'psd': X_psd.shape[1], 'psd_motor': X_psd_motor.shape[1],
        'lat': X_lat.shape[1] if has_lat else None, 'ratio': X_ratio.shape[1],
        'csp': CSP_COMPONENTS,
        'ts':  cov_sub.shape[1] * (cov_sub.shape[1] + 1) // 2 if cov_sub is not None else None,
        'tfr': X_tfr.shape[1], 'cov': cov_sub.shape[1] if cov_sub is not None else None,
    }

    rows = []
    for i, (feat, clf) in enumerate(COMBOS):
        name = f'{feat}_{clf}'
        if feat in ('ts', 'cov') and not HAS_PYRIEMANN:
            continue
        if feat == 'lat' and not has_lat:
            continue
        scores = oof[name]
        auc = roc_auc_score(y, scores)
        ci_lo, ci_hi, _ = bootstrap_auc_ci(y, scores, groups, n_boot=n_boot, seed=1000 + i)
        p, _ = permutation_p(y, scores, auc, n_perms=n_perms, seed=2000 + i)
        rows.append(dict(
            patient_id=pid, feature=feat, classifier=clf,
            n_features=n_features[feat], auc=auc,
            ci_lo=ci_lo, ci_hi=ci_hi, p_value=p,
        ))
        print(f'  [explore] {pid}: {name:9s} AUC={auc:.3f}  '
              f'95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]  p={p:.4f}')

    df_results = pd.DataFrame(rows)
    df_results.to_csv(out_dir / 'feature_comparison.csv', index=False)

    _plot_patient_comparison(pid, df_results, out_dir, n_groups)
    return df_results


def _plot_patient_comparison(pid, df_results, out_dir, n_groups):
    # Horizontal bars: with 29 combos, a vertical bar chart with rotated
    # x-tick labels needs ~1in/bar to avoid label collisions, which at the
    # PDF panel's ~7x4.2in box shrinks 12pt text to ~2.5pt (illegible).
    # Horizontal bars instead grow with height (cheap in this tall panel)
    # and an aspect ratio matched to the panel box keeps text near 7-8pt.
    n = len(df_results)
    fig_h = 0.21 * n
    fig_w = 1.67 * fig_h  # ~= panel box aspect (TEXT_W / _comparison_panel_h) in generate_reports.py
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    y = np.arange(n)
    labels = [COMBO_LABEL_COMPACT[(r.feature, r.classifier)] for r in df_results.itertuples()]
    colors = [CLASSIFIER_COLOR[r.classifier] for r in df_results.itertuples()]

    xerr = np.vstack([
        df_results['auc'] - df_results['ci_lo'],
        df_results['ci_hi'] - df_results['auc'],
    ])
    ax.barh(y, df_results['auc'], xerr=xerr, color=colors, alpha=0.8,
            capsize=3, ecolor='black')
    ax.axvline(0.5, color='k', ls='--', lw=1, label='Chance (0.5)')
    # Zoom the x-axis to the data range (with padding) rather than the full
    # 0-1 scale -- AUCs cluster in ~0.4-0.85, so the full scale compressed
    # bar-length differences to a thin band. Default (0.35, 0.95) widens
    # further only if a CI extends beyond it; extra right-padding for p-value text.
    xmin = max(0.0, min(0.35, float(df_results['ci_lo'].min()) - 0.03))
    xmax = min(1.0, max(0.95, float(df_results['ci_hi'].max()) + 0.05))
    for yi, r in zip(y, df_results.itertuples()):
        ax.text(r.ci_hi + 0.015, yi, f'p={r.p_value:.3f}', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax + 0.13)
    ax.set_xlabel('LOGO-CV AUC')
    ax.set_title(
        f'{pid}: Feature x Classifier comparison (bars: 95% CI, n={n_groups} pairs)',
        fontsize=12,
    )
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig.savefig(out_dir / f'{pid}_command_feature_comparison.png', dpi=150)
    plt.close(fig)


def _plot_summary_heatmap(all_results, out_path):
    pivot = all_results.pivot(index='patient_id',
                               columns=['feature', 'classifier'], values='auc')
    pval  = all_results.pivot(index='patient_id',
                               columns=['feature', 'classifier'], values='p_value')
    col_order = [(f, c) for f, c in COMBOS if (f, c) in pivot.columns]
    # Transpose: feature x classifier combos as rows, patients as columns --
    # 17 rows x 5 columns is much closer to the page's available aspect ratio
    # than 5 rows x 17 columns, so the rendered figure fills far more of the page.
    pivot = pivot[col_order].T
    pval  = pval[col_order].T

    fig, ax = plt.subplots(figsize=(1.3 * len(pivot.columns) + 3, 0.5 * len(pivot.index) + 1.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.4, vmax=0.85, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([COMBO_LABEL_COMPACT[(f, c)] for f, c in pivot.index], fontsize=12)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v, p = pivot.values[i, j], pval.values[i, j]
            star = '*' if p < 0.05 else ''
            ax.text(j, i, f'{v:.2f}{star}', ha='center', va='center', fontsize=11)
    plt.colorbar(im, ax=ax, label='LOGO-CV AUC', fraction=0.04, pad=0.02)
    ax.set_title('Command-following decoding: AUC by feature x classifier\n'
                  '(* = permutation p < 0.05)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patients', nargs='+', default=None)
    parser.add_argument('--n-perms', type=int, default=N_PERMS)
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    parser.add_argument('--plots-only', action='store_true',
                         help='Regenerate figures from cached feature_comparison_all.csv '
                              'without recomputing classifiers')
    args = parser.parse_args()

    if args.plots_only:
        combined = pd.read_csv(RESULTS_DIR / 'feature_comparison_all.csv')
        if args.patients:
            combined = combined[combined['patient_id'].isin(args.patients)]
        for pid, df_results in combined.groupby('patient_id'):
            out_dir = RESULTS_DIR / pid / 'command'
            _plot_patient_comparison(pid, df_results.reset_index(drop=True), out_dir, N_TRIAL_PAIRS)
        _plot_summary_heatmap(combined, RESULTS_DIR / 'feature_comparison_heatmap.png')
        print('Regenerated figures from cached results.')
        return

    sessions = discover_sessions()
    if args.patients:
        sessions = [s for s in sessions if s['patient_id'] in args.patients]

    all_results = []
    for s in sessions:
        pid = s['patient_id']
        if pid.lower().startswith(('jo', 'test')):
            continue
        try:
            raw, sfreq, available_eeg, df = load_session(s['edf'], s['csv'])
        except Exception as e:
            print(f'{pid}: ERROR loading session: {e} -- skipping'); continue
        if not has_paradigm(df, 'command'):
            continue
        out_dir = RESULTS_DIR / pid / 'command'
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_comparison_for_patient(
            pid, raw, sfreq, available_eeg, df, out_dir,
            n_perms=args.n_perms, n_boot=args.n_boot,
        )
        if result is not None:
            all_results.append(result)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(RESULTS_DIR / 'feature_comparison_all.csv', index=False)
        _plot_summary_heatmap(combined, RESULTS_DIR / 'feature_comparison_heatmap.png')
        print(f'\nWrote {RESULTS_DIR / "feature_comparison_all.csv"}')
        print(f'Wrote {RESULTS_DIR / "feature_comparison_heatmap.png"}')


if __name__ == '__main__':
    main()
