#!/usr/bin/env python3
"""
Exploratory comparison of feature-extraction methods x classifiers for the
command-following keep-vs-stop sub-epoch decoding task.

Standalone script — does not modify run_all.py, results.json, or PDF reports.
Reuses build_command_subepochs() from lib/command.py for the exact same
event reconstruction and 2s sub-epochs used by the production SVM/MDM.

Feature sets:
    psd  — multitaper band power, all channels (production baseline)
    csp  — Common Spatial Patterns log-variance
    ts   — Riemannian tangent-space vector (covariance -> Euclidean vector)
    tfr  — Morlet wavelet power, motor channels, time-resolved

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

    cov_sub = None
    if HAS_PYRIEMANN:
        cov_sub = Covariances(estimator='lwf').fit_transform(np.clip(data_sub, -CLIP_V, CLIP_V))

    oof = {f'{feat}_{clf}': np.full(n, np.nan) for feat, clf in COMBOS}

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(np.zeros(n), y, groups=groups):
        y_tr = y[train_idx]

        _fit_classifiers(oof, 'psd', X_psd[train_idx], X_psd[test_idx], test_idx, y_tr)
        _fit_classifiers(oof, 'tfr', X_tfr[train_idx], X_tfr[test_idx], test_idx, y_tr)

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
        'psd': X_psd.shape[1], 'csp': CSP_COMPONENTS,
        'ts':  cov_sub.shape[1] * (cov_sub.shape[1] + 1) // 2 if cov_sub is not None else None,
        'tfr': X_tfr.shape[1], 'cov': cov_sub.shape[1] if cov_sub is not None else None,
    }

    rows = []
    for i, (feat, clf) in enumerate(COMBOS):
        name = f'{feat}_{clf}'
        if feat in ('ts', 'cov') and not HAS_PYRIEMANN:
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
    fig, ax = plt.subplots(figsize=(1.1 * len(df_results) + 1, 8))
    x = np.arange(len(df_results))
    labels = [COMBO_LABEL_COMPACT[(r.feature, r.classifier)] for r in df_results.itertuples()]
    colors = [CLASSIFIER_COLOR[r.classifier] for r in df_results.itertuples()]

    yerr = np.vstack([
        df_results['auc'] - df_results['ci_lo'],
        df_results['ci_hi'] - df_results['auc'],
    ])
    ax.bar(x, df_results['auc'], yerr=yerr, color=colors, alpha=0.8,
           capsize=4, ecolor='black')
    ax.axhline(0.5, color='k', ls='--', lw=1, label='Chance (0.5)')
    for xi, r in zip(x, df_results.itertuples()):
        ax.text(xi, r.ci_hi + 0.02, f'p={r.p_value:.3f}', ha='center', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')
    ax.set_ylabel('LOGO-CV AUC')
    # Zoom the y-axis to the data range (with padding) rather than the full
    # 0-1 scale -- AUCs cluster in ~0.4-0.85, so the full scale compressed
    # bar-height differences to a thin band. Default (0.35, 0.95) widens
    # further only if a CI extends beyond it.
    ymin = max(0.0, min(0.35, float(df_results['ci_lo'].min()) - 0.03))
    ymax = min(1.0, max(0.95, float(df_results['ci_hi'].max()) + 0.05))
    ax.set_ylim(ymin, ymax)
    ax.set_title(
        f'{pid}: Feature x Classifier comparison '
        f'(error bars = bootstrap 95% CI over {n_groups} trial-pairs)'
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
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
        raw, sfreq, available_eeg, df = load_session(s['edf'], s['csv'])
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
