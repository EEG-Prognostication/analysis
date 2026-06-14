#!/usr/bin/env python3
"""
Pooled cross-patient command-following decoding via leave-one-patient-out
(LOPO) cross-validation.

Standalone, exploratory script — does not modify run_all.py, results.json, or
PDF reports. Pools all 5 analysable patients' (CON010/012/013/014/015) 2s
keep/stop sub-epochs (built by build_command_subepochs(), the same function
the production SVM/MDM and explore_command_classifiers.py use) and asks a
harder question than the existing within-patient LeaveOneGroupOut comparisons:
"if a classifier is trained on 4 patients, does it generalize to a 5th,
never-seen patient?"

Compared methods (all LOPO over the 5 patients):
    eegnet             — compact CNN (Lawhern et al. 2018) on raw 19-channel,
                          128 Hz, 2s sub-epochs
    eegnet_small       — same architecture, half the filters (F1=4, F2=8) and
                          10x the weight decay, testing whether eegnet's LOPO
                          gap vs. the hand-engineered features below is an
                          overfitting/data-size problem
    eegnet_pretrained  — eegnet, with block1/block2 (temporal + spatial
                          filters) pretrained on each fold's training
                          patients' oddball rare-vs-standard epochs before
                          fine-tuning on command keep/stop epochs --
                          cross-paradigm transfer using the much larger
                          oddball dataset to initialize the learned filters
    psd_lda    — production band-power features, Shrinkage LDA
    ratio_svm  — Mu/Beta Ratio features, Linear SVM
    ratio_lda  — Mu/Beta Ratio features, Shrinkage LDA
    ts_svm     — Riemannian tangent-space features, Linear SVM
    lat_svm    — C3-C4 Laterality features, Linear SVM

Each baseline is trained on the pooled 4-patient feature matrix and evaluated
on the 5th — the cross-patient analogue of the existing within-patient
feature_comparison_all.csv.

Cross-patient scale alignment: each patient's raw EEG channels and PSD-derived
features are z-scored using that patient's own mean/SD (computed across all of
that patient's sub-epochs) before pooling. This is a per-subject calibration
step standard in cross-subject BCI transfer learning — it removes
patient-specific amplitude/impedance offsets without touching the keep-vs-stop
label structure within a patient.

Usage:
    python explore_pooled_classifiers.py
    python explore_pooled_classifiers.py --max-epochs 50 --patience 10  # quick test
    python explore_pooled_classifiers.py --skip-interpretation          # skip spatial-filter figure
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from lib.command import build_command_subepochs
from run_all import RESULTS_DIR, discover_sessions, load_session
from explore_command_classifiers import (
    CLASSIFIER_COLOR,
    bootstrap_auc_ci,
    permutation_p,
    _extract_laterality_features,
    _extract_mu_beta_ratio_features,
)

mne.set_log_level('WARNING')

PATIENTS      = ['CON010', 'CON012', 'CON013', 'CON014', 'CON015']
TARGET_SFREQ  = 128.0
CLIP_V        = 200e-6

N_BOOT  = 2000
N_PERMS = 200

# EEGNet hyperparameters (Lawhern et al. 2018)
F1, D, F2     = 8, 2, 16
KERNEL_LENGTH = 64   # 0.5s @ 128Hz
DROPOUT       = 0.5
MAX_EPOCHS    = 200
PATIENCE      = 20
BATCH_SIZE    = 64
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
SEED          = 0

# eegnet_small: half the filters + stronger weight decay than the standard
# EEGNet above, testing whether the standard model's pooled-LOPO gap vs.
# psd_lda/ts_svm reflects overfitting on ~1,440 training sub-epochs/fold.
F1_SMALL, D_SMALL, F2_SMALL = 4, 2, 8
DROPOUT_SMALL      = 0.6
WEIGHT_DECAY_SMALL = 1e-3

# eegnet_pretrained: clip threshold for oddball epochs reused for
# cross-paradigm pretraining (matches CLIP_V used for command sub-epochs).
ODDBALL_CLIP_V = CLIP_V

COMBOS = ['eegnet', 'eegnet_small', 'eegnet_pretrained',
          'psd_lda', 'ratio_svm', 'ratio_lda', 'ts_svm', 'lat_svm']
COMBO_LABEL = {
    'eegnet':            'EEGNet (CNN)',
    'eegnet_small':      'EEGNet (Small)\nRegularized',
    'eegnet_pretrained': 'EEGNet\nOddball-Pretrained',
    'psd_lda':   'Band Power\nShrinkage LDA',
    'ratio_svm': 'Mu/Beta Ratio\nLinear SVM',
    'ratio_lda': 'Mu/Beta Ratio\nShrinkage LDA',
    'ts_svm':    'Tangent Space\nLinear SVM',
    'lat_svm':   'C3-C4 Laterality\nLinear SVM',
}
COMBO_LABEL_COMPACT = {k: v.replace('\n', ' - ') for k, v in COMBO_LABEL.items()}
COMBO_COLOR = {
    'eegnet':            '#e74c3c',
    'eegnet_small':      '#c0392b',
    'eegnet_pretrained': '#8e44ad',
    'psd_lda':   CLASSIFIER_COLOR['lda'],
    'ratio_svm': CLASSIFIER_COLOR['svm'],
    'ratio_lda': CLASSIFIER_COLOR['lda'],
    'ts_svm':    CLASSIFIER_COLOR['svm'],
    'lat_svm':   CLASSIFIER_COLOR['svm'],
}


# ── Data pooling ────────────────────────────────────────────────────────────

@dataclass
class PooledPatientData:
    patient_id: str
    X_raw:   np.ndarray            # (n_sub, n_ch, n_times) z-scored, 128 Hz, for EEGNet
    y:       np.ndarray            # (n_sub,) 1=keep, 0=stop
    groups:  np.ndarray            # (n_sub,) trial-pair index 0..47
    X_psd:   np.ndarray            # (n_sub, n_features) log10 band power, z-scored
    X_ratio: np.ndarray            # (n_sub, n_motor_ch) mu/beta ratio, z-scored
    X_lat:   np.ndarray | None      # (n_sub, n_bands) C3-C4 laterality, z-scored, or None
    cov:     np.ndarray | None      # (n_sub, n_ch, n_ch) Ledoit-Wolf covariance (native sfreq, clipped)
    ch_names: list


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + 1e-12)


def load_all_patients(patients: list[str] = PATIENTS) -> dict[str, PooledPatientData]:
    sessions = {s['patient_id']: s for s in discover_sessions()}
    out = {}
    for pid in patients:
        s = sessions.get(pid)
        if s is None:
            print(f'[pooled] {pid}: no session found -- skipping')
            continue
        raw, sfreq, available_eeg, df = load_session(s['edf'], s['csv'])
        built = build_command_subepochs(raw, sfreq, available_eeg, df)
        if built is None:
            print(f'[pooled] {pid}: no command rows -- skipping')
            continue

        data_clipped = np.clip(built.data_sub, -CLIP_V, CLIP_V)

        # Resample 2s sub-epochs from native sfreq (512 Hz) down to TARGET_SFREQ
        # so all patients share the same EEGNet input shape regardless of
        # recording sample rate.
        down = sfreq / TARGET_SFREQ
        data_rs = mne.filter.resample(data_clipped, down=down, axis=-1, verbose=False)

        # Per-patient, per-channel z-score (computed across all of this
        # patient's sub-epochs/timepoints) — see module docstring.
        mu = data_rs.mean(axis=(0, 2), keepdims=True)
        sd = data_rs.std(axis=(0, 2), keepdims=True)
        X_raw = ((data_rs - mu) / (sd + 1e-12)).astype(np.float32)

        ch_names = built.sub_epochs.ch_names
        n_bands  = len(built.SVM_BANDS)
        X_psd    = built.X_svm  # already log10

        X_ratio = _extract_mu_beta_ratio_features(
            X_psd, ch_names, built.MOTOR_CHANNELS, built.SVM_BAND_LABELS)

        X_lat = None
        if 'C3' in ch_names and 'C4' in ch_names:
            sub_sides = built.keep_meta_df['side'].values[built.sub_groups]
            X_lat = _extract_laterality_features(X_psd, ch_names, sub_sides, n_bands)

        cov = None
        if HAS_PYRIEMANN:
            cov = Covariances(estimator='lwf').fit_transform(data_clipped)

        out[pid] = PooledPatientData(
            patient_id=pid,
            X_raw=X_raw,
            y=built.sub_labels.astype(np.int64),
            groups=built.sub_groups,
            X_psd=_zscore_cols(X_psd),
            X_ratio=_zscore_cols(X_ratio),
            X_lat=_zscore_cols(X_lat) if X_lat is not None else None,
            cov=cov,
            ch_names=ch_names,
        )
        print(f'[pooled] {pid}: {X_raw.shape[0]} sub-epochs, '
              f'{X_raw.shape[1]} channels x {X_raw.shape[2]} samples @ {TARGET_SFREQ:.0f}Hz')
    return out


# ── EEGNet ──────────────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """Compact CNN for EEG decoding (Lawhern et al. 2018).

    Block 1: temporal conv (learns F1 bandpass-like filters) -> depthwise
    spatial conv (learns D spatial filters per temporal filter, one weight
    per EEG channel — these are the filters visualized as topomaps).
    Block 2: separable conv (depthwise + pointwise) for compact
    temporal-feature recombination. Flatten -> single logit.
    """

    def __init__(self, n_channels: int, n_times: int,
                 F1: int = F1, D: int = D, F2: int = F2,
                 kernel_length: int = KERNEL_LENGTH, dropout: float = DROPOUT):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),                                       # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            n_flat = self.block2(self.block1(dummy)).numel()
        self.classifier = nn.Linear(n_flat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.flatten(1)).squeeze(-1)  # logits


def train_eegnet(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
                 n_channels: int, n_times: int,
                 F1: int = F1, D: int = D, F2: int = F2, dropout: float = DROPOUT,
                 lr: float = LR, weight_decay: float = WEIGHT_DECAY,
                 max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                 seed: int = SEED, init_state: dict | None = None,
                 pos_weight: float | None = None) -> EEGNet:
    """Train an EEGNet. If init_state is given, its block1/block2 (temporal +
    spatial filter) weights are loaded before training -- the classifier head
    is left at its fresh random init since its input size depends on n_times
    and may differ from init_state's source model (see eegnet_pretrained)."""
    torch.manual_seed(seed)
    model = EEGNet(n_channels, n_times, F1=F1, D=D, F2=F2, dropout=dropout)
    if init_state is not None:
        block_state = {k: v for k, v in init_state.items() if not k.startswith('classifier.')}
        model.load_state_dict(block_state, strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xtr_t  = torch.from_numpy(Xtr).unsqueeze(1)
    ytr_t  = torch.from_numpy(ytr.astype(np.float32))
    Xval_t = torch.from_numpy(Xval).unsqueeze(1)
    yval_t = torch.from_numpy(yval.astype(np.float32))

    n = Xtr_t.shape[0]
    rng = np.random.default_rng(seed)
    best_val_loss, best_state, no_improve = np.inf, None, 0

    for _ in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            logits = model(Xtr_t[batch_idx])
            loss_fn(logits, ytr_t[batch_idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def eegnet_predict_proba(model: EEGNet, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X).unsqueeze(1))).numpy()


# ── Cross-paradigm pretraining (oddball epochs -> command fine-tuning) ─────

def load_oddball_epochs(patients: list[str], ch_names: list[str]
                         ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load cached oddball epochs (epochs-epo.fif, written by run_oddball)
    for cross-paradigm EEGNet pretraining. Reorders channels to match the
    command sub-epoch order, clips to +/-200uV, resamples to TARGET_SFREQ,
    and per-patient per-channel z-scores -- mirroring load_all_patients()'s
    X_raw preprocessing so both paradigms share a feature scale.

    Returns {} (and pretraining is skipped entirely) if any patient is
    missing its epochs-epo.fif or its channel set doesn't cover ch_names."""
    out = {}
    for pid in patients:
        epo_file = RESULTS_DIR / pid / 'oddball' / 'epochs-epo.fif'
        if not epo_file.exists():
            return {}
        ep = mne.read_epochs(epo_file, preload=True, verbose=False)
        if not set(ch_names).issubset(set(ep.ch_names)):
            return {}
        ep = ep.reorder_channels(ch_names)
        data = np.clip(ep.get_data(), -ODDBALL_CLIP_V, ODDBALL_CLIP_V)
        down = ep.info['sfreq'] / TARGET_SFREQ
        data_rs = mne.filter.resample(data, down=down, axis=-1, verbose=False)
        mu = data_rs.mean(axis=(0, 2), keepdims=True)
        sd = data_rs.std(axis=(0, 2), keepdims=True)
        X = ((data_rs - mu) / (sd + 1e-12)).astype(np.float32)
        y = (ep.events[:, 2] == 2).astype(np.int64)  # 1=rare, 0=standard
        out[pid] = (X, y)
    return out


def pretrain_eegnet_oddball(X: np.ndarray, y: np.ndarray, n_channels: int, n_times: int,
                             max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                             seed: int = SEED) -> dict:
    """Pretrain an EEGNet on oddball rare-vs-standard epochs (random 85/15
    split for early stopping; rare is the minority class so the loss uses
    pos_weight = n_standard/n_rare) and return its state_dict so
    train_eegnet() can transfer block1/block2 into a command model."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = max(1, int(0.15 * len(y)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    n_pos = int((y[train_idx] == 1).sum())
    pos_weight = float((len(train_idx) - n_pos) / max(n_pos, 1))
    model = train_eegnet(X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                          n_channels, n_times, max_epochs=max_epochs, patience=patience,
                          seed=seed, pos_weight=pos_weight)
    return model.state_dict()


# ── Baseline feature/classifier combos (LOPO, pooled training) ──────────────

def _fit_predict_lda(Xtr, ytr, Xte):
    scaler = RobustScaler().fit(Xtr)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(scaler.transform(Xtr), ytr)
    return lda.predict_proba(scaler.transform(Xte))[:, 1]


def _fit_predict_svm(Xtr, ytr, Xte):
    scaler = RobustScaler().fit(Xtr)
    svm = LinearSVC(max_iter=10000, dual='auto').fit(scaler.transform(Xtr), ytr)
    return svm.decision_function(scaler.transform(Xte))


def _fit_predict_ts_svm(cov_tr, ytr, cov_te):
    ts = TangentSpace(metric='riemann').fit(cov_tr)
    Xtr, Xte = ts.transform(cov_tr), ts.transform(cov_te)
    scaler = RobustScaler().fit(Xtr)
    svm = LinearSVC(max_iter=10000, dual='auto').fit(scaler.transform(Xtr), ytr)
    return svm.decision_function(scaler.transform(Xte))


# ── LOPO loop ─────────────────────────────────────────────────────────────────

def run_lopo(data: dict[str, PooledPatientData], oddball_data: dict | None = None,
              max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE) -> tuple[dict, list]:
    patients = list(data.keys())
    n_ch, n_times = data[patients[0]].X_raw.shape[1:]
    has_lat = all(data[p].X_lat is not None for p in patients)
    has_oddball = bool(oddball_data) and all(p in oddball_data for p in patients)

    combos = [c for c in COMBOS if c != 'lat_svm' or has_lat]
    combos = [c for c in combos if c != 'ts_svm' or HAS_PYRIEMANN]
    combos = [c for c in combos if c != 'eegnet_pretrained' or has_oddball]

    oof = {c: {} for c in combos}

    for i, test_pid in enumerate(patients):
        val_pid   = patients[(i + 1) % len(patients)]
        train_pids = [p for p in patients if p not in (test_pid, val_pid)]
        bl_pids    = train_pids + [val_pid]  # baselines use all 4 non-test patients
        print(f'[pooled] LOPO fold {i+1}/{len(patients)}: '
              f'test={test_pid}  val={val_pid}  train={train_pids}')

        Xtr  = np.concatenate([data[p].X_raw for p in train_pids], axis=0)
        ytr  = np.concatenate([data[p].y     for p in train_pids], axis=0)
        Xval = data[val_pid].X_raw
        yval = data[val_pid].y
        Xte  = data[test_pid].X_raw

        model = train_eegnet(Xtr, ytr, Xval, yval, n_ch, n_times,
                              max_epochs=max_epochs, patience=patience)
        oof['eegnet'][test_pid] = eegnet_predict_proba(model, Xte)

        model_small = train_eegnet(Xtr, ytr, Xval, yval, n_ch, n_times,
                                    F1=F1_SMALL, D=D_SMALL, F2=F2_SMALL,
                                    dropout=DROPOUT_SMALL, weight_decay=WEIGHT_DECAY_SMALL,
                                    max_epochs=max_epochs, patience=patience)
        oof['eegnet_small'][test_pid] = eegnet_predict_proba(model_small, Xte)

        if has_oddball:
            X_odd = np.concatenate([oddball_data[p][0] for p in train_pids], axis=0)
            y_odd = np.concatenate([oddball_data[p][1] for p in train_pids], axis=0)
            pretrained_state = pretrain_eegnet_oddball(X_odd, y_odd, n_ch, X_odd.shape[2],
                                                         max_epochs=max_epochs, patience=patience)
            model_pre = train_eegnet(Xtr, ytr, Xval, yval, n_ch, n_times,
                                      init_state=pretrained_state,
                                      max_epochs=max_epochs, patience=patience)
            oof['eegnet_pretrained'][test_pid] = eegnet_predict_proba(model_pre, Xte)

        ytr_bl = np.concatenate([data[p].y for p in bl_pids], axis=0)

        X_psd_tr = np.concatenate([data[p].X_psd for p in bl_pids], axis=0)
        oof['psd_lda'][test_pid] = _fit_predict_lda(X_psd_tr, ytr_bl, data[test_pid].X_psd)

        X_ratio_tr = np.concatenate([data[p].X_ratio for p in bl_pids], axis=0)
        oof['ratio_svm'][test_pid] = _fit_predict_svm(X_ratio_tr, ytr_bl, data[test_pid].X_ratio)
        oof['ratio_lda'][test_pid] = _fit_predict_lda(X_ratio_tr, ytr_bl, data[test_pid].X_ratio)

        if has_lat:
            X_lat_tr = np.concatenate([data[p].X_lat for p in bl_pids], axis=0)
            oof['lat_svm'][test_pid] = _fit_predict_svm(X_lat_tr, ytr_bl, data[test_pid].X_lat)

        if HAS_PYRIEMANN:
            cov_tr = np.concatenate([data[p].cov for p in bl_pids], axis=0)
            oof['ts_svm'][test_pid] = _fit_predict_ts_svm(cov_tr, ytr_bl, data[test_pid].cov)

    return oof, combos


def summarize_lopo(data: dict[str, PooledPatientData], oof: dict, combos: list,
                    n_perms: int = N_PERMS, n_boot: int = N_BOOT) -> pd.DataFrame:
    patients = list(data.keys())
    rows = []
    for ci, combo in enumerate(combos):
        all_y, all_scores, all_groups = [], [], []
        for gi, pid in enumerate(patients):
            y, scores, groups = data[pid].y, oof[combo][pid], data[pid].groups
            auc = roc_auc_score(y, scores)
            ci_lo, ci_hi, _ = bootstrap_auc_ci(y, scores, groups, n_boot=n_boot, seed=1000 + 100 * ci + gi)
            p, _ = permutation_p(y, scores, auc, n_perms=n_perms, seed=2000 + 100 * ci + gi)
            rows.append(dict(test_patient=pid, combo=combo, auc=auc,
                              ci_lo=ci_lo, ci_hi=ci_hi, p_value=p, n_groups=len(np.unique(groups))))
            all_y.append(y); all_scores.append(scores); all_groups.append(groups + 1000 * gi)

        all_y, all_scores, all_groups = map(np.concatenate, (all_y, all_scores, all_groups))
        auc = roc_auc_score(all_y, all_scores)
        ci_lo, ci_hi, _ = bootstrap_auc_ci(all_y, all_scores, all_groups, n_boot=n_boot, seed=3000 + ci)
        p, _ = permutation_p(all_y, all_scores, auc, n_perms=n_perms, seed=4000 + ci)
        rows.append(dict(test_patient='ALL', combo=combo, auc=auc,
                          ci_lo=ci_lo, ci_hi=ci_hi, p_value=p, n_groups=len(np.unique(all_groups))))
    return pd.DataFrame(rows)


# ── Plots ──────────────────────────────────────────────────────────────────

def _plot_lopo_heatmap(df: pd.DataFrame, combos: list, out_path: Path):
    patients = [p for p in df['test_patient'].unique() if p != 'ALL'] + ['ALL']
    pivot = df.pivot(index='combo', columns='test_patient', values='auc')[patients].loc[combos]
    pval  = df.pivot(index='combo', columns='test_patient', values='p_value')[patients].loc[combos]

    fig, ax = plt.subplots(figsize=(1.3 * len(patients) + 3, 0.5 * len(combos) + 1.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.4, vmax=0.85, aspect='auto')
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, fontsize=12)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels([COMBO_LABEL_COMPACT[c] for c in combos], fontsize=12)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v, p = pivot.values[i, j], pval.values[i, j]
            ax.text(j, i, f'{v:.2f}{"*" if p < 0.05 else ""}', ha='center', va='center', fontsize=11)
    plt.colorbar(im, ax=ax, label='LOPO AUC (held-out patient)', fraction=0.04, pad=0.02)
    ax.set_title('Cross-patient generalization: AUC by method x held-out patient\n'
                  '(ALL = all 5 held-out patients pooled; * = permutation p < 0.05)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_lopo_summary_bars(df: pd.DataFrame, combos: list, out_path: Path):
    sub = df[df['test_patient'] == 'ALL'].set_index('combo').loc[combos].reset_index()
    n = len(sub)
    fig, ax = plt.subplots(figsize=(8, 0.7 * n + 1.5))
    y = np.arange(n)
    labels = [COMBO_LABEL_COMPACT[c] for c in sub['combo']]
    colors = [COMBO_COLOR[c] for c in sub['combo']]
    xerr = np.vstack([sub['auc'] - sub['ci_lo'], sub['ci_hi'] - sub['auc']])
    ax.barh(y, sub['auc'], xerr=xerr, color=colors, alpha=0.8, capsize=3, ecolor='black')
    ax.axvline(0.5, color='k', ls='--', lw=1, label='Chance (0.5)')
    for yi, r in zip(y, sub.itertuples()):
        ax.text(r.ci_hi + 0.015, yi, f'p={r.p_value:.3f}', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    xmin = max(0.0, min(0.35, float(sub['ci_lo'].min()) - 0.03))
    xmax = min(1.0, max(0.95, float(sub['ci_hi'].max()) + 0.05))
    ax.set_xlim(xmin, xmax + 0.13)
    n_groups = int(sub['n_groups'].iloc[0])
    ax.set_xlabel(f'LOPO AUC (all 5 held-out patients pooled, n={n_groups} trial-pairs)')
    ax.set_title('Cross-patient generalization summary (95% CI)', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_interpretation_model(data: dict[str, PooledPatientData],
                                 max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
                                 seed: int = SEED) -> EEGNet:
    """Train one EEGNet on all 5 patients pooled (random 85/15 split for early
    stopping, not patient-aware) so its learned spatial filters can be
    visualized. Not used for any AUC reported in lopo_results.csv."""
    X = np.concatenate([d.X_raw for d in data.values()], axis=0)
    y = np.concatenate([d.y for d in data.values()], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = max(1, int(0.15 * len(y)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return train_eegnet(X[train_idx], y[train_idx], X[val_idx], y[val_idx],
                         X.shape[1], X.shape[2], max_epochs=max_epochs, patience=patience, seed=seed)


def _plot_eegnet_spatial_filters(model: EEGNet, ch_names: list, out_path: Path):
    """Topomaps of the F1*D depthwise spatial filters — one weight per EEG
    channel per filter, laid out as D rows (spatial filters) x F1 columns
    (temporal filters), mirroring plot_svm_patterns()'s percentile-scaled
    RdBu_r style."""
    weights = model.block1[2].weight.detach().numpy().squeeze(-1).squeeze(1)  # (F1*D, n_channels)
    n_filt = weights.shape[0]

    info = mne.create_info(ch_names, sfreq=TARGET_SFREQ, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, match_case=False, on_missing='warn')

    fig, axes = plt.subplots(D, F1, figsize=(2.2 * F1, 2.4 * D))
    for fi in range(n_filt):
        temporal_idx, spatial_idx = fi // D, fi % D
        ax = axes[spatial_idx, temporal_idx]
        w = weights[fi]
        scale = np.abs(w).max() or 1.0
        im, _ = mne.viz.plot_topomap(w, info, vlim=(-scale, scale), cmap='RdBu_r', axes=ax, show=False)
        ax.set_title(f'Temporal {temporal_idx}\nSpatial {spatial_idx}', fontsize=9)
    fig.suptitle('EEGNet Depthwise Spatial Filters (trained on all 5 patients pooled)', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-perms', type=int, default=N_PERMS)
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--skip-interpretation', action='store_true',
                         help='Skip the all-patients interpretation model and spatial-filter figure')
    args = parser.parse_args()

    out_dir = RESULTS_DIR / 'POOLED' / 'command'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[pooled] Loading and pooling {len(PATIENTS)} patients...')
    data = load_all_patients()
    if len(data) < 3:
        print(f'[pooled] Need at least 3 patients with command data; found {len(data)}.')
        return

    ch_names = next(iter(data.values())).ch_names
    oddball_data = load_oddball_epochs(list(data.keys()), ch_names)
    if oddball_data:
        print(f'[pooled] Loaded cached oddball epochs for cross-paradigm pretraining '
              f'({len(oddball_data)} patients).')
    else:
        print('[pooled] Oddball epochs unavailable -- skipping eegnet_pretrained.')

    print(f'\n[pooled] Running {len(data)}-fold leave-one-patient-out cross-validation...')
    oof, combos = run_lopo(data, oddball_data=oddball_data,
                            max_epochs=args.max_epochs, patience=args.patience)

    df_results = summarize_lopo(data, oof, combos, n_perms=args.n_perms, n_boot=args.n_boot)
    df_results.to_csv(out_dir / 'lopo_results.csv', index=False)
    print(f'\nWrote {out_dir / "lopo_results.csv"}')

    print('\n[pooled] Pooled (ALL held-out patients) results:')
    for _, r in df_results[df_results['test_patient'] == 'ALL'].iterrows():
        print(f'  {r.combo:12s} AUC={r.auc:.3f}  95% CI=[{r.ci_lo:.3f}, {r.ci_hi:.3f}]  p={r.p_value:.4f}')

    _plot_lopo_heatmap(df_results, combos, out_dir / 'lopo_heatmap.png')
    _plot_lopo_summary_bars(df_results, combos, out_dir / 'lopo_summary_bars.png')
    print(f'Wrote {out_dir / "lopo_heatmap.png"}')
    print(f'Wrote {out_dir / "lopo_summary_bars.png"}')

    if not args.skip_interpretation:
        print('\n[pooled] Training interpretation model on all patients pooled...')
        model = train_interpretation_model(data, max_epochs=args.max_epochs, patience=args.patience)
        _plot_eegnet_spatial_filters(model, next(iter(data.values())).ch_names,
                                      out_dir / 'eegnet_spatial_filters.png')
        print(f'Wrote {out_dir / "eegnet_spatial_filters.png"}')


if __name__ == '__main__':
    main()
