"""Pooled cross-patient analyses — batch helper.

Exports:
    run_pooled_oddball — LOPO XDAWN+MDM + normative distributions across all patients
    run_pooled_resting — cross-patient resting-state microstate comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np

try:
    from pyriemann.classification import MDM
    from pyriemann.estimation import XdawnCovariances
    from sklearn.pipeline import Pipeline as _Pipeline
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

DEFAULT_N_PERMS = 1000


def run_pooled_oddball(results_dir: Path, n_perms: int = DEFAULT_N_PERMS):
    """Cross-patient analysis pooling all available oddball epochs.

    Loads cleaned epochs (epochs-epo.fif) written by run_oddball, pools them,
    and runs:
      - Leave-one-patient-out (LOPO) CV with XDAWN+MDM
      - Normative distributions of P3b amplitude, SVM accuracy, Fischer score
    Output goes to results_dir/POOLED/oddball/.
    """
    if not HAS_PYRIEMANN:
        print('  [pooled] pyriemann not installed — skipping pooled analysis.')
        return

    out_dir = results_dir / 'POOLED' / 'oddball'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-patient epochs and metadata
    patient_epochs, patient_ids, patient_meta = [], [], []
    for pd_dir in sorted(results_dir.iterdir()):
        if pd_dir.name == 'POOLED' or not pd_dir.is_dir():
            continue
        epo_file  = pd_dir / 'oddball' / 'epochs-epo.fif'
        meta_file = pd_dir / 'oddball' / 'metadata.json'
        if not epo_file.exists():
            continue
        try:
            ep = mne.read_epochs(epo_file, preload=True, verbose=False)
            md = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            patient_epochs.append(ep)
            patient_ids.append(pd_dir.name)
            patient_meta.append(md)
        except Exception as e:
            print(f'  [pooled] Could not load {pd_dir.name}: {e}')

    if len(patient_epochs) < 3:
        print(f'  [pooled] Need at least 3 patients with saved epochs; '
              f'found {len(patient_epochs)}.')
        return
    print(f'  [pooled] Loaded epochs for: {patient_ids}')

    # Find common channels across all patients
    common_chs = set(patient_epochs[0].ch_names)
    for ep in patient_epochs[1:]:
        common_chs &= set(ep.ch_names)
    common_chs = sorted(common_chs)
    for i, ep in enumerate(patient_epochs):
        patient_epochs[i] = ep.pick(common_chs)

    # Build pooled arrays with patient group labels
    X_list, y_list, grp_list = [], [], []
    for gi, ep in enumerate(patient_epochs):
        data = ep.get_data()
        labs = ep.events[:, 2]
        X_list.append(data)
        y_list.append((labs == 2).astype(int))
        grp_list.append(np.full(len(labs), gi))
    X_pool = np.concatenate(X_list, axis=0)
    y_pool = np.concatenate(y_list)
    g_pool = np.concatenate(grp_list)

    # ── LOPO XDAWN+MDM ──────────────────────────────────────────────────────
    lopo = LeaveOneGroupOut()
    xdawn_pipe = _Pipeline([
        ('xdawn', XdawnCovariances(nfilter=3, estimator='lwf')),
        ('mdm',   MDM(metric='riemann')),
    ])
    lopo_preds = cross_val_predict(xdawn_pipe, X_pool, y_pool,
                                   cv=lopo, groups=g_pool)
    lopo_acc = float(np.mean(lopo_preds == y_pool))

    # Per-patient accuracy
    per_patient_acc = {}
    for gi, pid in enumerate(patient_ids):
        mask = g_pool == gi
        per_patient_acc[pid] = float(np.mean(lopo_preds[mask] == y_pool[mask]))
        print(f'  [pooled] LOPO {pid}: acc={per_patient_acc[pid]:.3f}')
    print(f'  [pooled] LOPO overall acc={lopo_acc:.3f}')

    # Permutation test
    N_POOLED_PERMS = max(20, n_perms // 10)
    rng_pool = np.random.default_rng(60)
    null_pool = []
    for _ in range(N_POOLED_PERMS):
        y_p = rng_pool.permutation(y_pool)
        null_pool.append(float(np.mean(
            cross_val_predict(xdawn_pipe, X_pool, y_p, cv=lopo, groups=g_pool) == y_p
        )))
    null_pool = np.array(null_pool)
    p_pool = float((np.sum(null_pool >= lopo_acc) + 1) / (N_POOLED_PERMS + 1))
    print(f'  [pooled] LOPO permutation p={p_pool:.3f}')

    # ── Normative distributions ─────────────────────────────────────────────
    def _get(md, *keys, default=None):
        v = md
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, default)
            else:
                return default
        return v

    norm = {
        'patient':       patient_ids,
        'p3b_amp_uv':    [_get(m, 'components', 'P3b', 'observed_uv') for m in patient_meta],
        'p3b_p':         [_get(m, 'components', 'P3b', 'p_value')     for m in patient_meta],
        'mmn_amp_uv':    [_get(m, 'components', 'MMN', 'observed_uv') for m in patient_meta],
        'fischer_score': [_get(m, 'fischer_score')                     for m in patient_meta],
        'svm_acc':       [_get(m, 'svm_result', 'accuracy')           for m in patient_meta],
        'n_rare':        [_get(m, 'n_rare_post_rejection')             for m in patient_meta],
        'lopo_acc':      [per_patient_acc[pid] for pid in patient_ids],
    }

    # Figure 1: per-patient LOPO accuracy bar
    fig_lopo, ax_lopo = plt.subplots(figsize=(8, 4))
    colors_bar = ['#e84040' if a >= 0.6 else '#5b9bd5'
                  for a in norm['lopo_acc']]
    ax_lopo.bar(norm['patient'], norm['lopo_acc'], color=colors_bar, alpha=0.8)
    ax_lopo.axhline(0.5, color='k', lw=1.2, ls='--', label='Chance (0.5)')
    ax_lopo.set_ylim(0, 1)
    ax_lopo.set_ylabel('LOPO accuracy')
    ax_lopo.set_title(
        f'Leave-one-patient-out XDAWN+MDM  '
        f'(overall={lopo_acc:.3f}, p={p_pool:.3f})',
        fontsize=11)
    ax_lopo.legend(fontsize=9)
    ax_lopo.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig_lopo.savefig(out_dir / 'pooled_lopo_accuracy.png', dpi=150)
    plt.close(fig_lopo)

    # Figure 2: normative scatter — P3b amplitude vs Fischer score
    fig_norm, axes_norm = plt.subplots(1, 3, figsize=(14, 4))
    # P3b amplitude
    p3b_vals = [v for v in norm['p3b_amp_uv'] if v is not None]
    pid_p3b  = [norm['patient'][i] for i, v in enumerate(norm['p3b_amp_uv']) if v is not None]
    axes_norm[0].barh(pid_p3b, p3b_vals, color='#f0b800', alpha=0.8)
    axes_norm[0].axvline(1.095, color='firebrick', lw=1.2, ls='--',
                          label='Shao threshold')
    axes_norm[0].set_xlabel('P3b amplitude (µV)')
    axes_norm[0].set_title('P3b Amplitude (Pz)')
    axes_norm[0].legend(fontsize=8)
    axes_norm[0].grid(True, alpha=0.3, axis='x')
    # Fischer score
    fis_vals = [v if v is not None else 0 for v in norm['fischer_score']]
    axes_norm[1].barh(norm['patient'], fis_vals, color='#70ad47', alpha=0.8)
    axes_norm[1].set_xlim(0, 4)
    axes_norm[1].set_xlabel('Fischer score (0–4)')
    axes_norm[1].set_title('Fischer Hierarchy Score')
    axes_norm[1].axvline(3, color='firebrick', lw=1.2, ls='--', label='Score >= 3')
    axes_norm[1].legend(fontsize=8)
    axes_norm[1].grid(True, alpha=0.3, axis='x')
    # Per-patient rare tone count
    n_rare_vals = [v if v is not None else 0 for v in norm['n_rare']]
    colors_rare = ['#e84040' if n < 10 else '#5b9bd5' for n in n_rare_vals]
    axes_norm[2].barh(norm['patient'], n_rare_vals, color=colors_rare, alpha=0.8)
    axes_norm[2].axvline(10, color='firebrick', lw=1.2, ls='--', label='N=10 floor')
    axes_norm[2].set_xlabel('Rare epochs kept')
    axes_norm[2].set_title('Rare Tone Count (red < 10)')
    axes_norm[2].legend(fontsize=8)
    axes_norm[2].grid(True, alpha=0.3, axis='x')
    fig_norm.suptitle('Cohort Normative Summary — Oddball P300', fontsize=12)
    plt.tight_layout()
    fig_norm.savefig(out_dir / 'pooled_normative_summary.png', dpi=150)
    plt.close(fig_norm)

    pooled_meta = {
        'patient_ids':    patient_ids,
        'lopo_acc':       lopo_acc,
        'lopo_p':         p_pool,
        'per_patient_acc': per_patient_acc,
        'normative':       {k: [float(v) if v is not None else None for v in vals]
                            if isinstance(vals[0] if vals else None, (int, float, type(None)))
                            else vals
                            for k, vals in norm.items()},
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(pooled_meta, f, indent=2)
    print(f'  [pooled] Saved figures to {out_dir}')


def run_pooled_resting(results_dir: Path):
    """Cross-patient comparison of resting-state microstate metrics.

    Loads each analysable patient's microstate summary (results/{pid}/resting/
    metadata.json) alongside their oddball Fischer score and command-following
    SVM AUC, reduces the per-class microstate stats to three cohort-comparable
    scalars (GEV, occurrence-weighted mean duration, coverage entropy), and
    plots a cohort summary plus an exploratory comparison against the
    task-evoked findings. Output goes to results_dir/POOLED/resting/.

    With only ~5 analysable patients, no correlation statistics are computed —
    this is a descriptive cohort summary, not a hypothesis test.
    """
    out_dir = results_dir / 'POOLED' / 'resting'
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_ids, gevs, mean_durations, entropies = [], [], [], []
    fischer_scores, cmd_aucs = [], []

    for pd_dir in sorted(results_dir.iterdir()):
        if pd_dir.name == 'POOLED' or not pd_dir.is_dir():
            continue
        rest_file = pd_dir / 'resting' / 'metadata.json'
        odd_file  = pd_dir / 'oddball' / 'metadata.json'
        cmd_file  = pd_dir / 'command' / 'results.json'
        if not (rest_file.exists() and odd_file.exists() and cmd_file.exists()):
            continue

        ms = json.loads(rest_file.read_text()).get('microstates')
        if not ms:
            continue
        odd_md = json.loads(odd_file.read_text())
        cmd_md = json.loads(cmd_file.read_text())

        classes    = ms['classes']
        coverage   = np.array([c['coverage_pct'] for c in classes]) / 100.0
        occurrence = np.array([c['occurrence_per_min'] for c in classes])
        duration   = np.array([c['mean_duration_ms'] for c in classes])

        # Occurrence-weighted mean duration: the average length of a microstate
        # segment across the whole recording (longer occurrence-weight = more
        # of those segments occurred).
        mean_duration = float(np.sum(occurrence * duration) / np.sum(occurrence))
        # Normalized Shannon entropy of class coverage: 0 = one class dominates
        # the recording, 1 = all four classes are equally represented.
        p = coverage[coverage > 0]
        entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(coverage)))

        patient_ids.append(pd_dir.name)
        gevs.append(ms['gev'])
        mean_durations.append(mean_duration)
        entropies.append(entropy)
        fischer_scores.append(odd_md.get('fischer_score'))
        cmd_aucs.append((cmd_md.get('svm_result') or {}).get('auc'))

    if len(patient_ids) < 3:
        print(f'  [pooled-resting] Need at least 3 patients with resting+oddball+command '
              f'results; found {len(patient_ids)}.')
        return
    print(f'  [pooled-resting] Patients: {patient_ids}')

    # Figure 1: cohort summary — GEV, mean microstate duration, coverage entropy
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].barh(patient_ids, gevs, color='#5b9bd5', alpha=0.8)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel('Global explained variance')
    axes[0].set_title('Microstate GEV')
    axes[0].grid(True, alpha=0.3, axis='x')

    axes[1].barh(patient_ids, mean_durations, color='#70ad47', alpha=0.8)
    axes[1].set_xlabel('Mean duration (ms)')
    axes[1].set_title('Microstate Duration')
    axes[1].grid(True, alpha=0.3, axis='x')

    axes[2].barh(patient_ids, entropies, color='#f0b800', alpha=0.8)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel('Coverage entropy (0–1)')
    axes[2].set_title('Microstate Repertoire Diversity')
    axes[2].grid(True, alpha=0.3, axis='x')

    fig.suptitle('Cohort Summary — Resting-State Microstates', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / 'pooled_microstate_summary.png', dpi=150)
    plt.close(fig)

    # Figure 2: exploratory comparison against task-evoked findings
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))

    # Alternate label offsets above/below each point so close-together values
    # (e.g. two patients sharing a Fischer score) don't render overlapping text.
    offsets = [(6, 6) if i % 2 == 0 else (6, -12) for i in range(len(patient_ids))]

    fis_vals = [v if v is not None else 0 for v in fischer_scores]
    axes2[0].scatter(gevs, fis_vals, s=70, color='#e84040')
    for x, y, pid, off in zip(gevs, fis_vals, patient_ids, offsets):
        axes2[0].annotate(pid, (x, y), textcoords='offset points', xytext=off, fontsize=9)
    axes2[0].set_xlabel('Microstate GEV')
    axes2[0].set_ylabel('Fischer hierarchy score (0–4)')
    axes2[0].set_title('GEV vs Oddball Fischer Score')
    axes2[0].set_ylim(-0.3, 4.3)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].scatter(gevs, cmd_aucs, s=70, color='#5b9bd5')
    for x, y, pid, off in zip(gevs, cmd_aucs, patient_ids, offsets):
        axes2[1].annotate(pid, (x, y), textcoords='offset points', xytext=off, fontsize=9)
    axes2[1].axhline(0.5, color='k', lw=1.2, ls='--', label='Chance (0.5)')
    axes2[1].set_xlabel('Microstate GEV')
    axes2[1].set_ylabel('Command-following SVM AUC')
    axes2[1].set_title('GEV vs Command-Following AUC')
    # Pad ymax so point labels for the highest-AUC patient aren't clipped by
    # the axes border, and ymin so the chance line stays comfortably in view.
    axes2[1].set_ylim(min(0.5, min(cmd_aucs)) - 0.02, max(cmd_aucs) + 0.04)
    axes2[1].legend(fontsize=9)
    axes2[1].grid(True, alpha=0.3)

    fig2.suptitle(f'Resting-State Microstates vs Task-Evoked Findings '
                   f'(n={len(patient_ids)}, descriptive only)', fontsize=11)
    plt.tight_layout()
    fig2.savefig(out_dir / 'pooled_microstate_vs_outcomes.png', dpi=150)
    plt.close(fig2)

    pooled_meta = {
        'patient_ids':      patient_ids,
        'gev':              [float(v) for v in gevs],
        'mean_duration_ms': [float(v) for v in mean_durations],
        'coverage_entropy': [float(v) for v in entropies],
        'fischer_score':    fischer_scores,
        'command_svm_auc':  [float(v) if v is not None else None for v in cmd_aucs],
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(pooled_meta, f, indent=2)
    print(f'  [pooled-resting] Saved figures to {out_dir}')
