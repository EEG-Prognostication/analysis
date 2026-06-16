#!/usr/bin/env python3
"""
Batch runner: run EEG analyses for all patients missing results.

Usage:
    python run_all.py                                  # run all missing analyses
    python run_all.py --force                          # re-run even if outputs exist
    python run_all.py --patients CON011 CON012         # specific patient(s)
    python run_all.py --analyses oddball language      # specific analyses
    python run_all.py --workers 4                      # parallel patients (default: 1)
    python run_all.py --fast                           # 100 permutations (dev mode)
    python run_all.py --perms 500                      # custom permutation count
    python run_all.py --plots-only                     # regenerate figures only (oddball)
    python run_all.py --video                          # generate topomap MP4s only
"""
from __future__ import annotations

import argparse
import multiprocessing as _mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless — no display needed
# Larger default text across every figure (oddball/language/command/resting/
# spindles/pooled) -- applies to figures created in this process and in the
# ProcessPoolExecutor workers below, which re-import this module on spawn.
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
})
import mne
import pandas as pd

ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from lib.io import DEFAULT_EEG_CHANNELS, align_stimulus_csv, load_raw_eeg_metadata
from lib.preprocessing import load_filtered_eeg
from lib.oddball import run_oddball, run_oddball_video, STATS_SENTINEL as _ODDBALL_SENTINEL
from lib.language import run_language, STATS_SENTINEL as _LANGUAGE_SENTINEL
from lib.command import run_command, STATS_SENTINEL as _COMMAND_SENTINEL
from lib.spindles import run_spindles, STATS_SENTINEL as _SPINDLES_SENTINEL
from lib.resting import run_resting, STATS_SENTINEL as _RESTING_SENTINEL
from lib.pooled import run_pooled_oddball, run_pooled_resting

mne.set_log_level('WARNING')

REPO_ROOT   = ANALYSIS_ROOT.parent
CSV_DIR     = REPO_ROOT / 'stimulus_software' / 'patient_data' / 'results'
EDF_DIR     = REPO_ROOT / 'stimulus_software' / 'patient_data' / 'edfs'
RESULTS_DIR = ANALYSIS_ROOT / 'results'

ALL_ANALYSES      = ['oddball', 'command']                 # run by default
OPTIONAL_ANALYSES = ['language', 'spindles', 'resting']    # only when explicitly requested
POOLED_ANALYSES   = ['pooled_oddball', 'pooled_resting']
DEFAULT_N_PERMS = 1000

_SENTINELS = {
    'oddball':  _ODDBALL_SENTINEL,
    'language': _LANGUAGE_SENTINEL,
    'command':  _COMMAND_SENTINEL,
    'spindles': _SPINDLES_SENTINEL,
    'resting':  _RESTING_SENTINEL,
}


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


def _sentinel_exists(patient_id: str, analysis: str) -> bool:
    sentinel = _SENTINELS.get(analysis)
    if not sentinel:
        return False
    return (RESULTS_DIR / patient_id / analysis / sentinel).exists()


def has_paradigm(df: pd.DataFrame, analysis: str) -> bool:
    if analysis in ('spindles', 'resting'):
        return True  # uses resting EEG only — no CSV marker needed
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


# ── Shared EEG load ───────────────────────────────────────────────────────────

def load_session(edf_path: Path, csv_path: Path):
    raw, sfreq, available_eeg = load_raw_eeg_metadata(
        edf_path, eeg_channels=DEFAULT_EEG_CHANNELS, bad_channels=[], preload=False, verbose=False
    )
    df, _ = align_stimulus_csv(csv_path, sfreq=sfreq, n_times=raw.n_times)
    return raw, sfreq, available_eeg, df


# ── Worker ────────────────────────────────────────────────────────────────────

def _run_patient_job(job: dict) -> str:
    """Worker function executed in a subprocess for one patient x one analysis."""
    matplotlib.use('Agg')
    mne.set_log_level('WARNING')
    pid             = job['patient_id']
    analysis        = job['analysis']
    n_perms         = job['n_perms']
    plots_only      = job['plots_only']
    force_rejection = job['force_rejection']
    force           = job['force']

    out_dir = RESULTS_DIR / pid / analysis
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw, sfreq, available_eeg, df = load_session(job['edf'], job['csv'])
        if analysis == 'oddball':
            run_oddball(pid, raw, sfreq, available_eeg, df, out_dir,
                        force=force, plots_only=plots_only, n_perms=n_perms,
                        force_rejection=force_rejection)
        elif analysis == 'language':
            run_language(pid, raw, sfreq, available_eeg, df, out_dir, force=force)
        elif analysis == 'command':
            run_command(pid, raw, sfreq, available_eeg, df, out_dir, force=force)
        elif analysis == 'spindles':
            run_spindles(pid, raw, sfreq, available_eeg, df, out_dir, force=force)
        elif analysis == 'resting':
            run_resting(pid, raw, sfreq, available_eeg, df, out_dir, force=force)
        return f'{pid}/{analysis}: OK'
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f'{pid}/{analysis}: ERROR — {e}\n{tb}'


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Batch EEG analysis runner')
    parser.add_argument('--force',   action='store_true', help='Re-run even if outputs exist')
    parser.add_argument('--force-rejection', action='store_true',
                        help='Re-fit autoreject even if cached model exists')
    parser.add_argument('--plots-only', action='store_true',
                        help='Regenerate figures only — skip permutations and SVM (oddball). '
                             'Requires a previous full run to have cached null_arrays.npz.')
    parser.add_argument('--video',   action='store_true',
                        help='Generate oddball topographic MP4 animation (20 fps). '
                             'Skips all other analyses.')
    parser.add_argument('--pooled',  action='store_true',
                        help='Run pooled cross-patient analyses (oddball + resting microstates) '
                             'after per-patient runs.')
    parser.add_argument('--patients', nargs='+', metavar='ID',
                        help='Limit to specific patient IDs')
    parser.add_argument('--analyses', nargs='+',
                        choices=ALL_ANALYSES + OPTIONAL_ANALYSES + POOLED_ANALYSES,
                        metavar='NAME',
                        help=f'Analyses to run. Default: {ALL_ANALYSES}. '
                             f'Optional (not run by default): {OPTIONAL_ANALYSES}')
    parser.add_argument('--workers', type=int, default=1, metavar='N',
                        help='Parallel worker processes (default: 1). '
                             'Set to number of CPU cores for maximum speedup.')
    parser.add_argument('--perms', type=int, default=DEFAULT_N_PERMS, metavar='N',
                        help=f'Permutation count (default: {DEFAULT_N_PERMS})')
    parser.add_argument('--fast', action='store_true',
                        help='Development mode: 100 permutations (overrides --perms)')
    args = parser.parse_args()

    n_perms = 100 if args.fast else args.perms

    sessions = discover_sessions()
    if args.patients:
        sessions = [s for s in sessions if s['patient_id'] in args.patients]
    sessions = [s for s in sessions if not s['patient_id'].lower().startswith(('jo', 'test'))]

    print(f'Sessions to consider: {[s["patient_id"] for s in sessions]}')
    if args.fast:
        print(f'  --fast mode: {n_perms} permutations')
    elif args.perms != DEFAULT_N_PERMS:
        print(f'  --perms {n_perms}')
    if args.workers > 1:
        print(f'  --workers {args.workers}: running patients in parallel')

    if args.video:
        for session in sessions:
            pid  = session['patient_id']
            date = session['date']
            video_path = RESULTS_DIR / pid / 'oddball' / f'{pid}_oddball_topomap.mp4'
            if not args.force and video_path.exists():
                print(f'\n{pid}: video exists -- skipping.')
                continue
            print(f'\n{"="*60}\n{pid} ({date}): generating topomap video\n{"="*60}')
            try:
                raw, sfreq, available_eeg, df = load_session(session['edf'], session['csv'])
            except Exception as e:
                print(f'  ERROR loading session: {e}'); continue
            if not has_paradigm(df, 'oddball'):
                print('  [oddball-video] No oddball rows — skipping.'); continue
            out_dir = RESULTS_DIR / pid / 'oddball'
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_oddball_video(pid, raw, sfreq, available_eeg, df, out_dir, force=args.force)
            except Exception as e:
                import traceback; print(f'  ERROR: {e}'); traceback.print_exc()
        print('\nDone.')
        return

    # Build per-patient jobs (pooled analyses are handled separately below)
    target_analyses = [a for a in (args.analyses or ALL_ANALYSES) if a not in POOLED_ANALYSES]
    jobs = []
    for session in sessions:
        pid  = session['patient_id']
        date = session['date']
        work = []
        for analysis in target_analyses:
            if args.force:
                work.append((analysis, False))   # (analysis, plots_only)
            elif _sentinel_exists(pid, analysis):
                if args.plots_only:
                    work.append((analysis, True))
                else:
                    print(f'{pid} ({date}) [{analysis}]: sentinel present — skipping '
                          f'(use --plots-only to regenerate figures)')
                    continue
            else:
                work.append((analysis, False))

        if not work:
            print(f'{pid} ({date}): all analyses complete — skipping.')
            continue

        analyses_str = [a for a, _ in work]
        print(f'{pid} ({date}): queuing {analyses_str}')
        for analysis, plots_only in work:
            jobs.append({
                'patient_id':      pid,
                'date':            date,
                'edf':             session['edf'],
                'csv':             session['csv'],
                'analysis':        analysis,
                'n_perms':         n_perms,
                'plots_only':      plots_only,
                'force_rejection': args.force_rejection,
                'force':           args.force,
            })

    if not jobs:
        print('Nothing to run.')
    elif args.workers > 1:
        ctx = _mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(_run_patient_job, job): job for job in jobs}
            for fut in as_completed(futures):
                print(fut.result())
    else:
        for job in jobs:
            pid      = job['patient_id']
            date     = job['date']
            analysis = job['analysis']
            print(f'\n{"="*60}\n{pid} ({date}): running {analysis}\n{"="*60}')
            msg = _run_patient_job(job)
            print(msg)

    # Pooled analyses run after all per-patient analyses complete
    if args.pooled or (args.analyses and 'pooled_oddball' in args.analyses):
        print(f'\n{"="*60}\nPooled oddball analysis\n{"="*60}')
        try:
            run_pooled_oddball(RESULTS_DIR, n_perms=n_perms)
        except Exception as e:
            import traceback; print(f'  ERROR in pooled: {e}'); traceback.print_exc()

    if args.pooled or (args.analyses and 'pooled_resting' in args.analyses):
        print(f'\n{"="*60}\nPooled resting-state microstate analysis\n{"="*60}')
        try:
            run_pooled_resting(RESULTS_DIR)
        except Exception as e:
            import traceback; print(f'  ERROR in pooled resting: {e}'); traceback.print_exc()

    print('\nDone.')


if __name__ == '__main__':
    main()
