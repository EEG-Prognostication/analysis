# analysis

EEG analysis pipeline for bedside cognitive assessment.

## Setup

```bash
cd analysis
source .venv/bin/activate
```

## Batch runner

`run_all.py` runs `oddball`, `language`, and `command` (the default analyses) for every patient that has both an EDF and a CSV. Results are written to `results/<PatientID>/`.

```bash
python run_all.py                               # all missing default analyses
python run_all.py --force                       # re-run even if outputs exist
python run_all.py --patients CON012 CON015      # specific patients
python run_all.py --analyses oddball language   # specific analyses
```

### Optional analyses

`spindles` and `resting` characterize the *passive resting* EEG (before the first stimulus) rather than task-evoked responses. They are **not** run by default — request them explicitly:

```bash
python run_all.py --analyses spindles resting   # passive/resting-state suite only
python run_all.py --analyses resting --patients CON014
```

After running, regenerate PDF reports:

```bash
python generate_reports.py
```

## Notebooks

Each notebook is self-contained. Set `SUBJECT_ID` and `SESSION_DATE` in the configuration cell and run.

| Notebook | Paradigm | Analysis | Positive finding |
| --- | --- | --- | --- |
| `oddball_p300.ipynb` | Oddball | P300 ERP at 300–600 ms + permutation test | Cognitive detection of deviant tone |
| `language_tracking.ipynb` | Language | ITPC at 0.78 / 1.56 / 3.125 Hz + permutation test | Neural entrainment to speech rhythm |
| `command_following.ipynb` | Motor command | Mu/beta ERD at C3/C4 + SVM | Lateralized motor imagery response |
| `voice_familiarity.ipynb` | Loved one voice | Familiarity ERP at 300–600 ms + permutation test | Implicit memory / emotional processing |

Requires a `manual_sync_pulse` + `sync_detection` row pair in the CSV for timestamp alignment.

The oddball analysis additionally reports two complementary, additive single-trial measures alongside the ERP: delta-gamma phase-amplitude coupling at Pz (`_oddball_pac.png`) and Lempel-Ziv complexity (`_oddball_lzc.png`), plus an XDAWN+Riemannian-MDM single-trial classifier run alongside the existing SVM (`_oddball_xdawn_null.png`).

The command-following SVM uses log-transformed PSD band-power features (EEG power is approximately log-normal) and the Riemannian MDM classifier clips raw amplitude to ±200µV before covariance estimation — both prevent rare electrode-artifact transients (e.g. a brief lead pop) from dominating the linear/covariance-based classifiers, without rejecting any epochs.

## Resting-state suite (optional)

Runs entirely on the passive EEG recorded before the first delivered stimulus — no task or patient cooperation required. Located via `first_paradigm_onset()` in `lib/io.py`, capped at 300 s, skipped if shorter than 30 s.

| Module | Measures | Output |
| --- | --- | --- |
| `lib/resting.py` (`--analyses resting`) | Permutation entropy, normalized Lempel-Ziv complexity, EEG microstates, weighted Symbolic Mutual Information (wSMI), background EEG quality screen | `results/<PatientID>/resting/` + `metadata.json` |
| `lib/spindles.py` (`--analyses spindles`) | 12–15 Hz sleep spindle detection (Hilbert envelope, rate + amplitude) | `results/<PatientID>/spindles/` + `metadata.json` |

`generate_reports.py` builds a dedicated `report_resting_state.pdf` when these have been run.

## Shared helpers

- `lib/io.py` — path resolution, EDF loading, CSV alignment, and `first_paradigm_onset()` (locates the pre-paradigm resting window)
- `lib/preprocessing.py` — bandpass/notch filter helpers

## Source data

EDF recordings and stimulus CSVs live in `stimulus_software/patient_data/` and are gitignored.
