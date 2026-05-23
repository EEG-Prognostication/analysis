# analysis

EEG analysis pipeline for bedside cognitive assessment.

## Setup

```bash
cd analysis
source .venv/bin/activate
```

## Batch runner

`run_all.py` runs all missing analyses for every patient that has both an EDF and a CSV. Results are written to `results/<PatientID>/`.

```bash
python run_all.py                               # all missing analyses
python run_all.py --force                       # re-run even if outputs exist
python run_all.py --patients CON012 CON015      # specific patients
python run_all.py --analyses oddball language   # specific analyses
```

After running, regenerate PDF reports:

```bash
python generate_reports.py
```

## Notebooks

Each notebook is self-contained. Set `SUBJECT_ID` and `SESSION_DATE` in the configuration cell and run.

| Notebook | Paradigm | Analysis | Positive finding |
| --- | --- | --- | --- |
| `oddball_p300_erp.ipynb` | Oddball | P300 ERP at 300–600 ms + permutation test | Cognitive detection of deviant tone |
| `language_tracking_itpc.ipynb` | Language | ITPC at 0.78 / 1.56 / 3.125 Hz + permutation test | Neural entrainment to speech rhythm |
| `command_following_erd_svm.ipynb` | Motor command | Mu/beta ERD at C3/C4 + SVM | Lateralized motor imagery response |
| `voice_familiarity_erp.ipynb` | Loved one voice | Familiarity ERP at 300–600 ms + permutation test | Implicit memory / emotional processing |

Requires a `manual_sync_pulse` + `sync_detection` row pair in the CSV for timestamp alignment.

## Shared helpers

- `lib/io.py` — path resolution, EDF loading, and CSV alignment
- `lib/preprocessing.py` — bandpass/notch filter helpers

## Source data

EDF recordings and stimulus CSVs live in `stimulus_software/patient_data/` and are gitignored.
