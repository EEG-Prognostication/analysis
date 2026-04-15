# analysis

EEG analysis notebooks for the Harborview bedside covert cognition assessment project.

## Scope

This repository is the working repo for analysis development.
Treat `stimulus_software`, `awaken-ai`, and `eeg-analysis` as reference repos only unless a task explicitly requires syncing logic back from them.

## Setup

Use the `stimulus_software/.venv` — it has MNE 1.11, scipy, pandas, and matplotlib.

```bash
cd ../stimulus_software
source .venv/bin/activate
cd ../analysis
jupyter notebook
```

## Notebooks

Each notebook is self-contained. Set `SUBJECT_ID`, `SESSION_DATE`, and any filename overrides in the configuration cell and run.
The oddball notebook now resolves paths relative to the repo so it can be reused on another machine without editing Joey-specific absolute paths.
All notebooks now share the same EDF + sync alignment helpers from `analysis/lib/` and save outputs to `results/<SUBJECT_ID>/`.

| Notebook | Paradigm | Analysis | Positive finding |
| --- | --- | --- | --- |
| `language_itpc.ipynb` | Language | ITPC at 0.78 / 1.56 / 3.125 Hz + permutation test | Neural entrainment to speech rhythm |
| `oddball_p300.ipynb` | Oddball | P300 ERP at 300–600 ms (Pz/Cz) + permutation test | Cognitive detection of deviant tone |
| `command_following.ipynb` | Motor command | Mu/beta ERD at C3/C4 + paired t-test | Lateralized motor imagery response |
| `voice_familiarity.ipynb` | Loved one voice | Familiarity ERP at 300–600 ms + permutation test | Implicit memory / emotional processing |

Requires a `manual_sync_pulse` + `sync_detection` row pair in the CSV for timestamp alignment. See notebook Section 3.

## Current focus

- Start with `notebooks/oddball_p300.ipynb` for single-subject exploratory P300 work.
- Use `awaken-ai/src/pipelines/p300_oddball.py` only as a reference implementation when notebook logic needs comparison against the production pipeline.

## Shared Helpers

- `lib/io.py` contains repo-relative path resolution, EDF metadata loading, and stimulus CSV alignment helpers.
- `lib/preprocessing.py` contains shared EEG channel loading and standard filtering helpers.
- `reports/` and `tests/` are scaffolded for future report builders and helper tests.

## Related repos

- [stimulus_software](../stimulus_software) — stimulus delivery and source data generation reference
- [awaken-ai](../awaken-ai) — production offline pipeline reference
- [eeg-analysis](../eeg-analysis) — predecessor research library reference
