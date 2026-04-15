"""Notebook-friendly I/O helpers for bedside EEG analysis."""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd


DEFAULT_EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "Cz",
    "C3",
    "C4",
    "T3",
    "T4",
    "T5",
    "T6",
    "Pz",
    "P3",
    "P4",
    "O1",
    "O2",
]

_CHANNEL_RENAME_MAP = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def resolve_analysis_root(start: str | Path = ".") -> Path:
    """Return the repository's ``analysis`` directory from any notebook cwd."""
    path = Path(start).resolve()
    candidates = [path, *path.parents]
    for candidate in candidates:
        if candidate.name == "analysis" and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("Could not locate the analysis root directory from the current working directory.")


def build_subject_paths(
    subject_id: str,
    session_date: str,
    edf_filename: str | None = None,
    csv_filename: str | None = None,
    analysis_root: str | Path | None = None,
) -> dict[str, Path]:
    """Build standard EDF/CSV/output paths for a subject session."""
    analysis_dir = resolve_analysis_root(analysis_root or Path.cwd())
    repo_root = analysis_dir.parent
    base_dir = repo_root / "stimulus_software" / "patient_data"
    edf_dir = base_dir / "edfs"
    csv_dir = base_dir / "results"

    edf_name = edf_filename or f"{subject_id}_clipped.EDF"
    csv_name = csv_filename or f"{subject_id}_{session_date}_stimulus_results.csv"

    edf_path = edf_dir / edf_name
    csv_path = csv_dir / csv_name
    out_dir = analysis_dir / "results" / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    return {
        "analysis_root": analysis_dir,
        "repo_root": repo_root,
        "base_dir": base_dir,
        "edf_dir": edf_dir,
        "csv_dir": csv_dir,
        "edf_path": edf_path,
        "csv_path": csv_path,
        "out_dir": out_dir,
    }


def load_raw_eeg_metadata(
    edf_path: str | Path,
    eeg_channels: list[str] | None = None,
    bad_channels: list[str] | None = None,
    preload: bool = False,
    verbose: bool | str = False,
) -> tuple[mne.io.BaseRaw, float, list[str]]:
    """Open the EDF, normalize channel names, and report available EEG channels."""
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=verbose)
    sfreq = raw.info["sfreq"]

    rename_map = {src: dst for src, dst in _CHANNEL_RENAME_MAP.items() if src in raw.ch_names}
    if rename_map:
        raw.rename_channels(rename_map)

    dc_channels = [ch for ch in raw.ch_names if "DC" in ch]
    if dc_channels:
        raw.set_channel_types({ch: "misc" for ch in dc_channels})

    bads = list(bad_channels or [])
    raw.info["bads"] = bads

    requested_eeg = list(eeg_channels or DEFAULT_EEG_CHANNELS)
    available_eeg = [ch for ch in requested_eeg if ch in raw.ch_names and ch not in bads]
    if not available_eeg:
        raise RuntimeError("No expected EEG channels found in the EDF after channel renaming.")

    return raw, sfreq, available_eeg


def align_stimulus_csv(
    csv_path: str | Path,
    sfreq: float,
    n_times: int,
) -> tuple[pd.DataFrame, float]:
    """Load the stimulus CSV and align event times into EDF sample space.

    Requires two rows to be present in the CSV:
      - ``stim_type == 'manual_sync_pulse'``: written by stimulus_software during the session
        when the clinician sends the sync pulse; ``start_time`` is the DAC wall-clock time of
        the pulse onset.
      - ``stim_type == 'sync_detection'``: records the EDF time (seconds from EDF start) at
        which the sync pulse was detected on the DC audio channel. This row is written
        *after* the session — it is not part of the live recording. The primary source is
        stimulus_software's EDF viewer (``lib/edf_viewer.py``): the clinician opens the EDF,
        detects the sync pulse on the DC channel, and the viewer saves the row back to the CSV.
        Other detection methods may produce this row too — as long as the format matches.
        The ``notes`` field must contain ``csv_pulse_idx=<n>`` to identify which
        ``manual_sync_pulse`` row to pair with (typically ``csv_pulse_idx=0``).

    Returns:
        df: DataFrame with added columns ``edf_start``, ``edf_end``, ``start_sample``, ``end_sample``
        time_offset: scalar offset such that ``edf_time = csv_dac_time + time_offset``
    """
    df = pd.read_csv(csv_path)
    df = df[df["stim_type"] != "stim_type"].copy()
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])

    sync_det = df[df["stim_type"] == "sync_detection"]
    sync_man = df[df["stim_type"] == "manual_sync_pulse"]

    if sync_det.empty:
        raise RuntimeError("No sync_detection row found in CSV.")
    if sync_man.empty:
        raise RuntimeError(
            "No manual_sync_pulse row found. This notebook assumes the stimulus CSV contains both "
            "manual_sync_pulse and sync_detection rows for EDF alignment."
        )

    sync_edf_sec = sync_det.iloc[0]["start_time"]
    notes = str(sync_det.iloc[0].get("notes", ""))
    pulse_idx = int(notes.split("csv_pulse_idx=")[1].split(",")[0].strip()) if "csv_pulse_idx=" in notes else 0
    sync_csv_dac_time = sync_man.iloc[pulse_idx]["start_time"]
    time_offset = sync_edf_sec - sync_csv_dac_time

    df["edf_start"] = df["start_time"] + time_offset
    df["edf_end"] = df["end_time"] + time_offset
    df["start_sample"] = (df["edf_start"] * sfreq).astype(int).clip(0, n_times - 1)
    df["end_sample"] = (df["edf_end"] * sfreq).astype(int).clip(0, n_times - 1)

    return df, time_offset
