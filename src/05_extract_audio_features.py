"""Extract reproducible clip-level audio features from PMEmo chorus MP3s.

The extraction contract comes from ``configs/features.yaml -> audio``:

* decode each clip as mono audio resampled to 22,050 Hz;
* calculate the predeclared frame-level librosa descriptors;
* summarize every descriptor dimension by mean, population standard
  deviation, 10th percentile, and 90th percentile; and
* write exactly one feature row per immutable PMEmo ``track_id``.

This script deliberately does *not* standardize features. Scaling belongs
inside the training folds in ``06_train_pmemo_models.py``; fitting a scaler to
this complete table would leak information from held-out songs.

Primary deliverables
--------------------
``data_processed/audio_features_pmemo.parquet``
    Modeling table containing ``track_id`` plus 293 numeric predictors.

``results/audio_extraction_pmemo_qc.csv``
    Audit table containing file path, decoded duration, signal checks, status,
    and any extraction error. QC fields are kept out of the model predictors.

Evaluation modes
----------------
Run a safe 20-track integration test (separate outputs):

    myenv/bin/python src/05_extract_audio_features.py --test

Run the complete PMEmo extraction after the test passes:

    myenv/bin/python src/05_extract_audio_features.py

Existing outputs are never overwritten unless ``--overwrite`` is supplied.
The ds002721 branch is intentionally not implemented here yet: Gate A in
``configs/analysis_plan.yaml`` is still unresolved, and bridge audio must use
the exact same frozen feature schema only after its provenance is verified.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
PMEMO_DIR = REPO_ROOT / "data_raw" / "PMEmo"
PMEMO_AUDIO_DIR = PMEMO_DIR / "chorus"
PMEMO_METADATA_PATH = PMEMO_DIR / "metadata.csv"
PMEMO_STATIC_LABELS_PATH = PMEMO_DIR / "annotations" / "static_annotations.csv"

DEFAULT_QC_OUTPUT = REPO_ROOT / "results" / "audio_extraction_pmemo_qc.csv"
TEST_FEATURE_OUTPUT = REPO_ROOT / "data_processed" / "audio_features_pmemo_test.parquet"
TEST_QC_OUTPUT = REPO_ROOT / "results" / "audio_extraction_pmemo_test_qc.csv"

# These frame settings were not specified numerically in features.yaml, so
# they are declared once here and recorded in the QC output. A 2,048-sample
# Hann window with a 512-sample hop is a conventional, transparent setting at
# 22,050 Hz (about 93 ms windows and 23 ms hops).
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
N_MFCC = 20
SPECTRAL_CONTRAST_BANDS = 6  # librosa returns n_bands + 1 = 7 dimensions

SUPPORTED_SUMMARY_STATS = ("mean", "std", "p10", "p90")
SUPPORTED_CONFIG_FEATURES = {
    "mfcc_20_plus_deltas",
    "rms_energy",
    "tempo_and_beat_strength",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_contrast",
    "spectral_flatness",
    "spectral_flux",
    "chroma_and_tonal_centroid",
    "zero_crossing_rate",
}

# 40 MFCC/delta dimensions + 1 RMS + 1 beat strength + 1 tempo +
# 6 scalar spectral/ZCR dimensions + 7 contrast + 12 chroma + 6 tonnetz.
# All except tempo receive four summaries: (40+1+1+6+7+12+6)*4 + 1 = 293.
EXPECTED_FEATURE_COUNT = 293


class ExtractionError(RuntimeError):
    """Raised when an audio file cannot produce a trustworthy feature row."""


@dataclass(frozen=True)
class ExtractionSettings:
    """Frozen numerical settings used identically for every audio corpus."""

    sample_rate_hz: int
    mono: bool
    summary_stats: tuple[str, ...]
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    n_mfcc: int = N_MFCC


def load_audio_config(config_path: Path = FEATURES_CONFIG_PATH) -> tuple[dict, ExtractionSettings]:
    """Load and validate the predeclared audio configuration.

    Failing early on an unknown feature/statistic is safer than silently
    producing a table whose schema no longer matches the analysis plan.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Feature configuration not found: {config_path}")

    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or "audio" not in config:
        raise ValueError(f"Missing top-level 'audio' section in {config_path}")

    audio_config = config["audio"]
    configured_features = set(audio_config.get("features", []))
    missing_features = SUPPORTED_CONFIG_FEATURES - configured_features
    unknown_features = configured_features - SUPPORTED_CONFIG_FEATURES
    if missing_features or unknown_features:
        raise ValueError(
            "Configured audio feature list does not match the implemented schema. "
            f"Missing={sorted(missing_features)}, unknown={sorted(unknown_features)}"
        )

    summary_stats = tuple(audio_config.get("clip_level_stats", []))
    if summary_stats != SUPPORTED_SUMMARY_STATS:
        raise ValueError(
            "clip_level_stats must remain ordered as "
            f"{list(SUPPORTED_SUMMARY_STATS)}, found {list(summary_stats)}"
        )

    if audio_config.get("channels") != "mono":
        raise ValueError("This extractor requires configs/features.yaml -> audio.channels: mono")

    sample_rate = int(audio_config.get("sample_rate_hz", 0))
    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate: {sample_rate}")

    settings = ExtractionSettings(
        sample_rate_hz=sample_rate,
        mono=True,
        summary_stats=summary_stats,
    )
    return audio_config, settings


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a config/CLI path relative to the repository when necessary."""

    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_pmemo_manifest(
    metadata_path: Path = PMEMO_METADATA_PATH,
    audio_dir: Path = PMEMO_AUDIO_DIR,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build the deterministic extraction manifest from PMEmo metadata.

    PMEmo's metadata is authoritative for the musicId-to-filename mapping.
    Sorting by integer track ID makes row order stable across machines and
    repeated runs.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(f"PMEmo metadata not found: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"PMEmo chorus directory not found: {audio_dir}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {"musicId", "fileName"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"PMEmo metadata is missing columns: {sorted(missing)}")

    manifest = metadata.loc[:, ["musicId", "fileName"]].copy()
    manifest = manifest.rename(columns={"musicId": "track_id", "fileName": "file_name"})
    manifest["track_id"] = pd.to_numeric(manifest["track_id"], errors="raise").astype(int)
    manifest["file_name"] = manifest["file_name"].astype(str)

    if manifest["track_id"].duplicated().any():
        duplicates = sorted(manifest.loc[manifest["track_id"].duplicated(), "track_id"].tolist())
        raise ValueError(f"Duplicate PMEmo track IDs in metadata: {duplicates[:10]}")

    manifest = manifest.sort_values("track_id", kind="stable").reset_index(drop=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit/--test-size must be a positive integer")
        manifest = manifest.head(limit).copy()

    manifest["audio_path"] = manifest["file_name"].map(lambda name: audio_dir / name)
    return manifest


def _as_2d(values: np.ndarray) -> np.ndarray:
    """Return feature values in (dimensions, frames) form."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] == 0:
        raise ExtractionError(f"Expected a nonempty 1D/2D feature array, found shape {array.shape}")
    if not np.isfinite(array).all():
        raise ExtractionError("Feature calculation produced NaN or infinite values")
    return array


def summarize_feature_matrix(
    prefix: str,
    values: np.ndarray,
    summary_stats: Iterable[str] = SUPPORTED_SUMMARY_STATS,
) -> dict[str, float]:
    """Flatten a frame-level matrix into deterministic clip-level summaries.

    Single-dimensional descriptors use names such as ``rms_mean``. Multiple
    dimensions are numbered from one with zero padding, for example
    ``mfcc_01_mean`` and ``chroma_12_p90``. ``std`` uses NumPy's population
    convention (ddof=0), which is appropriate because the frames constitute
    the complete observed clip rather than a sample requiring Bessel correction.
    """

    matrix = _as_2d(values)
    stats = tuple(summary_stats)
    if stats != SUPPORTED_SUMMARY_STATS:
        raise ValueError(f"Unsupported summary statistics/order: {stats}")

    result: dict[str, float] = {}
    width = max(2, len(str(matrix.shape[0])))
    for dimension_index, row in enumerate(matrix, start=1):
        dimension_name = prefix if matrix.shape[0] == 1 else f"{prefix}_{dimension_index:0{width}d}"
        result[f"{dimension_name}_mean"] = float(np.mean(row))
        result[f"{dimension_name}_std"] = float(np.std(row, ddof=0))
        result[f"{dimension_name}_p10"] = float(np.percentile(row, 10))
        result[f"{dimension_name}_p90"] = float(np.percentile(row, 90))
    return result


def calculate_spectral_flux(magnitude_spectrogram: np.ndarray) -> np.ndarray:
    """Calculate frame-to-frame spectral flux from normalized magnitudes.

    Normalizing every frame by its L1 magnitude prevents overall loudness from
    dominating this descriptor; RMS already represents energy. Flux is the
    Euclidean distance between adjacent normalized spectra. The first frame is
    assigned zero because it has no predecessor.
    """

    magnitude = _as_2d(magnitude_spectrogram)
    frame_totals = magnitude.sum(axis=0, keepdims=True)
    normalized = magnitude / np.maximum(frame_totals, np.finfo(np.float64).eps)
    changes = np.diff(normalized, axis=1)
    flux = np.sqrt(np.sum(changes * changes, axis=0))
    return np.concatenate(([0.0], flux))[np.newaxis, :]


def extract_audio_features(
    audio_path: Path,
    settings: ExtractionSettings,
) -> tuple[dict[str, float], dict[str, float | int | str]]:
    """Extract the complete feature schema and QC measurements for one clip."""

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # librosa performs both requirements from features.yaml at load time:
    # resampling to the fixed rate and stereo-to-mono downmixing.
    waveform, sample_rate = librosa.load(
        audio_path,
        sr=settings.sample_rate_hz,
        mono=settings.mono,
        dtype=np.float32,
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ExtractionError(f"Decoded waveform is empty or not mono: shape={waveform.shape}")
    if not np.isfinite(waveform).all():
        raise ExtractionError("Decoded waveform contains NaN or infinite samples")

    peak_amplitude = float(np.max(np.abs(waveform)))
    if peak_amplitude <= np.finfo(np.float32).eps:
        raise ExtractionError("Decoded waveform is effectively silent")

    duration_sec = float(waveform.size / sample_rate)
    if duration_sec < 1.0:
        raise ExtractionError(f"Decoded clip is unexpectedly short: {duration_sec:.3f} seconds")

    n_fft = settings.n_fft
    hop_length = settings.hop_length

    # Reuse one magnitude/power spectrogram for the spectral descriptors so
    # their frames share an identical FFT grid and hop length.
    magnitude = np.abs(
        librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
            pad_mode="constant",
        )
    )
    power = magnitude**2

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=settings.n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    # The predeclared "deltas" are first temporal derivatives only. Adding
    # delta-delta features would change the registered schema.
    mfcc_delta = librosa.feature.delta(mfcc, order=1, mode="interp")

    rms = librosa.feature.rms(
        S=magnitude,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    onset_strength = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
    )
    tempo_values = librosa.feature.tempo(
        onset_envelope=onset_strength,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )
    tempo_bpm = float(np.asarray(tempo_values).reshape(-1)[0])

    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude,
        sr=sample_rate,
        centroid=spectral_centroid,
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sample_rate,
        roll_percent=0.85,
    )
    spectral_contrast = librosa.feature.spectral_contrast(
        S=magnitude,
        sr=sample_rate,
        n_bands=SPECTRAL_CONTRAST_BANDS,
    )
    spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)
    spectral_flux = calculate_spectral_flux(magnitude)

    chroma = librosa.feature.chroma_stft(
        S=power,
        sr=sample_rate,
        n_chroma=12,
    )
    # Tonnetz is calculated from the harmonic component so that percussive
    # transients do not dominate the tonal-centroid representation.
    harmonic_waveform = librosa.effects.harmonic(waveform)
    harmonic_chroma = librosa.feature.chroma_cqt(
        y=harmonic_waveform,
        sr=sample_rate,
        hop_length=hop_length,
        n_chroma=12,
    )
    tonnetz = librosa.feature.tonnetz(chroma=harmonic_chroma, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        waveform,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    feature_matrices: tuple[tuple[str, np.ndarray], ...] = (
        ("mfcc", mfcc),
        ("mfcc_delta", mfcc_delta),
        ("rms", rms),
        ("beat_strength", onset_strength),
        ("spectral_centroid", spectral_centroid),
        ("spectral_bandwidth", spectral_bandwidth),
        ("spectral_rolloff", spectral_rolloff),
        ("spectral_contrast", spectral_contrast),
        ("spectral_flatness", spectral_flatness),
        ("spectral_flux", spectral_flux),
        ("chroma", chroma),
        ("tonnetz", tonnetz),
        ("zero_crossing_rate", zero_crossing_rate),
    )

    features: dict[str, float] = {}
    for prefix, matrix in feature_matrices:
        features.update(summarize_feature_matrix(prefix, matrix, settings.summary_stats))
    features["tempo_bpm"] = tempo_bpm

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ExtractionError(
            f"Feature schema has {len(features)} columns; expected {EXPECTED_FEATURE_COUNT}"
        )
    if not np.isfinite(np.fromiter(features.values(), dtype=np.float64)).all():
        raise ExtractionError("Final feature row contains NaN or infinite values")

    qc = {
        "duration_sec": duration_sec,
        "sample_rate_hz": int(sample_rate),
        "n_samples": int(waveform.size),
        "peak_amplitude": peak_amplitude,
        "waveform_rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
        "n_fft": n_fft,
        "hop_length": hop_length,
    }
    return features, qc


def extract_manifest(
    manifest: pd.DataFrame,
    settings: ExtractionSettings,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract every manifest row, returning feature and QC tables.

    A malformed track is recorded in QC and does not erase successful work.
    Final policy (fail or allow failures) is applied by ``main`` after outputs
    are safely written.
    """

    feature_rows: list[dict[str, float | int]] = []
    qc_rows: list[dict[str, float | int | str]] = []
    feature_schema: tuple[str, ...] | None = None

    records = manifest.to_dict(orient="records")
    iterator = tqdm(records, desc="Extracting PMEmo", unit="track", disable=not show_progress)
    for record in iterator:
        track_id = int(record["track_id"])
        file_name = str(record["file_name"])
        audio_path = Path(record["audio_path"])
        started = time.perf_counter()

        qc_row: dict[str, float | int | str] = {
            "track_id": track_id,
            "file_name": file_name,
            "audio_path": str(audio_path.relative_to(REPO_ROOT))
            if audio_path.is_relative_to(REPO_ROOT)
            else str(audio_path),
            "status": "failed",
            "error_type": "",
            "error_message": "",
        }

        try:
            features, signal_qc = extract_audio_features(audio_path, settings)
            current_schema = tuple(features)
            if feature_schema is None:
                feature_schema = current_schema
            elif current_schema != feature_schema:
                raise ExtractionError(f"Feature column order changed for track {track_id}")

            feature_rows.append({"track_id": track_id, **features})
            qc_row.update(signal_qc)
            qc_row["status"] = "ok"
        except Exception as exc:  # QC must retain the exact file-level failure.
            qc_row["error_type"] = type(exc).__name__
            qc_row["error_message"] = str(exc).replace("\n", " ")
        finally:
            qc_row["elapsed_sec"] = round(time.perf_counter() - started, 4)
            qc_rows.append(qc_row)

    features_df = pd.DataFrame(feature_rows)
    qc_df = pd.DataFrame(qc_rows)
    if not features_df.empty:
        features_df["track_id"] = features_df["track_id"].astype(int)
    return features_df, qc_df


def validate_feature_table(
    features_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    manifest: pd.DataFrame,
    static_labels_path: Path = PMEMO_STATIC_LABELS_PATH,
    require_complete_label_coverage: bool = False,
) -> dict[str, int]:
    """Validate uniqueness, schema, finiteness, extraction coverage, and labels."""

    if features_df.empty:
        raise ValueError("No feature rows were extracted successfully")
    if features_df.columns[0] != "track_id":
        raise ValueError("The first feature-table column must be track_id")
    if features_df["track_id"].duplicated().any():
        raise ValueError("Feature table contains duplicate track_id values")

    feature_columns = [column for column in features_df.columns if column != "track_id"]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Feature table has {len(feature_columns)} predictors; expected {EXPECTED_FEATURE_COUNT}"
        )

    numeric = features_df[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Feature table contains NaN or infinite predictor values")

    expected_ids = set(manifest["track_id"].astype(int))
    extracted_ids = set(features_df["track_id"].astype(int))
    if not extracted_ids.issubset(expected_ids):
        raise ValueError("Feature table contains track IDs not present in the extraction manifest")

    successful_qc_ids = set(qc_df.loc[qc_df["status"] == "ok", "track_id"].astype(int))
    if extracted_ids != successful_qc_ids:
        raise ValueError("Feature rows and successful QC rows do not identify the same tracks")

    labelled_in_manifest = 0
    missing_labelled_features = 0
    if static_labels_path.exists():
        labels = pd.read_csv(static_labels_path, usecols=["musicId"])
        label_ids = set(pd.to_numeric(labels["musicId"], errors="raise").astype(int))
        labelled_in_manifest = len(expected_ids & label_ids)
        # A test batch is responsible only for labelled tracks in that batch.
        # The full run is stricter: every static label ID must have a feature
        # row, including detection of a label accidentally absent from metadata.
        required_label_ids = label_ids if require_complete_label_coverage else (expected_ids & label_ids)
        missing_labelled_features = len(required_label_ids - extracted_ids)
        if require_complete_label_coverage and missing_labelled_features:
            raise ValueError(
                f"{missing_labelled_features} labelled PMEmo tracks lack extracted features"
            )

    return {
        "manifest_rows": len(manifest),
        "successful_rows": len(features_df),
        "failed_rows": int((qc_df["status"] != "ok").sum()),
        "feature_columns": len(feature_columns),
        "labelled_tracks_in_manifest": labelled_in_manifest,
        "missing_labelled_features": missing_labelled_features,
    }


def write_outputs(
    features_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    feature_output: Path,
    qc_output: Path,
    overwrite: bool,
) -> None:
    """Write then reopen both deliverables, refusing accidental overwrite."""

    existing = [path for path in (feature_output, qc_output) if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists ({formatted}); use --overwrite to replace it")

    feature_output.parent.mkdir(parents=True, exist_ok=True)
    qc_output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(feature_output, index=False)
    qc_df.to_csv(qc_output, index=False)

    # Reopening catches a truncated/corrupt file or an unexpected serialization
    # type before the script reports success.
    reopened_features = pd.read_parquet(feature_output)
    reopened_qc = pd.read_csv(qc_output)
    if reopened_features.shape != features_df.shape:
        raise IOError(f"Reopened Parquet shape changed: {reopened_features.shape} != {features_df.shape}")
    if reopened_qc.shape != qc_df.shape:
        raise IOError(f"Reopened QC CSV shape changed: {reopened_qc.shape} != {qc_df.shape}")


def ensure_output_targets_available(
    feature_output: Path,
    qc_output: Path,
    overwrite: bool,
) -> None:
    """Refuse an accidental overwrite before spending time on extraction."""

    existing = [path for path in (feature_output, qc_output) if path.exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists ({formatted}); use --overwrite to replace it")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface used for test and full extraction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Extract a deterministic small batch to separate test outputs.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=20,
        help="Number of sorted PMEmo tracks used by --test (default: 20).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Advanced debugging limit. Prefer --test for a safe evaluation run.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Override feature Parquet path.")
    parser.add_argument("--qc-output", type=Path, default=None, help="Override QC CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing outputs.")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if QC records failed tracks (not recommended for final extraction).",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress display.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run test or full extraction and print a concise validation summary."""

    args = build_argument_parser().parse_args(argv)
    audio_config, settings = load_audio_config()

    if args.test and args.limit is not None:
        raise ValueError("Use either --test/--test-size or --limit, not both")

    limit = args.test_size if args.test else args.limit
    manifest = load_pmemo_manifest(limit=limit)

    configured_output = resolve_repo_path(audio_config["output_file"])
    feature_output = resolve_repo_path(args.output) if args.output else (
        TEST_FEATURE_OUTPUT if args.test else configured_output
    )
    qc_output = resolve_repo_path(args.qc_output) if args.qc_output else (
        TEST_QC_OUTPUT if args.test else DEFAULT_QC_OUTPUT
    )

    # Perform this preflight before decoding hundreds of files. write_outputs
    # repeats the check to guard against a target appearing during extraction.
    ensure_output_targets_available(feature_output, qc_output, args.overwrite)

    mode = "TEST" if args.test or args.limit is not None else "FULL"
    print(f"Mode: {mode}")
    print(f"Tracks requested: {len(manifest)}")
    print(f"Audio input: {PMEMO_AUDIO_DIR.relative_to(REPO_ROOT)}")
    print(f"Feature output: {feature_output}")
    print(f"QC output: {qc_output}")
    print(
        f"Settings: mono, {settings.sample_rate_hz} Hz, n_fft={settings.n_fft}, "
        f"hop_length={settings.hop_length}"
    )

    features_df, qc_df = extract_manifest(
        manifest=manifest,
        settings=settings,
        show_progress=not args.no_progress,
    )
    summary = validate_feature_table(
        features_df=features_df,
        qc_df=qc_df,
        manifest=manifest,
        require_complete_label_coverage=(mode == "FULL"),
    )
    write_outputs(
        features_df=features_df,
        qc_df=qc_df,
        feature_output=feature_output,
        qc_output=qc_output,
        overwrite=args.overwrite,
    )

    print("\nValidation summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  feature_table_shape: {features_df.shape}")

    if summary["failed_rows"] and not args.allow_failures:
        raise RuntimeError(
            f"Extraction completed with {summary['failed_rows']} failed tracks. "
            f"Inspect {qc_output}; rerun only after resolving them."
        )

    print("Extraction completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExtractionError, FileNotFoundError, IOError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
