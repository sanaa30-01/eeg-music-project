"""
Bridge analysis: apply the FROZEN Stage B (PMEmo) audio models to the
audited ds002721 clips.

Feature extraction reuses 05_extract_audio_features.py's actual functions
directly (imported, not re-implemented) -- zero risk of a subtle mismatch
between how PMEmo's training features and ds002721's bridge features were
computed. A mismatch here would not crash; it would silently produce
meaningless predictions.

HARD CONSTRAINT: the Stage B models are never retrained or recalibrated
here -- loaded via joblib and only ever .predict()'d.
"""

import hashlib
import joblib
import importlib.util
import sys
from pathlib import Path
import soundfile as sf 
import librosa
from scipy.stats import pearsonr 

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "data_raw" / "eerola_soundtracks" / "Set1" # adjust if your folder's named differently
AUDIT_CSV = REPO_ROOT / "data_raw" / "ds002721_stimulus_audit.csv"
MANIFEST_PATH = REPO_ROOT / "data_processed" / "models" / "pmemo_model_manifest.yaml"
OUT_FEATURES = REPO_ROOT / "data_processed" / "audio_features_ds002721.parquet"
TRIMMED_AUDIO_DIR = REPO_ROOT / "data_raw" / "ds002721_audio_trimmed"
TARGET_DURATION_SEC = 12.0 


def _load_stage_b_extractor():
    """Import 05_extract_audio_features.py despite its filename starting
    with a digit (which makes a plain `import` statement invalid Python).
    Loading it this way reuses his ACTUAL functions verbatim -- no
    re-typing the feature logic by hand, which is exactly where a silent,
    unnoticed mismatch could creep in.
    """
    module_path = REPO_ROOT / "src" / "05_extract_audio_features.py"
    spec = importlib.util.spec_from_file_location("stage_b_extractor", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_b_extractor"] = module   # registering before exec_module runs; needed to resolve @dataclass' type hints 
    spec.loader.exec_module(module)
    return module

def trim_ds002721_clips(audit_csv: Path, source_audio_dir: Path,
                         trimmed_dir: Path, settings) -> None:
    """Trim every recovered ds002721 clip to its first 12 seconds.

    Writes trimmed copies to trimmed_dir (originals untouched) and updates
    the audit CSV's clip_start_sec/clip_end_sec/notes -- filling in fields
    that were left blank since the original Gate A pass. Clips shorter than
    the 12s target can't be trimmed to it; these are flagged distinctly
    rather than silently included at their shorter, mismatched length.
    """
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(audit_csv)

    for col in ("clip_start_sec", "clip_end_sec", "trim_status"):
        if col not in audit.columns:
            audit[col] = None

    for idx, row in audit[audit["extraction_status"] == "recovered"].iterrows():
        source_path = source_audio_dir / row["notes"]  # 'notes' holds the mp3 filename
        if not source_path.exists():
            continue

        waveform, sr = librosa.load(source_path, sr=settings.sample_rate_hz, mono=settings.mono)
        original_duration = len(waveform) / sr

        if original_duration < TARGET_DURATION_SEC:
            # can't trim to 12s -- shorter than the target window. Flag it;
            # do not silently write a shorter file that would still mismatch
            # PMEmo's expected clip length.
            audit.at[idx, "trim_status"] = f"too_short (source={original_duration:.2f}s)"
            audit.at[idx, "clip_start_sec"] = 0.0
            audit.at[idx, "clip_end_sec"] = round(original_duration, 2)
            continue

        n_samples_target = int(TARGET_DURATION_SEC * sr)
        trimmed_waveform = waveform[:n_samples_target]

        out_path = trimmed_dir / row["notes"]
        sf.write(out_path, trimmed_waveform, sr)

        audit.at[idx, "trim_status"] = "trimmed_first_12s"
        audit.at[idx, "clip_start_sec"] = 0.0
        audit.at[idx, "clip_end_sec"] = TARGET_DURATION_SEC

    audit.to_csv(audit_csv, index=False)

    n_trimmed = (audit["trim_status"] == "trimmed_first_12s").sum()
    n_too_short = audit["trim_status"].astype(str).str.startswith("too_short").sum()
    print(f"Trimmed: {n_trimmed}, too short to trim: {n_too_short}")
    if n_too_short:
        print(audit[audit["trim_status"].astype(str).str.startswith("too_short")]
              [["ds002721_stimulus_id", "trim_status"]]) 

def build_ds002721_manifest(audit_csv: Path, audio_dir: Path) -> pd.DataFrame:
    """One row per recovered ds002721 clip: stimulus ID + audio file path."""
    audit = pd.read_csv(audit_csv)
    recovered = audit[audit["extraction_status"] == "recovered"].copy()
    # 'notes' holds the mp3 filename, per 01_audit_data.py's build_ds002721_audit()
    recovered["audio_path"] = recovered["notes"].map(lambda fname: audio_dir / fname)
    missing = recovered[~recovered["audio_path"].map(lambda p: p.exists())]
    if len(missing):
        print(f"[WARN] {len(missing)} recovered clips have no matching file on disk -- excluding:")
        print(missing[["ds002721_stimulus_id", "audio_path"]])
    return recovered[recovered["audio_path"].map(lambda p: p.exists())][
        ["ds002721_stimulus_id", "audio_path"]
    ].reset_index(drop=True)


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"[FAIL] {MANIFEST_PATH} not found. Run 05/06 first.")
        sys.exit(1)

    manifest_yaml = yaml.safe_load(MANIFEST_PATH.read_text())
    expected_feature_order = manifest_yaml["feature_order"]
    expected_hash = manifest_yaml["feature_order_sha256"]

    stage_b = _load_stage_b_extractor()
    _audio_config, settings = stage_b.load_audio_config()

    trim_ds002721_clips(AUDIT_CSV, AUDIO_DIR, TRIMMED_AUDIO_DIR, settings) 

    ds_manifest = build_ds002721_manifest(AUDIT_CSV, TRIMMED_AUDIO_DIR)
    print(f"Extracting features for {len(ds_manifest)} ds002721 clips...")

    rows, durations, failures = [], [], []
    for _, row in ds_manifest.iterrows():
        try:
            features, qc = stage_b.extract_audio_features(row["audio_path"], settings)
            rows.append({"ds002721_stimulus_id": int(row["ds002721_stimulus_id"]), **features})
            durations.append(qc["duration_sec"])
        except Exception as exc:
            failures.append((row["ds002721_stimulus_id"], str(exc)))

    if failures:
        print(f"[WARN] {len(failures)} clips failed extraction:")
        for stim_id, err in failures:
            print(f"  {stim_id}: {err}")
    
    if not rows:
        print("[FAIL] No clips were successfully extracted -- check AUDIO_DIR "
              "and the failures list above before going further.")
        sys.exit(1)

    durations = np.array(durations)

    # THE DURATION CHECK -- this is the critical diagnostic. ds002721 clips
    # should be 12 seconds. If these numbers come back much larger and
    # spread out, that's strong evidence the files are untrimmed originals,
    # not the exact windows participants heard -- a real problem to fix
    # BEFORE trusting anything downstream of this script.
    durations = np.array(durations)
    print(f"\nClip duration check (should be ~12.0s if properly trimmed):")
    print(f"  min: {durations.min():.2f}s, max: {durations.max():.2f}s, "
          f"mean: {durations.mean():.2f}s, median: {np.median(durations):.2f}s")

    features_df = pd.DataFrame(rows)

    # Reorder to match the manifest EXACTLY -- this guarantees correctness
    # regardless of whether the hash check below matches, since forcing
    # this order is what actually determines what gets fed to .predict()
    missing_cols = set(expected_feature_order) - set(features_df.columns)
    if missing_cols:
        print(f"[FAIL] Missing columns the model expects: {sorted(missing_cols)}")
        sys.exit(1)
    features_df = features_df[["ds002721_stimulus_id"] + expected_feature_order]

    actual_hash = hashlib.sha256("\n".join(expected_feature_order).encode()).hexdigest()
    print(f"\nColumn-order hash matches manifest: {actual_hash == expected_hash}")
    if actual_hash != expected_hash:
        print("  (Mismatch here likely just means the hash was computed a different way than "
              "assumed -- the explicit column reordering above is what actually guarantees "
              "correctness, not this hash.)")

    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(OUT_FEATURES, index=False)
    print(f"\n[OK] Wrote {len(features_df)} rows to {OUT_FEATURES.relative_to(REPO_ROOT)}")


# ============================================================
# 1. LOAD THE FROZEN STAGE B MODELS -- read-only from here on
# ============================================================
valence_model = joblib.load(REPO_ROOT / "data_processed" / "models" / "pmemo_valence_model.joblib")
arousal_model = joblib.load(REPO_ROOT / "data_processed" / "models" / "pmemo_arousal_model.joblib")

# ============================================================
# 2. APPLY THEM TO THE 299 TRIMMED ds002721 CLIPS
# ============================================================
audio_features = pd.read_parquet(OUT_FEATURES)  # the file you already built
predictor_cols = [c for c in audio_features.columns if c != "ds002721_stimulus_id"]

# THE actual bridge moment: a model that has never seen ds002721, film
# soundtracks, or EEG data of any kind, predicting purely from audio
audio_features["predicted_valence"] = valence_model.predict(audio_features[predictor_cols])
audio_features["predicted_arousal"] = arousal_model.predict(audio_features[predictor_cols])

predictions = audio_features[["ds002721_stimulus_id", "predicted_valence", "predicted_arousal"]]

# ============================================================
# 3. BUILD THE REAL, POPULATION-AVERAGE SELF-REPORT PER CLIP
# ============================================================
# Same composite definitions as Stage A -- kept identical on purpose, so
# "valence" means the same thing on both sides of this comparison
trials = pd.read_parquet("data_processed/trials_ds002721.parquet")
trials["valence_like_composite"] = trials[["pleasant", "happy", "tender"]].mean(axis=1)
trials["arousal_like_composite"] = trials[["energetic", "tense", "angry", "fearful"]].mean(axis=1)

clip_level = trials.groupby("ds002721_stimulus_id").agg(
    actual_valence=("valence_like_composite", "mean"),
    actual_arousal=("arousal_like_composite", "mean"),
    n_participants=("participant_id", "nunique"),
).reset_index()

# ============================================================
# 4. JOIN PREDICTIONS TO REALITY
# ============================================================
bridge = predictions.merge(clip_level, on="ds002721_stimulus_id", how="inner")
print(f"Clips in bridge analysis: {len(bridge)}")

bridge.to_csv("results/bridge_predictions.csv", index=False)

# ============================================================
# 5. PRIMARY vs EXPLORATORY SPLIT BY RATER COUNT
# ============================================================
# threshold can be debated on
RATER_THRESHOLD = 5

well_rated = bridge[bridge["n_participants"] >= RATER_THRESHOLD]
thin = bridge[bridge["n_participants"] < RATER_THRESHOLD]

print(f"\n--- PRIMARY (>={RATER_THRESHOLD} raters, {len(well_rated)} clips) ---")
for outcome in ["valence", "arousal"]:
    r, p = pearsonr(well_rated[f"actual_{outcome}"], well_rated[f"predicted_{outcome}"])
    print(f"  {outcome}: r={r:.3f}, p={p:.4f}")

print(f"\n--- EXPLORATORY (<{RATER_THRESHOLD} raters, {len(thin)} clips) ---")
for outcome in ["valence", "arousal"]:
    r, p = pearsonr(thin[f"actual_{outcome}"], thin[f"predicted_{outcome}"])
    print(f"  {outcome}: r={r:.3f}, p={p:.4f}") 