"""
Extract clip-level audio features (PMEmo, and ds002721 clips IF Gate A passes).

Settings: configs/features.yaml -> audio (22050 Hz mono, MFCCs+deltas, RMS,
tempo/beat, spectral centroid/bandwidth/rolloff/contrast/flatness/flux,
chroma+tonal centroid, ZCR; clip-level mean/std/p10/p90).

Reads: data_raw/pmemo/ (chorus MP3s)
       data_raw/ds002721_audio/ (ONLY if Gate A passed — see configs/analysis_plan.yaml)
Writes: data_processed/audio_features_pmemo.parquet   (key: track_id)
        data_processed/audio_features_ds002721.parquet (if applicable)

Standardize using TRAINING-FOLD means only (configs/features.yaml ->
audio.standardization) — never fit the scaler on data that includes test rows.

If PMEmo's own precomputed openSMILE features are used for the MVP baseline,
keep that feature set in a clearly separate column namespace / separate file
from the independently-extracted librosa set here. Do not merge them silently
(see configs/analysis_plan.yaml -> exclusions).

TODO (Days 15-19)
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CFG = yaml.safe_load((REPO_ROOT / "configs" / "features.yaml").read_text())


def main() -> None:
    raise NotImplementedError("Days 15-19 task. See configs/features.yaml -> audio.")


if __name__ == "__main__":
    main()
