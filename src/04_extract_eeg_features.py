"""
Extract the predeclared EEG feature set from preprocessed epochs.

Settings: configs/features.yaml -> eeg.psd, eeg.bands_hz, eeg.derived
(Welch PSD, relative log band power per region, frontal alpha asymmetry).
Max 13 predictors per outcome (configs/features.yaml -> eeg.max_predictors_per_outcome)
— do not silently add more.

Reads: data_interim/epochs_ds002721/
Writes: data_processed/eeg_features.parquet

TODO (Days 11-12):
  - Welch PSD per epoch (2s windows, 50% overlap)
  - Relative log power: theta/alpha/beta (+ optional gamma) x
    frontal/central/parietal region means
  - Frontal alpha asymmetry: log(alpha_F4) - log(alpha_F3), document sign convention
  - One row per (participant, clip) matching trials_ds002721.parquet
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CFG = yaml.safe_load((REPO_ROOT / "configs" / "features.yaml").read_text())


def main() -> None:
    raise NotImplementedError("Days 11-12 task. See configs/features.yaml -> eeg.")


if __name__ == "__main__":
    main()
