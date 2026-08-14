"""
Filter, epoch, and artefact-reject ds002721 EEG.

Settings come from configs/features.yaml (eeg.filtering, eeg.epoching,
eeg.artefact_rejection) — do not hardcode values here.

Steps (see configs/features.yaml for exact parameters):
  1. Load raw BIDS EEG via mne_bids
  2. Inspect power spectrum before deciding on notch filter (50 vs 60 Hz)
  3. Band-pass filter (default 1-45 Hz)
  4. Set average reference (after confirming montage / bad channels)
  5. Epoch: primary window 1-11s post-onset (excludes onset transient and
     rating interval)
  6. Reject epochs beyond the configured amplitude threshold
  7. Log retained trial counts PER PARTICIPANT and PER CLIP — this is a
     required Week 2 deliverable (trial-retention figure)

Writes: data_interim/epochs_ds002721/  (per-subject -epo.fif files)
        results/trial_retention.csv

TODO (Days 8-10): implement per configs/features.yaml
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CFG = yaml.safe_load((REPO_ROOT / "configs" / "features.yaml").read_text())


def main() -> None:
    eeg_cfg = FEATURES_CFG["eeg"]
    raise NotImplementedError(
        f"Days 8-10 task. Use eeg config: {eeg_cfg}"
    )


if __name__ == "__main__":
    main()
