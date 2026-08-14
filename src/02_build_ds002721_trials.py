"""
Build one row per participant x clip trial from the ds002721 BIDS tree.

Reads: data_raw/ds002721/ (BIDS: events.tsv per subject, self-report ratings)
Writes: data_processed/trials_ds002721.parquet
        (+ a data dictionary describing every column)

Load with mne_bids.read_raw_bids() / BIDSPath — do not hand-parse file paths.
One row = one (participant, clip) pair, columns = the 8 Likert ratings
(pleasant, energetic, tense, angry, fearful, happy, sad, tender) plus
participant_id, clip_id, and any trial-level metadata needed downstream
(e.g. presentation order, if relevant to QC).

TODO (Days 6-7):
  - Load BIDS events/ratings with mne_bids
  - Validate rating scales (expected 1-9) and trial counts per participant
  - Merge into trials_ds002721.parquet
  - Write data dictionary alongside it
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIDS_ROOT = REPO_ROOT / "data_raw" / "ds002721"
OUT_PATH = REPO_ROOT / "data_processed" / "trials_ds002721.parquet"


def main() -> None:
    raise NotImplementedError(
        "Days 6-7 task. Load ratings/events via mne_bids, build one row per "
        "participant x clip, write to data_processed/trials_ds002721.parquet."
    )


if __name__ == "__main__":
    main()
