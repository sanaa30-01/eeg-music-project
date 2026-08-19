"""
Build one row per participant x clip trial from ds002721 events.tsv files.

Trial structure (from the events.json codebook + what we learned during
Gate A):
  786          fixation cross onset
  788          music start
  301+         which clip played (subtract 300, zero-pad 3 digits -> mp3 filename)
  800-807      "Question 0N presented" (N=1..8: pleasant, energetic, tense,
                angry, fearful, happy, sad, tender, in that fixed order)
  901-909      "Answer" -- response VALUE, 1 (strongly disagree) .. 9 (strongly agree)
  833-841      a second, redundant encoding of the same button press -- not used here

One trial = everything between one stimulus code and the next (or end of run).
Each question's rating = the value of the first Answer (901-909) event that
follows that question's Question event.

Writes: data_processed/trials_ds002721.parquet
        data_processed/trials_ds002721_data_dictionary.md
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DS002721_DIR = REPO_ROOT / "data_raw" / "ds002721"
OUT_PARQUET = REPO_ROOT / "data_processed" / "trials_ds002721.parquet"
OUT_DICT = REPO_ROOT / "data_processed" / "trials_ds002721_data_dictionary.md"

NON_STIMULUS_CODES = {
    0, 264, 266, 277,
    786, 788,
    800, 801, 802, 803, 804, 805, 806, 807,
    833, 834, 835, 836, 837, 838, 839, 840, 841,
    901, 902, 903, 904, 905, 906, 907, 908, 909,
    257, 259, 260, 263, 32768, 1092,
    33568, 33569, 33570, 33571, 33572, 33573, 33574, 33575,
}

QUESTION_MAP = {
    800: "pleasant", 801: "energetic", 802: "tense", 803: "angry",
    804: "fearful", 805: "happy", 806: "sad", 807: "tender",
}


def is_stimulus_code(code: int) -> bool:
    return code >= 301 and code not in NON_STIMULUS_CODES

def find_answer(df: pd.DataFrame, question_onset: float):
    later = df[
        (df["onset"] > question_onset)
        & (df["trial_type"].between(901, 909) | df["trial_type"].between(833, 841))
    ].sort_values("onset")
    if not len(later):
        return None
    code = int(later.iloc[0]["trial_type"])
    if 901 <= code <= 909:
        return code - 900
    else:  # 833-841
        return code - 832


def parse_run(events_path: Path, participant_id: str, run_id: str) -> list[dict]:
    df = pd.read_csv(events_path, sep="\t").sort_values("onset").reset_index(drop=True)
    trials = []
    current = None

    for _, row in df.iterrows():
        code = int(row["trial_type"])
        onset = row["onset"]

        if is_stimulus_code(code):
            if current is not None:
                trials.append(current)
            current = {
                "participant_id": participant_id,
                "run": run_id,
                "ds002721_stimulus_id": code,
                "clip_mp3": f"{code - 300:03d}.mp3",
                "music_onset_sec": None,
                **{col: None for col in QUESTION_MAP.values()},
            }
            continue

        if current is None:
            continue  # events before the first stimulus code -- skip

        if code == 788 and current["music_onset_sec"] is None:
            current["music_onset_sec"] = onset
        elif code in QUESTION_MAP:
            value = find_answer(df, onset)
            current[QUESTION_MAP[code]] = value

    if current is not None:
        trials.append(current)
    return trials


def main() -> None:
    if not DS002721_DIR.exists():
        print(f"[FAIL] {DS002721_DIR} not found. Download ds002721 first.")
        sys.exit(1)

    all_trials = []
    for events_path in sorted(DS002721_DIR.glob("sub-*/eeg/*events.tsv")):
        parts = events_path.name.split("_")
        participant_id = parts[0]
        run_id = next((p for p in parts if p.startswith("task-")), "unknown")
        all_trials.extend(parse_run(events_path, participant_id, run_id))

    trials_df = pd.DataFrame(all_trials)
    n_missing = trials_df[list(QUESTION_MAP.values())].isna().any(axis=1).sum()
    print(f"Built {len(trials_df)} trials across {trials_df['participant_id'].nunique()} participants.")
    print(f"Trials with at least one missing rating: {n_missing}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_parquet(OUT_PARQUET, index=False)
    print(f"[OK] Wrote {OUT_PARQUET.relative_to(REPO_ROOT)}")

    OUT_DICT.write_text(
        "# trials_ds002721.parquet -- data dictionary\n\n"
        "One row per (participant, clip) trial.\n\n"
        "| Column | Meaning |\n|---|---|\n"
        "| participant_id | e.g. sub-01 |\n"
        "| run | BIDS run/task label the trial came from |\n"
        "| ds002721_stimulus_id | raw event code (301+) identifying the clip |\n"
        "| clip_mp3 | filename in data_raw/eerola_soundtracks/ |\n"
        "| music_onset_sec | onset (sec) of the '788 Music played' event |\n"
        "| pleasant, energetic, tense, angry, fearful, happy, sad, tender | "
        "1-9 self-report ratings from the 901-909 Answer codes |\n\n"
        "Ratings are null where no matching Answer event was found -- check "
        "the missing-rating count printed at build time before treating "
        "this table as complete.\n"
    )
    print(f"[OK] Wrote {OUT_DICT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main() 
