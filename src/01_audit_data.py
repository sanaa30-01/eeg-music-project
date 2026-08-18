"""
Gate A: data availability audit.

--check-pmemo
   Confirms PMEmo metadata/annotation CSVs are present and loads them.

--init-ds002721-audit
   Scans every subject's events.tsv, extracts real stimulus codes (anything
   NOT in NON_STIMULUS_CODES), checks each one against the recovered audio
   folder, and writes data_raw/ds002721_stimulus_audit.csv automatically.
   Also records how many participants heard each clip, since clips are a
   random per-participant draw from a shared pool, not a fixed shared set.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data_raw"
PMEMO_DIR = DATA_RAW / "pmemo"
AUDIO_DIR = DATA_RAW / "eerola_soundtracks" / "Set1"  # adjust if your folder is named differently
DS002721_DIR = DATA_RAW / "ds002721"
AUDIT_CSV = DATA_RAW / "ds002721_stimulus_audit.csv"
ANALYSIS_PLAN = REPO_ROOT / "configs" / "analysis_plan.yaml"

NON_STIMULUS_CODES = {
    0, 264, 266, 277,  # undocumented, below the stimulus range -- not clips
    786, 788,
    800, 801, 802, 803, 804, 805, 806, 807,
    833, 834, 835, 836, 837, 838, 839, 840, 841,
    901, 902, 903, 904, 905, 906, 907, 908, 909,
    257, 259, 260, 263, 32768, 1092,
    33568, 33569, 33570, 33571, 33572, 33573, 33574, 33575,
}


def check_pmemo() -> None:
    print("\n=== PMEmo metadata check ===")
    if not PMEMO_DIR.exists():
        print(f"[FAIL] {PMEMO_DIR} does not exist yet.")
        print("       Download PMEmo per data_raw/README_access.md, then re-run.")
        return

    candidate_metadata = list(PMEMO_DIR.rglob("*metadata*.csv"))
    candidate_static = list(PMEMO_DIR.rglob("*static*annotation*.csv")) + list(
        PMEMO_DIR.rglob("*static*label*.csv")
    )

    if not candidate_metadata:
        print("[FAIL] No metadata CSV found under data_raw/pmemo/.")
        return

    meta_path = candidate_metadata[0]
    meta = pd.read_csv(meta_path)
    print(f"[OK] Loaded metadata: {meta_path.relative_to(REPO_ROOT)} ({len(meta)} rows)")

    if candidate_static:
        static_path = candidate_static[0]
        static = pd.read_csv(static_path)
        print(f"[OK] Loaded static annotations: {static_path.relative_to(REPO_ROOT)} ({len(static)} rows)")
        if len(static) != 767:
            print(f"[WARN] Expected 767 annotated songs per the PMEmo paper, found {len(static)}.")
    else:
        print("[WARN] No static annotation CSV found.")


def build_ds002721_audit() -> None:
    print("\n=== ds002721 stimulus audit (Gate A) ===")
    if not DS002721_DIR.exists():
        print(f"[FAIL] {DS002721_DIR} does not exist yet. Download ds002721 first.")
        return
    if not AUDIO_DIR.exists():
        print(f"[FAIL] {AUDIO_DIR} does not exist yet. Download the Eerola/Vuoskoski audio first.")
        return

    per_clip_participants: dict[int, set[str]] = {}
    for f in sorted(DS002721_DIR.glob("sub-*/eeg/*events.tsv")):
        sub = f.name.split("_")[0]
        df = pd.read_csv(f, sep="\t")
        music = df[(~df["trial_type"].isin(NON_STIMULUS_CODES)) & (df["trial_type"] >= 301)]
        for code in music["trial_type"].unique():
            per_clip_participants.setdefault(int(code), set()).add(sub)

    rows = []
    for i, (code, subs) in enumerate(sorted(per_clip_participants.items()), start=1):
        fname = f"{code - 300:03d}.mp3"
        recovered = (AUDIO_DIR / fname).exists()
        rows.append({
            "trial_id": i,
            "ds002721_stimulus_id": code,
            "source_title": "",
            "source_album_or_release": "",
            "source_url": "https://osf.io/nmr6w/",
            "license_or_terms": "Eerola & Vuoskoski (2011) Soundtracks corpus -- research use",
            "clip_start_sec": "",
            "clip_end_sec": "",
            "checksum_sha256": "",
            "extraction_status": "recovered" if recovered else "not_found",
            "n_participants": len(subs),
            "notes": fname,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(AUDIT_CSV, index=False)

    recovered_n = int((df_out["extraction_status"] == "recovered").sum())
    print(f"[OK] Wrote {len(df_out)} rows to {AUDIT_CSV.relative_to(REPO_ROOT)}")
    print(f"Recovered: {recovered_n} / {len(df_out)}")
    print(f"Clips with only 1 participant: {int((df_out['n_participants'] == 1).sum())}")
    print(f"Clips with 5+ participants: {int((df_out['n_participants'] >= 5).sum())}")

    try:
        plan = yaml.safe_load(ANALYSIS_PLAN.read_text())
        if plan.get("gate_a_decision_rule", {}).get("decision") is None:
            print("[REMINDER] configs/analysis_plan.yaml -> gate_a_decision_rule.decision is still null.")
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-pmemo", action="store_true")
    parser.add_argument("--init-ds002721-audit", action="store_true",
                         help="Build the ds002721 stimulus audit CSV from real events.tsv data.")
    args = parser.parse_args()

    if not (args.check_pmemo or args.init_ds002721_audit):
        parser.print_help()
        sys.exit(1)

    if args.check_pmemo:
        check_pmemo()
    if args.init_ds002721_audit:
        build_ds002721_audit()


if __name__ == "__main__":
    main()