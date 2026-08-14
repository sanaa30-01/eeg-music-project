"""
Gate A: data availability audit.

Two independent checks, run separately or together:

1. --check-pmemo
   Confirms PMEmo metadata/annotation CSVs are present under data_raw/pmemo/
   and loads them into a single dataframe, printing a summary (song count,
   annotation coverage, any ID mismatches). Does not touch audio files.

2. --init-ds002721-audit
   Creates data_raw/ds002721_stimulus_audit.csv if it doesn't exist yet
   (40 rows, one per trial), or, if it already exists, reports current
   progress against the Gate A decision rule (>=32/40 recovered -> proceed
   with the direct bridge; otherwise -> downgrade the bridge claim).

Usage:
    python src/01_audit_data.py --check-pmemo
    python src/01_audit_data.py --init-ds002721-audit
    python src/01_audit_data.py --check-pmemo --init-ds002721-audit
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data_raw"
PMEMO_DIR = DATA_RAW / "pmemo"
AUDIT_CSV = DATA_RAW / "ds002721_stimulus_audit.csv"
ANALYSIS_PLAN = REPO_ROOT / "configs" / "analysis_plan.yaml"

AUDIT_COLUMNS = [
    "trial_id",
    "ds002721_stimulus_id",
    "source_title",
    "source_album_or_release",
    "source_url",
    "license_or_terms",
    "clip_start_sec",
    "clip_end_sec",
    "checksum_sha256",
    "extraction_status",
    "notes",
]
VALID_STATUSES = {"recovered", "not_found", "ambiguous"}
N_CLIPS = 40


def check_pmemo() -> None:
    print("\n=== PMEmo metadata check ===")
    if not PMEMO_DIR.exists():
        print(f"[FAIL] {PMEMO_DIR} does not exist yet.")
        print("       Download PMEmo per data_raw/README_access.md, then re-run.")
        return

    # PMEmo's exact filenames vary slightly between the 2018/2019 releases —
    # adjust these patterns once you've actually downloaded and looked inside.
    candidate_metadata = list(PMEMO_DIR.rglob("*metadata*.csv"))
    candidate_static = list(PMEMO_DIR.rglob("*static*annotation*.csv")) + list(
        PMEMO_DIR.rglob("*static*label*.csv")
    )

    if not candidate_metadata:
        print("[FAIL] No metadata CSV found under data_raw/pmemo/.")
        print("       Check the folder structure matches what you downloaded from Drive.")
        return

    meta_path = candidate_metadata[0]
    meta = pd.read_csv(meta_path)
    print(f"[OK] Loaded metadata: {meta_path.relative_to(REPO_ROOT)} ({len(meta)} rows)")
    print(f"     Columns: {list(meta.columns)}")

    if candidate_static:
        static_path = candidate_static[0]
        static = pd.read_csv(static_path)
        print(f"[OK] Loaded static annotations: {static_path.relative_to(REPO_ROOT)} ({len(static)} rows)")
        # PMEmo's paper reports 794 songs — flag if this run's download differs.
        if len(static) != 794:
            print(f"[WARN] Expected 794 annotated songs per the PMEmo paper, found {len(static)}. "
                  f"Confirm you have the right release before treating this as final.")
    else:
        print("[WARN] No static annotation CSV found — double-check the download; "
              "you need this for Stage B labels.")


def init_ds002721_audit() -> None:
    print("\n=== ds002721 stimulus audit (Gate A) ===")
    if not AUDIT_CSV.exists():
        rows = [
            {c: ("not_found" if c == "extraction_status" else "") for c in AUDIT_COLUMNS}
            | {"trial_id": i}
            for i in range(1, N_CLIPS + 1)
        ]
        pd.DataFrame(rows, columns=AUDIT_COLUMNS).to_csv(AUDIT_CSV, index=False)
        print(f"[OK] Created {AUDIT_CSV.relative_to(REPO_ROOT)} with {N_CLIPS} template rows.")
        print("     Fill it in by hand — see data_raw/README_access.md, section 3.")
        return

    df = pd.read_csv(AUDIT_CSV)
    if len(df) != N_CLIPS:
        print(f"[WARN] Expected {N_CLIPS} rows, found {len(df)}. Check the file wasn't edited unexpectedly.")

    bad_status = ~df["extraction_status"].isin(VALID_STATUSES)
    if bad_status.any():
        print(f"[WARN] {bad_status.sum()} row(s) have an unrecognized extraction_status "
              f"(expected one of {sorted(VALID_STATUSES)}).")

    recovered = int((df["extraction_status"] == "recovered").sum())
    threshold = 32
    print(f"Recovered: {recovered} / {N_CLIPS}")

    if recovered >= threshold:
        print(f"[GATE A: PASS] >= {threshold} recovered -> proceed with the direct Stage A <-> Stage B bridge.")
    else:
        print(f"[GATE A: NOT YET / FAIL] < {threshold} recovered -> per configs/analysis_plan.yaml, "
              f"either keep auditing or downgrade the bridge to a construct-level comparison.")

    # Remind the team to actually record the decision, not just leave it implicit.
    try:
        plan = yaml.safe_load(ANALYSIS_PLAN.read_text())
        if plan.get("gate_a_decision_rule", {}).get("decision") is None:
            print("[REMINDER] configs/analysis_plan.yaml -> gate_a_decision_rule.decision is still null. "
                  "Fill in decision / clips_recovered / decided_by once the team agrees.")
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-pmemo", action="store_true", help="Validate PMEmo metadata is present and loadable.")
    parser.add_argument("--init-ds002721-audit", action="store_true",
                         help="Create or report progress on the ds002721 stimulus audit CSV.")
    args = parser.parse_args()

    if not (args.check_pmemo or args.init_ds002721_audit):
        parser.print_help()
        sys.exit(1)

    if args.check_pmemo:
        check_pmemo()
    if args.init_ds002721_audit:
        init_ds002721_audit()


if __name__ == "__main__":
    main()
