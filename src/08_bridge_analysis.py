"""
Apply the FROZEN Stage B (PMEmo) model to recovered ds002721 clips and
compare against ds002721 self-report / EEG.

HARD CONSTRAINT (configs/splits.yaml -> bridge_analysis.constraint):
the Stage B model is never retrained, tuned, or recalibrated here. Load it
from data_processed/models/pmemo_final_model.joblib and only run .predict().

Only proceeds meaningfully if Gate A passed — check
configs/analysis_plan.yaml -> gate_a_decision_rule.decision before running
the full bridge; if it's "fail", this script should still run but the
results get written up as a construct-level comparison, NOT a direct
model-to-EEG validation (see README "What we will not claim").

Reads: data_processed/models/pmemo_final_model.joblib
       data_processed/audio_features_ds002721.parquet
       data_processed/trials_ds002721.parquet (for self-report composites)
       data_processed/eeg_features.parquet (for the secondary EEG association test)
Writes: data_processed/bridge_predictions.parquet
        results/bridge_scatterplots.png
        results/alignment_log.csv

TODO (Days 22-23)
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_CFG = yaml.safe_load((REPO_ROOT / "configs" / "analysis_plan.yaml").read_text())


def main() -> None:
    gate_decision = PLAN_CFG["gate_a_decision_rule"]["decision"]
    if gate_decision is None:
        raise RuntimeError(
            "configs/analysis_plan.yaml -> gate_a_decision_rule.decision is still null. "
            "Finish the Gate A audit (src/01_audit_data.py) and record the decision first."
        )
    raise NotImplementedError(f"Days 22-23 task. Gate A decision on file: {gate_decision!r}")


if __name__ == "__main__":
    main()
