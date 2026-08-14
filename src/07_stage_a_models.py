"""
Fit Stage A models: EEG features -> ds002721 self-report composites.

Split protocol: configs/splits.yaml -> ds002721_stage_a
(primary = leave-one-participant-out, secondary = leave-one-clip-out).

Models: participant-mean baseline (must-have, honest baseline) and Ridge
(must-have); Elastic Net and logistic-on-quadrants are nice-to-have /
contingency only (see README core model suite table).

Primary outcomes: configs/analysis_plan.yaml -> primary_outcomes
(valence_like_composite, arousal_like_composite). Report all 8 individual
ratings as secondary outcomes with FDR correction
(configs/analysis_plan.yaml -> multiple_comparisons).

Reads: data_processed/eeg_features.parquet, data_processed/trials_ds002721.parquet
Writes: results/stage_a_model_comparison.csv
        results/stage_a_coefficient_stability.png

TODO (Days 13-14)
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_CFG = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text())
PLAN_CFG = yaml.safe_load((REPO_ROOT / "configs" / "analysis_plan.yaml").read_text())


def main() -> None:
    raise NotImplementedError("Days 13-14 task. See configs/splits.yaml -> ds002721_stage_a.")


if __name__ == "__main__":
    main()
