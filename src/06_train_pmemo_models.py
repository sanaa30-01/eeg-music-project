"""
Train and select the Stage B (audio-only) PMEmo models.

Split protocol: configs/splits.yaml -> pmemo_internal_model_selection /
pmemo_final_estimate (GroupKFold by song/clip ID, nested CV for the final
estimate). NEVER let windows from one song cross folds.

Models (see README / project doc "core model suite" for the full table):
  - Must-have: Ridge / Elastic Net (audio), SVR-RBF (audio)
  - Nice-to-have: Random Forest, OpenL3/VGGish embeddings + Ridge
  - Cut first if behind schedule: late fusion audio+EEG model

Model selection rule: configs/splits.yaml -> model_selection_rule
(primary metric = concordance correlation coefficient, tie-break = RMSE).
LOCK the winning model before generating any ds002721 predictions
(configs/analysis_plan.yaml -> H3_bridge condition) — save it to
data_processed/models/pmemo_final_model.joblib and do not retune afterward.

Reads: data_processed/audio_features_pmemo.parquet
Writes: results/pmemo_model_comparison.csv
        results/pmemo_predicted_vs_observed.png
        data_processed/models/pmemo_final_model.joblib

TODO (Days 15-19)
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_CFG = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text())


def main() -> None:
    raise NotImplementedError("Days 15-19 task. See configs/splits.yaml -> pmemo_*.")


if __name__ == "__main__":
    main()
