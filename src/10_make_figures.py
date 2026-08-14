"""
Generate the final figure/table set for the report.

Required figures (see project doc "Required figures" / README):
  1. Pipeline diagram (PMEmo -> features -> frozen model -> ds002721
     predictions; ds002721 EEG -> features -> self-report; bridge comparison)
  2. Dataset/label compatibility table (condensed from the project doc)
  3. EEG preprocessing/QC figure (raw vs filtered trace, retained trials,
     PSD example)
  4. EEG feature-vs-emotion scatterplots: frontal alpha asymmetry vs
     valence-like; frontal/central beta or theta power vs arousal-like
  5. Stage A model performance table (baseline vs Ridge, MAE/RMSE/Pearson r,
     95% bootstrap CIs)
  6. Main music-model comparison table (Ridge vs SVR, valence vs arousal,
     identical grouped nested-CV folds)
  7. Predicted-vs-observed bridge plots (with regression line, r, CI, clip count)
  8. Error-analysis figure (absolute error by low/mid/high outcome range)
  9. Limitations/ethical-use table

Reads: results/*.csv, data_processed/*.parquet
Writes: reports/figures/*.png, reports/tables/*.csv

TODO (Days 28-29)
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raise NotImplementedError("Days 28-29 task. See docstring for the required figure/table list.")


if __name__ == "__main__":
    main()
