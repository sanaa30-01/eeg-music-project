"""
Bootstrap confidence intervals, permutation tests, and FDR correction for
final reported metrics.

Settings: configs/splits.yaml -> uncertainty
  - 2000 bootstrap resamples, resampled BY SONG for PMEmo metrics and BY CLIP
    for bridge correlations (never row-level resampling)
  - 1000-permutation test for the key bridge correlations
  - Benjamini-Hochberg FDR across the 8 secondary self-report outcomes
    (configs/analysis_plan.yaml -> multiple_comparisons)

Reads: results/*.csv (model outputs from 06, 07, 08)
Writes: results/final_metrics_with_ci.csv

TODO (Days 24-25)
"""

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_CFG = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text())


def main() -> None:
    raise NotImplementedError("Days 24-25 task. See configs/splits.yaml -> uncertainty.")


if __name__ == "__main__":
    main()
