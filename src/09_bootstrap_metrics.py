"""
Bootstrap confidence intervals and permutation tests for Stage A and the
bridge analysis.

Per configs/splits.yaml -> uncertainty:
  - 2,000 bootstrap resamples for point-estimate confidence intervals
  - 1,000-permutation test for the key bridge correlations
  - Stage A resamples by GROUP (participant for LOPO, clip for LOCO), never
    by individual trial -- trials within one group aren't independent, so
    resampling rows directly would break the same dependency structure the
    leave-one-group-out evaluation was designed to respect in the first place.
  - Bridge resamples by ROW directly -- each row there already IS one
    independent clip, so no grouping is needed.

Reads:  data_processed/stage_a_predictions.parquet  (from 07_stage_a_models.py)
        results/bridge_predictions.csv               (from 08_bridge_analysis.py)
Writes: results/stage_a_metrics_with_ci.csv
        results/bridge_metrics_with_ci.csv

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_A_PATH = REPO_ROOT / "data_processed" / "stage_a_predictions.parquet"
BRIDGE_PATH = REPO_ROOT / "results" / "bridge_predictions.csv"
STAGE_A_OUT = REPO_ROOT / "results" / "stage_a_metrics_with_ci.csv"
BRIDGE_OUT = REPO_ROOT / "results" / "bridge_metrics_with_ci.csv"

RNG_SEED = 2026
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000
RATER_THRESHOLD = 5  # CONFIRM with partner -- see note in 08_bridge_analysis.py


# ============================================================
# BRIDGE: simple row-level resampling
# ============================================================
def bootstrap_ci_simple(actual: pd.Series, predicted: pd.Series,
                         n_resamples: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float]:
    """95% bootstrap CI for Pearson r, resampling rows with replacement.
    Valid here because each row already IS one independent unit (one clip).
    """
    rng = np.random.default_rng(seed)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    n = len(actual)

    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(actual[idx])) < 2 or len(np.unique(predicted[idx])) < 2:
            continue  # a degenerate resample can't have a defined correlation
        r, _ = pearsonr(actual[idx], predicted[idx])
        stats.append(r)

    stats = np.array(stats)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def permutation_test_correlation(actual: pd.Series, predicted: pd.Series,
                                  n_permutations: int = N_PERMUTATIONS, seed: int = RNG_SEED) -> float:
    """Two-sided permutation p-value: how often does shuffling the pairing
    between actual and predicted produce a correlation at least as extreme
    as the real one, purely by chance? Makes no distributional assumptions,
    unlike pearsonr's own built-in p-value.
    """
    rng = np.random.default_rng(seed)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    observed_r, _ = pearsonr(actual, predicted)

    shuffled_rs = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(predicted)
        r, _ = pearsonr(actual, shuffled)
        shuffled_rs.append(r)

    shuffled_rs = np.array(shuffled_rs)
    return float(np.mean(np.abs(shuffled_rs) >= np.abs(observed_r)))


# ============================================================
# STAGE A: grouped resampling (whole participants or whole clips move together)
# ============================================================
def bootstrap_ci_grouped(df: pd.DataFrame, group_col: str, actual_col: str,
                          predicted_col: str, n_resamples: int = N_BOOTSTRAP,
                          seed: int = RNG_SEED) -> tuple[float, float]:
    """95% bootstrap CI for Pearson r, resampling whole GROUPS (e.g. every
    trial belonging to one participant, together) rather than individual
    rows -- preserves the dependency structure from the original
    leave-one-group-out evaluation.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    n_groups = len(groups)
    rows_by_group = {g: df[df[group_col] == g] for g in groups}  # precompute once

    stats = []
    for _ in range(n_resamples):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        resampled = pd.concat([rows_by_group[g] for g in sampled_groups], ignore_index=True)
        if resampled[actual_col].nunique() < 2 or resampled[predicted_col].nunique() < 2:
            continue
        r, _ = pearsonr(resampled[actual_col], resampled[predicted_col])
        stats.append(r)

    stats = np.array(stats)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_bridge_analysis(bridge: pd.DataFrame) -> pd.DataFrame:
    """CIs and permutation p-values for both bridge subsets and outcomes."""
    well_rated = bridge[bridge["n_participants"] >= RATER_THRESHOLD]
    thin = bridge[bridge["n_participants"] < RATER_THRESHOLD]

    rows = []
    for subset_name, subset in [("primary (>=5 raters)", well_rated), ("exploratory (<5 raters)", thin)]:
        for outcome in ["valence", "arousal"]:
            actual = subset[f"actual_{outcome}"]
            predicted = subset[f"predicted_{outcome}"]
            observed_r, pearson_p = pearsonr(actual, predicted)
            ci_low, ci_high = bootstrap_ci_simple(actual, predicted)
            perm_p = permutation_test_correlation(actual, predicted)

            print(f"[bridge] {subset_name} -- {outcome}: r={observed_r:.3f}, "
                  f"95% CI=[{ci_low:.3f}, {ci_high:.3f}], permutation p={perm_p:.4f}")

            rows.append({
                "subset": subset_name, "outcome": outcome, "n_clips": len(subset),
                "pearson_r": observed_r, "pearson_p": pearson_p,
                "ci_low": ci_low, "ci_high": ci_high, "permutation_p": perm_p,
            })

    return pd.DataFrame(rows)


def run_stage_a_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """CIs for Stage A's primary (LOPO) and secondary (LOCO) results."""
    rows = []
    evaluations = [
        ("LOPO", "participant_id", "primary_pred"),
        ("LOCO", "ds002721_stimulus_id", "loco_pred"),
    ]

    for eval_name, group_col, pred_suffix in evaluations:
        for outcome in ["valence", "arousal"]:
            actual_col = f"{outcome}_like_composite"
            predicted_col = f"{outcome}_{pred_suffix}"
            if predicted_col not in data.columns:
                print(f"[WARN] {predicted_col} not found in stage_a_predictions.parquet -- skipping")
                continue

            observed_r, _ = pearsonr(data[actual_col], data[predicted_col])
            ci_low, ci_high = bootstrap_ci_grouped(data, group_col, actual_col, predicted_col)

            print(f"[stage A] {eval_name} -- {outcome}: r={observed_r:.3f}, "
                  f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]")

            rows.append({
                "evaluation": eval_name, "outcome": outcome,
                "pearson_r": observed_r, "ci_low": ci_low, "ci_high": ci_high,
            })

    return pd.DataFrame(rows)


def main() -> None:
    if not STAGE_A_PATH.exists():
        print(f"[FAIL] {STAGE_A_PATH} not found. Run 07_stage_a_models.py first "
              f"(with the data.to_parquet(...) line added at the end).")
        sys.exit(1)
    if not BRIDGE_PATH.exists():
        print(f"[FAIL] {BRIDGE_PATH} not found. Run 08_bridge_analysis.py first.")
        sys.exit(1)

    data = pd.read_parquet(STAGE_A_PATH)
    bridge = pd.read_csv(BRIDGE_PATH)

    print("=" * 60)
    print("BRIDGE ANALYSIS")
    print("=" * 60)
    bridge_results = run_bridge_analysis(bridge)
    bridge_results.to_csv(BRIDGE_OUT, index=False)
    print(f"[OK] Wrote {BRIDGE_OUT.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 60)
    print("STAGE A (LOPO primary, LOCO secondary)")
    print("=" * 60)
    stage_a_results = run_stage_a_analysis(data)
    stage_a_results.to_csv(STAGE_A_OUT, index=False)
    print(f"[OK] Wrote {STAGE_A_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()