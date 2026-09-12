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
(configs/analysis_plan.yaml -> multiple_comparisons) -- NOT YET DONE, see
09_bootstrap_metrics.py.

Reads: data_processed/eeg_features.parquet, data_processed/trials_ds002721.parquet
Writes: results/stage_a_model_comparison.csv

STILL TODO: 8 individual secondary outcomes; bootstrap CIs and FDR
correction (belongs in 09_bootstrap_metrics.py, not here).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error


# ============================================================
# 1. LOAD AND MERGE
# ============================================================
# Two separate tables need to become one: EEG features (one row per
# participant x clip, from 04_extract_eeg_features.py) and the self-report
# ratings (one row per participant x clip, from 02_build_ds002721_trials.py).
# Joining on (participant_id, ds002721_stimulus_id) lines up "this person's
# brain response to this clip" with "this person's rating of this clip".
eeg_features = pd.read_parquet("data_processed/eeg_features.parquet")
trials = pd.read_parquet("data_processed/trials_ds002721.parquet")

data = eeg_features.merge(
    trials,
    on=["participant_id", "ds002721_stimulus_id"],
    how="inner"
)
print(f"Merged rows: {len(data)}")  # expect close to 927 (eeg_features' row count)


# ============================================================
# 2. HANDLE THE SUB-31 DUPLICATE
# ============================================================
# sub-31 heard clip 463 twice (once in run 2, once in run 5), with
# genuinely different ratings each time. A merge on (participant, clip)
# can't tell which EEG epoch belongs with which rating for this pair --
# there's no run-level key connecting them -- so both ambiguous instances
# get dropped rather than guessed at.
dupes_in_trials = trials[trials.duplicated(subset=["participant_id", "ds002721_stimulus_id"], keep=False)]
print(f"Duplicate (participant, clip) pairs in trials: {len(dupes_in_trials)}")
print(dupes_in_trials.sort_values(["participant_id", "ds002721_stimulus_id"]))

dupes_in_eeg = eeg_features[eeg_features.duplicated(subset=["participant_id", "ds002721_stimulus_id"], keep=False)]
print(f"\nDuplicate (participant, clip) pairs in eeg_features: {len(dupes_in_eeg)}")

is_dupe = (data["participant_id"] == "sub-31") & (data["ds002721_stimulus_id"] == 463)
print(f"Dropping {is_dupe.sum()} ambiguous merged rows (sub-31 heard clip 463 twice, "
      f"can't disambiguate without a run-level join key)")
data = data[~is_dupe].reset_index(drop=True)
print(f"Remaining rows after dedup: {len(data)}")  # expect 925


# ============================================================
# 3. BUILD THE COMPOSITE OUTCOMES
# ============================================================
# Per configs/analysis_plan.yaml: valence-like and arousal-like are each
# a plain average of a few of the 8 individual 1-9 self-report ratings.
# .mean(axis=1) silently skips NaN by default -- so a trial missing just
# one rating (e.g. one of the 124 run-boundary-truncated trials from
# 02_build_ds002721_trials.py) still gets a valid composite from whichever
# of its ratings ARE present, rather than becoming NaN itself.
data["valence_like_composite"] = data[["pleasant", "happy", "tender"]].mean(axis=1)
data["arousal_like_composite"] = data[["energetic", "tense", "angry", "fearful"]].mean(axis=1)

print(f"Rows with a valid valence composite: {data['valence_like_composite'].notna().sum()}")
print(f"Rows with a valid arousal composite: {data['arousal_like_composite'].notna().sum()}")


# ============================================================
# 4. SHARED METRICS FUNCTION
# ============================================================
# Used for every model below (baseline, LOPO, LOCO) so all results are
# scored identically and comparably. Four different lenses on the same
# predictions:
#   mae     -- average absolute error, in original 1-9 rating units
#   rmse    -- similar to MAE but penalizes big misses more heavily
#              (squares the errors before averaging, then un-squares)
#   pearson_r    -- does the prediction move in the same LINEAR direction
#                   as the truth? Ranges -1 (perfectly backwards) to
#                   +1 (perfectly aligned), 0 = no linear relationship
#   spearman_rho -- same idea as pearson_r, but based on RANK order rather
#                   than exact values -- less sensitive to outliers
def compute_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    """MAE, RMSE, Pearson r, Spearman rho for one set of predictions."""
    mae = (actual - predicted).abs().mean()
    rmse = root_mean_squared_error(actual, predicted)
    pearson_r, _ = pearsonr(actual, predicted)
    spearman_rho, _ = spearmanr(actual, predicted)
    return {"mae": mae, "rmse": rmse, "pearson_r": pearson_r, "spearman_rho": spearman_rho}


# ============================================================
# 5. BASELINE: LEAVE-ONE-OUT PARTICIPANT MEAN
# ============================================================
# The trivial "model" every real model has to beat: predict each trial
# using this SAME participant's average rating on their OTHER trials
# (never including the trial being predicted, or its own value would leak
# into its own "prediction"). This deliberately gives the baseline access
# to a person's own rating history -- that's what makes it a meaningful
# bar: it tests whether EEG adds anything BEYOND already knowing how this
# specific person tends to rate things.
def leave_one_out_participant_mean(df: pd.DataFrame, outcome_col: str) -> np.ndarray:
    """For each row, predict using this participant's mean on OTHER rows only."""
    predictions = np.zeros(len(df))
    for participant in df["participant_id"].unique():
        mask = df["participant_id"] == participant
        values = df.loc[mask, outcome_col].values
        n = len(values)
        total = values.sum()
        # removing just this trial's own contribution from the sum, then
        # averaging over the remaining (n - 1) trials -- the actual
        # "leave-one-out" trick, vectorized across the whole participant
        predictions[mask] = (total - values) / (n - 1)
    return predictions


data["valence_baseline_pred"] = leave_one_out_participant_mean(data, "valence_like_composite")
data["arousal_baseline_pred"] = leave_one_out_participant_mean(data, "arousal_like_composite")

valence_mae = (data["valence_like_composite"] - data["valence_baseline_pred"]).abs().mean()
arousal_mae = (data["arousal_like_composite"] - data["arousal_baseline_pred"]).abs().mean()
print(f"Baseline MAE -- valence: {valence_mae:.3f}, arousal: {arousal_mae:.3f}")

valence_baseline_metrics = compute_metrics(data["valence_like_composite"], data["valence_baseline_pred"])
arousal_baseline_metrics = compute_metrics(data["arousal_like_composite"], data["arousal_baseline_pred"])
print(f"Baseline metrics -- valence: {valence_baseline_metrics}, arousal: {arousal_baseline_metrics}")


# every eeg_features column except the two ID columns is a real predictor:
# frontal/central/parietal x theta/alpha/beta, plus frontal_alpha_asymmetry
# (10 features total -- confirmed earlier against the 13-predictor config limit)
feature_cols = [c for c in eeg_features.columns if c not in ("participant_id", "ds002721_stimulus_id")]


# ============================================================
# 6. UNTUNED RIDGE (alpha=1.0) -- first real model, no tuning yet
# ============================================================
# Leave-one-participant-out: hold out one person entirely, train on
# everyone else, predict the held-out person, repeat for every participant.
# This tests generalization to a NEW PERSON -- if the model just memorized
# quirks of the people it trained on, it would do fine in training but
# poorly here, which is exactly what this loop is designed to catch.
def lopo_ridge(df: pd.DataFrame, outcome_col: str, alpha: float = 1.0) -> np.ndarray:
    """Leave-one-participant-out Ridge predictions for one outcome."""
    # start as all-NaN, not zeros -- a bug that left rows unfilled would
    # show up obviously as NaN later, rather than silently looking like a
    # real (wrong) prediction of exactly 0
    predictions = np.full(len(df), np.nan)
    participants = df["participant_id"].unique()

    for held_out in participants:
        train_mask = df["participant_id"] != held_out
        test_mask = df["participant_id"] == held_out

        # Pipeline bundles imputing + modeling so BOTH only ever get fit on
        # training data. A few epochs have NaN frontal_alpha_asymmetry
        # (when F3/F4 weren't both present) -- Ridge can't handle NaN
        # directly, so SimpleImputer fills it in first, using only the
        # training fold's median (never the held-out participant's own
        # data, which would leak test information into the fill values).
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=alpha)),
        ])

        pipeline.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, outcome_col])
        predictions[test_mask] = pipeline.predict(df.loc[test_mask, feature_cols])

    return predictions


data["valence_ridge_pred"] = lopo_ridge(data, "valence_like_composite")
data["arousal_ridge_pred"] = lopo_ridge(data, "arousal_like_composite")

valence_ridge_metrics = compute_metrics(data["valence_like_composite"], data["valence_ridge_pred"])
arousal_ridge_metrics = compute_metrics(data["arousal_like_composite"], data["arousal_ridge_pred"])
print(f"Ridge metrics -- valence: {valence_ridge_metrics}, arousal: {arousal_ridge_metrics}")

valence_ridge_mae = (data["valence_like_composite"] - data["valence_ridge_pred"]).abs().mean()
arousal_ridge_mae = (data["arousal_like_composite"] - data["arousal_ridge_pred"]).abs().mean()
print(f"Ridge MAE  -- valence: {valence_ridge_mae:.3f}, arousal: {arousal_ridge_mae:.3f}")
print(f"Baseline   -- valence: {valence_mae:.3f}, arousal: {arousal_mae:.3f}")
print(f"Beats baseline? valence: {valence_ridge_mae < valence_mae}, arousal: {arousal_ridge_mae < arousal_mae}")


# ============================================================
# 7. TUNED RIDGE (scaled features, alpha chosen by inner CV) -- PRIMARY H1 RESULT
# ============================================================
def lopo_ridge_tuned(df: pd.DataFrame, outcome_col: str,
                      alphas=(0.1, 1.0, 10.0, 100.0), n_inner_splits=5):
    """LOPO Ridge with feature scaling and alpha chosen by grouped inner
    cross-validation -- entirely within the training participants, never
    touching the held-out person.
    """
    predictions = np.full(len(df), np.nan)
    chosen_alphas = []
    participants = df["participant_id"].unique()

    for held_out in participants:
        train_mask = df["participant_id"] != held_out
        test_mask = df["participant_id"] == held_out
        train_df = df.loc[train_mask]

        # GroupKFold on the INNER split too -- so tuning never breaks one
        # participant's trials across an inner train/validation boundary,
        # same principle as the outer LOPO loop, one level deeper
        groups = train_df["participant_id"].values
        n_splits = min(n_inner_splits, train_df["participant_id"].nunique())
        inner_cv = GroupKFold(n_splits=n_splits)

        # StandardScaler added between imputer and model: Ridge's penalty
        # punishes large coefficients equally regardless of WHY they're
        # large. A feature with a naturally wider numeric range can end up
        # unfairly favored/penalized purely due to scale, not real
        # importance -- scaling to mean=0/std=1 removes that distortion.
        # Fit only on whatever data GridSearchCV hands it during each inner
        # fold, and refit on the full training set for the final model --
        # never on the held-out participant.
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ])

        # tries every alpha in the grid, scores each by (negative) MAE
        # across the inner folds, keeps the winner -- entirely inside
        # train_df, before this fold's held-out participant is ever touched
        search = GridSearchCV(
            pipeline,
            param_grid={"model__alpha": list(alphas)},
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
        )
        search.fit(train_df[feature_cols], train_df[outcome_col], groups=groups)

        chosen_alphas.append(search.best_params_["model__alpha"])
        # best_estimator_ is already refit on the FULL training set with
        # the winning alpha -- this is what predicts the held-out participant
        predictions[test_mask] = search.best_estimator_.predict(df.loc[test_mask, feature_cols])

    return predictions, chosen_alphas


# PRIMARY result: the preregistered grid from configs/analysis_plan.yaml.
# This is what gets reported as the actual H1 answer.
data["valence_primary_pred"], valence_primary_alphas = lopo_ridge_tuned(
    data, "valence_like_composite", alphas=(0.1, 1.0, 10.0, 100.0)
)
data["arousal_primary_pred"], arousal_primary_alphas = lopo_ridge_tuned(
    data, "arousal_like_composite", alphas=(0.1, 1.0, 10.0, 100.0)
)

valence_primary_metrics = compute_metrics(data["valence_like_composite"], data["valence_primary_pred"])
arousal_primary_metrics = compute_metrics(data["arousal_like_composite"], data["arousal_primary_pred"])
print(f"Primary metrics -- valence: {valence_primary_metrics}, arousal: {arousal_primary_metrics}")

valence_primary_mae = (data["valence_like_composite"] - data["valence_primary_pred"]).abs().mean()
arousal_primary_mae = (data["arousal_like_composite"] - data["arousal_primary_pred"]).abs().mean()
print(f"PRIMARY (scaled, preregistered grid) MAE -- valence: {valence_primary_mae:.3f}, arousal: {arousal_primary_mae:.3f}")
print(f"Baseline                                  -- valence: {valence_mae:.3f}, arousal: {arousal_mae:.3f}")
print(f"Beats baseline? valence: {valence_primary_mae < valence_mae}, arousal: {arousal_primary_mae < arousal_mae}")
print(f"\nAlphas chosen (valence): {valence_primary_alphas}")
print(f"Alphas chosen (arousal): {arousal_primary_alphas}")


# ============================================================
# 8. WIDE-GRID RIDGE -- EXPLORATORY ONLY, not the H1 answer
# ============================================================
# Same function, wider search (up to 5000) -- built to check whether the
# preregistered grid's ceiling (100) was artificially capping performance.
# Findings: valence picked 5000 in every fold and still didn't beat
# baseline; arousal's chosen alpha scattered unpredictably across folds.
# Kept and labeled explicitly as exploratory, per the plan's own rule that
# a post-hoc search shouldn't quietly replace the preregistered primary result.
wide_alphas = (0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0)

data["valence_ridge_wide_pred"], valence_wide_alphas = lopo_ridge_tuned(
    data, "valence_like_composite", alphas=wide_alphas
)
data["arousal_ridge_wide_pred"], arousal_wide_alphas = lopo_ridge_tuned(
    data, "arousal_like_composite", alphas=wide_alphas
)

valence_wide_mae = (data["valence_like_composite"] - data["valence_ridge_wide_pred"]).abs().mean()
arousal_wide_mae = (data["arousal_like_composite"] - data["arousal_ridge_wide_pred"]).abs().mean()
print(f"Wide-grid Ridge MAE -- valence: {valence_wide_mae:.3f}, arousal: {arousal_wide_mae:.3f}")
print(f"Baseline            -- valence: {valence_mae:.3f}, arousal: {arousal_mae:.3f}")
print(f"Beats baseline? valence: {valence_wide_mae < valence_mae}, arousal: {arousal_wide_mae < arousal_mae}")
print(f"\nAlphas chosen (valence): {valence_wide_alphas}")
print(f"Alphas chosen (arousal): {arousal_wide_alphas}")


# ============================================================
# 9. LEAVE-ONE-CLIP-OUT -- secondary generalization test
# ============================================================
# A DIFFERENT question from LOPO. LOPO tests "can this generalize to a new
# PERSON" -- but a held-out person's clips still appear in training (other
# people heard them too), so a model could in principle learn "clip 305
# tends to get high arousal" from other participants, without EEG doing
# any real work, and LOPO wouldn't catch that. LOCO closes this gap: hold
# out ALL trials for one clip, train on every other clip, so the model has
# literally never seen this clip before, for anyone.
#
# Design choice, not explicitly spelled out in the configs: the INNER
# tuning loop groups by PARTICIPANT rather than by clip, since removing
# one clip's few rows still leaves nearly every participant intact in
# training -- consistent with this project's general "don't split one
# person's trials across a boundary" principle. 
def loco_ridge_tuned(df: pd.DataFrame, outcome_col: str,
                      alphas=(0.1, 1.0, 10.0, 100.0), n_inner_splits=5):
    """Leave-one-clip-out Ridge: hold out one clip's trials, train on every
    other clip, predict the held-out clip. Repeated for every unique clip.
    """
    predictions = np.full(len(df), np.nan)
    clip_ids = df["ds002721_stimulus_id"].unique()

    for held_out_clip in clip_ids:
        train_mask = df["ds002721_stimulus_id"] != held_out_clip
        test_mask = df["ds002721_stimulus_id"] == held_out_clip
        train_df = df.loc[train_mask]

        groups = train_df["participant_id"].values
        n_splits = min(n_inner_splits, train_df["participant_id"].nunique())
        inner_cv = GroupKFold(n_splits=n_splits)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ])

        search = GridSearchCV(
            pipeline,
            param_grid={"model__alpha": list(alphas)},
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
        )
        search.fit(train_df[feature_cols], train_df[outcome_col], groups=groups)
        predictions[test_mask] = search.best_estimator_.predict(df.loc[test_mask, feature_cols])

    return predictions


# NOTE: 307 outer folds (one per unique clip), each running a full inner
# grid search -- this will take noticeably longer than the 30-fold LOPO
# runs above. 
data["valence_loco_pred"] = loco_ridge_tuned(data, "valence_like_composite")
data["arousal_loco_pred"] = loco_ridge_tuned(data, "arousal_like_composite")

valence_loco_metrics = compute_metrics(data["valence_like_composite"], data["valence_loco_pred"])
arousal_loco_metrics = compute_metrics(data["arousal_like_composite"], data["arousal_loco_pred"])
print(f"Valence -- leave-one-clip-out: {valence_loco_metrics}")
print(f"Arousal -- leave-one-clip-out: {arousal_loco_metrics}")


# ============================================================
# 10. FINAL COMPARISON TABLE
# ============================================================
results_table = pd.DataFrame([
    {"outcome": "valence", "model": "participant-mean baseline", "mae": valence_baseline_metrics["mae"],
     "rmse": valence_baseline_metrics["rmse"], "pearson_r": valence_baseline_metrics["pearson_r"],
     "spearman_rho": valence_baseline_metrics["spearman_rho"], "alpha": None},
    {"outcome": "valence", "model": "ridge (alpha=1.0, untuned)", "mae": valence_ridge_metrics["mae"],
     "rmse": valence_ridge_metrics["rmse"], "pearson_r": valence_ridge_metrics["pearson_r"],
     "spearman_rho": valence_ridge_metrics["spearman_rho"], "alpha": 1.0},
    {"outcome": "valence", "model": "ridge (tuned, PRIMARY, grid to 100)", "mae": valence_primary_metrics["mae"],
     "rmse": valence_primary_metrics["rmse"], "pearson_r": valence_primary_metrics["pearson_r"],
     "spearman_rho": valence_primary_metrics["spearman_rho"], "alpha": "100.0 (all folds)"},
    {"outcome": "valence", "model": "ridge (tuned, EXPLORATORY, grid to 5000)", "mae": valence_wide_mae,
     "rmse": None, "pearson_r": None, "spearman_rho": None, "alpha": "5000.0 (all folds)"},
    {"outcome": "valence", "model": "ridge (tuned, leave-one-clip-out)", "mae": valence_loco_metrics["mae"],
     "rmse": valence_loco_metrics["rmse"], "pearson_r": valence_loco_metrics["pearson_r"],
     "spearman_rho": valence_loco_metrics["spearman_rho"], "alpha": "grid to 100, per-fold"},

    {"outcome": "arousal", "model": "participant-mean baseline", "mae": arousal_baseline_metrics["mae"],
     "rmse": arousal_baseline_metrics["rmse"], "pearson_r": arousal_baseline_metrics["pearson_r"],
     "spearman_rho": arousal_baseline_metrics["spearman_rho"], "alpha": None},
    {"outcome": "arousal", "model": "ridge (alpha=1.0, untuned)", "mae": arousal_ridge_metrics["mae"],
     "rmse": arousal_ridge_metrics["rmse"], "pearson_r": arousal_ridge_metrics["pearson_r"],
     "spearman_rho": arousal_ridge_metrics["spearman_rho"], "alpha": 1.0},
    {"outcome": "arousal", "model": "ridge (tuned, PRIMARY, grid to 100)", "mae": arousal_primary_metrics["mae"],
     "rmse": arousal_primary_metrics["rmse"], "pearson_r": arousal_primary_metrics["pearson_r"],
     "spearman_rho": arousal_primary_metrics["spearman_rho"], "alpha": "100.0 (all folds)"},
    {"outcome": "arousal", "model": "ridge (tuned, EXPLORATORY, grid to 5000)", "mae": arousal_wide_mae,
     "rmse": None, "pearson_r": None, "spearman_rho": None, "alpha": "scattered, see notes"},
    {"outcome": "arousal", "model": "ridge (tuned, leave-one-clip-out)", "mae": arousal_loco_metrics["mae"],
     "rmse": arousal_loco_metrics["rmse"], "pearson_r": arousal_loco_metrics["pearson_r"],
     "spearman_rho": arousal_loco_metrics["spearman_rho"], "alpha": "grid to 100, per-fold"},
])

results_table.to_csv("results/stage_a_model_comparison.csv", index=False)
print(results_table) 