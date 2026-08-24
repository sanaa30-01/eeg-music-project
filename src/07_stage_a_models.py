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

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold 

eeg_features = pd.read_parquet("data_processed/eeg_features.parquet")
trials = pd.read_parquet("data_processed/trials_ds002721.parquet")

# Join on (participant, clip) -- each of these should uniquely identify one
# trial, since a participant never hears the same clip twice
data = eeg_features.merge(
    trials,
    on=["participant_id", "ds002721_stimulus_id"],
    how="inner"
)
print(f"Merged rows: {len(data)}")  # sanity check: should be close to 927, the eeg_features row count

# Per configs/analysis_plan.yaml -- primary outcomes are two composites,
# each the plain average of a few of the 8 individual self-report ratings
data["valence_like_composite"] = data[["pleasant", "happy", "tender"]].mean(axis=1)
data["arousal_like_composite"] = data[["energetic", "tense", "angry", "fearful"]].mean(axis=1)

# There were 124 trials with exactly one missing rating so this code accounts for that.
# If a trial is missing just "tender", it still gets a valid composite from pleasant+happy alone. 
print(f"Rows with a valid valence composite: {data['valence_like_composite'].notna().sum()}")
print(f"Rows with a valid arousal composite: {data['arousal_like_composite'].notna().sum()}") 

#checking for duplicates in trials
dupes_in_trials = trials[trials.duplicated(subset=["participant_id", "ds002721_stimulus_id"], keep=False)]
print(f"Duplicate (participant, clip) pairs in trials: {len(dupes_in_trials)}")
print(dupes_in_trials.sort_values(["participant_id", "ds002721_stimulus_id"]))

#checking for duplicates in eeg_features
dupes_in_eeg = eeg_features[eeg_features.duplicated(subset=["participant_id", "ds002721_stimulus_id"], keep=False)]
print(f"\nDuplicate (participant, clip) pairs in eeg_features: {len(dupes_in_eeg)}")

#dropping ambiguous merged rows
is_dupe = (data["participant_id"] == "sub-31") & (data["ds002721_stimulus_id"] == 463)
print(f"Dropping {is_dupe.sum()} ambiguous merged rows (sub-31 heard clip 463 twice, can't disambiguate without a run-level join key)")
data = data[~is_dupe].reset_index(drop=True)
print(f"Remaining rows: {len(data)}")

def leave_one_out_participant_mean(df: pd.DataFrame, outcome_col: str) -> np.ndarray:
    """For each row, predict using this participant's mean on OTHER rows only."""
    predictions = np.zeros(len(df))
    for participant in df["participant_id"].unique():
        mask = df["participant_id"] == participant
        values = df.loc[mask, outcome_col].values
        n = len(values)
        total = values.sum()
        # for each trial, the "leave this one out" mean is:
        # (all other ratings' total) / (all other ratings' count)
        predictions[mask] = (total - values) / (n - 1)
    return predictions

data["valence_baseline_pred"] = leave_one_out_participant_mean(data, "valence_like_composite")
data["arousal_baseline_pred"] = leave_one_out_participant_mean(data, "arousal_like_composite")

# baseline error -- this is the number any real model needs to beat
valence_mae = (data["valence_like_composite"] - data["valence_baseline_pred"]).abs().mean()
arousal_mae = (data["arousal_like_composite"] - data["arousal_baseline_pred"]).abs().mean()
print(f"Baseline MAE -- valence: {valence_mae:.3f}, arousal: {arousal_mae:.3f}")


# every eeg_features column except the two ID columns is a real predictor --
# frontal/central/parietal x theta/alpha/beta, plus frontal_alpha_asymmetry
# (10 features total, confirmed earlier against the 13-predictor config limit)
feature_cols = [c for c in eeg_features.columns if c not in ("participant_id", "ds002721_stimulus_id")]


def lopo_ridge(df: pd.DataFrame, outcome_col: str, alpha: float = 1.0) -> np.ndarray:
    """Leave-one-participant-out Ridge predictions for one outcome.

    Same evaluation logic as the baseline function: hold out one
    participant entirely, train only on everyone else, predict on the
    held-out person, repeat for every participant. This is what makes the
    result comparable to the baseline MAE -- both are tested the same way,
    on a person the "model" (Ridge, or the participant-mean trick) never
    saw during its own training.
    """
    # start as all-NaN rather than zeros -- if a bug ever left some rows
    # unfilled, NaN would surface as an obvious problem later (e.g. in the
    # MAE calculation), whereas leftover zeros could silently look like a
    # real, oddly specific prediction, and be far harder to notice
    predictions = np.full(len(df), np.nan)
    participants = df["participant_id"].unique()

    for held_out in participants:
        train_mask = df["participant_id"] != held_out
        test_mask = df["participant_id"] == held_out

        # Pipeline bundles imputing + modeling into one object so both
        # steps only ever get fit on the TRAINING data -- calling
        # .fit() below fits the imputer's median AND the Ridge model
        # together, using only train_mask rows; .predict() then reuses
        # that same fitted imputer on the test rows, rather than
        # recalculating a median from the test participant's own data.
        # This matters because imputing test data using test data's own
        # median would leak a small amount of test-set information into
        # a step that's supposed to be blind to it.
        #
        # SimpleImputer is needed at all because a few epochs can have
        # NaN features -- specifically frontal_alpha_asymmetry, which
        # comes back NaN in the rare case F3 or F4 wasn't present in a
        # given participant's channel list. Ridge itself can't handle
        # NaN inputs directly, so they need to be filled in first.
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=alpha)),
        ])

        # fit on every OTHER participant's rows
        pipeline.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, outcome_col])

        # predict on ONLY the held-out participant's rows, store those
        # predictions back into the matching positions of the full array
        predictions[test_mask] = pipeline.predict(df.loc[test_mask, feature_cols])

    return predictions


# run the full LOPO procedure once per outcome -- valence and arousal are
# modeled completely independently, each gets its own set of 31 (one per
# held-out participant) fit/predict cycles
data["valence_ridge_pred"] = lopo_ridge(data, "valence_like_composite")
data["arousal_ridge_pred"] = lopo_ridge(data, "arousal_like_composite")

# same MAE calculation as the baseline: mean absolute difference between
# real rating and predicted rating, on the original 1-9 scale
valence_ridge_mae = (data["valence_like_composite"] - data["valence_ridge_pred"]).abs().mean()
arousal_ridge_mae = (data["arousal_like_composite"] - data["arousal_ridge_pred"]).abs().mean()

print(f"Ridge MAE  -- valence: {valence_ridge_mae:.3f}, arousal: {arousal_ridge_mae:.3f}")
print(f"Baseline   -- valence: {valence_mae:.3f}, arousal: {arousal_mae:.3f}")

# the actual H1 test: is Ridge's error LOWER than the baseline's error?
print(f"Beats baseline? valence: {valence_ridge_mae < valence_mae}, arousal: {arousal_ridge_mae < arousal_mae}") 

# LOPO Ridge with alpha chosen by grouped inner cross-validation to optimize performance on the training set
def lopo_ridge_tuned(df: pd.DataFrame, outcome_col: str,
                      alphas=(0.1, 1.0, 10.0, 100.0), n_inner_splits=5):
    """LOPO Ridge, but with alpha chosen by grouped inner cross-validation
    on the training participants only -- never touching the held-out person.
    """
    predictions = np.full(len(df), np.nan)
    chosen_alphas = []  # track which alpha wins each fold -- useful diagnostic
    participants = df["participant_id"].unique()

    for held_out in participants:
        train_mask = df["participant_id"] != held_out
        test_mask = df["participant_id"] == held_out
        train_df = df.loc[train_mask]

        # GroupKFold ensures the INNER split also never breaks one
        # participant's trials across inner-train/inner-validation --
        # same "don't let a person leak across a boundary" principle as
        # the outer LOPO loop, just applied one level deeper
        groups = train_df["participant_id"].values
        n_splits = min(n_inner_splits, train_df["participant_id"].nunique())
        inner_cv = GroupKFold(n_splits=n_splits)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge()),
        ])

        # GridSearchCV tries every alpha in the grid, scores each by
        # (negative) MAE across the inner folds, and keeps the best one --
        # entirely within train_df, before ever seeing the held-out person
        search = GridSearchCV(
            pipeline,
            param_grid={"model__alpha": list(alphas)},
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
        )
        search.fit(train_df[feature_cols], train_df[outcome_col], groups=groups)

        chosen_alphas.append(search.best_params_["model__alpha"])
        # best_estimator_ is already refit on the FULL training set using
        # the winning alpha -- this is what actually predicts the held-out person
        predictions[test_mask] = search.best_estimator_.predict(df.loc[test_mask, feature_cols])

    return predictions, chosen_alphas


data["valence_ridge_tuned_pred"], valence_alphas = lopo_ridge_tuned(data, "valence_like_composite")
data["arousal_ridge_tuned_pred"], arousal_alphas = lopo_ridge_tuned(data, "arousal_like_composite")

valence_tuned_mae = (data["valence_like_composite"] - data["valence_ridge_tuned_pred"]).abs().mean()
arousal_tuned_mae = (data["arousal_like_composite"] - data["arousal_ridge_tuned_pred"]).abs().mean()

print(f"Tuned Ridge MAE -- valence: {valence_tuned_mae:.3f}, arousal: {arousal_tuned_mae:.3f}")
print(f"Baseline        -- valence: {valence_mae:.3f}, arousal: {arousal_mae:.3f}")
print(f"Beats baseline? valence: {valence_tuned_mae < valence_mae}, arousal: {arousal_tuned_mae < arousal_mae}")
print(f"\nAlphas chosen (valence): {valence_alphas}")
print(f"Alphas chosen (arousal): {arousal_alphas}")

# tried to extend the grid search to 5000.0 to see if it would improve the model's performance so that it can regularize more
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

# creating a table to compare the models and their performance
results_table = pd.DataFrame([
    {"outcome": "valence", "model": "participant-mean baseline", "mae": valence_mae, "alpha": None},
    {"outcome": "valence", "model": "ridge (alpha=1.0, untuned)", "mae": valence_ridge_mae, "alpha": 1.0},
    {"outcome": "valence", "model": "ridge (tuned, grid to 100)", "mae": valence_tuned_mae, "alpha": "100.0 (all folds)"},
    {"outcome": "valence", "model": "ridge (tuned, grid to 5000)", "mae": valence_wide_mae, "alpha": "5000.0 (all folds)"},
    {"outcome": "arousal", "model": "participant-mean baseline", "mae": arousal_mae, "alpha": None},
    {"outcome": "arousal", "model": "ridge (alpha=1.0, untuned)", "mae": arousal_ridge_mae, "alpha": 1.0},
    {"outcome": "arousal", "model": "ridge (tuned, grid to 100)", "mae": arousal_tuned_mae, "alpha": "100.0 (all folds)"},
    {"outcome": "arousal", "model": "ridge (tuned, grid to 5000)", "mae": arousal_wide_mae, "alpha": "scattered, see note"},
])
results_table.to_csv("results/stage_a_model_comparison.csv", index=False)
print(results_table) 