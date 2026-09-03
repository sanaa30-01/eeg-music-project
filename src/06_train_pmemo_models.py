"""Train, compare, select, and freeze PMEmo audio-only regression models.

This is the project's Stage B analysis. It predicts PMEmo's crowd-aggregated
static induced-valence and induced-arousal ratings from the 293 handcrafted
audio predictors created by ``05_extract_audio_features.py``. ds002721, EEG,
lyrics, dynamic annotations, openSMILE features, and deep embeddings are
deliberately outside this script's scope.

The central statistical rule is nested grouped cross-validation:

1. One deterministic five-fold *outer* split estimates generalization to
   unheard tracks.
2. For every outer training set, a five-fold *inner* split selects model
   hyperparameters without seeing the outer test tracks.
3. Imputation, constant-feature removal, and scaling live inside sklearn
   Pipelines, so each operation learns from training data only.
4. Every outer-test prediction is saved. Reported performance is calculated
   from these out-of-fold predictions, never from predictions on fitted data.

Normal full run::

    myenv/bin/python src/06_train_pmemo_models.py

Fast software/integration test with isolated outputs::

    myenv/bin/python src/06_train_pmemo_models.py --quick

Strictly optional post-primary Random Forest comparison::

    myenv/bin/python src/06_train_pmemo_models.py --random-forest

Add ``--quick`` to that command for the reduced optional software test. The
forest branch reuses the frozen fold file and writes separately named outputs;
it never changes the primary model comparison or saved winning pipelines.

Use ``--overwrite`` only when intentionally replacing an earlier run. The
script refuses existing targets before starting expensive computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# Matplotlib normally writes a font cache below the user's home directory.
# That location is not always writable in reproducible/sandboxed environments,
# so point it at a disposable system directory before importing matplotlib.
MPL_CACHE = Path(tempfile.gettempdir()) / "eeg_music_project_matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import joblib
import matplotlib

matplotlib.use("Agg")  # Generate PNGs without requiring a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PLAN_PATH = REPO_ROOT / "configs" / "analysis_plan.yaml"
FEATURES_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"
SPLITS_CONFIG_PATH = REPO_ROOT / "configs" / "splits.yaml"
FEATURES_PATH = REPO_ROOT / "data_processed" / "audio_features_pmemo.parquet"
LABELS_PATH = REPO_ROOT / "data_raw" / "PMEmo" / "annotations" / "static_annotations.csv"

EXPECTED_AUDIO_ROWS = 794
EXPECTED_LABELLED_ROWS = 767
EXPECTED_FEATURE_COUNT = 293
TARGET_COLUMNS = {
    "valence": "Valence(mean)",
    "arousal": "Arousal(mean)",
}
MODEL_NAMES = ("dummy_mean", "ridge", "elastic_net", "svr_rbf")
OPTIONAL_RANDOM_FOREST_NAME = "random_forest_optional"


@dataclass(frozen=True)
class ModelSpec:
    """One model family, its leakage-safe pipeline, and tuning candidates."""

    name: str
    pipeline: Pipeline
    parameter_grid: dict[str, list[Any]]


@dataclass(frozen=True)
class OutputPaths:
    """All artifacts for either the quick run or the final scientific run."""

    folds: Path
    predictions: Path
    fold_metrics: Path
    comparison: Path
    figure: Path
    valence_model: Path
    arousal_model: Path
    manifest: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.folds,
            self.predictions,
            self.fold_metrics,
            self.comparison,
            self.figure,
            self.valence_model,
            self.arousal_model,
            self.manifest,
        )


@dataclass(frozen=True)
class OptionalRandomForestPaths:
    """Artifacts kept separate from the frozen primary Stage B analysis.

    Random Forest was proposed only after the primary models had been fitted.
    Separate filenames prevent this exploratory analysis from silently changing
    the preregistered comparison, winners, or final Ridge/Elastic-Net models.
    """

    predictions: Path
    fold_metrics: Path
    augmented_comparison: Path
    figure: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every optional output so overwrite checks remain complete."""

        return (self.predictions, self.fold_metrics, self.augmented_comparison, self.figure)


def load_splits_config(path: Path = SPLITS_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the split/model-selection rules used by this stage."""

    if not path.exists():
        raise FileNotFoundError(f"Split configuration not found: {path}")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    internal = config.get("pmemo_internal_model_selection", {})
    final = config.get("pmemo_final_estimate", {})
    selection = config.get("model_selection_rule", {})
    if internal.get("method") != "GroupKFold" or int(internal.get("n_splits", 0)) != 5:
        raise ValueError("PMEmo internal selection must remain five-fold GroupKFold")
    if final.get("method") != "nested_grouped_cv" or int(final.get("outer_splits", 0)) != 5:
        raise ValueError("PMEmo final estimate must remain five-fold nested grouped CV")
    if selection.get("primary_metric") != "concordance_correlation_coefficient":
        raise ValueError("The primary model-selection metric must remain CCC")
    if selection.get("tie_breaker") != "rmse":
        raise ValueError("The model-selection tie-breaker must remain RMSE")
    return config


def load_and_validate_inputs(
    feature_path: Path = FEATURES_PATH,
    label_path: Path = LABELS_PATH,
    expected_audio_rows: int = EXPECTED_AUDIO_ROWS,
    expected_labelled_rows: int = EXPECTED_LABELLED_ROWS,
    expected_feature_count: int = EXPECTED_FEATURE_COUNT,
) -> tuple[pd.DataFrame, list[str]]:
    """Read PMEmo predictors/targets and enforce their complete data contract.

    Returning the feature names separately makes it hard for identifiers or
    target columns to enter X accidentally later. The one-to-one merge also
    catches duplicated IDs before they can silently multiply observations.
    """

    if not feature_path.exists():
        raise FileNotFoundError(f"Extracted PMEmo feature table not found: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"PMEmo static-label file not found: {label_path}")

    features = pd.read_parquet(feature_path)
    labels = pd.read_csv(label_path)
    if "track_id" not in features:
        raise ValueError("Feature table must contain track_id")
    required_labels = {"musicId", *TARGET_COLUMNS.values()}
    missing_labels = required_labels - set(labels.columns)
    if missing_labels:
        raise ValueError(f"Static-label table is missing columns: {sorted(missing_labels)}")

    if len(features) != expected_audio_rows:
        raise ValueError(f"Expected {expected_audio_rows} audio rows, found {len(features)}")
    if features["track_id"].nunique() != expected_audio_rows or features["track_id"].duplicated().any():
        raise ValueError("PMEmo feature track_id values must be unique and complete")
    if len(labels) != expected_labelled_rows:
        raise ValueError(f"Expected {expected_labelled_rows} labelled rows, found {len(labels)}")
    if labels["musicId"].nunique() != expected_labelled_rows or labels["musicId"].duplicated().any():
        raise ValueError("PMEmo label musicId values must be unique and complete")

    feature_columns = [column for column in features.columns if column != "track_id"]
    if len(feature_columns) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} audio predictors, found {len(feature_columns)}"
        )
    if not all(pd.api.types.is_numeric_dtype(features[column]) for column in feature_columns):
        raise ValueError("Every audio predictor must be numeric")
    if not np.isfinite(features[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Audio predictors contain missing or infinite values")

    for target_column in TARGET_COLUMNS.values():
        labels[target_column] = pd.to_numeric(labels[target_column], errors="raise")
    if not np.isfinite(labels[list(TARGET_COLUMNS.values())].to_numpy(dtype=float)).all():
        raise ValueError("PMEmo targets contain missing or infinite values")

    data = features.merge(
        labels.loc[:, ["musicId", *TARGET_COLUMNS.values()]],
        left_on="track_id",
        right_on="musicId",
        how="inner",
        validate="one_to_one",
    )
    if len(data) != expected_labelled_rows:
        raise ValueError(
            f"Validated feature/label join should contain {expected_labelled_rows} tracks, "
            f"found {len(data)}"
        )
    data = data.sort_values("track_id", kind="stable").reset_index(drop=True)

    forbidden_predictors = {"track_id", "musicId", *TARGET_COLUMNS.values()}
    if forbidden_predictors & set(feature_columns):
        raise ValueError("An identifier or outcome was accidentally included as a predictor")
    return data, feature_columns


def concordance_correlation_coefficient(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Return Lin's concordance correlation coefficient (CCC).

    CCC rewards correlation while penalizing differences in mean and scale.
    Degenerate constant inputs are handled explicitly: identical constants are
    perfectly concordant; any other zero-denominator case receives zero rather
    than NaN, keeping CV scoring and final tables finite and auditable.
    """

    true = np.asarray(list(y_true), dtype=float).reshape(-1)
    pred = np.asarray(list(y_pred), dtype=float).reshape(-1)
    if true.size != pred.size or true.size == 0:
        raise ValueError("CCC inputs must be nonempty arrays of equal length")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("CCC inputs must contain only finite values")

    true_centered = true - true.mean()
    pred_centered = pred - pred.mean()
    covariance = float(np.mean(true_centered * pred_centered))
    denominator = float(
        np.mean(true_centered**2)
        + np.mean(pred_centered**2)
        + (true.mean() - pred.mean()) ** 2
    )
    if denominator <= np.finfo(float).eps:
        return 1.0 if np.allclose(true, pred) else 0.0
    return float(2.0 * covariance / denominator)


def _safe_correlation(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    correlation_function: Callable[[np.ndarray, np.ndarray], Any],
) -> float:
    """Calculate a finite correlation, defining constant predictions as zero."""

    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.size < 2 or np.ptp(true) <= np.finfo(float).eps or np.ptp(pred) <= np.finfo(float).eps:
        return 0.0
    value = float(correlation_function(true, pred).statistic)
    return value if np.isfinite(value) else 0.0


def calculate_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Calculate every preregistered metric from one set of held-out predictions."""

    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("Metric inputs must be nonempty arrays of identical shape")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must contain only finite values")
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(root_mean_squared_error(true, pred)),
        "pearson_r": _safe_correlation(true, pred, pearsonr),
        "spearman_rho": _safe_correlation(true, pred, spearmanr),
        "ccc": concordance_correlation_coefficient(true, pred),
    }


def make_group_splits(groups: pd.Series, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic group-disjoint splits and verify every boundary."""

    group_array = np.asarray(groups)
    if len(np.unique(group_array)) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique groups")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    placeholder = np.zeros((len(group_array), 1))
    splits = list(splitter.split(placeholder, groups=group_array))

    test_positions: list[int] = []
    for train_indices, test_indices in splits:
        train_groups = set(group_array[train_indices])
        test_groups = set(group_array[test_indices])
        if train_groups & test_groups:
            raise AssertionError("A group appears in both train and test data")
        test_positions.extend(test_indices.tolist())
    if sorted(test_positions) != list(range(len(group_array))):
        raise AssertionError("Grouped folds must test every row exactly once")
    return splits


def make_outer_fold_table(data: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Create the one shared outer-fold assignment used by every analysis."""

    splits = make_group_splits(data["track_id"], n_splits=n_splits, seed=seed)
    fold_number = np.full(len(data), -1, dtype=int)
    for fold, (_, test_indices) in enumerate(splits):
        fold_number[test_indices] = fold
    if (fold_number < 0).any():
        raise AssertionError("At least one track did not receive an outer fold")
    table = pd.DataFrame({"track_id": data["track_id"].astype(int), "outer_fold": fold_number})
    return table.sort_values("track_id", kind="stable").reset_index(drop=True), splits


def load_saved_outer_splits(
    data: pd.DataFrame,
    fold_path: Path,
    expected_n_splits: int,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """Reconstruct split indices from the already frozen primary fold file.

    The optional Random Forest must be evaluated on exactly the same held-out
    tracks as the core models. Regenerating a split with the same seed would be
    deterministic, but loading the saved artifact gives a stronger guarantee:
    the afterthought analysis literally consumes the previously frozen mapping.
    """

    if not fold_path.exists():
        raise FileNotFoundError(
            f"Primary outer-fold file is required before optional Random Forest: {fold_path}"
        )
    fold_table = pd.read_csv(fold_path)
    if list(fold_table.columns) != ["track_id", "outer_fold"]:
        raise ValueError("Saved outer-fold table must contain only track_id and outer_fold")
    if fold_table["track_id"].duplicated().any():
        raise ValueError("Saved outer-fold table contains duplicate track IDs")

    expected_ids = set(data["track_id"].astype(int))
    saved_ids = set(fold_table["track_id"].astype(int))
    if saved_ids != expected_ids or len(fold_table) != len(data):
        raise ValueError("Saved outer folds do not cover this modeling dataset exactly")
    observed_folds = sorted(fold_table["outer_fold"].astype(int).unique().tolist())
    if observed_folds != list(range(expected_n_splits)):
        raise ValueError(
            f"Expected fold numbers 0..{expected_n_splits - 1}, found {observed_folds}"
        )

    # Align fold labels to the validated modeling-table row order before making
    # integer indices for sklearn. Each test group appears in one fold only;
    # all remaining rows become that fold's training set.
    fold_by_track = fold_table.set_index("track_id")["outer_fold"]
    row_folds = data["track_id"].map(fold_by_track)
    if row_folds.isna().any():
        raise ValueError("At least one modeling track has no saved outer fold")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_positions = np.arange(len(data), dtype=int)
    for fold in range(expected_n_splits):
        test_indices = all_positions[row_folds.to_numpy(dtype=int) == fold]
        train_indices = all_positions[row_folds.to_numpy(dtype=int) != fold]
        train_groups = set(data.iloc[train_indices]["track_id"])
        test_groups = set(data.iloc[test_indices]["track_id"])
        if not test_groups or train_groups & test_groups:
            raise AssertionError(f"Invalid saved group boundary in outer fold {fold}")
        splits.append((train_indices, test_indices))
    return fold_table.sort_values("track_id", kind="stable").reset_index(drop=True), splits


def make_model_specs(quick: bool, seed: int) -> dict[str, ModelSpec]:
    """Construct the four model families and their registered search spaces."""

    common_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    scaled_steps = [*common_steps, ("scaler", RobustScaler())]

    full_grids = {
        "ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        "elastic_net": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
        "svr_rbf": {
            "model__C": [0.1, 1, 10],
            "model__epsilon": [0.03, 0.1, 0.2],
            "model__gamma": ["scale", 0.01, 0.1],
        },
    }
    # Quick mode tests the complete control flow with at least two candidates
    # per tuned family while avoiding the cost of the full scientific search.
    quick_grids = {
        "ridge": {"model__alpha": [1.0, 10.0]},
        "elastic_net": {"model__alpha": [0.01, 0.1], "model__l1_ratio": [0.5]},
        "svr_rbf": {"model__C": [1, 10], "model__epsilon": [0.1], "model__gamma": ["scale"]},
    }
    grids = quick_grids if quick else full_grids

    return {
        "dummy_mean": ModelSpec(
            name="dummy_mean",
            pipeline=Pipeline([*common_steps, ("model", DummyRegressor(strategy="mean"))]),
            parameter_grid={},
        ),
        "ridge": ModelSpec(
            name="ridge",
            pipeline=Pipeline([*scaled_steps, ("model", Ridge())]),
            parameter_grid=grids["ridge"],
        ),
        "elastic_net": ModelSpec(
            name="elastic_net",
            pipeline=Pipeline(
                [
                    *scaled_steps,
                    (
                        "model",
                        ElasticNet(max_iter=100_000, tol=1e-4, random_state=seed),
                    ),
                ]
            ),
            parameter_grid=grids["elastic_net"],
        ),
        "svr_rbf": ModelSpec(
            name="svr_rbf",
            pipeline=Pipeline([*scaled_steps, ("model", SVR(kernel="rbf"))]),
            parameter_grid=grids["svr_rbf"],
        ),
    }


def make_optional_random_forest_spec(quick: bool, seed: int) -> ModelSpec:
    """Build the strictly optional tree model and its exploratory search grid.

    Random Forest still receives fold-local median imputation and constant-
    predictor removal. It deliberately omits RobustScaler because decision
    trees split on ordered thresholds and are invariant to monotonic rescaling.

    The full grid is the exploratory grid documented after completion of the
    core analysis. Quick mode uses four lightweight candidates so developers
    can test the full path without waiting for thousands of fitted trees.
    """

    full_grid = {
        "model__n_estimators": [300, 600],
        "model__max_features": ["sqrt", 0.3, 0.7],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_depth": [None, 10, 20],
    }
    quick_grid = {
        "model__n_estimators": [50],
        "model__max_features": ["sqrt", 0.5],
        "model__min_samples_leaf": [1, 3],
        "model__max_depth": [None],
    }
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            (
                "model",
                RandomForestRegressor(
                    random_state=seed,
                    # Parallelism belongs to GridSearchCV. Keeping each forest
                    # single-threaded prevents nested process oversubscription.
                    n_jobs=1,
                ),
            ),
        ]
    )
    return ModelSpec(
        name=OPTIONAL_RANDOM_FOREST_NAME,
        pipeline=pipeline,
        parameter_grid=quick_grid if quick else full_grid,
    )


CCC_SCORER = make_scorer(concordance_correlation_coefficient, greater_is_better=True)
SCORING = {
    "ccc": CCC_SCORER,
    "neg_rmse": "neg_root_mean_squared_error",
}


def select_ccc_then_rmse(cv_results: dict[str, Any]) -> int:
    """Choose highest mean validation CCC, then lowest RMSE for exact ties."""

    ccc = np.asarray(cv_results["mean_test_ccc"], dtype=float)
    negative_rmse = np.asarray(cv_results["mean_test_neg_rmse"], dtype=float)
    finite = np.isfinite(ccc) & np.isfinite(negative_rmse)
    if not finite.any():
        raise ValueError("Every hyperparameter candidate produced non-finite CV scores")
    best_ccc = np.max(ccc[finite])
    tied = finite & np.isclose(ccc, best_ccc, rtol=0.0, atol=1e-12)
    candidates = np.flatnonzero(tied)
    # sklearn negates RMSE because scorers are maximized; the largest negative
    # value is therefore the smallest ordinary RMSE.
    return int(candidates[np.argmax(negative_rmse[candidates])])


def _json_safe(value: Any) -> Any:
    """Convert numpy/scikit values into ordinary JSON/YAML-compatible types."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def tune_and_fit(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    inner_splits: int,
    seed: int,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune one model entirely within supplied training data, then refit it."""

    if not spec.parameter_grid:
        estimator = clone(spec.pipeline)
        estimator.fit(X, y)
        return estimator, {}

    split_indices = make_group_splits(groups, n_splits=inner_splits, seed=seed)
    search = GridSearchCV(
        estimator=clone(spec.pipeline),
        param_grid=spec.parameter_grid,
        scoring=SCORING,
        refit=select_ccc_then_rmse,
        cv=split_indices,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X, y)
    return search.best_estimator_, _json_safe(search.best_params_)


def run_nested_cross_validation(
    data: pd.DataFrame,
    feature_columns: list[str],
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    model_specs: dict[str, ModelSpec],
    inner_n_splits: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate complete out-of-fold predictions and per-fold metrics."""

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    total_fits = len(TARGET_COLUMNS) * len(model_specs) * len(outer_splits)
    completed = 0

    X_all = data[feature_columns]
    groups_all = data["track_id"]
    for outcome, target_column in TARGET_COLUMNS.items():
        y_all = data[target_column]
        for model_name, spec in model_specs.items():
            for outer_fold, (train_indices, test_indices) in enumerate(outer_splits):
                X_train = X_all.iloc[train_indices]
                X_test = X_all.iloc[test_indices]
                y_train = y_all.iloc[train_indices]
                y_test = y_all.iloc[test_indices]
                train_groups = groups_all.iloc[train_indices]

                estimator, best_parameters = tune_and_fit(
                    spec=spec,
                    X=X_train,
                    y=y_train,
                    groups=train_groups,
                    inner_splits=inner_n_splits,
                    seed=seed + outer_fold,
                    n_jobs=n_jobs,
                )
                predicted = np.asarray(estimator.predict(X_test), dtype=float)
                if predicted.shape != y_test.to_numpy().shape or not np.isfinite(predicted).all():
                    raise ValueError(f"Invalid predictions from {model_name}/{outcome}/fold {outer_fold}")

                parameters_json = json.dumps(best_parameters, sort_keys=True)
                for row_position, observed, prediction in zip(test_indices, y_test, predicted, strict=True):
                    prediction_rows.append(
                        {
                            "track_id": int(data.iloc[row_position]["track_id"]),
                            "outer_fold": int(outer_fold),
                            "outcome": outcome,
                            "model": model_name,
                            "observed": float(observed),
                            "predicted": float(prediction),
                            "absolute_error": float(abs(observed - prediction)),
                            "best_parameters": parameters_json,
                        }
                    )

                metrics = calculate_metrics(y_test, predicted)
                metric_rows.append(
                    {
                        "outcome": outcome,
                        "model": model_name,
                        "outer_fold": int(outer_fold),
                        "n_test_tracks": int(len(test_indices)),
                        **metrics,
                        "best_parameters": parameters_json,
                    }
                )
                completed += 1
                print(
                    f"[{completed:02d}/{total_fits}] {outcome:7s} | {model_name:11s} | "
                    f"outer fold {outer_fold + 1}/{len(outer_splits)} | "
                    f"CCC={metrics['ccc']:.4f} RMSE={metrics['rmse']:.4f}",
                    flush=True,
                )

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["outcome", "model", "outer_fold", "track_id"], kind="stable"
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["outcome", "model", "outer_fold"], kind="stable"
    ).reset_index(drop=True)
    return predictions, fold_metrics


def validate_oof_predictions(
    predictions: pd.DataFrame,
    fold_table: pd.DataFrame,
    expected_models: Iterable[str] = MODEL_NAMES,
    expected_outcomes: Iterable[str] = TARGET_COLUMNS,
) -> None:
    """Prove complete coverage and identical outer folds for every comparison."""

    expected_fold_map = fold_table.set_index("track_id")["outer_fold"].sort_index()
    expected_ids = set(expected_fold_map.index.astype(int))
    duplicate_key = ["track_id", "outcome", "model"]
    if predictions.duplicated(duplicate_key).any():
        raise ValueError("A track has more than one OOF prediction for a model/outcome")

    for outcome in expected_outcomes:
        for model_name in expected_models:
            subset = predictions[
                (predictions["outcome"] == outcome) & (predictions["model"] == model_name)
            ]
            if set(subset["track_id"].astype(int)) != expected_ids or len(subset) != len(expected_ids):
                raise ValueError(f"Incomplete OOF coverage for {outcome}/{model_name}")
            actual_fold_map = subset.set_index("track_id")["outer_fold"].sort_index()
            if not actual_fold_map.equals(expected_fold_map):
                raise ValueError(f"Outer-fold assignments changed for {outcome}/{model_name}")

    numeric = predictions[["observed", "predicted", "absolute_error"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("OOF prediction output contains non-finite values")


def bootstrap_metric_intervals(
    observed: np.ndarray,
    predicted: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    """Calculate percentile 95% intervals by resampling whole tracks."""

    rng = np.random.default_rng(seed)
    names = ("mae", "rmse", "pearson_r", "spearman_rho", "ccc")
    distributions = {name: np.empty(n_resamples, dtype=float) for name in names}
    n_tracks = len(observed)
    for resample in range(n_resamples):
        indices = rng.integers(0, n_tracks, size=n_tracks)
        metrics = calculate_metrics(observed[indices], predicted[indices])
        for name in names:
            distributions[name][resample] = metrics[name]

    intervals: dict[str, float] = {}
    for name, values in distributions.items():
        intervals[f"{name}_ci_low"] = float(np.percentile(values, 2.5))
        intervals[f"{name}_ci_high"] = float(np.percentile(values, 97.5))
    return intervals


def build_model_comparison(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    model_names: Iterable[str] = MODEL_NAMES,
) -> pd.DataFrame:
    """Combine pooled OOF metrics, fold variability, and bootstrap intervals."""

    rows: list[dict[str, Any]] = []
    combination_index = 0
    for outcome in TARGET_COLUMNS:
        for model_name in model_names:
            subset = predictions[
                (predictions["outcome"] == outcome) & (predictions["model"] == model_name)
            ]
            observed = subset["observed"].to_numpy(dtype=float)
            predicted = subset["predicted"].to_numpy(dtype=float)
            pooled = calculate_metrics(observed, predicted)
            folds = fold_metrics[
                (fold_metrics["outcome"] == outcome) & (fold_metrics["model"] == model_name)
            ]
            intervals = bootstrap_metric_intervals(
                observed,
                predicted,
                n_resamples=n_bootstrap,
                seed=seed + combination_index,
            )
            rows.append(
                {
                    "outcome": outcome,
                    "model": model_name,
                    "n_tracks": int(len(subset)),
                    **pooled,
                    "mean_fold_ccc": float(folds["ccc"].mean()),
                    "std_fold_ccc": float(folds["ccc"].std(ddof=0)),
                    "mean_fold_rmse": float(folds["rmse"].mean()),
                    **intervals,
                    "bootstrap_resamples": int(n_bootstrap),
                }
            )
            combination_index += 1

    comparison = pd.DataFrame(rows)
    numeric = comparison.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Model-comparison table contains non-finite metrics")
    return comparison.sort_values(["outcome", "model"], kind="stable").reset_index(drop=True)


def choose_winners(comparison: pd.DataFrame) -> dict[str, str]:
    """Select each target's model by mean fold CCC, then pooled RMSE."""

    winners: dict[str, str] = {}
    for outcome in TARGET_COLUMNS:
        ranked = comparison[comparison["outcome"] == outcome].sort_values(
            ["mean_fold_ccc", "rmse", "model"],
            ascending=[False, True, True],
            kind="stable",
        )
        if ranked.empty:
            raise ValueError(f"No comparison rows are available for {outcome}")
        winners[outcome] = str(ranked.iloc[0]["model"])
    return winners


def make_output_paths(quick: bool) -> OutputPaths:
    """Keep quick-test artifacts completely separate from reportable results."""

    suffix = "_test" if quick else ""
    model_directory = REPO_ROOT / "data_processed" / "models" / ("test" if quick else "")
    return OutputPaths(
        folds=REPO_ROOT / "data_processed" / "splits" / f"pmemo_outer_folds{suffix}.csv",
        predictions=REPO_ROOT / "results" / f"pmemo_out_of_fold_predictions{suffix}.csv",
        fold_metrics=REPO_ROOT / "results" / f"pmemo_fold_metrics{suffix}.csv",
        comparison=REPO_ROOT / "results" / f"pmemo_model_comparison{suffix}.csv",
        figure=REPO_ROOT / "results" / f"pmemo_predicted_vs_observed{suffix}.png",
        valence_model=model_directory / "pmemo_valence_model.joblib",
        arousal_model=model_directory / "pmemo_arousal_model.joblib",
        manifest=model_directory / "pmemo_model_manifest.yaml",
    )


def make_optional_random_forest_paths(quick: bool) -> OptionalRandomForestPaths:
    """Name optional artifacts so they cannot be mistaken for primary outputs."""

    suffix = "_test" if quick else ""
    return OptionalRandomForestPaths(
        predictions=REPO_ROOT
        / "results"
        / f"pmemo_random_forest_out_of_fold_predictions{suffix}.csv",
        fold_metrics=REPO_ROOT / "results" / f"pmemo_random_forest_fold_metrics{suffix}.csv",
        augmented_comparison=REPO_ROOT
        / "results"
        / f"pmemo_model_comparison_with_optional_random_forest{suffix}.csv",
        figure=REPO_ROOT
        / "results"
        / f"pmemo_random_forest_predicted_vs_observed{suffix}.png",
    )


def ensure_outputs_available(
    paths: OutputPaths | OptionalRandomForestPaths,
    overwrite: bool,
) -> None:
    """Stop before model fitting if any target would be overwritten accidentally."""

    existing = [path for path in paths.all_paths() if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Output targets already exist; use --overwrite:\n  {joined}")


def make_prediction_figure(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot only held-out predictions, with common axes within each outcome."""

    figure, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    for row, outcome in enumerate(TARGET_COLUMNS):
        outcome_data = predictions[predictions["outcome"] == outcome]
        lower = float(min(outcome_data["observed"].min(), outcome_data["predicted"].min()))
        upper = float(max(outcome_data["observed"].max(), outcome_data["predicted"].max()))
        padding = max((upper - lower) * 0.05, 0.01)
        limits = (lower - padding, upper + padding)

        for column, model_name in enumerate(MODEL_NAMES):
            axis = axes[row, column]
            subset = outcome_data[outcome_data["model"] == model_name]
            metrics = calculate_metrics(subset["observed"], subset["predicted"])
            axis.scatter(subset["observed"], subset["predicted"], s=13, alpha=0.45, edgecolors="none")
            axis.plot(limits, limits, linestyle="--", linewidth=1.2, color="black", label="identity")
            if np.ptp(subset["observed"].to_numpy()) > 0:
                slope, intercept = np.polyfit(subset["observed"], subset["predicted"], deg=1)
                x_line = np.array(limits)
                axis.plot(x_line, slope * x_line + intercept, linewidth=1.4, color="#d95f02")
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            axis.set_title(f"{outcome.title()} - {model_name}")
            axis.text(
                0.03,
                0.97,
                f"n={len(subset)}\nCCC={metrics['ccc']:.3f}\nRMSE={metrics['rmse']:.3f}",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            if row == 1:
                axis.set_xlabel("Observed PMEmo rating")
            if column == 0:
                axis.set_ylabel("Out-of-fold prediction")

    figure.suptitle("PMEmo static emotion prediction - held-out tracks only", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_optional_random_forest_figure(predictions: pd.DataFrame, output_path: Path) -> None:
    """Render the exploratory forest alone, using held-out predictions only."""

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for axis, outcome in zip(axes, TARGET_COLUMNS, strict=True):
        subset = predictions[predictions["outcome"] == outcome]
        metrics = calculate_metrics(subset["observed"], subset["predicted"])
        lower = float(min(subset["observed"].min(), subset["predicted"].min()))
        upper = float(max(subset["observed"].max(), subset["predicted"].max()))
        padding = max((upper - lower) * 0.05, 0.01)
        limits = (lower - padding, upper + padding)
        axis.scatter(
            subset["observed"],
            subset["predicted"],
            s=14,
            alpha=0.45,
            edgecolors="none",
        )
        axis.plot(limits, limits, linestyle="--", linewidth=1.2, color="black")
        slope, intercept = np.polyfit(subset["observed"], subset["predicted"], deg=1)
        x_line = np.asarray(limits)
        axis.plot(x_line, slope * x_line + intercept, linewidth=1.4, color="#d95f02")
        axis.set(xlim=limits, ylim=limits, title=outcome.title())
        axis.set_xlabel("Observed PMEmo rating")
        axis.set_ylabel("Random Forest out-of-fold prediction")
        axis.text(
            0.03,
            0.97,
            f"n={len(subset)}\nCCC={metrics['ccc']:.3f}\nRMSE={metrics['rmse']:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    figure.suptitle("OPTIONAL AFTERTHOUGHT - Random Forest held-out predictions", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_optional_random_forest_outputs(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    forest_comparison: pd.DataFrame,
    primary_comparison_path: Path,
    paths: OptionalRandomForestPaths,
) -> pd.DataFrame:
    """Save isolated forest results and a clearly labelled augmented table.

    The original ``pmemo_model_comparison.csv`` is read but never modified.
    Its rows are copied into a new table and labelled ``primary_core``; the two
    forest rows are labelled ``optional_afterthought`` and explicitly declared
    ineligible to change the frozen primary model selection.
    """

    if not primary_comparison_path.exists():
        raise FileNotFoundError(
            "The primary comparison must exist before running the optional analysis: "
            f"{primary_comparison_path}"
        )
    primary = pd.read_csv(primary_comparison_path)
    expected_primary_pairs = {
        (outcome, model_name) for outcome in TARGET_COLUMNS for model_name in MODEL_NAMES
    }
    observed_primary_pairs = set(zip(primary["outcome"], primary["model"], strict=True))
    if observed_primary_pairs != expected_primary_pairs:
        raise ValueError("Primary comparison does not contain the expected frozen core models")

    primary = primary.assign(
        analysis_role="primary_core",
        eligible_for_primary_selection=True,
        interpretation="Prespecified Stage B comparison",
    )
    forest_labelled = forest_comparison.assign(
        analysis_role="optional_afterthought",
        eligible_for_primary_selection=False,
        interpretation="Exploratory only; does not revise frozen primary winners",
    )
    augmented = pd.concat([primary, forest_labelled], ignore_index=True)
    # Force optional rows to the bottom of each outcome block. Alphabetical
    # sorting would put "optional" before "primary", contradicting the intended
    # visual hierarchy even though the statistical labels were still correct.
    augmented["_role_order"] = augmented["analysis_role"].map(
        {"primary_core": 0, "optional_afterthought": 1}
    )
    augmented = augmented.sort_values(
        ["outcome", "_role_order", "model"], kind="stable"
    ).drop(columns="_role_order")

    tables = {
        paths.predictions: predictions,
        paths.fold_metrics: fold_metrics,
        paths.augmented_comparison: augmented,
    }
    for path, table in tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        if pd.read_csv(path).shape != table.shape:
            raise IOError(f"Optional output failed reopen validation: {path}")
    return augmented.reset_index(drop=True)


def _selected_feature_names(pipeline: Pipeline, feature_columns: list[str]) -> list[str]:
    """Recover names surviving the fitted constant-feature filter."""

    support = pipeline.named_steps["variance"].get_support()
    return np.asarray(feature_columns, dtype=object)[support].tolist()


def fit_and_save_final_models(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_specs: dict[str, ModelSpec],
    winners: dict[str, str],
    inner_splits: int,
    seed: int,
    n_jobs: int,
    paths: OutputPaths,
    fold_table: pd.DataFrame,
    quick: bool,
) -> dict[str, Any]:
    """Retune each chosen family on all available data, save, and reopen it."""

    X = data[feature_columns]
    groups = data["track_id"]
    model_paths = {"valence": paths.valence_model, "arousal": paths.arousal_model}
    final_models: dict[str, Any] = {}

    for outcome, target_column in TARGET_COLUMNS.items():
        model_name = winners[outcome]
        fitted, parameters = tune_and_fit(
            spec=model_specs[model_name],
            X=X,
            y=data[target_column],
            groups=groups,
            inner_splits=inner_splits,
            seed=seed + 100,
            n_jobs=n_jobs,
        )
        path = model_paths[outcome]
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, path)

        # A saved model is not accepted merely because joblib.dump returned.
        # Reopen it and require finite predictions in the registered schema.
        reopened = joblib.load(path)
        check_predictions = np.asarray(reopened.predict(X.iloc[:5]), dtype=float)
        if check_predictions.shape != (5,) or not np.isfinite(check_predictions).all():
            raise IOError(f"Reopened {outcome} model failed prediction validation")
        final_models[outcome] = {
            "target_column": target_column,
            "model_family": model_name,
            "hyperparameters": parameters,
            "artifact": str(path.relative_to(REPO_ROOT)),
            "features_after_variance_filter": _selected_feature_names(reopened, feature_columns),
        }

    feature_hash = hashlib.sha256("\n".join(feature_columns).encode("utf-8")).hexdigest()
    manifest = {
        "analysis_scope": "PMEmo static induced valence/arousal only; ds002721 and EEG deferred",
        "run_mode": "quick_test" if quick else "full",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "training_sample_count": int(len(data)),
        "training_track_ids": data["track_id"].astype(int).tolist(),
        "source_files": {
            "features": str(FEATURES_PATH.relative_to(REPO_ROOT)),
            "labels": str(LABELS_PATH.relative_to(REPO_ROOT)),
            # Record all three reviewed configuration files even though only
            # splits.yaml directly controls this script. This makes it clear
            # which project-level analysis and extraction declarations were
            # in force when the frozen models were produced.
            "analysis_plan_config": str(ANALYSIS_PLAN_PATH.relative_to(REPO_ROOT)),
            "features_config": str(FEATURES_CONFIG_PATH.relative_to(REPO_ROOT)),
            "splits_config": str(SPLITS_CONFIG_PATH.relative_to(REPO_ROOT)),
            "training_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        },
        "feature_count": len(feature_columns),
        "feature_order_sha256": feature_hash,
        "feature_order": feature_columns,
        "split_protocol": {
            "method": "nested GroupKFold",
            "group": "track_id",
            "outer_splits": int(fold_table["outer_fold"].nunique()),
            "inner_splits": inner_splits,
            "outer_fold_file": str(paths.folds.relative_to(REPO_ROOT)),
        },
        "selection_rule": {
            "primary": "highest mean outer-fold CCC",
            "tie_breaker": "lowest pooled out-of-fold RMSE",
            "final_hyperparameters": "CCC-first inner grouped CV on all training tracks",
        },
        "models": final_models,
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        },
    }
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(yaml.safe_dump(_json_safe(manifest), sort_keys=False, width=100))

    reopened_manifest = yaml.safe_load(paths.manifest.read_text())
    if reopened_manifest["feature_order"] != feature_columns:
        raise IOError("Saved model manifest changed the registered feature order")
    return manifest


def write_and_reopen_tables(
    fold_table: pd.DataFrame,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    """Write all CSV deliverables and prove they reopen with unchanged shapes."""

    tables = {
        paths.folds: fold_table,
        paths.predictions: predictions,
        paths.fold_metrics: fold_metrics,
        paths.comparison: comparison,
    }
    for path, table in tables.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        reopened = pd.read_csv(path)
        if reopened.shape != table.shape:
            raise IOError(f"Reopened table shape changed for {path}: {reopened.shape} != {table.shape}")


def run_optional_random_forest_analysis(
    data: pd.DataFrame,
    feature_columns: list[str],
    inner_n_splits: int,
    outer_n_splits: int,
    bootstrap_resamples: int,
    seed: int,
    n_jobs: int,
    quick: bool,
    overwrite: bool,
    started: float,
) -> int:
    """Run the post-primary Random Forest without changing primary artifacts."""

    primary_paths = make_output_paths(quick=quick)
    optional_paths = make_optional_random_forest_paths(quick=quick)
    ensure_outputs_available(optional_paths, overwrite=overwrite)
    fold_table, outer_splits = load_saved_outer_splits(
        data=data,
        fold_path=primary_paths.folds,
        expected_n_splits=outer_n_splits,
    )
    forest_spec = make_optional_random_forest_spec(quick=quick, seed=seed)
    model_specs = {forest_spec.name: forest_spec}
    candidate_count = int(np.prod([len(values) for values in forest_spec.parameter_grid.values()]))

    print("Mode: OPTIONAL RANDOM FOREST AFTERTHOUGHT" + (" - QUICK TEST" if quick else ""))
    print("Primary Stage B outputs and winners will not be modified.")
    print(f"Modeling tracks: {len(data)}")
    print(f"Predictors: {len(feature_columns)}")
    print(f"Reused outer-fold file: {primary_paths.folds.relative_to(REPO_ROOT)}")
    print(f"Outer/inner folds: {outer_n_splits}/{inner_n_splits}")
    print(f"Hyperparameter candidates: {candidate_count}")
    print(f"Bootstrap resamples: {bootstrap_resamples}")

    predictions, fold_metrics = run_nested_cross_validation(
        data=data,
        feature_columns=feature_columns,
        outer_splits=outer_splits,
        model_specs=model_specs,
        inner_n_splits=inner_n_splits,
        seed=seed,
        n_jobs=n_jobs,
    )
    validate_oof_predictions(
        predictions,
        fold_table,
        expected_models=[OPTIONAL_RANDOM_FOREST_NAME],
    )
    forest_comparison = build_model_comparison(
        predictions,
        fold_metrics,
        n_bootstrap=bootstrap_resamples,
        seed=seed + 500,
        model_names=[OPTIONAL_RANDOM_FOREST_NAME],
    )
    augmented = write_optional_random_forest_outputs(
        predictions=predictions,
        fold_metrics=fold_metrics,
        forest_comparison=forest_comparison,
        primary_comparison_path=primary_paths.comparison,
        paths=optional_paths,
    )
    make_optional_random_forest_figure(predictions, optional_paths.figure)
    if not optional_paths.figure.exists() or optional_paths.figure.stat().st_size == 0:
        raise IOError("Optional Random Forest figure was not written correctly")

    # Confirm that the combined afterthought table contains all eight original
    # rows plus exactly two optional rows, one per target. This is a reporting
    # comparison only; no winner-selection or final-model fitting is invoked.
    optional_rows = augmented[augmented["analysis_role"] == "optional_afterthought"]
    if len(augmented) != 10 or len(optional_rows) != 2:
        raise AssertionError("Augmented comparison has incorrect primary/optional coverage")
    if optional_rows["eligible_for_primary_selection"].astype(bool).any():
        raise AssertionError("Optional Random Forest was incorrectly made selection-eligible")

    elapsed = time.perf_counter() - started
    display = forest_comparison[
        ["outcome", "model", "mae", "rmse", "pearson_r", "spearman_rho", "ccc", "mean_fold_ccc"]
    ]
    print("\nStrictly optional Random Forest results")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Elapsed seconds: {elapsed:.1f}")
    print("Optional outputs passed validation; primary artifacts remain frozen.")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Define explicit full/quick modes and safe resource controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run an isolated reduced-grid software test.")
    parser.add_argument(
        "--random-forest",
        action="store_true",
        help=(
            "Run only the strictly optional post-primary Random Forest analysis, "
            "reusing frozen folds and writing separate outputs."
        ),
    )
    parser.add_argument("--quick-size", type=int, default=90, help="Tracks sampled in quick mode (default: 90).")
    parser.add_argument("--overwrite", action="store_true", help="Replace this mode's existing artifacts.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs inside each hyperparameter search (default: all available cores).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute validation, nested CV, selection, final fitting, and artifact QA."""

    args = build_argument_parser().parse_args(argv)
    started = time.perf_counter()
    config = load_splits_config()
    seed = int(config.get("random_seed", 2026))
    full_outer_splits = int(config["pmemo_final_estimate"]["outer_splits"])
    full_inner_splits = int(config["pmemo_internal_model_selection"]["n_splits"])
    bootstrap_resamples = int(config["uncertainty"]["bootstrap"]["n_resamples"])

    data, feature_columns = load_and_validate_inputs()
    if args.quick:
        if args.quick_size < 12 or args.quick_size > len(data):
            raise ValueError(f"--quick-size must be between 12 and {len(data)}")
        # This deterministic random subset is for software validation only and
        # can never be confused with final results because every output path is
        # suffixed/test-scoped.
        data = data.sample(n=args.quick_size, random_state=seed).sort_values(
            "track_id", kind="stable"
        ).reset_index(drop=True)
        outer_n_splits = 3
        inner_n_splits = 2
        bootstrap_resamples = 200
    else:
        outer_n_splits = full_outer_splits
        inner_n_splits = full_inner_splits

    # The optional branch returns before any primary output path is checked or
    # written. Consequently, requesting Random Forest cannot refit the core
    # models, alter their winners, or overwrite their scientific artifacts.
    if args.random_forest:
        return run_optional_random_forest_analysis(
            data=data,
            feature_columns=feature_columns,
            inner_n_splits=inner_n_splits,
            outer_n_splits=outer_n_splits,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            n_jobs=args.n_jobs,
            quick=args.quick,
            overwrite=args.overwrite,
            started=started,
        )

    paths = make_output_paths(quick=args.quick)
    ensure_outputs_available(paths, overwrite=args.overwrite)
    model_specs = make_model_specs(quick=args.quick, seed=seed)
    fold_table, outer_splits = make_outer_fold_table(data, outer_n_splits, seed)

    mode = "QUICK SOFTWARE TEST" if args.quick else "FULL SCIENTIFIC RUN"
    print(f"Mode: {mode}")
    print(f"Modeling tracks: {len(data)}")
    print(f"Predictors: {len(feature_columns)}")
    print(f"Outer/inner folds: {outer_n_splits}/{inner_n_splits}")
    print(f"Models: {', '.join(model_specs)}")
    print(f"Targets: {', '.join(TARGET_COLUMNS)}")
    print(f"Bootstrap resamples: {bootstrap_resamples}")

    predictions, fold_metrics = run_nested_cross_validation(
        data=data,
        feature_columns=feature_columns,
        outer_splits=outer_splits,
        model_specs=model_specs,
        inner_n_splits=inner_n_splits,
        seed=seed,
        n_jobs=args.n_jobs,
    )
    validate_oof_predictions(predictions, fold_table)
    comparison = build_model_comparison(
        predictions,
        fold_metrics,
        n_bootstrap=bootstrap_resamples,
        seed=seed,
    )
    winners = choose_winners(comparison)

    # Fold assignments are written before the model manifest so the manifest
    # points to a real, validated artifact. All remaining outputs are then
    # reopened or otherwise checked before success is reported.
    write_and_reopen_tables(fold_table, predictions, fold_metrics, comparison, paths)
    make_prediction_figure(predictions, paths.figure)
    if not paths.figure.exists() or paths.figure.stat().st_size == 0:
        raise IOError("Predicted-versus-observed figure was not written correctly")
    fit_and_save_final_models(
        data=data,
        feature_columns=feature_columns,
        model_specs=model_specs,
        winners=winners,
        inner_splits=inner_n_splits,
        seed=seed,
        n_jobs=args.n_jobs,
        paths=paths,
        fold_table=fold_table,
        quick=args.quick,
    )

    elapsed = time.perf_counter() - started
    print("\nModel comparison (pooled out-of-fold metrics)")
    display_columns = ["outcome", "model", "mae", "rmse", "pearson_r", "spearman_rho", "ccc", "mean_fold_ccc"]
    print(comparison[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected valence model: {winners['valence']}")
    print(f"Selected arousal model: {winners['arousal']}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print("All Stage B artifacts passed validation.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, IOError, RuntimeError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
