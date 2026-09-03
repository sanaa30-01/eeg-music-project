"""Unit tests for PMEmo Stage B metrics, validation, folds, and artifacts."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "06_train_pmemo_models.py"
SPEC = importlib.util.spec_from_file_location("train_pmemo_models", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - defensive import guard
    raise ImportError(f"Unable to load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConcordanceTests(unittest.TestCase):
    """Check the primary selection metric, including degenerate inputs."""

    def test_identical_values_have_perfect_concordance(self) -> None:
        values = [0.1, 0.4, 0.8, 1.0]
        self.assertAlmostEqual(MODULE.concordance_correlation_coefficient(values, values), 1.0)

    def test_reversed_values_have_negative_concordance(self) -> None:
        true = [0.1, 0.4, 0.8, 1.0]
        predicted = list(reversed(true))
        self.assertLess(MODULE.concordance_correlation_coefficient(true, predicted), 0.0)

    def test_constant_inputs_are_finite_and_safe(self) -> None:
        self.assertEqual(MODULE.concordance_correlation_coefficient([2, 2], [2, 2]), 1.0)
        self.assertEqual(MODULE.concordance_correlation_coefficient([2, 2], [3, 3]), 0.0)
        metrics = MODULE.calculate_metrics([1, 2, 3], [2, 2, 2])
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertEqual(metrics["pearson_r"], 0.0)


class DataAndSplitTests(unittest.TestCase):
    """Exercise one-to-one input validation and group-disjoint folds."""

    def test_small_validated_join_excludes_ids_and_targets_from_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.parquet"
            label_path = root / "labels.csv"
            pd.DataFrame(
                {
                    "track_id": [1, 2, 3, 4],
                    "feature_a": [0.1, 0.2, 0.3, 0.4],
                    "feature_b": [1.0, 0.5, -0.5, -1.0],
                }
            ).to_parquet(feature_path, index=False)
            pd.DataFrame(
                {
                    "musicId": [1, 2, 3],
                    "Arousal(mean)": [0.2, 0.4, 0.6],
                    "Valence(mean)": [0.3, 0.5, 0.7],
                }
            ).to_csv(label_path, index=False)

            data, feature_columns = MODULE.load_and_validate_inputs(
                feature_path,
                label_path,
                expected_audio_rows=4,
                expected_labelled_rows=3,
                expected_feature_count=2,
            )

        self.assertEqual(len(data), 3)
        self.assertEqual(feature_columns, ["feature_a", "feature_b"])
        self.assertTrue(set(feature_columns).isdisjoint({"track_id", "musicId", *MODULE.TARGET_COLUMNS.values()}))

    def test_outer_splits_have_no_group_overlap_and_cover_every_track(self) -> None:
        groups = pd.Series(np.repeat(np.arange(12), 2))
        splits = MODULE.make_group_splits(groups, n_splits=3, seed=2026)
        tested_positions: list[int] = []
        for train_indices, test_indices in splits:
            self.assertTrue(set(groups.iloc[train_indices]).isdisjoint(set(groups.iloc[test_indices])))
            tested_positions.extend(test_indices.tolist())
        self.assertEqual(sorted(tested_positions), list(range(len(groups))))

    def test_saved_outer_folds_are_reused_exactly(self) -> None:
        """The optional model must consume, not regenerate, frozen fold IDs."""

        data = pd.DataFrame({"track_id": [20, 10, 40, 30], "feature": [1.0, 2.0, 3.0, 4.0]})
        saved = pd.DataFrame({"track_id": [10, 20, 30, 40], "outer_fold": [1, 0, 1, 0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            saved.to_csv(path, index=False)
            reopened, splits = MODULE.load_saved_outer_splits(data, path, expected_n_splits=2)

        pd.testing.assert_frame_equal(reopened, saved)
        observed_test_ids = [set(data.iloc[test_indices].track_id) for _, test_indices in splits]
        self.assertEqual(observed_test_ids, [{20, 40}, {10, 30}])

    def test_oof_validator_requires_one_prediction_per_track_and_shared_folds(self) -> None:
        folds = pd.DataFrame({"track_id": [1, 2, 3], "outer_fold": [0, 1, 2]})
        rows = []
        for outcome in MODULE.TARGET_COLUMNS:
            for model in MODULE.MODEL_NAMES:
                for track_id, fold in zip(folds.track_id, folds.outer_fold, strict=True):
                    rows.append(
                        {
                            "track_id": track_id,
                            "outer_fold": fold,
                            "outcome": outcome,
                            "model": model,
                            "observed": 0.5,
                            "predicted": 0.5,
                            "absolute_error": 0.0,
                        }
                    )
        predictions = pd.DataFrame(rows)
        MODULE.validate_oof_predictions(predictions, folds)
        with self.assertRaises(ValueError):
            MODULE.validate_oof_predictions(predictions.iloc[:-1], folds)


class ArtifactTests(unittest.TestCase):
    """Verify that a serialized sklearn model can be reopened and used."""

    def test_model_artifact_reopens_and_predicts(self) -> None:
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.linspace(0.2, 0.8, 6)
        model = Pipeline([("model", DummyRegressor(strategy="mean"))]).fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            reopened = joblib.load(path)
            predictions = reopened.predict(X)
        self.assertEqual(predictions.shape, y.shape)
        self.assertTrue(np.isfinite(predictions).all())


class OptionalRandomForestTests(unittest.TestCase):
    """Keep the post-primary forest technically and statistically separate."""

    def test_optional_forest_is_not_a_primary_model_and_is_not_scaled(self) -> None:
        spec = MODULE.make_optional_random_forest_spec(quick=False, seed=2026)
        self.assertNotIn(MODULE.OPTIONAL_RANDOM_FOREST_NAME, MODULE.MODEL_NAMES)
        self.assertEqual(list(spec.pipeline.named_steps), ["imputer", "variance", "model"])
        self.assertEqual(spec.parameter_grid["model__n_estimators"], [300, 600])
        self.assertEqual(spec.parameter_grid["model__max_features"], ["sqrt", 0.3, 0.7])

    def test_optional_output_paths_cannot_overwrite_primary_outputs(self) -> None:
        primary = set(MODULE.make_output_paths(quick=False).all_paths())
        optional = set(MODULE.make_optional_random_forest_paths(quick=False).all_paths())
        self.assertTrue(primary.isdisjoint(optional))


if __name__ == "__main__":
    unittest.main()
