# PMEmo Stages 5–6: deliverables and handoff

Last updated: 2026-09-02

## Synopsis

The mathematics/programming contribution for PMEmo Stages 5 and 6 is complete.
Stage 5 converted all 794 PMEmo chorus excerpts into a fixed table of 293
handcrafted audio predictors. All predictors are finite, and all 767 tracks
with static valence/arousal labels are covered. Stage 6 used those 767 tracks
for leakage-safe nested grouped cross-validation with one shared five-fold
outer split (seed 2026). The primary comparison included a mean Dummy baseline,
Ridge, Elastic Net, and RBF-SVR. Ridge won valence; Elastic Net won arousal.

Random Forest was subsequently tested as a strictly optional afterthought on
the same frozen folds. It did not beat the primary winner for either outcome,
so it was not promoted and no Random Forest final model was saved. ds002721,
EEG integration, bridge analysis, dynamic labels, lyrics/openSMILE features,
and deep-learning models remain outside the completed PMEmo-only scope.

## Final results

| Outcome | Frozen model | Mean outer-fold CCC | Pooled OOF CCC | RMSE | MAE |
|---|---|---:|---:|---:|---:|
| Valence | Ridge (`alpha=10`) | 0.6768 | 0.6759 | 0.1160 | 0.0903 |
| Arousal | Elastic Net (`alpha=0.01`, `l1_ratio=0.1`) | 0.8335 | 0.8344 | 0.0994 | 0.0775 |

Optional Random Forest reached mean outer-fold CCC 0.6556 for valence and
0.8166 for arousal. These are supporting sensitivity results, not replacements
for the frozen models.

## Deliverables to retain

| Purpose | File |
|---|---|
| Stage 5 implementation | `src/05_extract_audio_features.py` |
| Stage 6 implementation | `src/06_train_pmemo_models.py` |
| Stage 5 tests | `tests/test_05_extract_audio_features.py` |
| Stage 6 and optional-model tests | `tests/test_06_train_pmemo_models.py` |
| Final 293-feature table | `data_processed/audio_features_pmemo.parquet` |
| Extraction QC | `results/audio_extraction_pmemo_qc.csv` |
| Frozen outer folds | `data_processed/splits/pmemo_outer_folds.csv` |
| Primary OOF predictions | `results/pmemo_out_of_fold_predictions.csv` |
| Primary fold metrics | `results/pmemo_fold_metrics.csv` |
| Primary model comparison | `results/pmemo_model_comparison.csv` |
| Primary OOF figure | `results/pmemo_predicted_vs_observed.png` |
| Frozen valence pipeline | `data_processed/models/pmemo_valence_model.joblib` |
| Frozen arousal pipeline | `data_processed/models/pmemo_arousal_model.joblib` |
| Feature order, IDs, parameters, versions | `data_processed/models/pmemo_model_manifest.yaml` |
| Optional Random Forest OOF predictions | `results/pmemo_random_forest_out_of_fold_predictions.csv` |
| Optional Random Forest fold metrics | `results/pmemo_random_forest_fold_metrics.csv` |
| Separate augmented comparison | `results/pmemo_model_comparison_with_optional_random_forest.csv` |
| Optional Random Forest figure | `results/pmemo_random_forest_predicted_vs_observed.png` |
| Detailed mathematical progress | `MODEL_FITTING_PROGRESS.md` |

Large processed tables, figures, and model binaries are intentionally ignored
by Git where configured, but they remain required local scientific artifacts.

## Reproduction commands

Run from the repository root with the established environment:

```bash
/Users/srijan/eeg-music-project/myenv/bin/python -m unittest discover -s tests -v
/Users/srijan/eeg-music-project/myenv/bin/python src/05_extract_audio_features.py --overwrite
/Users/srijan/eeg-music-project/myenv/bin/python src/06_train_pmemo_models.py --overwrite
```

The optional Random Forest analysis is separate and expensive (the completed
run took about 22 minutes 45 seconds):

```bash
/Users/srijan/eeg-music-project/myenv/bin/python src/06_train_pmemo_models.py --random-forest --overwrite
```

Use `results/pmemo_out_of_fold_predictions.csv`, never predictions made on the
final models' 767 training tracks, when reporting generalization performance.
Use `data_processed/models/pmemo_model_manifest.yaml` whenever applying a saved
model because it records the exact required feature order.

## Completion checks

- 794 unique feature-table tracks and 293 finite predictors.
- 767 unique labelled/modeling tracks.
- One OOF prediction per track, outcome, and primary model.
- No train/test group overlap and identical outer folds for every comparison.
- All metrics finite and both frozen pipelines reopen successfully.
- Twelve repository tests passed after the optional Random Forest addition.
- Quick/test artifacts were deleted after validation; source tests were kept.
