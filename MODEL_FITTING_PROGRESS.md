# Mathematics student: model-fitting plan and progress

Last audited: 2026-09-02

This is the working tracker for the mathematics/programming role. It turns
the project brief into a short sequence of model-fitting decisions and records
what is actually reproducible from the repository. Update this file whenever a
milestone changes; do not infer completion from the presence of a script alone.

## Current position

The project has completed an initial EEG preprocessing, feature extraction, and
Stage A Ridge run. That run is **provisional**, because its outcome definitions
and model specification do not yet match the project brief. The current agreed
scope is PMEmo only: ds002721, EEG integration, and bridge analysis are deferred.
PMEmo Stage B static-label model fitting is now complete and reproducible.

**Current PMEmo-only Stage B progress: complete.** The wider project remains
incomplete because Stage A corrections and the deliberately deferred EEG/bridge
work have not been performed.

| Workstream | Status | Evidence / remaining issue |
|---|---|---|
| Repository and analysis plan | In progress | Structure and configs exist; the plan is not frozen and package versions are not pinned. |
| ds002721 trial table | Previously run, not locally reproducible now | Downstream results show it existed, but `data_processed/trials_ds002721.parquet` and raw ds002721 files are absent. |
| EEG preprocessing and features | Previously run | 927 of 1,119 attempted epochs retained (82.8%); 31 participants. Raw/intermediate/processed files needed to rerun are absent. |
| Stage A baseline and Ridge | Provisional | `results/stage_a_model_comparison.csv` exists, but only MAE is reported and important specification corrections are required below. |
| Exact-audio Gate A | Incomplete | Audit says 307/307 recovered, but all checksum, timing, and source-title fields are blank; source audio is not currently present. |
| PMEmo access | Ready | 794 chorus audio files are present under `data_raw/PMEmo/`. Current scripts expect lowercase `data_raw/pmemo/`. |
| PMEmo audio features | Complete | 794/794 tracks extracted successfully into 293 predictors; all 767 labelled tracks covered; separate QC report has zero failures. |
| Stage B PMEmo models | Complete | Nested grouped CV compared Dummy, Ridge, Elastic Net, and RBF-SVR for both static targets. Ridge won valence; Elastic Net won arousal. |
| Bridge, uncertainty, figures | Not started | Scripts 08-10 are stubs; Gate A and frozen Stage B model are prerequisites. |

## Corrections required before Stage A can be called final

1. **Use the brief's predeclared composites.** Compute z-scores using parameters
   learned from each training fold only:

   - valence-like = `(z(pleasant) + z(happy) + z(tender) - z(sad) - z(angry) - z(fearful)) / 6`
   - arousal-like = `(z(energetic) + z(tense)) / 2`

   The current code instead averages raw pleasant/happy/tender ratings for
   valence and raw energetic/tense/angry/fearful ratings for arousal.

2. **Make the baseline match the prediction question.** The present
   `leave_one_out_participant_mean` uses other labels belonging to the held-out
   participant, while Ridge is trained without that participant. Treat this as
   an oracle/personal-history baseline, not an equally blind LOPO baseline.
   Add an honest training-fold grand-mean baseline; if useful, report the
   participant-history and clip-mean baselines separately and label them.

3. **Scale EEG predictors inside every training fold.** Add `StandardScaler`
   after imputation and before Ridge, as specified in the brief. Do not scale
   once on the full dataset.

4. **Keep the preregistered alpha grid** `[0.1, 1, 10, 100]` for the primary
   result. The exploratory grid extended to 5,000 after seeing results must be
   labelled exploratory, not substituted for the primary analysis.

5. **Complete the evaluation.** For both outcomes, save out-of-fold predictions
   and report MAE, RMSE, Pearson r, Spearman rho, and confidence intervals.
   Add leave-one-clip-out as the secondary generalisation test. Then evaluate
   the eight individual ratings with Benjamini-Hochberg FDR correction.

Current provisional MAE results should therefore not be interpreted as the H1
answer. They show Ridge worse than the current participant-history baseline:
valence 1.210 vs 1.149 and arousal 1.087 vs 0.919 for the preregistered alpha
grid, but the comparison changes after the corrections above.

## Streamlined execution order

Work strictly top to bottom. Optional models wait until all must-have rows are
complete.

### 1. Restore and freeze inputs

- [ ] Restore/download ds002721 raw data and the Eerola soundtrack corpus to
  the documented ignored directories.
- [ ] Fix the PMEmo path/case mismatch (`PMEmo` versus `pmemo`) once, preferably
  through a config value used by scripts 01, 05, and 06.
- [ ] Rebuild the trial table and EEG features, or restore the exact processed
  artifacts used for the committed Stage A CSV.
- [ ] Freeze `configs/analysis_plan.yaml` with a date and log the composite and
  Gate A corrections.
- [ ] Pin the working environment and save deterministic split assignments.

**Done when:** another machine can regenerate the Stage A modeling table from
documented inputs without manual path edits.

### 2. Repair and finish Stage A

- [ ] Implement fold-safe outcome construction, honest baselines, fold-safe
  feature scaling, and nested LOPO Ridge tuning.
- [ ] Save one out-of-fold prediction row per participant-trial-model-outcome.
- [ ] Add the leave-one-clip-out sensitivity analysis.
- [ ] Produce the full metric table and grouped bootstrap intervals.
- [ ] State the result as an association/generalisation result, whether null or
  positive; do not tune further merely to beat the baseline.

**Done when:** H1 can be answered from a single table with a valid baseline,
five metrics, uncertainty intervals, and explicit sample counts.

### 3. Resolve Gate A using the dataset that actually exists

The brief assumes 40 shared clips, but the repository audit found **307 unique
stimulus IDs** drawn unevenly across participants; the retention table contains
296 attempted IDs and only 263 with at least one retained EEG trial. Do not use
the literal `32/40` rule without reconciling this mismatch.

- [ ] Verify the event-code-to-file mapping against authoritative metadata.
- [ ] Record duration/timing, SHA-256, source/title, terms, and extraction result
  for every candidate bridge clip.
- [ ] Predeclare a revised coverage rule based on unique clips and minimum
  participant/EEG support, then record pass/fail in the analysis plan.

**Done when:** the exact audio input for every bridge prediction is auditable,
and the team has signed one unambiguous go/no-go decision.

### 4. Fit and freeze Stage B

- [x] Implement `src/05_extract_audio_features.py`; produce one immutable row
  per PMEmo track and a feature-schema/QC report.
- [x] Save a fixed outer five-fold grouped split by track/song ID.
- [x] Fit separate valence and arousal Dummy, Ridge, Elastic Net, and RBF-SVR pipelines. Imputation,
  scaling, and hyperparameter tuning must occur inside the training folds.
- [x] Save outer-fold predictions and compute MAE, RMSE, Pearson r, Spearman
  rho, and concordance correlation coefficient (CCC).
- [x] Select by mean validation CCC, break ties by RMSE, refit on all PMEmo,
  and save a locked model plus its feature schema and split IDs.

**Done when:** `results/pmemo_model_comparison.csv`, out-of-fold predictions,
and frozen valence/arousal model artifacts can be regenerated by one command.

### 5. Bridge and uncertainty

- [ ] Extract the identical Stage B feature schema from audited ds002721 audio.
- [ ] Predict with the frozen PMEmo models; never recalibrate on ds002721.
- [ ] Aggregate self-reports and EEG features by clip under a stated minimum
  support rule.
- [ ] Compute bridge correlations with 2,000 clip bootstrap resamples and 1,000
  label permutations; report clip count and missingness for every test.
- [ ] Generate final tables, error analysis, and precise null/positive wording.

**Done when:** H2 and, only if Gate A passes, H3 have reproducible estimates,
uncertainty, and leakage checks.

## Deliberately postponed

Random Forest has now been evaluated once as a strictly optional post-primary
analysis; do not promote it into the frozen primary selection or final saved
models. Do not spend time yet on OpenL3/VGGish, DEAM, dynamic PMEmo labels,
late fusion, personalisation, or classification. They do not unblock the
minimum viable mathematical analysis.

## Completed Stage B run (2026-09-02)

- Validated 794 unique extracted PMEmo tracks, all 293 predictors finite, and
  767 unique statically labelled tracks. A validated one-to-one merge yielded
  exactly 767 modeling rows; identifiers and targets were excluded from the
  predictor matrix.
- Used one deterministic five-fold outer assignment (seed 2026) for every
  model and both outcomes. Each tuned model used five-fold grouped inner CV.
  Selection used highest CCC, with lower RMSE as the tie-breaker.
- Valence winner: Ridge. Mean outer-fold CCC = 0.6768; pooled OOF CCC = 0.6759,
  RMSE = 0.1160, MAE = 0.0903, Pearson r = 0.7007, and Spearman rho = 0.6984.
  The all-data refit selected `alpha=10.0`.
- Arousal winner: Elastic Net. Mean outer-fold CCC = 0.8335; pooled OOF CCC =
  0.8344, RMSE = 0.0994, MAE = 0.0775, Pearson r = 0.8432, and Spearman rho =
  0.8494. The all-data refit selected `alpha=0.01` and `l1_ratio=0.1`.
- The final full run took 28.2 seconds in `myenv` (Python 3.14.6, scikit-learn 1.9.0).
  Seven unit tests and the 90-track quick integration run passed. The final
  independent audit passed 69 assertions: 767 unique tracks per model/outcome,
  no missing or duplicate predictions, no train/test group overlap, finite
  metrics, exact shared folds, exact 293-column feature order, and successful
  reopening and prediction from both saved pipelines.
- Random Forest was not part of this primary run; it was evaluated later as the
  isolated optional analysis documented below. No ds002721 processing, dynamic
  labels, EEG/bridge integration, lyrics/openSMILE features, or deep-learning
  models were added.
- A NumPy/joblib deprecation warning appeared only while the unit test reopened
  a small temporary artifact; it did not affect fitting or final validation.

## Optional Random Forest afterthought (2026-09-02)

- Added an opt-in `--random-forest` branch to the Stage B script. It reloads the
  exact frozen outer-fold CSV and returns before any primary result, winner, or
  final-model code can run. It uses median imputation and constant-predictor
  removal inside each fold, but correctly omits scaling for the tree estimator.
- The full exploratory grid contained 54 combinations: 300/600 trees; `sqrt`,
  0.3, or 0.7 maximum feature fractions; minimum leaf sizes 1/3/5; and maximum
  depths unrestricted/10/20. Tuning used the same five-fold inner CCC-first,
  RMSE-tie-break procedure within each of the five frozen outer training sets.
- Valence Random Forest: mean outer-fold CCC = 0.6556, pooled OOF CCC = 0.6536,
  RMSE = 0.1140, MAE = 0.0907, Pearson r = 0.7120, and Spearman rho = 0.7110.
- Arousal Random Forest: mean outer-fold CCC = 0.8166, pooled OOF CCC = 0.8174,
  RMSE = 0.1010, MAE = 0.0792, Pearson r = 0.8383, and Spearman rho = 0.8400.
- Random Forest did not exceed the frozen primary winner on either outcome:
  valence Ridge mean-fold CCC remained 0.6768, and arousal Elastic Net remained
  0.8335. Therefore the original winners and joblib artifacts remain unchanged.
- The full optional run took 1,365.3 seconds (22 minutes 45.3 seconds). Twelve
  tests passed. Validation confirmed 767 OOF predictions per outcome, no
  duplicate/missing rows, finite metrics, identical frozen folds, two optional
  comparison rows explicitly ineligible for primary selection, and unchanged
  SHA-256 checksums for all protected primary comparison/prediction/model files.

## Progress log

| Date | Change | Next decision |
|---|---|---|
| 2026-08-29 | Audited the brief and repository; recorded current outputs, stubs, reproducibility gaps, composite mismatch, baseline mismatch, and the 40-versus-307 stimulus discrepancy. | Repair/freeze Stage A specification before starting Stage B. |
| 2026-08-29 | Implemented and tested PMEmo extraction; 794 tracks produced 293 finite predictors each, zero QC failures, and complete coverage of 767 static-label tracks. | Freeze grouped splits, then implement Stage B Ridge/SVR fitting. |
| 2026-09-02 | Completed PMEmo-only Stage B with leakage-safe nested grouped CV, reusable folds, OOF predictions and metrics, 2,000-resample intervals, saved winning pipelines/manifest, and an OOF-only diagnostic figure. | Use the frozen PMEmo models only when the team explicitly resumes a separately approved downstream stage. |
| 2026-09-02 | Evaluated Random Forest as an isolated optional afterthought using the frozen folds. It underperformed the primary winner for both targets, so Ridge/Elastic Net remain locked. | Retain the optional files as sensitivity evidence; do not revise the primary selection. |
| 2026-09-02 | Consolidated Stages 5–6 in `PMEMO_STAGES_5_6_DELIVERABLES.md`; updated README status and removed generated quick/test and temporary artifacts while retaining source tests. | Treat the PMEmo-only mathematics/programming handoff as complete. |
