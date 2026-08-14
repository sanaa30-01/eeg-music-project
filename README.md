# From Music Audio to Felt Affect

**Auditing audio-only emotion predictions against EEG and self-reports**

## Research question

Can an audio-only model trained on an accessible induced-emotion music dataset
(PMEmo) predict population-level valence and arousal ratings for separate
music excerpts, and do those predictions agree with independent self-report
and EEG evidence (OpenNeuro ds002721)?

See `configs/analysis_plan.yaml` for the full preregistration: hypotheses,
primary/secondary outcomes, exclusions, and the exact claim language this
project is allowed to make.

**Scope, one line:** this project does not claim music has a universal
emotional effect, that perceived emotion equals felt emotion, or that EEG
"validates" an emotion model. Its strongest possible conclusion is about
construct-level agreement between an audio model and population-average
self-report/EEG correlates in a separate dataset. See "What we will not
claim" below before writing anything up.

## Team roles (adjust names)

- **Person A** — pipeline/ML: repo, environment, PMEmo audio modeling, stats,
  reproducibility.
- **Person B** — EEG/cognitive science: ds002721 methods, EEG QC, self-report
  construct definitions, interpretation and write-up.
- Joint: the Day 4 audio-audit gate decision, model selection sign-off,
  limitations section.

## Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate music-eeg-affect

# or pip
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Freeze the environment once it's working:

```bash
pip freeze > requirements.lock.txt   # commit this alongside requirements.txt
```

Global random seed for every script: **2026**. Every train/test split gets
saved to `data_processed/splits/` — never regenerate a split silently.

## Getting the data

**You must obtain all three datasets yourself — no raw audio or raw EEG is
stored in this repository.** Full instructions, links, licenses, and exactly
what you're looking for are in [`data_raw/README_access.md`](data_raw/README_access.md).

Short version:
1. OpenNeuro **ds002721** — EEG + self-reports, openly downloadable, but the
   40 film-music audio clips are **not** included. Getting the matching audio
   is the single biggest risk in this project — see Gate A below.
2. **PMEmo** — 794 chorus excerpts + induced valence/arousal labels, ~1.3GB,
   hosted on Google Drive (linked from the PMEmo GitHub repo).
3. **DEAM** (optional, external benchmark only) — Creative Commons audio +
   perceived valence/arousal annotations.

## Gate A — the audio-access audit (do this by Day 4, before anything else)

Before writing any modeling code, audit whether the 40 ds002721 film-music
clips can actually be recovered as playable audio:

```bash
python src/01_audit_data.py --check-pmemo --init-ds002721-audit
```

This (a) sanity-checks that PMEmo metadata loaded correctly, and (b) creates/
validates `data_raw/ds002721_stimulus_audit.csv`, which you then fill in by
hand: one row per clip, with source URL, license/terms, clip timing,
checksum, and extraction status.

**Decision rule:**
- **≥32 / 40 clips** recoverable → proceed with the direct Stage A ↔ Stage B
  bridge as planned.
- **<32 / 40** → keep Stage A (EEG/self-report) and Stage B (PMEmo audio
  model) as independent analyses; downgrade the bridge claim to a
  construct-level comparison, not direct model-to-EEG validation.

Record the go/no-go decision itself (not just the CSV) — it's a required
Week 1 deliverable.

## Pipeline (run order)

```
01_audit_data.py            Gate A: PMEmo metadata check + ds002721 audit scaffold
02_build_ds002721_trials.py One row per participant × clip, from BIDS events/ratings
03_preprocess_eeg.py        Filter, epoch, artefact-reject (see configs/features.yaml)
04_extract_eeg_features.py  Predeclared band-power + frontal asymmetry features
05_extract_audio_features.py PMEmo (and, if Gate A passes, ds002721) audio features
06_train_pmemo_models.py    Ridge / SVR, grouped nested CV, model selection by CCC
07_stage_a_models.py        EEG → self-report regression, leave-one-participant-out
08_bridge_analysis.py       Apply frozen PMEmo model to ds002721 clips; compare
09_bootstrap_metrics.py     Bootstrap CIs, permutation tests, FDR correction
10_make_figures.py          Final tables and figures
```

Each script reads its settings from `configs/*.yaml` rather than hardcoded
values — edit the configs, not the scripts, when changing features or splits.

## What we will not claim

The final result should be phrased at the level of:

> "An audio-only model trained on PMEmo's crowd-aggregated induced
> valence–arousal labels showed [degree of] agreement with population-average
> induced-affect ratings in a separate EEG music-listening dataset, and these
> ratings had [degree of] association with a predeclared set of EEG spectral
> features."

Not:
- "The model predicts what any individual will feel."
- "Audio features cause EEG changes."
- "A song has one true emotional effect."
- "A model trained on perceived emotion predicts induced emotion."
- "EEG validates an emotion model clinically or universally."

## Status

- [ ] Repo + environment set up
- [ ] Analysis plan agreed (`configs/analysis_plan.yaml`)
- [ ] Gate A audit complete — go/no-go decided
- [ ] Stage A (EEG) models fit
- [ ] Stage B (PMEmo) models fit and frozen
- [ ] Bridge analysis complete
- [ ] Report drafted
