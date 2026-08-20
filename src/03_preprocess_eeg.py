"""
Filter, epoch, and artefact-reject ds002721 EEG for all participants.

Settings come from configs/features.yaml (eeg.filtering, eeg.epoching,
eeg.artefact_rejection, eeg.bad_channels) -- confirmed through manual
inspection of sub-01 earlier; see rough.py for that exploration.

Pipeline per run (see configs/features.yaml for exact parameters):
  1. Load raw EEG
  2. Notch filter (removes electrical hum -- confirmed 50Hz)
  3. Band-pass filter (keeps the 1-45Hz range where real EEG rhythms live)
  4. Drop bad channels (T3, T4 -- consistently noisy, confirmed on sub-01)
  5. Average reference (recomputed AFTER dropping bad channels, so their
     noise doesn't get smeared into the average)
  6. Epoch: 1-11s post music-onset (skips the onset transient, stops before
     the rating questions)
  7. Reject epochs on amplitude, but EXCLUDING FP1/FP2 from that check --
     those channels fail on almost every trial due to normal blinks, which
     would gut the dataset if included in the rejection decision.
  8. Concatenate all of one participant's runs into one Epochs object,
     save as data_interim/epochs_ds002721/{participant}-epo.fif
  9. Log retained-trial counts, per participant AND per clip, to
     results/trial_retention.csv 

CAVEAT: bad_channels (T3, T4) was only visually confirmed for sub-01.
Applying it to everyone assumes electrode issues generalize across
participants, which hasn't been individually verified.
"""

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DS002721_DIR = REPO_ROOT / "data_raw" / "ds002721"
TRIALS_PATH = REPO_ROOT / "data_processed" / "trials_ds002721.parquet"
EPOCHS_OUT_DIR = REPO_ROOT / "data_interim" / "epochs_ds002721"
RETENTION_CSV = REPO_ROOT / "results" / "trial_retention.csv"
FEATURES_CFG = yaml.safe_load((REPO_ROOT / "configs" / "features.yaml").read_text())["eeg"]


def process_run(edf_path: Path, run_trials: pd.DataFrame) -> mne.Epochs | None:
    """Turn one run's raw EDF file into clean, trial-locked epochs.

    Returns None if this run has no usable trials (e.g. runs 1/6, which
    are practice/non-music runs -- 0 rows in run_trials for those).
    """
    if len(run_trials) == 0:
        return None 

    # --- load ---
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # --- filter: notch removes the 50Hz electrical hum, bandpass keeps
    #     the 1-45Hz range where real brain rhythms (theta/alpha/beta/gamma)
    #     actually live, and discards slow drift + some high-freq noise ---
    raw.notch_filter(freqs=FEATURES_CFG["filtering"]["notch_hz"], verbose=False)
    l_freq, h_freq = FEATURES_CFG["filtering"]["bandpass_hz"]
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # --- drop bad channels BEFORE referencing, so their noise doesn't
    #     get folded into the average every other channel is measured against ---
    bad_channels = [c for c in FEATURES_CFG["bad_channels"] if c in raw.ch_names]
    if bad_channels:
        raw.drop_channels(bad_channels)

    # --- average reference: recompute every channel's voltage relative to
    #     the mean of all (good) channels, instead of whatever physical
    #     reference electrode the hardware happened to use ---
    raw.set_eeg_reference(ref_channels="average", verbose=False)

    # --- drop any trial with no logged music-onset time (run-boundary
    #     cutoff -- the run ended before this trial's music-start event fired) ---
    run_trials = run_trials.dropna(subset=["music_onset_sec"])
    if len(run_trials) == 0:
        return None

    # --- build the events array MNE needs: [sample_number, 0, event_code]
    #     per trial. Sample number = seconds * sampling rate, since MNE
    #     wants indices into the data array, not raw seconds. ---
    sfreq = raw.info["sfreq"]
    events = np.column_stack([
        (run_trials["music_onset_sec"] * sfreq).astype(int),
        np.zeros(len(run_trials), dtype=int),
        run_trials["ds002721_stimulus_id"].astype(int),
    ])

    # --- epoch: cut the continuous recording into one 10s chunk per trial,
    #     starting 1s after music onset (skips the onset-transient response)
    #     and ending at 11s (stops before the rating questions begin) ---
    tmin, tmax = FEATURES_CFG["epoching"]["primary_window_sec"]
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None,
                         preload=True, verbose=False)

    # --- artefact rejection: reject an epoch if any channel EXCEPT FP1/FP2
    #     swings more than reject_uv (converted from microvolts to volts,
    #     since MNE stores data in volts internally) within the epoch.
    #     FP1/FP2 are excluded from this check because they sit right above
    #     the eyes and fail almost every trial due to ordinary blinking --
    #     their actual values are still kept in every epoch that passes,
    #     just not used to decide whether the epoch survives. ---
    reject_v = FEATURES_CFG["artefact_rejection"]["reject_uv"] * 1e-6
    excluded = FEATURES_CFG["artefact_rejection"]["exclude_from_check"]
    check_idx = [i for i, c in enumerate(epochs.ch_names) if c not in excluded]

    data = epochs.get_data()
    good_mask = np.array([
        (epoch[check_idx].max(axis=1) - epoch[check_idx].min(axis=1)).max() < reject_v
        for epoch in data
    ])

    return epochs[good_mask]


def main() -> None:
    if not TRIALS_PATH.exists():
        print(f"[FAIL] {TRIALS_PATH} not found. Run 02_build_ds002721_trials.py first.")
        sys.exit(1)

    trials = pd.read_parquet(TRIALS_PATH)
    EPOCHS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    RETENTION_CSV.parent.mkdir(parents=True, exist_ok=True)

    per_participant_rows = []
    per_clip_attempted: dict[int, int] = {}
    per_clip_retained: dict[int, int] = {}

    for sub in sorted(trials["participant_id"].unique()):
        sub_epochs_list = []
        n_attempted, n_retained = 0, 0

        for run in sorted(trials[trials["participant_id"] == sub]["run"].unique()):
            run_trials = trials[(trials["participant_id"] == sub) & (trials["run"] == run)]
            edf_path = DS002721_DIR / sub / "eeg" / f"{sub}_{run}_eeg.edf"
            if not edf_path.exists():
                continue

            # tally ATTEMPTED before rejection, so a run that loses everything
            # to artefact rejection still gets counted as attempted, not skipped
            attempted_codes = run_trials.dropna(subset=["music_onset_sec"])["ds002721_stimulus_id"]
            n_attempted += len(attempted_codes)
            for code in attempted_codes:
                per_clip_attempted[int(code)] = per_clip_attempted.get(int(code), 0) + 1

            clean_epochs = process_run(edf_path, run_trials)
            if clean_epochs is None:
                continue
            if len(clean_epochs) == 0:
                print(f"  [WARN] {sub} {run}: 0/{len(attempted_codes)} "
                      f"epochs survived artefact rejection -- skipping this run")
                continue
            sub_epochs_list.append(clean_epochs)

            # tally RETAINED only for epochs that actually survived
            retained_codes = [int(c) for c in clean_epochs.events[:, 2]]
            for code in retained_codes:
                per_clip_retained[code] = per_clip_retained.get(code, 0) + 1
            n_retained += len(clean_epochs)

        if sub_epochs_list:
            # NOTE: concatenating epochs across runs assumes MNE merges each
            # run's event_id dictionary cleanly (different runs use different
            # subsets of the ~307 possible clip codes). This should work, but
            # hasn't been verified beyond a couple of participants -- check
            # the printed epoch count against n_retained below if anything
            # looks off.
            sub_epochs = mne.concatenate_epochs(sub_epochs_list, verbose=False)
            out_path = EPOCHS_OUT_DIR / f"{sub}-epo.fif"
            sub_epochs.save(out_path, overwrite=True, verbose=False)
            print(f"{sub}: kept {n_retained} / {n_attempted} epochs "
                  f"({100 * n_retained / n_attempted:.1f}%) -> {out_path.name}")
        else:
            print(f"{sub}: no usable epochs found "
                  f"(attempted: {n_attempted}, retained: 0)")

        per_participant_rows.append({
            "participant_id": sub,
            "attempted": n_attempted,
            "retained": n_retained,
            "retained_pct": round(100 * n_retained / n_attempted, 1) if n_attempted else None,
        })

    # --- write the two retention summaries ---
    pd.DataFrame(per_participant_rows).to_csv(
        RETENTION_CSV.with_name("trial_retention_per_participant.csv"), index=False
    )

    per_clip_rows = [
        {
            "ds002721_stimulus_id": code,
            "attempted": per_clip_attempted.get(code, 0),
            "retained": per_clip_retained.get(code, 0),
        }
        for code in sorted(per_clip_attempted)
    ]
    pd.DataFrame(per_clip_rows).to_csv(
        RETENTION_CSV.with_name("trial_retention_per_clip.csv"), index=False
    )

    total_attempted = sum(r["attempted"] for r in per_participant_rows)
    total_retained = sum(r["retained"] for r in per_participant_rows)
    print(f"\nTOTAL: kept {total_retained} / {total_attempted} "
          f"({100 * total_retained / total_attempted:.1f}%)")

if __name__ == "__main__":
    main()