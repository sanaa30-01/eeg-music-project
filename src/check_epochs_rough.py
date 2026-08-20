import mne
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

"""
Filter, epoch, and artefact-reject ds002721 EEG.

Settings come from configs/features.yaml (eeg.filtering, eeg.epoching,
eeg.artefact_rejection) — do not hardcode values here.

Steps (see configs/features.yaml for exact parameters):
  1. Load raw BIDS EEG via mne_bids
  2. Inspect power spectrum before deciding on notch filter (50 vs 60 Hz)
  3. Band-pass filter (default 1-45 Hz)
  4. Set average reference (after confirming montage / bad channels)
  5. Epoch: primary window 1-11s post-onset (excludes onset transient and
     rating interval)
  6. Reject epochs beyond the configured amplitude threshold
  7. Log retained trial counts PER PARTICIPANT and PER CLIP — this is a
     required Week 2 deliverable (trial-retention figure)

Writes: data_interim/epochs_ds002721/  (per-subject -epo.fif files)
        results/trial_retention.csv

TODO (Days 8-10): implement per configs/features.yaml
"""

raw = mne.io.read_raw_edf(
    "data_raw/ds002721/sub-01/eeg/sub-01_task-run3_eeg.edf",
    preload=True
)

print(raw.info)          # metadata: channel names, sampling rate, etc.
print(raw.ch_names)      # the actual list of 19 (or however many) electrode names
raw.plot(duration=20, n_channels=19, block=True)   # opens an interactive scrolling viewer

psd = raw.compute_psd(fmax=80) # plot the power spectral density of the raw data to check where the main interference artifacts are
psd.plot(show=False)
plt.show(block=True)

#filtering using notch filter to remove the 50 Hz interference artifact
#filtering using bandpass filter to remove the low and high frequency noise
raw_filtered = raw.copy()
raw_filtered.notch_filter(freqs=50)
raw_filtered.filter(l_freq=1, h_freq=45)

raw_filtered.plot(duration=20, n_channels=19, start=250, block=True)

raw_filtered.info['bads'] = ['T3', 'T4']  #identified bad channels 

#setting the average reference because right now the reference is whatever the physical reference electrode the recording systme picked, which is not neutral
#average referencing is used instead to make sure the signal is referenced to something more neutral 
raw_filtered.drop_channels(['T3', 'T4']) 
eeg_reference = raw_filtered.set_eeg_reference(ref_channels='average')
eeg_reference.plot(duration=20, n_channels=19, block=True)
plt.show(block=True)


#building one run's epochs
trials = pd.read_parquet("data_processed/trials_ds002721.parquet")
run_trials = trials[(trials["participant_id"] == "sub-01") & (trials["run"] == "task-run3")]

sfreq = raw_filtered.info["sfreq"]  # should be 1000.0

#dropping trials with no music_onset_sec because they are likely run-boundary cutoff
n_before = len(run_trials)
run_trials = run_trials.dropna(subset=["music_onset_sec"])
n_dropped = n_before - len(run_trials)
if n_dropped:
    print(f"[WARN] Dropped {n_dropped} trial(s) with no music_onset_sec (likely run-boundary cutoff)")

#mne needs a list of events to build the epochs --> events array is required 
events = np.column_stack([
    (run_trials["music_onset_sec"] * sfreq).astype(int),   # multiplied by sampling frequency to get the sample number because mne needs indices not seconds
    np.zeros(len(run_trials), dtype=int),                   # unused, MNE wants a placeholder
    run_trials["ds002721_stimulus_id"].astype(int),          # event code because we need to know which music clip each epoch is
])

#building the epochs
#time window is 1-11 seconds post-onset and pre-questionnaire
epochs = mne.Epochs(raw_filtered, events, tmin=1.0, tmax=11.0, baseline=None, preload=True)

#rejecting epochs based on amplitude threshold
data = epochs.get_data()
ch_names = epochs.ch_names
#checking the indices of the channels that are not FP1 and FP2 because we want to exclude the frontal channels from the amplitude threshold calculation
#because the frontal channels are more likely to be contaminated by movement artifacts (eye blinks, etc.)
check_idx = [i for i, c in enumerate(ch_names) if c not in ("FP1", "FP2")]

#calculating the amplitude threshold for each epoch and checking if it is less than 100 microvolts
#if it is, then the epoch is kept, otherwise it is rejected
#the amplitude threshold is calculated by subtracting the minimum value from the maximum value of the epoch for the non-frontal channels
good_mask = np.array([
    (epoch[check_idx].max(axis=1) - epoch[check_idx].min(axis=1)).max() < 100e-6
    for epoch in data
])

print(f"Kept {good_mask.sum()} / {len(good_mask)} epochs ({100 * good_mask.mean():.1f}%)")
#keeping the epochs that are good
epochs_clean = epochs[good_mask] 

print(epochs_clean)
print(epochs_clean.ch_names) 
epochs_clean.plot(n_epochs=5, n_channels=17, block=True)  # 17, since T3/T4 are dropped 

trials = pd.read_parquet("data_processed/trials_ds002721.parquet")
ds_root = Path("data_raw/ds002721")
peak_amplitudes = []  # in microvolts

for sub in sorted(trials["participant_id"].unique())[:5]:
    for run in sorted(trials[trials["participant_id"] == sub]["run"].unique()):
        run_trials = trials[(trials["participant_id"] == sub) & (trials["run"] == run)].dropna(subset=["music_onset_sec"])
        if len(run_trials) == 0:
            continue
        edf_path = ds_root / sub / "eeg" / f"{sub}_{run}_eeg.edf"
        if not edf_path.exists():
            continue

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.drop_channels(["T3", "T4"])
        raw.notch_filter(freqs=50, verbose=False)
        raw.filter(l_freq=1, h_freq=45, verbose=False)
        raw.set_eeg_reference(ref_channels="average", verbose=False)

        sfreq = raw.info["sfreq"]
        events = np.column_stack([
            (run_trials["music_onset_sec"] * sfreq).astype(int),
            np.zeros(len(run_trials), dtype=int),
            run_trials["ds002721_stimulus_id"].astype(int),
        ])
        # no reject= here -- keep everything, so we can measure actual amplitudes
        epochs = mne.Epochs(raw, events, tmin=1.0, tmax=11.0, baseline=None, preload=True, verbose=False)
        data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times), in volts
        for epoch in data:
            peak_to_peak = (epoch.max(axis=1) - epoch.min(axis=1)) * 1e6  # convert to microvolts, per channel
            peak_amplitudes.append(peak_to_peak.max())  # worst channel in this epoch

peak_amplitudes = np.array(peak_amplitudes)
print(f"Median peak-to-peak amplitude: {np.median(peak_amplitudes):.1f} µV")
print(f"75th percentile: {np.percentile(peak_amplitudes, 75):.1f} µV")
print(f"90th percentile: {np.percentile(peak_amplitudes, 90):.1f} µV")
print(f"Max: {peak_amplitudes.max():.1f} µV")
print(f"% of epochs that would pass at 150µV: {100 * (peak_amplitudes < 150).mean():.1f}%")
print(f"% of epochs that would pass at 200µV: {100 * (peak_amplitudes < 200).mean():.1f}%")

# Same setup as before, but track which channel(s) exceed threshold, and separately
# compute what the rejection rate WOULD be if FP1/FP2 were excluded from consideration
fp_only_offender_count = 0
other_offender_count = 0
would_pass_without_fp = 0
total = 0

for sub in sorted(trials["participant_id"].unique())[:5]:
    for run in sorted(trials[trials["participant_id"] == sub]["run"].unique()):
        run_trials = trials[(trials["participant_id"] == sub) & (trials["run"] == run)].dropna(subset=["music_onset_sec"])
        if len(run_trials) == 0:
            continue
        edf_path = ds_root / sub / "eeg" / f"{sub}_{run}_eeg.edf"
        if not edf_path.exists():
            continue

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.drop_channels(["T3", "T4"])
        raw.notch_filter(freqs=50, verbose=False)
        raw.filter(l_freq=1, h_freq=45, verbose=False)
        raw.set_eeg_reference(ref_channels="average", verbose=False)

        sfreq = raw.info["sfreq"]
        events = np.column_stack([
            (run_trials["music_onset_sec"] * sfreq).astype(int),
            np.zeros(len(run_trials), dtype=int),
            run_trials["ds002721_stimulus_id"].astype(int),
        ])
        epochs = mne.Epochs(raw, events, tmin=1.0, tmax=11.0, baseline=None, preload=True, verbose=False)
        data = epochs.get_data()
        ch_names = epochs.ch_names
        fp_idx = [ch_names.index(c) for c in ["FP1", "FP2"]]
        other_idx = [i for i in range(len(ch_names)) if i not in fp_idx]

        for epoch in data:
            p2p = (epoch.max(axis=1) - epoch.min(axis=1)) * 1e6
            total += 1
            if p2p[other_idx].max() < 100:
                would_pass_without_fp += 1

print(f"Total epochs: {total}")
print(f"% that would PASS at 100µV if FP1/FP2 were excluded from the check: {100 * would_pass_without_fp / total:.1f}%")