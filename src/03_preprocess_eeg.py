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

import mne
import matplotlib.pyplot as plt
import pandas as pd



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