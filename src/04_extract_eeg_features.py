"""
Extract the predeclared EEG feature set from preprocessed epochs.

Settings: configs/features.yaml -> eeg.psd, eeg.bands_hz, eeg.derived
(Welch PSD, relative log band power per region, frontal alpha asymmetry).
Max 13 predictors per outcome (configs/features.yaml -> eeg.max_predictors_per_outcome)
— do not silently add more.

Reads: data_interim/epochs_ds002721/
Writes: data_processed/eeg_features.parquet

TODO (Days 11-12):
  - Welch PSD per epoch (2s windows, 50% overlap)
  - Relative log power: theta/alpha/beta (+ optional gamma) x
    frontal/central/parietal region means
  - Frontal alpha asymmetry: log(alpha_F4) - log(alpha_F3), document sign convention
  - One row per (participant, clip) matching trials_ds002721.parquet
"""

import mne
import matplotlib.pyplot as plt
import numpy as np

epochs = mne.read_epochs("data_interim/epochs_ds002721/sub-01-epo.fif", preload=True)
print(epochs)

one_epoch = epochs[0]  # just the first trial, to look at individually
psd = one_epoch.compute_psd(method="welch", fmin=1, fmax=45, n_fft=2000, n_overlap=1000)
psd.plot()
plt.show(block=True) 

# get the raw power values and frequency bins from the PSD you already computed
power, freqs = psd.get_data(return_freqs=True)
power = power[0]  # (1, 17, 89) -> (17, 89): drop the epoch dimension, since there's only one epoch
print(power.shape)  # should be (17 channels, N frequency bins)

bands = {"theta": (4, 7), "alpha": (8, 12), "beta": (13, 30), "gamma": (30, 45)}

total_power = power.sum(axis=1)  # sum across all frequencies, per channel -- the denominator

for band_name, (fmin, fmax) in bands.items():
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    band_power = power[:, band_mask].sum(axis=1)      # sum within just this band, per channel
    relative_power = band_power / total_power           # band's share of this epoch's total power
    log_relative_power = np.log(relative_power)          # log-transform because the data is skewed 

    print(f"{band_name}: mean across channels = {log_relative_power.mean():.3f}")