import pandas as pd
import mne 
import matplotlib.pyplot as plt

df = pd.read_parquet("data_processed/trials_ds002721.parquet")
question_cols = ["pleasant", "energetic", "tense", "angry", "fearful", "happy", "sad", "tender"]

df["n_missing"] = df[question_cols].isna().sum(axis=1)
df["trial_num_in_run"] = df.groupby(["participant_id", "run"]).cumcount() + 1
df["trials_in_run"] = df.groupby(["participant_id", "run"])["run"].transform("count")
df["is_last_in_run"] = df["trial_num_in_run"] == df["trials_in_run"]

missing = df[df["n_missing"] > 0]

print("Missing trials that ARE the last trial in their run:", missing["is_last_in_run"].sum())
print("Missing trials that are NOT the last trial in their run:", (~missing["is_last_in_run"]).sum())

for col in question_cols:
    print(col, "->", missing[col].isna().sum(), "missing")


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