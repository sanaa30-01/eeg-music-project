import pandas as pd
from pathlib import Path

files = sorted(Path("data_raw/ds002721/sub-02/eeg").glob("*events.tsv"))
total = 0
for f in files:
    df = pd.read_csv(f, sep="\t")
    music_trials = df[(df["trial_type"] >= 301) & (df["trial_type"] <= 360)]
    print(f.name, "->", len(music_trials), "music trials")
    total += len(music_trials)
print("TOTAL:", total)

