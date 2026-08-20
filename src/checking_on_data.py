import pandas as pd
from pathlib import Path
from collections import defaultdict
from collections import Counter 

NON_STIMULUS_CODES = {0, 264, 266, 277, 786, 788, 800, 801, 802, 803, 804, 805, 806, 807,
                       833, 834, 835, 836, 837, 838, 839, 840, 841,
                       901, 902, 903, 904, 905, 906, 907, 908, 909,
                       257, 259, 260, 263, 32768, 1092,
                       33568, 33569, 33570, 33571, 33572, 33573, 33574, 33575}

ds_root = Path("data_raw/ds002721")
per_subject_clips = defaultdict(set)

for f in sorted(ds_root.glob("sub-*/eeg/*events.tsv")):
    sub = f.name.split("_")[0]
    df = pd.read_csv(f, sep="\t")
    music = df[(~df["trial_type"].isin(NON_STIMULUS_CODES)) & (df["trial_type"] >= 301)]
    per_subject_clips[sub] |= set(music["trial_type"].unique())

for sub, clips in sorted(per_subject_clips.items()):
    print(sub, "->", len(clips), "distinct clips") 


clip_participant_count = Counter()
for sub, clips in per_subject_clips.items():
    for c in clips:
        clip_participant_count[c] += 1

counts = sorted(clip_participant_count.values())
print("Distinct clips used study-wide:", len(clip_participant_count))
print("Clips seen by only 1 participant:", sum(1 for c in counts if c == 1))
print("Clips seen by 5+ participants:", sum(1 for c in counts if c >= 5))
print("Median participants per clip:", counts[len(counts)//2]) 