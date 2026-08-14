# Data access guide

**Nothing in this folder is committed to git except this file and the audit
CSV template** (see `.gitignore`). You obtain all raw data yourself, locally.
Never upload raw audio or raw EEG to GitHub — copyright and dataset terms
both prohibit redistribution.

| Dataset | URL | License / access notes | Approx. size | What you need |
|---|---|---|---|---|
| OpenNeuro ds002721 | https://openneuro.org/datasets/ds002721 | Open (OpenNeuro standard terms); BIDS-formatted EEG, events, self-reports. **Audio stimuli are NOT included** — the dataset paper points to a separate soundtrack source (see Gate A below). | EEG portion: a few GB | `eeg/`, `sub-*_events.tsv`, self-report ratings |
| PMEmo | Code/metadata: https://github.com/HuiZhangDB/PMEmo <br> Data (2018 original): Google Drive, linked from that repo's README <br> Data (2019 updated, adds lyrics/extra physio features): Google Drive, linked from that repo's README | Publicly available for research; cite Zhang et al. 2018 (ICMR) if used. Full songs are NOT included for copyright reasons — only manually-selected chorus excerpts (MP3). | ~1.3 GB | `metadata.csv` (song/chorus timing), static + dynamic valence/arousal annotation CSVs, chorus MP3s, precomputed audio features |
| DEAM (optional, external benchmark only) | https://cvml.unige.ch/databases/DEAM/ | Creative Commons audio; annotations for perceived (not induced) valence/arousal | Audio ~1.3GB, features ~600MB, annotations ~5MB | annotation CSVs, precomputed openSMILE features (audio optional if using precomputed features) |

## Step-by-step

### 1. PMEmo (do this first — no gate, no ambiguity)
1. Clone or browse the code repo for context: `git clone https://github.com/HuiZhangDB/PMEmo.git`
2. Follow the Google Drive link in that repo's README (there are two versions —
   **use the 2019 "updated" version** unless you specifically need the original
   2018 release; the updated version has the same core valence/arousal labels
   plus extras you can ignore).
3. Download into `data_raw/pmemo/`, keeping the original folder structure
   (metadata CSV, annotation CSVs, chorus MP3s, precomputed features CSV).
4. Sanity check: `python ../src/01_audit_data.py --check-pmemo` should report
   794 songs with matching metadata and annotation rows.

### 2. ds002721 (EEG + self-report — straightforward)
1. Install `openneuro-py` (already in `environment.yml`) or DataLad.
2. Download via `openneuro-py download --dataset ds002721 --target data_raw/ds002721/`
   — **do not manually rearrange files inside the BIDS tree.**
3. This gets you EEG, events, and self-report ratings. It does **not** get
   you the 40 music clips themselves.

### 3. ds002721 audio (Gate A — the actual risk in this project)
The ds002721 data paper (Daly et al., 2020) references a separate soundtrack
source for the 40 twelve-second film-music excerpts. Your job by Day 4:
1. Open `ds002721_stimulus_audit.csv` (template below) — one row per trial/
   stimulus ID.
2. For each of the 40 clips, find and record: the source (soundtrack
   album/release), a URL where it can be legally streamed or downloaded,
   license/terms, the exact clip start/end timestamp as used in the
   experiment, a checksum once extracted, and extraction status
   (`recovered` / `not_found` / `ambiguous`).
3. Do **not** download from piracy sites, YouTube rips, or anything without
   clear terms — if a clip can't be legally sourced, mark it `not_found` and
   move on.
4. Tally the `recovered` count and follow the decision rule in the top-level
   README (≥32/40 → direct bridge; <32/40 → downgrade the bridge claim).

### 4. DEAM (optional — only if pursuing the external benchmark)
Download annotations + precomputed features from the link above. Audio itself
is optional — you likely only need it if you want to extract your own
features instead of using the precomputed openSMILE set.

## Folder layout once populated (not committed)

```
data_raw/
├── README_access.md          (committed)
├── ds002721_stimulus_audit.csv  (committed — the audit table itself, no audio)
├── ds002721/                 (gitignored — BIDS tree from openneuro-py)
├── ds002721_audio/            (gitignored — recovered clips, Gate A output)
├── pmemo/                    (gitignored — metadata, annotations, chorus MP3s)
└── deam/                     (gitignored — optional)
```
