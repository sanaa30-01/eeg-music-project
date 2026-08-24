"""
Extract predeclared EEG features (region-mean band power, frontal alpha
asymmetry) from cleaned epochs.

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

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EPOCHS_DIR = REPO_ROOT / "data_interim" / "epochs_ds002721"
OUT_PATH = REPO_ROOT / "data_processed" / "eeg_features.parquet"
FEATURES_CFG = yaml.safe_load((REPO_ROOT / "configs" / "features.yaml").read_text())["eeg"]

# {"theta": [4,7], "alpha": [8,12], "beta": [13,30], "gamma": [30,45]}
BANDS = FEATURES_CFG["bands_hz"]

# per configs/features.yaml -> eeg.derived: region_mean_power only uses
# theta/alpha/beta, not gamma -- gamma is still computed above (in case you
# want it later) but deliberately left out of the region-averaging step
REGION_BANDS = ["theta", "alpha", "beta"]

# The saved epochs use uppercase channel names (FP1, FP2, ...) but
# configs/features.yaml writes them mixed-case (Fp1, Fp2, ...). Upper-casing
# both sides here means the region lookups below actually match instead of
# silently finding zero channels.
CHANNEL_GROUPS = {
    region: [c.upper() for c in names]
    for region, names in FEATURES_CFG["channels"].items()
}


def extract_features_for_participant(epo_path: Path) -> pd.DataFrame:
    """Turn one participant's saved epochs into one feature row per epoch."""

    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    ch_names_upper = [c.upper() for c in epochs.ch_names]

    # Welch PSD for every epoch in this participant's file AT ONCE, not
    # looped one epoch at a time. Returns power with shape
    # (n_epochs, n_channels, n_freqs) -- same n_fft/n_overlap settings as
    # the single-epoch check earlier (2s windows, 50% overlap, per config).
    psd = epochs.compute_psd(method="welch", fmin=1, fmax=45, n_fft=2000, n_overlap=1000, verbose=False)
    power, freqs = psd.get_data(return_freqs=True)

    # total power per (epoch, channel), summed across the WHOLE 1-45Hz
    # range -- this is the denominator for relative power, same idea as
    # the single-epoch calculation, just keeping the epoch dimension now
    # instead of collapsing it.
    total_power = power.sum(axis=2)  # shape: (n_epochs, n_channels)

    # For each band: sum power within that band's frequency range, divide
    # by total power (-> relative power, cancels out per-electrode/per-person
    # differences in raw amplitude that have nothing to do with brain state),
    # then log-transform (-> better-behaved distribution for the linear
    # models used later). Stored per band so both region-averaging and the
    # F3/F4 asymmetry calculation below can reuse the same numbers.
    log_relative = {}
    for band_name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)               # which frequency bins fall in this band
        band_power = power[:, :, mask].sum(axis=2)              # (n_epochs, n_channels)
        relative = band_power / total_power                       # this band's share of total power
        log_relative[band_name] = np.log(relative)                 # (n_epochs, n_channels)

    # Build one output row per epoch (= per clip this participant heard).
    rows = []
    for i in range(len(epochs)):
        row = {
            "participant_id": epo_path.stem.replace("-epo", ""),
            # events[:, 2] is the stimulus code column we built ourselves
            # back in 03_preprocess_eeg.py -- tells us which clip this epoch is
            "ds002721_stimulus_id": int(epochs.events[i, 2]),
        }

        # Region-mean power: average this epoch's per-channel log-relative
        # power over just the channels belonging to each region
        # (frontal/central/parietal), for theta/alpha/beta. This is a
        # dimensionality reduction -- 17 channels' worth of a band collapses
        # into one number per region, since nearby electrodes tend to pick
        # up correlated activity anyway.
        for region, region_channels in CHANNEL_GROUPS.items():
            idx = [j for j, c in enumerate(ch_names_upper) if c in region_channels]
            for band in REGION_BANDS:
                # .mean() over the matched channel indices for this epoch;
                # NaN if somehow none of this region's channels are present
                # (shouldn't happen here since T3/T4 aren't in any region group,
                # but guards against silent miscounting if that ever changes)
                row[f"{region}_{band}"] = log_relative[band][i, idx].mean() if idx else np.nan

        # Frontal alpha asymmetry: F4 minus F3, ALPHA BAND ONLY -- this is
        # a targeted left-vs-right comparison, not a region average, so it's
        # computed separately rather than folded into the frontal region
        # loop above (averaging F3 and F4 together would cancel out exactly
        # the left-right difference this feature exists to capture).
        if "F3" in ch_names_upper and "F4" in ch_names_upper:
            f3_idx = ch_names_upper.index("F3")
            f4_idx = ch_names_upper.index("F4")
            row["frontal_alpha_asymmetry"] = log_relative["alpha"][i, f4_idx] - log_relative["alpha"][i, f3_idx]
        else:
            row["frontal_alpha_asymmetry"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    epo_files = sorted(EPOCHS_DIR.glob("*-epo.fif"))
    if not epo_files:
        print(f"[FAIL] No epoch files found in {EPOCHS_DIR}. Run 03_preprocess_eeg.py first.")
        sys.exit(1)

    # process one participant's file at a time, collect each as its own
    # small DataFrame, then concatenate once at the end -- avoids repeatedly
    # growing one giant DataFrame row-by-row, which gets slow
    all_rows = []
    for epo_path in epo_files:
        df = extract_features_for_participant(epo_path)
        all_rows.append(df)
        print(f"{epo_path.stem}: {len(df)} epochs -> features extracted")

    features_df = pd.concat(all_rows, ignore_index=True)

    # sanity check against configs/features.yaml's own predeclared limit --
    # this is the "no more than 13 predictors per outcome" rule from the
    # project plan, checked automatically rather than trusted by eye
    feature_cols = [c for c in features_df.columns if c not in ("participant_id", "ds002721_stimulus_id")]
    n_predictors = len(feature_cols)
    print(f"\nTotal feature columns: {n_predictors} (config limit: {FEATURES_CFG['max_predictors_per_outcome']})")
    if n_predictors > FEATURES_CFG["max_predictors_per_outcome"]:
        print("[WARN] Exceeds the predeclared max_predictors_per_outcome -- check configs/features.yaml")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(OUT_PATH, index=False)
    print(f"[OK] Wrote {len(features_df)} rows to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()