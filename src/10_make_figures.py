"""
Generate the final figure/table set for the report.

Reads: results/*.csv, results/*.png (partner's Stage B outputs),
       data_processed/*.parquet, configs/*.yaml
Writes: reports/figures/*.png, reports/tables/*.csv

Each figure/table function is independent and wrapped in try/except in
main() -- one missing input file (e.g. raw EEG not present on a given
machine) shouldn't prevent every other figure from generating.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
TABLES_DIR = REPO_ROOT / "reports" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. PIPELINE DIAGRAM
# ============================================================
def fig1_pipeline_diagram():
    """Box-and-arrow schematic of the ACTUAL pipeline run, including the
    detours the original plan didn't anticipate (Gate A's real clip count,
    the trim step). Built with plain matplotlib patches/arrows -- no extra
    dependencies needed.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color="#dbe9f6"):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)
        return (x + w / 2, y, x + w / 2, y + h)  # (bottom-center, top-center) for arrows

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", lw=1.2))

    # -- Stage B branch (top) --
    b1 = box(0.3, 5.3, 2.2, 1.2, "PMEmo\n794 tracks,\n767 labelled")
    b2 = box(3.0, 5.3, 2.2, 1.2, "05_extract_audio\n293 features/track")
    b3 = box(5.7, 5.3, 2.2, 1.2, "06_train_pmemo\nRidge/ENet, frozen")
    arrow(b1[2], b1[3] - 0.6, b2[0], b2[3] - 0.6)
    arrow(b2[2], b2[3] - 0.6, b3[0], b3[3] - 0.6)

    # -- Gate A branch (middle) --
    g1 = box(0.3, 3.4, 2.2, 1.2, "ds002721 events\n60->307 unique\nstimulus codes found")
    g2 = box(3.0, 3.4, 2.2, 1.2, "Eerola/Vuoskoski\naudio recovered\n307/307")
    g3 = box(5.7, 3.4, 2.2, 1.2, "Trimmed to first 12s\n299 usable\n(8 too short)")
    arrow(g1[2], g1[3] - 0.6, g2[0], g2[3] - 0.6)
    arrow(g2[2], g2[3] - 0.6, g3[0], g3[3] - 0.6)

    # -- Stage A branch (bottom) --
    a1 = box(0.3, 1.5, 2.2, 1.2, "ds002721 EEG\n31 participants")
    a2 = box(3.0, 1.5, 2.2, 1.2, "03/04: filter, epoch,\nreject, band power")
    a3 = box(5.7, 1.5, 2.2, 1.2, "07: Ridge vs\nparticipant baseline")
    arrow(a1[2], a1[3] - 0.6, a2[0], a2[3] - 0.6)
    arrow(a2[2], a2[3] - 0.6, a3[0], a3[3] - 0.6)

    # -- Bridge, where Stage B + Gate A meet --
    bridge = box(8.7, 3.4, 3.5, 2.2,
                 "08_bridge_analysis\nFrozen PMEmo model applied\nto 299 trimmed clips\nvs. real self-reports\n(76 clips >=5 raters = primary)",
                 color="#f6dbc4")
    arrow(b3[2], b3[3] - 0.6, bridge[0] - 0.3, 4.5 + 1.1)
    arrow(g3[2], g3[3] - 0.6, bridge[0] - 0.3, 4.5)

    # Stage A's result is reported alongside but does NOT feed into the
    # bridge computation itself -- shown as a dashed line, not a solid arrow
    ax.annotate("", xy=(bridge[0] - 0.3, 4.0), xytext=(a3[2], a3[3] - 0.6),
                 arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="dashed", color="gray"))
    ax.text(7.3, 0.9, "(H3 secondary EEG-linkage test not run --\nH1 found no usable EEG-arousal relationship to build it on)",
            fontsize=7, style="italic", color="gray")

    ax.set_title("Actual pipeline as built (not the original 40-clip assumption)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_pipeline_diagram.png", dpi=150)
    plt.close(fig)
    print("[OK] fig1_pipeline_diagram")


# ============================================================
# 2. DATASET/LABEL COMPATIBILITY TABLE
# ============================================================
def table2_dataset_compatibility():
    """Static reference table -- condensed from the original project doc,
    not derived from any results file. Kept as a table function so it lives
    alongside the other report deliverables rather than only in prose.
    """
    rows = [
        {"dataset": "OpenNeuro ds002721", "construct": "Induced/felt emotion",
         "role": "Stage A; audio recovered via Gate A for bridge",
         "actual_scale": "31 participants, 307-clip pool (~40/person, randomly drawn)"},
        {"dataset": "PMEmo", "construct": "Induced/felt emotion",
         "role": "Stage B training + internal validation",
         "actual_scale": "794 tracks, 767 with static valence/arousal labels"},
        {"dataset": "DEAM", "construct": "Perceived emotion",
         "role": "Not used -- deferred, per team scope decision",
         "actual_scale": "N/A"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "02_dataset_compatibility.csv", index=False)
    print("[OK] table2_dataset_compatibility")


# ============================================================
# 3. EEG PREPROCESSING/QC FIGURE
# ============================================================
def fig3_eeg_qc(participant="sub-01", run="task-run3"):
    """Three-panel figure: raw vs filtered trace (same participant/run used
    for illustration throughout this project), retained-trial counts across
    all participants, and one example PSD. Reruns filtering fresh here
    rather than depending on any saved intermediate, since raw EEG isn't
    something previous scripts kept around long-term.
    """
    edf_path = REPO_ROOT / "data_raw" / "ds002721" / participant / "eeg" / f"{participant}_{run}_eeg.edf"
    if not edf_path.exists():
        print(f"[SKIP] fig3_eeg_qc -- {edf_path} not found on this machine")
        return

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw_filtered = raw.copy()
    raw_filtered.notch_filter(freqs=50, verbose=False)
    raw_filtered.filter(l_freq=1, h_freq=45, verbose=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1: raw vs filtered, 10 seconds, one representative channel (Fz)
    sfreq = raw.info["sfreq"]
    window = slice(0, int(10 * sfreq))
    times = np.arange(window.stop) / sfreq
    ch_idx = raw.ch_names.index("Fz")
    axes[0].plot(times, raw.get_data()[ch_idx, window] * 1e6, label="Raw", alpha=0.7, linewidth=0.8)
    axes[0].plot(times, raw_filtered.get_data()[ch_idx, window] * 1e6, label="Filtered", linewidth=0.8)
    axes[0].set_title(f"Raw vs filtered (Fz, {participant})")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].legend(fontsize=8)

    # Panel 2: retained-trial percentage per participant, from the CSV
    # 03_preprocess_eeg.py already produced -- this panel doesn't reprocess
    # anyone's EEG, just visualizes the existing retention table
    retention_path = REPO_ROOT / "results" / "trial_retention_per_participant.csv"
    if retention_path.exists():
        retention = pd.read_csv(retention_path)
        axes[1].bar(range(len(retention)), retention["retained_pct"].fillna(0))
        axes[1].axhline(retention["retained_pct"].mean(), color="red", linestyle="--",
                         linewidth=1, label=f"mean={retention['retained_pct'].mean():.1f}%")
        axes[1].set_title("Retained trials per participant (%)")
        axes[1].set_xlabel("Participant (sorted by ID)")
        axes[1].set_ylabel("% epochs retained")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "trial_retention_per_participant.csv\nnot found", ha="center")

    # Panel 3: PSD of the filtered signal, same 10s window, same channel --
    # illustrates the 1/f pattern discussed throughout Stage A
    psd_data = raw_filtered.compute_psd(fmax=45, verbose=False)
    power, freqs = psd_data.get_data(return_freqs=True)
    axes[2].plot(freqs, 10 * np.log10(power[ch_idx]))
    axes[2].set_title(f"PSD, filtered ({participant}, Fz)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Power (dB)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_eeg_qc.png", dpi=150)
    plt.close(fig)
    print("[OK] fig3_eeg_qc")


# ============================================================
# 4. EEG FEATURE-VS-EMOTION SCATTERPLOTS
# ============================================================
def fig4_feature_emotion_scatter():
    """Two scatterplots per the original spec: frontal alpha asymmetry vs
    valence-like, and frontal beta power vs arousal-like -- with a simple
    linear fit line and Pearson r annotated on each panel.
    """
    features_path = REPO_ROOT / "data_processed" / "eeg_features.parquet"
    trials_path = REPO_ROOT / "data_processed" / "trials_ds002721.parquet"
    if not (features_path.exists() and trials_path.exists()):
        print("[SKIP] fig4_feature_emotion_scatter -- inputs not found")
        return

    eeg = pd.read_parquet(features_path)
    trials = pd.read_parquet(trials_path)
    data = eeg.merge(trials, on=["participant_id", "ds002721_stimulus_id"], how="inner")
    data["valence_like_composite"] = data[["pleasant", "happy", "tender"]].mean(axis=1)
    data["arousal_like_composite"] = data[["energetic", "tense", "angry", "fearful"]].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    pairs = [
        ("frontal_alpha_asymmetry", "valence_like_composite", "Frontal alpha asymmetry", "Valence-like composite"),
        ("frontal_beta", "arousal_like_composite", "Frontal beta power (log-relative)", "Arousal-like composite"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        valid = data[[xcol, ycol]].dropna()
        ax.scatter(valid[xcol], valid[ycol], alpha=0.3, s=15)
        if len(valid) > 2:
            slope, intercept = np.polyfit(valid[xcol], valid[ycol], 1)
            x_line = np.linspace(valid[xcol].min(), valid[xcol].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1.5)
            r, _ = pearsonr(valid[xcol], valid[ycol])
            ax.set_title(f"r = {r:.3f}  (n={len(valid)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_feature_emotion_scatter.png", dpi=150)
    plt.close(fig)
    print("[OK] fig4_feature_emotion_scatter")


# ============================================================
# 5. STAGE A MODEL PERFORMANCE TABLE (with bootstrap CIs merged in)
# ============================================================
def table5_stage_a_performance():
    """Merges stage_a_model_comparison.csv (point estimates) with
    stage_a_metrics_with_ci.csv (bootstrap CIs) into one reportable table.
    These two files only share outcome+evaluation-type as a natural key,
    so the merge is done by hand-matching model name substrings rather than
    a clean join column -- worth double-checking the output row-by-row.
    """
    comparison_path = REPO_ROOT / "results" / "stage_a_model_comparison.csv"
    ci_path = REPO_ROOT / "results" / "stage_a_metrics_with_ci.csv"
    if not (comparison_path.exists() and ci_path.exists()):
        print("[SKIP] table5_stage_a_performance -- inputs not found")
        return

    comparison = pd.read_csv(comparison_path)
    ci = pd.read_csv(ci_path)

    # attach CI columns only to the two rows the CI file actually covers
    # (PRIMARY/LOPO and leave-one-clip-out) -- baseline, untuned, and
    # exploratory-grid rows don't have a matching CI computation
    comparison["ci_low"] = None
    comparison["ci_high"] = None
    for _, ci_row in ci.iterrows():
        eval_type = ci_row["evaluation"]  # "LOPO" or "LOCO"
        outcome = ci_row["outcome"]
        if eval_type == "LOPO":
            mask = (comparison["outcome"] == outcome) & comparison["model"].str.contains("PRIMARY")
        else:  # LOCO
            mask = (comparison["outcome"] == outcome) & comparison["model"].str.contains("leave-one-clip-out")
        comparison.loc[mask, "ci_low"] = ci_row["ci_low"]
        comparison.loc[mask, "ci_high"] = ci_row["ci_high"]

    comparison.to_csv(TABLES_DIR / "05_stage_a_performance.csv", index=False)
    print("[OK] table5_stage_a_performance")


# ============================================================
# 6. MAIN MUSIC-MODEL (STAGE B) COMPARISON TABLE
# ============================================================
def table6_stage_b_comparison():
    """This table already exists in full -- your partner's
    results/pmemo_model_comparison.csv. This function copies it into the
    report tables folder rather than regenerating it, since regenerating
    would mean re-running his training script for no benefit.
    """
    source = REPO_ROOT / "results" / "pmemo_model_comparison.csv"
    if not source.exists():
        print("[SKIP] table6_stage_b_comparison -- run 05/06_extract/train first")
        return
    df = pd.read_csv(source)
    df.to_csv(TABLES_DIR / "06_stage_b_comparison.csv", index=False)
    print("[OK] table6_stage_b_comparison (copied from Stage B's own output)")


# ============================================================
# 7. PREDICTED-VS-OBSERVED BRIDGE PLOTS
# ============================================================
def fig7_bridge_scatter():
    """Predicted (frozen PMEmo model) vs observed (real ds002721 self-report)
    per clip, split into primary (>=5 raters) and exploratory subsets, with
    regression line, r, and clip count annotated -- matching the CI
    threshold used in 08/09.
    """
    bridge_path = REPO_ROOT / "results" / "bridge_predictions.csv"
    ci_path = REPO_ROOT / "results" / "bridge_metrics_with_ci.csv"
    if not bridge_path.exists():
        print("[SKIP] fig7_bridge_scatter -- run 08_bridge_analysis.py first")
        return

    bridge = pd.read_csv(bridge_path)
    ci = pd.read_csv(ci_path) if ci_path.exists() else None
    RATER_THRESHOLD = 5  # must match 08/09 -- confirm this is still the agreed value

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    subsets = [("primary (>=5 raters)", bridge[bridge["n_participants"] >= RATER_THRESHOLD]),
               ("exploratory (<5 raters)", bridge[bridge["n_participants"] < RATER_THRESHOLD])]

    for row_idx, (subset_name, subset) in enumerate(subsets):
        for col_idx, outcome in enumerate(["valence", "arousal"]):
            ax = axes[row_idx, col_idx]
            actual = subset[f"actual_{outcome}"]
            predicted = subset[f"predicted_{outcome}"]
            ax.scatter(predicted, actual, alpha=0.4, s=20)

            if len(subset) > 2:
                slope, intercept = np.polyfit(predicted, actual, 1)
                x_line = np.linspace(predicted.min(), predicted.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=1.5)
                r, _ = pearsonr(actual, predicted)

                ci_text = ""
                if ci is not None:
                    match = ci[(ci["subset"] == subset_name) & (ci["outcome"] == outcome)]
                    if len(match):
                        ci_text = f", 95% CI=[{match['ci_low'].values[0]:.2f}, {match['ci_high'].values[0]:.2f}]"

                ax.set_title(f"{subset_name}, {outcome}\nr={r:.3f}{ci_text}, n={len(subset)}", fontsize=9)

            ax.set_xlabel(f"Predicted {outcome} (frozen PMEmo model)")
            ax.set_ylabel(f"Actual {outcome} (ds002721 self-report)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_bridge_scatter.png", dpi=150)
    plt.close(fig)
    print("[OK] fig7_bridge_scatter")


# ============================================================
# 8. ERROR-ANALYSIS FIGURE
# ============================================================
def fig8_error_analysis():
    """Absolute error by low/mid/high outcome tercile, for Stage A's
    PRIMARY (LOPO) predictions -- trial-level granularity here (925 rows)
    supports meaningful terciles, unlike the bridge's 76-clip primary set.
    """
    stage_a_path = REPO_ROOT / "data_processed" / "stage_a_predictions.parquet"
    if not stage_a_path.exists():
        print("[SKIP] fig8_error_analysis -- run 07_stage_a_models.py with the .to_parquet() line added")
        return

    data = pd.read_parquet(stage_a_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, outcome in zip(axes, ["valence", "arousal"]):
        actual_col = f"{outcome}_like_composite"
        pred_col = f"{outcome}_primary_pred"
        subset = data[[actual_col, pred_col]].dropna()

        # tercile bins by the ACTUAL rating, not the prediction -- this
        # asks "how well does the model do for genuinely low/mid/high
        # rated clips", which is the more meaningful framing
        subset["tercile"] = pd.qcut(subset[actual_col], 3, labels=["Low", "Mid", "High"])
        subset["abs_error"] = (subset[actual_col] - subset[pred_col]).abs()

        means = subset.groupby("tercile", observed=True)["abs_error"].mean()
        stds = subset.groupby("tercile", observed=True)["abs_error"].std()
        ax.bar(means.index.astype(str), means.values, yerr=stds.values, capsize=4)
        ax.set_title(f"{outcome}: MAE by actual-rating tercile")
        ax.set_ylabel("Mean absolute error")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_error_analysis.png", dpi=150)
    plt.close(fig)
    print("[OK] fig8_error_analysis")


# ============================================================
# 9. LIMITATIONS / ETHICAL-USE TABLE
# ============================================================
def table9_limitations():
    """Rewritten from the actual, specific things discovered during this
    project -- not the generic template from the original plan. Every row
    here traces back to something concretely found and documented along
    the way, not a hypothetical risk.
    """
    rows = [
        {"category": "Sample representativeness",
         "note": "ds002721: 31 participants, likely WEIRD-population sample per the source dataset's own demographics. No cultural generalization claimed."},
        {"category": "Participant exclusions",
         "note": "sub-29 excluded entirely (0/40 epochs survived artefact rejection). sub-02 retained despite severe loss (3/36 epochs, 91.7% loss) -- flagged, not treated as equally reliable as other participants."},
        {"category": "Stimulus coverage",
         "note": "Clips are a random ~40-per-participant draw from a 307-clip pool, not a shared fixed set. Median 2 raters/clip; 117/307 clips rated by exactly 1 person. Bridge analysis split into primary (>=5 raters, 76 clips) and exploratory (<5 raters, 223 clips) to avoid treating single-rater 'averages' as population estimates."},
        {"category": "Audio provenance",
         "note": "ds002721 does not include source audio. Recovered via the Eerola & Vuoskoski (2011) Soundtracks corpus, matched by event-code arithmetic (code-300=track number), cross-validated by exact filename match. 307/307 codes matched to existing files."},
        {"category": "Clip duration assumption",
         "note": "Source audio files vary in length (10-37s); ds002721 plays fixed 12s excerpts, but no end-of-music event code exists to recover the exact original trim point. All clips trimmed to their first 12 seconds as a documented, uniform assumption, not a verified match to the original excerpt. 8/307 clips were shorter than 12s and excluded rather than force-included at a mismatched length."},
        {"category": "EEG feature scope",
         "note": "13-channel montage (2 dropped for consistent noise, confirmed on 1 participant and generalized to all 31 -- not individually re-verified per participant), 3 broad bands, linear models only. FP1/FP2 excluded from artefact-rejection amplitude checks specifically (not from the features themselves) to avoid discarding most trials to ordinary blinking."},
        {"category": "H1 (EEG -> self-report)",
         "note": "Null under leave-one-participant-out for both composites (properly scaled, nested-CV tuned, preregistered alpha grid). Two CI-confirmed non-null findings: LOPO valence shows a small but real NEGATIVE correlation (r=-0.127, 95% CI excludes zero); LOCO arousal shows a small but real POSITIVE correlation (r=0.222, CI excludes zero). The frontal_alpha_asymmetry coefficient's sign was checked directly across folds and found directionally consistent with theory (positive), ruling out a sign-convention bug as the source of the negative valence result -- the full explanation across all 10 features was not pursued further."},
        {"category": "H3 (bridge)",
         "note": "Primary test null and CI-confirmed robust: all four correlations (primary/exploratory x valence/arousal) have bootstrap CIs comfortably containing zero, and permutation p-values are all non-significant (p>0.19). Secondary EEG-linkage test not conducted, since H1 found no reliable EEG-arousal relationship under LOPO to build it on."},
        {"category": "Secondary outcomes",
         "note": "The 8 individual self-report ratings (beyond the 2 composites) were not modeled; Benjamini-Hochberg FDR correction across them, as specified in the analysis plan, has not been performed."},
        {"category": "Non-clinical, non-causal",
         "note": "No individual-level, diagnostic, or causal claims. Associational only. Audio-EEG-self-report correlations do not establish mechanism."},
        {"category": "Data/licensing",
         "note": "No raw audio or raw EEG redistributed. PMEmo and Eerola/Vuoskoski corpora used under their respective research-use terms."},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "09_limitations_ethical_use.csv", index=False)
    print("[OK] table9_limitations")


def main():
    steps = [
        fig1_pipeline_diagram,
        table2_dataset_compatibility,
        fig3_eeg_qc,
        fig4_feature_emotion_scatter,
        table5_stage_a_performance,
        table6_stage_b_comparison,
        fig7_bridge_scatter,
        fig8_error_analysis,
        table9_limitations,
    ]
    for step in steps:
        try:
            step()
        except Exception as exc:
            print(f"[FAIL] {step.__name__}: {exc}")

    print(f"\nDone. Check {FIGURES_DIR.relative_to(REPO_ROOT)} and {TABLES_DIR.relative_to(REPO_ROOT)}.")


if __name__ == "__main__":
    main()