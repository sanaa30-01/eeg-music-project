"""Focused tests for the PMEmo audio-feature extraction contract.

These tests use a temporary synthetic WAV and never write into data_processed
or results. The separate ``--test`` CLI mode provides the real-PMEmo
integration evaluation.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "05_extract_audio_features.py"
SPEC = importlib.util.spec_from_file_location("extract_audio_features", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise ImportError(f"Unable to load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
# Dataclasses resolve type metadata through sys.modules while the dynamically
# loaded module is executing, so register it exactly as a normal import would.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AudioFeatureExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        _, self.settings = MODULE.load_audio_config()

    def test_summary_statistics_and_names_are_deterministic(self) -> None:
        values = np.array([[0.0, 1.0, 2.0, 3.0], [2.0, 4.0, 6.0, 8.0]])
        summary = MODULE.summarize_feature_matrix("example", values)

        self.assertEqual(len(summary), 8)
        self.assertEqual(list(summary)[:4], [
            "example_01_mean",
            "example_01_std",
            "example_01_p10",
            "example_01_p90",
        ])
        self.assertAlmostEqual(summary["example_01_mean"], 1.5)
        self.assertAlmostEqual(summary["example_02_p90"], 7.4)

    def test_synthetic_audio_produces_complete_finite_schema(self) -> None:
        sample_rate = self.settings.sample_rate_hz
        duration = 6.0
        time_axis = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate

        # A two-tone signal plus periodic impulses exercises tonal, spectral,
        # energy, onset, tempo, and zero-crossing descriptors.
        waveform = 0.20 * np.sin(2 * np.pi * 220.0 * time_axis)
        waveform += 0.10 * np.sin(2 * np.pi * 440.0 * time_axis)
        for onset_sec in np.arange(0.5, duration, 0.5):
            start = int(onset_sec * sample_rate)
            waveform[start : start + 100] += np.hanning(100) * 0.5
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "synthetic.wav"
            sf.write(audio_path, waveform, sample_rate)
            features, qc = MODULE.extract_audio_features(audio_path, self.settings)

        self.assertEqual(len(features), MODULE.EXPECTED_FEATURE_COUNT)
        self.assertTrue(np.isfinite(np.fromiter(features.values(), dtype=float)).all())
        self.assertEqual(qc["sample_rate_hz"], sample_rate)
        self.assertAlmostEqual(qc["duration_sec"], duration, places=2)
        self.assertIn("mfcc_01_mean", features)
        self.assertIn("spectral_flux_p90", features)
        self.assertIn("chroma_12_std", features)
        self.assertIn("tonnetz_06_mean", features)
        self.assertIn("tempo_bpm", features)


if __name__ == "__main__":
    unittest.main()
