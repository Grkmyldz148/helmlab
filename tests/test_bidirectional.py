"""Tests for bidirectional optimization infrastructure (Part D)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from helmlab.helmlab import Helmlab
from helmlab.feedback.generator import ColorPairGenerator
from helmlab.feedback.collector import FeedbackDataset


# ═══════════════════════════════════════════════════════════════════════
# Test Color Generator
# ═══════════════════════════════════════════════════════════════════════

class TestGenerator:
    """Tests for ColorPairGenerator."""

    @pytest.fixture
    def gen(self):
        p = Helmlab()
        return ColorPairGenerator(p)

    def test_critical_zone_pairs_nonempty(self, gen):
        """critical_zone_pairs returns non-empty list."""
        pairs = gen.critical_zone_pairs(n_per_zone=5)
        assert len(pairs) > 0

    def test_critical_zone_pairs_count(self, gen):
        """critical_zone_pairs returns 3 * n_per_zone pairs (3 zones)."""
        pairs = gen.critical_zone_pairs(n_per_zone=10)
        assert len(pairs) == 30  # 3 zones * 10

    def test_critical_zone_valid_hex(self, gen):
        """All critical zone pair colors are valid hex."""
        pairs = gen.critical_zone_pairs(n_per_zone=5)
        for p in pairs:
            assert p["hex1"].startswith("#") and len(p["hex1"]) == 7
            assert p["hex2"].startswith("#") and len(p["hex2"]) == 7

    def test_critical_zone_correct_zones(self, gen):
        """Critical zone pairs have expected zone labels."""
        pairs = gen.critical_zone_pairs(n_per_zone=5)
        zones = {p["zone"] for p in pairs}
        assert "blue_region" in zones
        assert "hk_boundary" in zones
        assert "achromatic_chromatic" in zones

    def test_uniform_random_count(self, gen):
        """uniform_random_pairs returns correct count."""
        pairs = gen.uniform_random_pairs(n=50)
        assert len(pairs) == 50

    def test_uniform_random_in_srgb(self, gen):
        """uniform_random_pairs colors are valid sRGB hex."""
        from helmlab.utils.srgb_convert import hex_to_srgb
        pairs = gen.uniform_random_pairs(n=20)
        for p in pairs:
            srgb = hex_to_srgb(p["hex1"])
            assert np.all(srgb >= 0.0) and np.all(srgb <= 1.0)

    def test_adaptive_pairs_with_residuals(self, gen):
        """adaptive_pairs focuses on high-residual region."""
        residuals = [
            {"hex1": "#ff0000", "hex2": "#00ff00", "residual": 5.0},
            {"hex1": "#0000ff", "hex2": "#ffff00", "residual": 10.0},
            {"hex1": "#808080", "hex2": "#808090", "residual": 0.1},
        ]
        pairs = gen.adaptive_pairs(residuals, n=10)
        assert len(pairs) == 10
        assert all(p["zone"] == "adaptive" for p in pairs)

    def test_export_for_survey(self, gen, tmp_path):
        """export_for_survey creates valid JSON file."""
        pairs = gen.uniform_random_pairs(n=5)
        path = str(tmp_path / "survey.json")
        gen.export_for_survey(pairs, path)
        with open(path) as f:
            data = json.load(f)
        assert data["n_pairs"] == 5
        assert len(data["pairs"]) == 5


# ═══════════════════════════════════════════════════════════════════════
# Feedback Dataset
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackDataset:
    """Tests for FeedbackDataset."""

    def test_add_load_roundtrip(self, tmp_path):
        """add_judgement → load roundtrip."""
        path = str(tmp_path / "fb.json")
        fb = FeedbackDataset(path)
        fb.add_judgement("#ff0000", "#00ff00", 75.0, "obs1")
        fb.add_judgement("#0000ff", "#ffff00", 60.0, "obs1")

        data = fb.load()
        assert data["XYZ_1"].shape == (2, 3)
        assert data["XYZ_2"].shape == (2, 3)
        assert data["DV"].shape == (2,)
        # Check DV values
        np.testing.assert_allclose(data["DV"], [75.0, 60.0])

    def test_merge_with_combvd(self, tmp_path):
        """merge_with_combvd combines datasets correctly."""
        path = str(tmp_path / "fb.json")
        fb = FeedbackDataset(path)
        fb.add_judgement("#ff0000", "#00ff00", 75.0, "obs1")

        combvd = {
            "XYZ_1": np.array([[0.5, 0.5, 0.5]]),
            "XYZ_2": np.array([[0.3, 0.3, 0.3]]),
            "DV": np.array([50.0]),
        }
        merged = fb.merge_with_combvd(combvd)
        assert merged["XYZ_1"].shape == (2, 3)
        assert merged["DV"].shape == (2,)
        assert merged["DV"][0] == 50.0  # COMBVD first
        assert merged["DV"][1] == 75.0  # feedback appended

    def test_combvd_compatible_output(self, tmp_path):
        """load() output has correct COMBVD format."""
        path = str(tmp_path / "fb.json")
        fb = FeedbackDataset(path)
        fb.add_judgement("#808080", "#c0c0c0", 30.0, "obs1")
        data = fb.load()
        assert "XYZ_1" in data
        assert "XYZ_2" in data
        assert "DV" in data
        assert data["XYZ_1"].ndim == 2
        assert data["XYZ_2"].ndim == 2

    def test_stats(self, tmp_path):
        """stats returns correct counts."""
        path = str(tmp_path / "fb.json")
        fb = FeedbackDataset(path)
        fb.add_judgement("#ff0000", "#00ff00", 75.0, "obs1")
        fb.add_judgement("#ff0000", "#00ff00", 80.0, "obs2")
        fb.add_judgement("#0000ff", "#ffff00", 60.0, "obs1")
        s = fb.stats()
        assert s["n_judgements"] == 3
        assert s["n_pairs"] == 2
        assert s["n_observers"] == 2


# ═══════════════════════════════════════════════════════════════════════
# Bidirectional Cost Function
# ═══════════════════════════════════════════════════════════════════════

class TestBidirectionalCost:
    """Tests for extended cost function."""

    def test_feedback_lambda_zero_matches_standard(self):
        """With fb_lambda=0, objective matches standard COMBVD-only."""
        from scripts.optimize_bidirectional import make_objective_with_feedback
        from helmlab.spaces.analytical import AnalyticalSpace

        space = AnalyticalSpace()

        # Tiny fake COMBVD
        combvd = {
            "XYZ_1": np.array([[0.5, 0.5, 0.5], [0.3, 0.3, 0.3]]),
            "XYZ_2": np.array([[0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]),
            "DV": np.array([10.0, 20.0]),
        }
        fb = {
            "XYZ_1": np.array([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]]),
            "XYZ_2": np.array([[0.9, 0.9, 0.9], [0.7, 0.7, 0.7]]),
            "DV": np.array([50.0, 10.0]),
        }

        obj_no_fb = make_objective_with_feedback(combvd, feedback=fb, fb_lambda=0.0)
        obj_with_fb = make_objective_with_feedback(combvd, feedback=fb, fb_lambda=0.1)

        cost_no = obj_no_fb(space)
        cost_with = obj_with_fb(space)

        # With feedback, cost should be different (feedback adds to cost)
        assert cost_no != cost_with
        # No-feedback cost should be purely COMBVD STRESS
        obj_pure = make_objective_with_feedback(combvd, feedback=None)
        assert abs(obj_pure(space) - cost_no) < 1e-10
