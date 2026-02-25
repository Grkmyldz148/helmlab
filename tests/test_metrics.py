"""Tests for STRESS and Delta E metrics."""

import numpy as np
import pytest

from colorspace.metrics.stress import stress


class TestSTRESS:
    def test_perfect_prediction(self):
        """STRESS should be 0 for perfect prediction."""
        DV = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        DE = DV.copy()
        s = stress(DV, DE)
        assert s == pytest.approx(0.0, abs=1e-10)

    def test_perfect_with_scale(self):
        """STRESS should be 0 for proportional prediction (scale-invariant)."""
        DV = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        DE = DV * 7.3  # any scale factor
        s = stress(DV, DE)
        assert s == pytest.approx(0.0, abs=1e-10)

    def test_random_is_positive(self):
        rng = np.random.default_rng(42)
        DV = rng.uniform(0.1, 10, size=100)
        DE = rng.uniform(0.1, 10, size=100)
        s = stress(DV, DE)
        assert 0 < s <= 100

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            stress(np.array([1, 2]), np.array([1, 2, 3]))
