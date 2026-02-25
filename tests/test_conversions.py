"""Tests for color conversion utilities."""

import numpy as np
import pytest

from colorspace.utils.conversions import (
    Lab_to_LCH, LCH_to_Lab,
    xyY_to_XYZ, XYZ_to_xyY,
    Lab_to_XYZ, XYZ_to_Lab,
)


class TestLabLCH:
    def test_round_trip(self, sample_Lab):
        rt = LCH_to_Lab(Lab_to_LCH(sample_Lab))
        np.testing.assert_allclose(rt, sample_Lab, atol=1e-12)

    def test_known_white(self):
        Lab = np.array([[100.0, 0.0, 0.0]])
        LCH = Lab_to_LCH(Lab)
        assert LCH[0, 0] == pytest.approx(100.0)
        assert LCH[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_batch_shape(self):
        Lab = np.random.randn(50, 3)
        LCH = Lab_to_LCH(Lab)
        assert LCH.shape == (50, 3)
        rt = LCH_to_Lab(LCH)
        np.testing.assert_allclose(rt, Lab, atol=1e-12)


class TestXyYXYZ:
    def test_round_trip(self, sample_XYZ):
        # Skip black (0,0,0) which is degenerate for xyY
        valid = sample_XYZ[:-1]
        xyY = XYZ_to_xyY(valid)
        rt = xyY_to_XYZ(xyY)
        np.testing.assert_allclose(rt, valid, atol=1e-12)

    def test_d65_chromaticity(self):
        XYZ = np.array([[0.95047, 1.0, 1.08883]])
        xyY = XYZ_to_xyY(XYZ)
        assert xyY[0, 0] == pytest.approx(0.3127, abs=1e-3)
        assert xyY[0, 1] == pytest.approx(0.3290, abs=1e-3)


class TestLabXYZ:
    def test_round_trip(self, sample_XYZ):
        # Skip black for numerical stability
        valid = sample_XYZ[sample_XYZ[:, 1] > 0.001]
        Lab = XYZ_to_Lab(valid)
        rt = Lab_to_XYZ(Lab)
        np.testing.assert_allclose(rt, valid, atol=1e-10)

    def test_white_is_L100(self):
        from colorspace.config import D65_WHITE
        Lab = XYZ_to_Lab(D65_WHITE.reshape(1, 3))
        assert Lab[0, 0] == pytest.approx(100.0, abs=1e-10)
        assert Lab[0, 1] == pytest.approx(0.0, abs=1e-10)
        assert Lab[0, 2] == pytest.approx(0.0, abs=1e-10)
