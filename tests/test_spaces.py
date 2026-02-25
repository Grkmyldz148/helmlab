"""Tests for color space wrappers."""

import numpy as np
import pytest

from colorspace.spaces.registry import get_space, all_spaces


SPACE_NAMES = ["oklch", "ipt", "jzazbz", "cam16ucs"]


class TestRegistry:
    def test_get_space(self):
        space = get_space("oklch")
        assert space.name == "OKLCH"

    def test_unknown_space(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_space("nonexistent")

    def test_all_spaces(self):
        spaces = all_spaces()
        names = {s.name for s in spaces}
        assert "OKLCH" in names
        assert "IPT" in names

    def test_aliases(self):
        s1 = get_space("oklch")
        s2 = get_space("oklab")
        assert s1 is s2


@pytest.mark.parametrize("space_name", SPACE_NAMES)
class TestRoundTrip:
    def test_round_trip_accuracy(self, space_name):
        """Test round-trip on realistic sRGB-range colors."""
        space = get_space(space_name)
        # Generate colors in a realistic range (sRGB white point neighborhood)
        rng = np.random.default_rng(42)
        # Typical monitor XYZ: X in [0.01, 0.95], Y in [0.01, 1.0], Z in [0.01, 1.09]
        XYZ = np.column_stack([
            rng.uniform(0.05, 0.90, 500),
            rng.uniform(0.05, 0.90, 500),
            rng.uniform(0.05, 0.90, 500),
        ])
        errors = space.round_trip_error(XYZ)
        # p99 tolerance: most values should round-trip well
        p99 = np.percentile(errors, 99)
        assert p99 < 1e-6, f"{space_name} round-trip p99 error: {p99:.2e}"

    def test_batch_shape(self, space_name):
        space = get_space(space_name)
        XYZ = np.random.rand(100, 3) * 0.8 + 0.05
        coords = space.from_XYZ(XYZ)
        assert coords.shape == (100, 3)
        rt = space.to_XYZ(coords)
        assert rt.shape == (100, 3)


@pytest.mark.parametrize("space_name", SPACE_NAMES)
class TestDistance:
    def test_self_distance_is_zero(self, space_name):
        space = get_space(space_name)
        XYZ = np.array([[0.5, 0.5, 0.5]])
        d = space.distance(XYZ, XYZ)
        assert d[0] == pytest.approx(0.0, abs=1e-10)

    def test_distance_positive(self, space_name):
        space = get_space(space_name)
        XYZ_1 = np.array([[0.3, 0.3, 0.3]])
        XYZ_2 = np.array([[0.6, 0.6, 0.6]])
        d = space.distance(XYZ_1, XYZ_2)
        assert d[0] > 0
