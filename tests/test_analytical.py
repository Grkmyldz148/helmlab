"""Tests for the analytical parametric color space (v2, enriched pipeline)."""

import json
import tempfile

import numpy as np
import pytest

from colorspace.spaces.analytical import AnalyticalParams, AnalyticalSpace, oklab_params
from colorspace.spaces.registry import get_space


class TestAnalyticalParams:
    def test_oklab_defaults(self):
        p = oklab_params()
        assert p.M1.shape == (3, 3)
        assert p.gamma.shape == (3,)
        assert p.M2.shape == (3, 3)
        assert p.hk_weight == 0.0
        # Enrichment defaults
        assert p.L_corr_p1 == 0.0
        assert p.L_corr_p2 == 0.0
        assert p.cs_cos1 == 0.0
        assert p.lc1 == 0.0

    def test_serialization_roundtrip(self):
        p = oklab_params()
        p.L_corr_p1 = 0.1
        p.cs_cos1 = -0.2
        p.lc1 = 0.3
        p.hk_sin1 = 0.05
        d = p.to_dict()
        p2 = AnalyticalParams.from_dict(d)
        np.testing.assert_array_almost_equal(p.M1, p2.M1)
        np.testing.assert_array_almost_equal(p.gamma, p2.gamma)
        np.testing.assert_array_almost_equal(p.M2, p2.M2)
        assert p.hk_weight == p2.hk_weight
        assert p.L_corr_p1 == p2.L_corr_p1
        assert p.cs_cos1 == p2.cs_cos1
        assert p.lc1 == p2.lc1
        assert p.hk_sin1 == p2.hk_sin1

    def test_save_load(self, tmp_path):
        p = oklab_params()
        p.cs_sin1 = 0.15
        p.lc2 = -0.3
        path = tmp_path / "params.json"
        p.save(path)
        p2 = AnalyticalParams.load(path)
        np.testing.assert_array_almost_equal(p.M1, p2.M1)
        assert p.cs_sin1 == p2.cs_sin1
        assert p.lc2 == p2.lc2

    def test_json_format(self, tmp_path):
        p = oklab_params()
        path = tmp_path / "params.json"
        p.save(path)
        with open(path) as f:
            d = json.load(f)
        assert "M1" in d
        assert "gamma" in d
        assert "M2" in d
        assert "L_corr_p1" in d
        assert "cs_cos1" in d
        assert "lc1" in d
        assert "hk_sin1" in d
        assert len(d["M1"]) == 3
        assert len(d["gamma"]) == 3

    def test_backward_compat_v1_json(self):
        """Loading a v1 JSON (without enrichment fields) should work."""
        v1_dict = {
            "M1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "gamma": [0.333, 0.333, 0.333],
            "M2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "hk_weight": 0.5,
            "hk_power": 0.8,
            "hk_hue_mod": 0.1,
        }
        p = AnalyticalParams.from_dict(v1_dict)
        assert p.hk_weight == 0.5
        assert p.L_corr_p1 == 0.0  # default
        assert p.cs_cos1 == 0.0
        assert p.lc1 == 0.0
        assert p.hk_sin1 == 0.0


class TestAnalyticalSpace:
    @pytest.fixture
    def space(self):
        return AnalyticalSpace(oklab_params())

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_from_XYZ_shape(self, space, xyz_batch):
        coords = space.from_XYZ(xyz_batch)
        assert coords.shape == (500, 3)

    def test_to_XYZ_shape(self, space):
        coords = np.random.rand(100, 3) * 0.5
        XYZ = space.to_XYZ(coords)
        assert XYZ.shape == (100, 3)

    def test_round_trip_no_hk(self, space, xyz_batch):
        """Without H-K (hk_weight=0), round-trip should be near-perfect."""
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"Max round-trip error: {errors.max():.2e}"

    def test_round_trip_with_hk(self, xyz_batch):
        """With H-K correction, round-trip should still be exact."""
        params = oklab_params()
        params.hk_weight = 0.1
        params.hk_power = 0.5
        params.hk_hue_mod = 0.05
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"Max round-trip error with H-K: {errors.max():.2e}"

    def test_single_color(self, space):
        """Test with a single color (no batch dim issues)."""
        XYZ = np.array([0.5, 0.5, 0.5])
        coords = space.from_XYZ(XYZ)
        assert coords.shape == (3,)
        rt = space.to_XYZ(coords)
        np.testing.assert_allclose(XYZ, rt, atol=1e-10)

    def test_2d_batch(self, space):
        """Test with 2D batch input."""
        XYZ = np.random.rand(10, 3) * 0.8 + 0.05
        coords = space.from_XYZ(XYZ)
        assert coords.shape == (10, 3)
        rt = space.to_XYZ(coords)
        np.testing.assert_allclose(XYZ, rt, atol=1e-10)


class TestEnrichmentRoundTrip:
    """Test round-trip accuracy with all enrichment stages active."""

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_cubic_L_only(self, xyz_batch):
        params = oklab_params()
        params.L_corr_p1 = 0.2
        params.L_corr_p2 = -0.15
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-8, f"Cubic L round-trip error: {errors.max():.2e}"

    def test_chroma_scale_only(self, xyz_batch):
        params = oklab_params()
        params.cs_cos1 = 0.3
        params.cs_sin1 = -0.2
        params.cs_cos2 = 0.1
        params.cs_sin2 = 0.05
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"Chroma scale round-trip error: {errors.max():.2e}"

    def test_L_chroma_scale_only(self, xyz_batch):
        params = oklab_params()
        params.lc1 = 0.5
        params.lc2 = -0.3
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"L-chroma scale round-trip error: {errors.max():.2e}"

    def test_enhanced_hk_only(self, xyz_batch):
        params = oklab_params()
        params.hk_weight = 0.2
        params.hk_power = 0.6
        params.hk_hue_mod = 0.1
        params.hk_sin1 = 0.05
        params.hk_cos2 = -0.08
        params.hk_sin2 = 0.03
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"Enhanced H-K round-trip error: {errors.max():.2e}"

    def test_hue_correction_only(self, xyz_batch):
        params = oklab_params()
        params.hue_cos1 = 0.15
        params.hue_sin1 = -0.1
        params.hue_cos2 = 0.08
        params.hue_sin2 = 0.05
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-8, f"Hue correction round-trip error: {errors.max():.2e}"

    def test_quartic_L_only(self, xyz_batch):
        """Quartic L correction (p3) round-trip."""
        params = oklab_params()
        params.L_corr_p1 = 0.2
        params.L_corr_p2 = -0.15
        params.L_corr_p3 = 0.1
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-8, f"Quartic L round-trip error: {errors.max():.2e}"

    def test_hlc_only(self, xyz_batch):
        """Hue×Lightness chroma interaction round-trip."""
        params = oklab_params()
        params.hlc_cos1 = 0.3
        params.hlc_sin1 = -0.2
        params.hlc_cos2 = 0.1
        params.hlc_sin2 = 0.05
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"HLC round-trip error: {errors.max():.2e}"

    def test_hue_lightness_only(self, xyz_batch):
        """Hue-dependent lightness scaling round-trip."""
        params = oklab_params()
        params.hl_cos1 = 0.1
        params.hl_sin1 = -0.05
        params.hl_cos2 = 0.08
        params.hl_sin2 = 0.03
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"Hue-lightness round-trip error: {errors.max():.2e}"

    def test_chroma_power_only(self, xyz_batch):
        """Nonlinear chroma power round-trip (v6)."""
        params = oklab_params()
        params.cp_cos1 = 0.1
        params.cp_sin1 = -0.05
        params.cp_cos2 = 0.08
        params.cp_sin2 = 0.03
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-10, f"Chroma power round-trip error: {errors.max():.2e}"

    def test_dark_L_only(self, xyz_batch):
        """Adaptive dark L compression round-trip (v6)."""
        params = oklab_params()
        params.lp_dark = 0.3
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-8, f"Dark L round-trip error: {errors.max():.2e}"

    def test_all_enrichments_active(self, xyz_batch):
        """Full enriched pipeline with all v6 stages active."""
        params = oklab_params()
        params.L_corr_p1 = 0.15
        params.L_corr_p2 = -0.1
        params.L_corr_p3 = 0.05
        params.cs_cos1 = 0.2
        params.cs_sin1 = -0.15
        params.cs_cos2 = 0.08
        params.cs_sin2 = 0.05
        params.cs_cos3 = -0.03
        params.cs_sin3 = 0.02
        params.lc1 = 0.3
        params.lc2 = -0.2
        params.hk_weight = 0.15
        params.hk_power = 0.5
        params.hk_hue_mod = 0.1
        params.hk_sin1 = 0.05
        params.hk_cos2 = -0.06
        params.hk_sin2 = 0.03
        params.hue_cos1 = 0.1
        params.hue_sin1 = -0.05
        params.hue_cos2 = 0.04
        params.hue_sin2 = 0.02
        params.hue_cos3 = 0.02
        params.hue_sin3 = -0.01
        params.hlc_cos1 = 0.15
        params.hlc_sin1 = -0.1
        params.hlc_cos2 = 0.05
        params.hlc_sin2 = 0.03
        params.hl_cos1 = 0.08
        params.hl_sin1 = -0.04
        params.hl_cos2 = 0.03
        params.hl_sin2 = 0.02
        params.cp_cos1 = 0.08
        params.cp_sin1 = -0.04
        params.cp_cos2 = 0.03
        params.cp_sin2 = 0.02
        params.lp_dark = 0.15
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-7, f"Full v6 enriched round-trip error: {errors.max():.2e}"


class TestDistance:
    @pytest.fixture
    def space(self):
        return AnalyticalSpace(oklab_params())

    def test_self_distance_zero(self, space):
        XYZ = np.array([[0.5, 0.5, 0.5]])
        d = space.distance(XYZ, XYZ)
        assert d[0] == pytest.approx(0.0, abs=1e-10)

    def test_distance_positive(self, space):
        XYZ_1 = np.array([[0.3, 0.3, 0.3]])
        XYZ_2 = np.array([[0.6, 0.6, 0.6]])
        d = space.distance(XYZ_1, XYZ_2)
        assert d[0] > 0

    def test_distance_symmetric(self, space):
        rng = np.random.default_rng(123)
        XYZ_1 = rng.uniform(0.1, 0.8, (50, 3))
        XYZ_2 = rng.uniform(0.1, 0.8, (50, 3))
        d12 = space.distance(XYZ_1, XYZ_2)
        d21 = space.distance(XYZ_2, XYZ_1)
        np.testing.assert_allclose(d12, d21, atol=1e-12)


class TestRegistry:
    def test_analytical_in_registry(self):
        space = get_space("analytical")
        assert space.name == "Analytical"


class TestOklabEquivalence:
    """With Oklab params and hk_weight=0, should match Oklab closely."""

    def test_forward_matches_oklab(self):
        from colorspace.spaces.oklch import OKLCH

        analytical = AnalyticalSpace(oklab_params())
        oklab = OKLCH()

        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.05, 0.90, (200, 3))

        coords_a = analytical.from_XYZ(XYZ)
        coords_o = oklab.from_XYZ(XYZ)

        # Should be very close (not exact due to floating point)
        np.testing.assert_allclose(coords_a, coords_o, atol=1e-6)

    def test_stress_matches_oklab(self):
        """STRESS with Oklab params should be ~same as OKLCH."""
        from colorspace.data.combvd import load_combvd
        from colorspace.metrics.stress import stress

        analytical = AnalyticalSpace(oklab_params())
        oklab = get_space("oklch")

        combvd = load_combvd()
        XYZ_1 = combvd["XYZ_1"]
        XYZ_2 = combvd["XYZ_2"]
        DV = combvd["DV"]

        stress_a = stress(DV, analytical.distance(XYZ_1, XYZ_2))
        stress_o = stress(DV, oklab.distance(XYZ_1, XYZ_2))

        # Should be within 1 point of each other
        assert abs(stress_a - stress_o) < 1.0, (
            f"Analytical STRESS={stress_a:.2f}, OKLCH STRESS={stress_o:.2f}"
        )


class TestHKEffect:
    """Test H-K correction behavior."""

    def test_hk_increases_lightness_for_chromatic(self):
        """Chromatic colors should have higher L with positive hk_weight."""
        params = oklab_params()
        params.hk_weight = 0.1
        params.hk_power = 0.5
        space_hk = AnalyticalSpace(params)
        space_no_hk = AnalyticalSpace(oklab_params())

        # A chromatic color (saturated red-ish)
        XYZ = np.array([[0.4, 0.2, 0.05]])
        coords_hk = space_hk.from_XYZ(XYZ)
        coords_no = space_no_hk.from_XYZ(XYZ)

        # L should be higher with H-K
        assert coords_hk[0, 0] > coords_no[0, 0]
        # a, b channels should be unchanged
        np.testing.assert_allclose(coords_hk[0, 1:], coords_no[0, 1:], atol=1e-12)

    def test_hk_no_effect_on_achromatic(self):
        """For achromatic colors (a=b~0), H-K should have negligible effect."""
        params = oklab_params()
        params.hk_weight = 0.1
        params.hk_power = 0.5
        space_hk = AnalyticalSpace(params)
        space_no_hk = AnalyticalSpace(oklab_params())

        # D65 white point (achromatic)
        XYZ = np.array([[0.95047, 1.0, 1.08883]])
        coords_hk = space_hk.from_XYZ(XYZ)
        coords_no = space_no_hk.from_XYZ(XYZ)

        # Difference should be tiny (chroma near zero)
        diff = abs(coords_hk[0, 0] - coords_no[0, 0])
        assert diff < 0.01, f"H-K effect on achromatic: {diff:.6f}"

    def test_enhanced_hk_harmonics(self):
        """Multiple H-K harmonics should create hue-dependent L variation."""
        params = oklab_params()
        params.hk_weight = 0.1
        params.hk_power = 0.5
        params.hk_sin1 = 0.2
        params.hk_cos2 = 0.1
        space = AnalyticalSpace(params)

        # Test two chromatic colors at different hues
        XYZ_red = np.array([[0.4, 0.2, 0.05]])
        XYZ_blue = np.array([[0.15, 0.1, 0.5]])
        L_red = space.from_XYZ(XYZ_red)[0, 0]
        L_blue = space.from_XYZ(XYZ_blue)[0, 0]
        # They should have different H-K boosts due to harmonics
        assert L_red != pytest.approx(L_blue, abs=0.01)


class TestEmbeddedHK:
    """Test embedded H-K (Phase 2: chroma-dependent lightness at step 3.7)."""

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_embedded_hk_round_trip(self, xyz_batch):
        """Round-trip with H-K at step 3.7 + all enrichment stages."""
        params = oklab_params()
        # H-K params
        params.hk_weight = 0.2
        params.hk_power = 0.6
        params.hk_hue_mod = 0.1
        params.hk_sin1 = 0.05
        params.hk_cos2 = -0.08
        params.hk_sin2 = 0.03
        # Cubic L (operates on H-K-adjusted L)
        params.L_corr_p1 = 0.15
        params.L_corr_p2 = -0.1
        # Chroma scaling
        params.cs_cos1 = 0.2
        params.cs_sin1 = -0.15
        # L-dep chroma (uses H-K-adjusted L)
        params.lc1 = 0.3
        params.lc2 = -0.2
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-8, f"Embedded H-K round-trip error: {errors.max():.2e}"

    def test_embedded_hk_uses_raw_chroma(self):
        """H-K at step 3.7 should use raw chroma (before chroma scaling stages).

        With a large chroma scaling factor, the H-K boost should NOT change
        because H-K now operates before chroma scaling.
        """
        # Space with H-K only (no chroma scaling)
        p1 = oklab_params()
        p1.hk_weight = 0.1
        p1.hk_power = 0.5
        space1 = AnalyticalSpace(p1)

        # Space with H-K + large chroma scaling
        p2 = oklab_params()
        p2.hk_weight = 0.1
        p2.hk_power = 0.5
        p2.cs_cos1 = 0.5  # large chroma scaling
        space2 = AnalyticalSpace(p2)

        XYZ = np.array([[0.4, 0.2, 0.05]])
        coords1 = space1.from_XYZ(XYZ)
        coords2 = space2.from_XYZ(XYZ)

        # L should be the same because H-K uses raw chroma (before chroma scaling)
        # The cubic L correction is inactive, so L flows through unchanged
        # after H-K. Only chroma (a, b) differ due to chroma scaling.
        np.testing.assert_allclose(coords1[0, 0], coords2[0, 0], atol=1e-12,
                                   err_msg="H-K should use raw chroma, not post-scaling")

    def test_embedded_hk_affects_downstream_L(self):
        """H-K at step 3.7 should affect downstream L-dependent stages.

        L-dependent chroma scaling (step 6) uses L that includes H-K boost.
        """
        # Space with L-dep chroma only (no H-K)
        p_base = oklab_params()
        p_base.lc1 = 0.5
        space_base = AnalyticalSpace(p_base)

        # Space with L-dep chroma + H-K (H-K changes L, which changes chroma)
        p_hk = oklab_params()
        p_hk.lc1 = 0.5
        p_hk.hk_weight = 0.2
        p_hk.hk_power = 0.5
        space_hk = AnalyticalSpace(p_hk)

        XYZ = np.array([[0.4, 0.2, 0.05]])
        coords_base = space_base.from_XYZ(XYZ)
        coords_hk = space_hk.from_XYZ(XYZ)

        # a, b should differ because H-K changes L which changes L-dep chroma scale
        assert not np.allclose(coords_base[0, 1:], coords_hk[0, 1:], atol=1e-6), \
            "Embedded H-K should affect downstream chroma through L dependency"


class TestHueDependentDarkL:
    """Test hue-dependent dark L compression (v13)."""

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_dark_L_hue_round_trip(self, xyz_batch):
        """Hue-dependent dark L compression round-trip."""
        params = oklab_params()
        params.lp_dark = 0.3
        params.lp_dark_hcos = 0.15
        params.lp_dark_hsin = -0.1
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-8, f"Hue-dep dark L round-trip error: {errors.max():.2e}"

    def test_dark_L_hue_with_all_enrichments(self, xyz_batch):
        """Hue-dep dark L + all other enrichments round-trip."""
        params = oklab_params()
        params.L_corr_p1 = 0.15
        params.L_corr_p2 = -0.1
        params.L_corr_p3 = 0.05
        params.cs_cos1 = 0.2
        params.cs_sin1 = -0.15
        params.cs_cos2 = 0.08
        params.cs_sin2 = 0.05
        params.cs_cos3 = -0.03
        params.cs_sin3 = 0.02
        params.lc1 = 0.3
        params.lc2 = -0.2
        params.hk_weight = 0.15
        params.hk_power = 0.5
        params.hk_hue_mod = 0.1
        params.hk_sin1 = 0.05
        params.hk_cos2 = -0.06
        params.hk_sin2 = 0.03
        params.hue_cos1 = 0.1
        params.hue_sin1 = -0.05
        params.hue_cos2 = 0.04
        params.hue_sin2 = 0.02
        params.hue_cos3 = 0.02
        params.hue_sin3 = -0.01
        params.hlc_cos1 = 0.15
        params.hlc_sin1 = -0.1
        params.hlc_cos2 = 0.05
        params.hlc_sin2 = 0.03
        params.hl_cos1 = 0.08
        params.hl_sin1 = -0.04
        params.hl_cos2 = 0.03
        params.hl_sin2 = 0.02
        params.cp_cos1 = 0.08
        params.cp_sin1 = -0.04
        params.cp_cos2 = 0.03
        params.cp_sin2 = 0.02
        params.lp_dark = 0.15
        params.lp_dark_hcos = 0.1
        params.lp_dark_hsin = -0.08
        space = AnalyticalSpace(params)
        errors = space.round_trip_error(xyz_batch)
        assert errors.max() < 1e-7, f"Full v13 round-trip error: {errors.max():.2e}"

    def test_backward_compat_zero_hue_params(self, xyz_batch):
        """With lp_dark_hcos=0 and lp_dark_hsin=0, v13 should match v12."""
        params_v12 = oklab_params()
        params_v12.lp_dark = 0.3
        space_v12 = AnalyticalSpace(params_v12)

        params_v13 = oklab_params()
        params_v13.lp_dark = 0.3
        params_v13.lp_dark_hcos = 0.0
        params_v13.lp_dark_hsin = 0.0
        space_v13 = AnalyticalSpace(params_v13)

        coords_v12 = space_v12.from_XYZ(xyz_batch)
        coords_v13 = space_v13.from_XYZ(xyz_batch)
        np.testing.assert_allclose(coords_v12, coords_v13, atol=1e-12)

    def test_hue_dep_dark_L_changes_output(self):
        """Hue-dependent params should produce different L for different hues."""
        params = oklab_params()
        params.lp_dark = 0.3
        params.lp_dark_hcos = 0.2
        space = AnalyticalSpace(params)

        # Two dark colors at different hues
        XYZ_red = np.array([[0.08, 0.04, 0.01]])   # dark reddish
        XYZ_blue = np.array([[0.02, 0.02, 0.10]])   # dark bluish
        L_red = space.from_XYZ(XYZ_red)[0, 0]
        L_blue = space.from_XYZ(XYZ_blue)[0, 0]

        # Without hue dep, compare
        params_nohue = oklab_params()
        params_nohue.lp_dark = 0.3
        space_nohue = AnalyticalSpace(params_nohue)
        L_red_nh = space_nohue.from_XYZ(XYZ_red)[0, 0]
        L_blue_nh = space_nohue.from_XYZ(XYZ_blue)[0, 0]

        # The hue-dep version should shift red and blue differently
        diff_red = L_red - L_red_nh
        diff_blue = L_blue - L_blue_nh
        assert abs(diff_red - diff_blue) > 1e-4, \
            f"Hue-dep dark L should affect red/blue differently: diff_red={diff_red:.6f}, diff_blue={diff_blue:.6f}"

    def test_serialization_new_params(self):
        """New v13 params should survive serialization round-trip."""
        params = oklab_params()
        params.lp_dark = 0.3
        params.lp_dark_hcos = 0.15
        params.lp_dark_hsin = -0.1
        d = params.to_dict()
        assert "lp_dark_hcos" in d
        assert "lp_dark_hsin" in d
        p2 = AnalyticalParams.from_dict(d)
        assert p2.lp_dark_hcos == 0.15
        assert p2.lp_dark_hsin == -0.1

    def test_backward_compat_v12_json(self):
        """Loading a v12 JSON (without lp_dark_hcos/hsin) should work."""
        v12_dict = {
            "M1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "gamma": [0.333, 0.333, 0.333],
            "M2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "hk_weight": 0.5,
            "hk_power": 0.8,
            "hk_hue_mod": 0.1,
            "lp_dark": 0.3,
        }
        p = AnalyticalParams.from_dict(v12_dict)
        assert p.lp_dark == 0.3
        assert p.lp_dark_hcos == 0.0
        assert p.lp_dark_hsin == 0.0


class TestDistLinear:
    """Test dist_linear parameter (v14: S-curve bias reduction)."""

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_backward_compat_zero(self):
        """dist_linear=0 should produce identical distances to v12 (pure compression)."""
        params_v12 = oklab_params()
        params_v12.dist_power = 0.9
        params_v12.dist_wC = 1.1
        params_v12.dist_compress = 1.5
        space_v12 = AnalyticalSpace(params_v12)

        params_v14 = oklab_params()
        params_v14.dist_power = 0.9
        params_v14.dist_wC = 1.1
        params_v14.dist_compress = 1.5
        params_v14.dist_linear = 0.0
        space_v14 = AnalyticalSpace(params_v14)

        rng = np.random.default_rng(99)
        XYZ_1 = rng.uniform(0.05, 0.90, (100, 3))
        XYZ_2 = rng.uniform(0.05, 0.90, (100, 3))

        d_v12 = space_v12.distance(XYZ_1, XYZ_2)
        d_v14 = space_v14.distance(XYZ_1, XYZ_2)
        np.testing.assert_allclose(d_v12, d_v14, atol=1e-15)

    def test_linear_one_is_identity(self):
        """dist_linear=1 should cancel compression: DE*(1+c*DE)/(1+c*DE) = DE."""
        params_compressed = oklab_params()
        params_compressed.dist_power = 0.9
        params_compressed.dist_wC = 1.1
        params_compressed.dist_compress = 1.5
        params_compressed.dist_linear = 1.0
        space_compressed = AnalyticalSpace(params_compressed)

        params_raw = oklab_params()
        params_raw.dist_power = 0.9
        params_raw.dist_wC = 1.1
        # No compression
        space_raw = AnalyticalSpace(params_raw)

        rng = np.random.default_rng(77)
        XYZ_1 = rng.uniform(0.05, 0.90, (100, 3))
        XYZ_2 = rng.uniform(0.05, 0.90, (100, 3))

        d_linear = space_compressed.distance(XYZ_1, XYZ_2)
        d_raw = space_raw.distance(XYZ_1, XYZ_2)
        np.testing.assert_allclose(d_linear, d_raw, rtol=1e-12)

    def test_monotonicity(self):
        """Distance should be monotonic for all α ∈ {0, 0.2, 0.5, 0.8, 1.0}."""
        for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
            params = oklab_params()
            params.dist_compress = 1.5
            params.dist_linear = alpha
            space = AnalyticalSpace(params)

            # Create pairs with increasing raw distance
            base = np.array([[0.5, 0.5, 0.5]])
            targets = np.array([[0.5 + i * 0.05, 0.5, 0.5] for i in range(1, 10)])
            bases = np.tile(base, (9, 1))

            d = space.distance(bases, targets)
            # Each successive distance should be larger
            for i in range(len(d) - 1):
                assert d[i + 1] > d[i], (
                    f"Monotonicity violated at α={alpha}: d[{i}]={d[i]:.6f} >= d[{i+1}]={d[i+1]:.6f}"
                )

    def test_serialization_dist_linear(self):
        """dist_linear should survive dict round-trip."""
        params = oklab_params()
        params.dist_compress = 1.5
        params.dist_linear = 0.3
        d = params.to_dict()
        assert "dist_linear" in d
        assert d["dist_linear"] == 0.3
        p2 = AnalyticalParams.from_dict(d)
        assert p2.dist_linear == 0.3
        assert p2.dist_compress == 1.5

    def test_backward_compat_v13_json(self):
        """Loading a v13 JSON (without dist_linear) should default to 0.0."""
        v13_dict = {
            "M1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "gamma": [0.333, 0.333, 0.333],
            "M2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "hk_weight": 0.5,
            "hk_power": 0.8,
            "hk_hue_mod": 0.1,
            "lp_dark": 0.3,
            "lp_dark_hcos": 0.1,
            "lp_dark_hsin": -0.05,
            "dist_compress": 1.5,
        }
        p = AnalyticalParams.from_dict(v13_dict)
        assert p.dist_linear == 0.0
        assert p.dist_compress == 1.5
        assert p.lp_dark_hcos == 0.1

    def test_dist_linear_no_rt_impact(self, xyz_batch):
        """dist_linear only affects distance, not forward/inverse transforms."""
        params_a = oklab_params()
        params_a.dist_compress = 1.5
        params_a.dist_linear = 0.0
        space_a = AnalyticalSpace(params_a)

        params_b = oklab_params()
        params_b.dist_compress = 1.5
        params_b.dist_linear = 0.5
        space_b = AnalyticalSpace(params_b)

        coords_a = space_a.from_XYZ(xyz_batch)
        coords_b = space_b.from_XYZ(xyz_batch)
        np.testing.assert_allclose(coords_a, coords_b, atol=1e-15)

        rt_a = space_a.round_trip_error(xyz_batch)
        rt_b = space_b.round_trip_error(xyz_batch)
        np.testing.assert_allclose(rt_a, rt_b, atol=1e-15)


class TestDistPostPower:
    """Test dist_post_power parameter (v14b: post-compress power for S-curve fix)."""

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_backward_compat_one(self):
        """dist_post_power=1.0 should match v12 exactly."""
        params_v12 = oklab_params()
        params_v12.dist_power = 0.9
        params_v12.dist_wC = 1.1
        params_v12.dist_compress = 1.5
        space_v12 = AnalyticalSpace(params_v12)

        params_pp = oklab_params()
        params_pp.dist_power = 0.9
        params_pp.dist_wC = 1.1
        params_pp.dist_compress = 1.5
        params_pp.dist_post_power = 1.0
        space_pp = AnalyticalSpace(params_pp)

        rng = np.random.default_rng(99)
        XYZ_1 = rng.uniform(0.05, 0.90, (100, 3))
        XYZ_2 = rng.uniform(0.05, 0.90, (100, 3))

        d_v12 = space_v12.distance(XYZ_1, XYZ_2)
        d_pp = space_pp.distance(XYZ_1, XYZ_2)
        np.testing.assert_allclose(d_v12, d_pp, atol=1e-15)

    def test_post_power_increases_large_de(self):
        """q>1 should expand large DE more than small DE."""
        params_base = oklab_params()
        params_base.dist_compress = 1.5
        space_base = AnalyticalSpace(params_base)

        params_pp = oklab_params()
        params_pp.dist_compress = 1.5
        params_pp.dist_post_power = 1.2
        space_pp = AnalyticalSpace(params_pp)

        # Small distance pair
        XYZ_a = np.array([[0.5, 0.5, 0.5]])
        XYZ_b = np.array([[0.51, 0.5, 0.5]])
        d_small_base = space_base.distance(XYZ_a, XYZ_b)[0]
        d_small_pp = space_pp.distance(XYZ_a, XYZ_b)[0]

        # Large distance pair
        XYZ_c = np.array([[0.5, 0.5, 0.5]])
        XYZ_d = np.array([[0.9, 0.2, 0.1]])
        d_large_base = space_base.distance(XYZ_c, XYZ_d)[0]
        d_large_pp = space_pp.distance(XYZ_c, XYZ_d)[0]

        # Ratio should increase for large, decrease for small (after STRESS scaling)
        ratio_small = d_small_pp / d_small_base
        ratio_large = d_large_pp / d_large_base
        assert ratio_large > ratio_small, (
            f"Post-power should expand large DE more: small_ratio={ratio_small:.4f}, large_ratio={ratio_large:.4f}"
        )

    def test_monotonicity(self):
        """Distance should remain monotonic for q > 0."""
        for q in [0.8, 1.0, 1.1, 1.2, 1.3]:
            params = oklab_params()
            params.dist_compress = 1.5
            params.dist_post_power = q
            space = AnalyticalSpace(params)

            base = np.array([[0.5, 0.5, 0.5]])
            targets = np.array([[0.5 + i * 0.05, 0.5, 0.5] for i in range(1, 10)])
            bases = np.tile(base, (9, 1))

            d = space.distance(bases, targets)
            for i in range(len(d) - 1):
                assert d[i + 1] > d[i], (
                    f"Monotonicity violated at q={q}: d[{i}]={d[i]:.6f} >= d[{i+1}]={d[i+1]:.6f}"
                )

    def test_serialization(self):
        """dist_post_power should survive dict round-trip."""
        params = oklab_params()
        params.dist_post_power = 1.15
        d = params.to_dict()
        assert "dist_post_power" in d
        p2 = AnalyticalParams.from_dict(d)
        assert p2.dist_post_power == 1.15

    def test_backward_compat_old_json(self):
        """Loading old JSON without dist_post_power should default to 1.0."""
        old_dict = {
            "M1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "gamma": [0.333, 0.333, 0.333],
            "M2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dist_compress": 1.5,
        }
        p = AnalyticalParams.from_dict(old_dict)
        assert p.dist_post_power == 1.0

    def test_no_rt_impact(self, xyz_batch):
        """dist_post_power only affects distance, not transforms."""
        params_a = oklab_params()
        space_a = AnalyticalSpace(params_a)

        params_b = oklab_params()
        params_b.dist_post_power = 1.2
        space_b = AnalyticalSpace(params_b)

        coords_a = space_a.from_XYZ(xyz_batch)
        coords_b = space_b.from_XYZ(xyz_batch)
        np.testing.assert_allclose(coords_a, coords_b, atol=1e-15)


class TestDistSLSC:
    """Test pair-dependent distance weights (v14c: SL/SC)."""

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_backward_compat_zero(self):
        """dist_sl=0 and dist_sc=0 should match v12 exactly."""
        params_v12 = oklab_params()
        params_v12.dist_power = 0.9
        params_v12.dist_wC = 1.1
        params_v12.dist_compress = 1.5
        space_v12 = AnalyticalSpace(params_v12)

        params_new = oklab_params()
        params_new.dist_power = 0.9
        params_new.dist_wC = 1.1
        params_new.dist_compress = 1.5
        params_new.dist_sl = 0.0
        params_new.dist_sc = 0.0
        space_new = AnalyticalSpace(params_new)

        rng = np.random.default_rng(99)
        XYZ_1 = rng.uniform(0.05, 0.90, (100, 3))
        XYZ_2 = rng.uniform(0.05, 0.90, (100, 3))

        d_v12 = space_v12.distance(XYZ_1, XYZ_2)
        d_new = space_new.distance(XYZ_1, XYZ_2)
        np.testing.assert_allclose(d_v12, d_new, atol=1e-15)

    def test_sl_changes_dark_vs_mid(self):
        """Positive dist_sl should change distance ratio between dark and mid-L pairs."""
        params = oklab_params()
        params.dist_sl = 2.0  # strong effect
        space = AnalyticalSpace(params)
        space_base = AnalyticalSpace(oklab_params())

        # Dark pair (low L_avg)
        XYZ_dark1 = np.array([[0.05, 0.03, 0.02]])
        XYZ_dark2 = np.array([[0.08, 0.05, 0.03]])
        # Mid pair (L_avg near 0.5)
        XYZ_mid1 = np.array([[0.40, 0.40, 0.40]])
        XYZ_mid2 = np.array([[0.50, 0.50, 0.50]])

        d_dark = space.distance(XYZ_dark1, XYZ_dark2)[0]
        d_mid = space.distance(XYZ_mid1, XYZ_mid2)[0]
        d_dark_base = space_base.distance(XYZ_dark1, XYZ_dark2)[0]
        d_mid_base = space_base.distance(XYZ_mid1, XYZ_mid2)[0]

        # SL > 1 for dark pairs → dL/SL < dL → smaller DE for dark
        # SL = 1 for mid pairs → no change
        ratio = d_dark / d_mid
        ratio_base = d_dark_base / d_mid_base
        assert ratio != pytest.approx(ratio_base, rel=0.01), \
            "dist_sl should change dark vs mid-L distance ratio"

    def test_sc_changes_chromatic_vs_achromatic(self):
        """Positive dist_sc should reduce chroma contribution for chromatic pairs."""
        params = oklab_params()
        params.dist_sc = 2.0  # strong effect
        space = AnalyticalSpace(params)
        space_base = AnalyticalSpace(oklab_params())

        # Achromatic pair (low C_avg)
        XYZ_ach1 = np.array([[0.40, 0.40, 0.40]])
        XYZ_ach2 = np.array([[0.50, 0.50, 0.50]])
        # Chromatic pair (high C_avg)
        XYZ_chr1 = np.array([[0.60, 0.30, 0.05]])
        XYZ_chr2 = np.array([[0.50, 0.25, 0.08]])

        d_ach = space.distance(XYZ_ach1, XYZ_ach2)[0]
        d_chr = space.distance(XYZ_chr1, XYZ_chr2)[0]
        d_ach_base = space_base.distance(XYZ_ach1, XYZ_ach2)[0]
        d_chr_base = space_base.distance(XYZ_chr1, XYZ_chr2)[0]

        # SC > 1 for chromatic → chroma contribution reduced
        ratio = d_chr / d_ach
        ratio_base = d_chr_base / d_ach_base
        assert ratio < ratio_base, \
            f"dist_sc should reduce chromatic distance ratio: {ratio:.4f} vs base {ratio_base:.4f}"

    def test_monotonicity(self):
        """Distance should remain monotonic with SL/SC active."""
        for sl, sc in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            params = oklab_params()
            params.dist_sl = sl
            params.dist_sc = sc
            space = AnalyticalSpace(params)

            base = np.array([[0.5, 0.5, 0.5]])
            targets = np.array([[0.5 + i * 0.05, 0.5, 0.5] for i in range(1, 10)])
            bases = np.tile(base, (9, 1))

            d = space.distance(bases, targets)
            for i in range(len(d) - 1):
                assert d[i + 1] > d[i], (
                    f"Monotonicity violated at sl={sl},sc={sc}: d[{i}]={d[i]:.6f} >= d[{i+1}]={d[i+1]:.6f}"
                )

    def test_serialization(self):
        """dist_sl and dist_sc should survive dict round-trip."""
        params = oklab_params()
        params.dist_sl = 1.5
        params.dist_sc = 0.8
        d = params.to_dict()
        assert "dist_sl" in d
        assert "dist_sc" in d
        p2 = AnalyticalParams.from_dict(d)
        assert p2.dist_sl == 1.5
        assert p2.dist_sc == 0.8

    def test_backward_compat_old_json(self):
        """Loading old JSON without dist_sl/dist_sc should default to 0.0."""
        old_dict = {
            "M1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "gamma": [0.333, 0.333, 0.333],
            "M2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dist_compress": 1.5,
            "dist_post_power": 1.1,
        }
        p = AnalyticalParams.from_dict(old_dict)
        assert p.dist_sl == 0.0
        assert p.dist_sc == 0.0

    def test_no_rt_impact(self, xyz_batch):
        """SL/SC only affect distance, not forward/inverse transforms."""
        params_a = oklab_params()
        space_a = AnalyticalSpace(params_a)

        params_b = oklab_params()
        params_b.dist_sl = 1.5
        params_b.dist_sc = 0.8
        space_b = AnalyticalSpace(params_b)

        coords_a = space_a.from_XYZ(xyz_batch)
        coords_b = space_b.from_XYZ(xyz_batch)
        np.testing.assert_allclose(coords_a, coords_b, atol=1e-15)

        rt_a = space_a.round_trip_error(xyz_batch)
        rt_b = space_b.round_trip_error(xyz_batch)
        np.testing.assert_allclose(rt_a, rt_b, atol=1e-15)


class TestDistSLSCHue:
    """Test hue-dependent SL/SC distance weights (v15)."""

    @pytest.fixture
    def xyz_batch(self):
        rng = np.random.default_rng(42)
        return rng.uniform(0.05, 0.90, (500, 3))

    def test_backward_compat_zero_hue(self):
        """All hue params=0 should match v14c behavior exactly."""
        params_v14c = oklab_params()
        params_v14c.dist_power = 0.9
        params_v14c.dist_wC = 1.1
        params_v14c.dist_compress = 1.5
        params_v14c.dist_sl = 0.5
        params_v14c.dist_sc = 0.8
        space_v14c = AnalyticalSpace(params_v14c)

        params_v15 = oklab_params()
        params_v15.dist_power = 0.9
        params_v15.dist_wC = 1.1
        params_v15.dist_compress = 1.5
        params_v15.dist_sl = 0.5
        params_v15.dist_sc = 0.8
        # All hue Fourier params default to 0.0
        space_v15 = AnalyticalSpace(params_v15)

        rng = np.random.default_rng(99)
        XYZ_1 = rng.uniform(0.05, 0.90, (100, 3))
        XYZ_2 = rng.uniform(0.05, 0.90, (100, 3))

        d_v14c = space_v14c.distance(XYZ_1, XYZ_2)
        d_v15 = space_v15.distance(XYZ_1, XYZ_2)
        np.testing.assert_allclose(d_v14c, d_v15, atol=1e-15)

    def test_sl_hue_changes_by_region(self):
        """Positive dist_sl_hcos1 should change L-weighting differently for blue vs red."""
        params = oklab_params()
        params.dist_sl = 0.5
        params.dist_sl_hcos1 = 0.4  # cos(h): positive near 0°, negative near 180°
        space = AnalyticalSpace(params)

        params_base = oklab_params()
        params_base.dist_sl = 0.5
        space_base = AnalyticalSpace(params_base)

        # Red-ish pair (hue near 0°, cos(h) > 0 → sl_coeff > dist_sl)
        XYZ_red1 = np.array([[0.40, 0.20, 0.05]])
        XYZ_red2 = np.array([[0.45, 0.22, 0.06]])
        # Cyan-ish pair (hue near 180°, cos(h) < 0 → sl_coeff < dist_sl)
        XYZ_cyan1 = np.array([[0.10, 0.20, 0.30]])
        XYZ_cyan2 = np.array([[0.12, 0.22, 0.33]])

        d_red = space.distance(XYZ_red1, XYZ_red2)[0]
        d_cyan = space.distance(XYZ_cyan1, XYZ_cyan2)[0]
        d_red_base = space_base.distance(XYZ_red1, XYZ_red2)[0]
        d_cyan_base = space_base.distance(XYZ_cyan1, XYZ_cyan2)[0]

        # Hue modulation should shift red and cyan differently vs uniform SL
        diff_red = d_red - d_red_base
        diff_cyan = d_cyan - d_cyan_base
        assert abs(diff_red - diff_cyan) > 1e-6, \
            f"Hue-dep SL should affect red/cyan differently: diff_red={diff_red:.8f}, diff_cyan={diff_cyan:.8f}"

    def test_sc_hue_changes_by_region(self):
        """Positive dist_sc_hcos1 should change C-weighting differently for blue vs red."""
        params = oklab_params()
        params.dist_sc = 0.8
        params.dist_sc_hcos1 = 0.5
        space = AnalyticalSpace(params)

        params_base = oklab_params()
        params_base.dist_sc = 0.8
        space_base = AnalyticalSpace(params_base)

        # Chromatic red pair
        XYZ_red1 = np.array([[0.50, 0.25, 0.05]])
        XYZ_red2 = np.array([[0.55, 0.28, 0.06]])
        # Chromatic cyan pair
        XYZ_cyan1 = np.array([[0.10, 0.25, 0.40]])
        XYZ_cyan2 = np.array([[0.12, 0.28, 0.44]])

        d_red = space.distance(XYZ_red1, XYZ_red2)[0]
        d_cyan = space.distance(XYZ_cyan1, XYZ_cyan2)[0]
        d_red_base = space_base.distance(XYZ_red1, XYZ_red2)[0]
        d_cyan_base = space_base.distance(XYZ_cyan1, XYZ_cyan2)[0]

        diff_red = d_red - d_red_base
        diff_cyan = d_cyan - d_cyan_base
        assert abs(diff_red - diff_cyan) > 1e-6, \
            f"Hue-dep SC should affect red/cyan differently: diff_red={diff_red:.8f}, diff_cyan={diff_cyan:.8f}"

    def test_serialization_hue_params(self):
        """Dict round-trip for all 8 new hue SL/SC params."""
        params = oklab_params()
        params.dist_sl = 0.5
        params.dist_sc = 0.8
        params.dist_sl_hcos1 = 0.1
        params.dist_sl_hsin1 = -0.2
        params.dist_sl_hcos2 = 0.3
        params.dist_sl_hsin2 = -0.4
        params.dist_sc_hcos1 = 0.15
        params.dist_sc_hsin1 = -0.25
        params.dist_sc_hcos2 = 0.35
        params.dist_sc_hsin2 = -0.45
        d = params.to_dict()
        for key in ["dist_sl_hcos1", "dist_sl_hsin1", "dist_sl_hcos2", "dist_sl_hsin2",
                     "dist_sc_hcos1", "dist_sc_hsin1", "dist_sc_hcos2", "dist_sc_hsin2"]:
            assert key in d, f"Missing key: {key}"
        p2 = AnalyticalParams.from_dict(d)
        assert p2.dist_sl_hcos1 == 0.1
        assert p2.dist_sl_hsin1 == -0.2
        assert p2.dist_sl_hcos2 == 0.3
        assert p2.dist_sl_hsin2 == -0.4
        assert p2.dist_sc_hcos1 == 0.15
        assert p2.dist_sc_hsin1 == -0.25
        assert p2.dist_sc_hcos2 == 0.35
        assert p2.dist_sc_hsin2 == -0.45

    def test_backward_compat_old_json(self):
        """Loading v14c JSON (without hue SL/SC) should default new params to 0.0."""
        v14c_dict = {
            "M1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "gamma": [0.333, 0.333, 0.333],
            "M2": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dist_compress": 1.5,
            "dist_post_power": 1.1,
            "dist_sl": 0.18,
            "dist_sc": 0.80,
        }
        p = AnalyticalParams.from_dict(v14c_dict)
        assert p.dist_sl == 0.18
        assert p.dist_sc == 0.80
        assert p.dist_sl_hcos1 == 0.0
        assert p.dist_sl_hsin1 == 0.0
        assert p.dist_sl_hcos2 == 0.0
        assert p.dist_sl_hsin2 == 0.0
        assert p.dist_sc_hcos1 == 0.0
        assert p.dist_sc_hsin1 == 0.0
        assert p.dist_sc_hcos2 == 0.0
        assert p.dist_sc_hsin2 == 0.0

    def test_no_rt_impact(self, xyz_batch):
        """Hue SL/SC only affects distance, not forward/inverse transforms."""
        params_a = oklab_params()
        space_a = AnalyticalSpace(params_a)

        params_b = oklab_params()
        params_b.dist_sl = 0.5
        params_b.dist_sc = 0.8
        params_b.dist_sl_hcos1 = 0.3
        params_b.dist_sc_hcos1 = 0.4
        space_b = AnalyticalSpace(params_b)

        coords_a = space_a.from_XYZ(xyz_batch)
        coords_b = space_b.from_XYZ(xyz_batch)
        np.testing.assert_allclose(coords_a, coords_b, atol=1e-15)

        rt_a = space_a.round_trip_error(xyz_batch)
        rt_b = space_b.round_trip_error(xyz_batch)
        np.testing.assert_allclose(rt_a, rt_b, atol=1e-15)


class TestChromaScaling:
    """Test hue-dependent and L-dependent chroma scaling."""

    def test_chroma_scale_changes_distance(self):
        """Hue-dependent chroma scaling should change distance for chromatic pairs."""
        base = AnalyticalSpace(oklab_params())
        params = oklab_params()
        params.cs_cos1 = 0.3
        scaled = AnalyticalSpace(params)

        XYZ_1 = np.array([[0.4, 0.2, 0.05]])
        XYZ_2 = np.array([[0.35, 0.2, 0.08]])
        d_base = base.distance(XYZ_1, XYZ_2)[0]
        d_scaled = scaled.distance(XYZ_1, XYZ_2)[0]
        assert d_base != pytest.approx(d_scaled, rel=0.01)

    def test_L_chroma_changes_distance(self):
        """L-dependent chroma scaling should change distance."""
        base = AnalyticalSpace(oklab_params())
        params = oklab_params()
        params.lc1 = 0.5
        scaled = AnalyticalSpace(params)

        XYZ_1 = np.array([[0.4, 0.2, 0.05]])
        XYZ_2 = np.array([[0.35, 0.2, 0.08]])
        d_base = base.distance(XYZ_1, XYZ_2)[0]
        d_scaled = scaled.distance(XYZ_1, XYZ_2)[0]
        assert d_base != pytest.approx(d_scaled, rel=0.01)


class TestNeutralCorrection:
    """Tests for v19 end-of-pipeline neutral axis correction."""

    @pytest.fixture
    def params_unequal_gamma(self):
        """Params with unequal gammas (like v14 optimized)."""
        p = oklab_params()
        p.gamma = np.array([0.39, 0.42, 0.43])
        return p

    def test_nc_flag_default_off(self):
        """Neutral correction should be off by default."""
        space = AnalyticalSpace(oklab_params())
        assert space._neutral_correction is False

    def test_nc_flag_on(self):
        """Neutral correction can be enabled."""
        space = AnalyticalSpace(oklab_params(), neutral_correction=True)
        assert space._neutral_correction is True

    def test_grays_achromatic_with_nc(self, params_unequal_gamma):
        """Grays should map to a=b≈0 when NC is enabled."""
        space = AnalyticalSpace(params_unequal_gamma, neutral_correction=True)
        D65 = np.array([0.95047, 1.0, 1.08883])
        Y_vals = np.linspace(0.01, 1.0, 20)
        grays = np.outer(Y_vals, D65)
        lab = space.from_XYZ(grays)
        C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        assert np.max(C) < 1e-3, f"Gray chroma too high: {np.max(C):.2e}"

    def test_grays_not_achromatic_without_nc(self, params_unequal_gamma):
        """Grays should NOT map to a=b=0 with unequal gammas and NC off."""
        space = AnalyticalSpace(params_unequal_gamma, neutral_correction=False)
        D65 = np.array([0.95047, 1.0, 1.08883])
        grays = np.outer(np.array([0.1, 0.5, 0.9]), D65)
        lab = space.from_XYZ(grays)
        C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        # With unequal gammas and no NC, grays should have nonzero chroma
        assert np.max(C) > 0.01

    def test_equal_gammas_low_chroma(self):
        """With equal gammas, grays have very low chroma (NC optional)."""
        p = oklab_params()  # gammas are all 1/3
        space = AnalyticalSpace(p, neutral_correction=False)
        D65 = np.array([0.95047, 1.0, 1.08883])
        grays = np.outer(np.linspace(0.01, 1.0, 10), D65)
        lab = space.from_XYZ(grays)
        C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        # Oklab M2 not perfectly aligned to D65 — small residual is expected
        assert np.max(C) < 1e-3

    def test_round_trip_with_nc(self, params_unequal_gamma):
        """Round-trip should be preserved with NC enabled."""
        space = AnalyticalSpace(params_unequal_gamma, neutral_correction=True)
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.05, 0.90, (100, 3))
        lab = space.from_XYZ(XYZ)
        XYZ_back = space.to_XYZ(lab)
        np.testing.assert_allclose(XYZ, XYZ_back, atol=1e-4)

    def test_round_trip_grays_with_nc(self, params_unequal_gamma):
        """Round-trip for grays should work with NC."""
        space = AnalyticalSpace(params_unequal_gamma, neutral_correction=True)
        D65 = np.array([0.95047, 1.0, 1.08883])
        grays = np.outer(np.linspace(0.05, 0.95, 10), D65)
        lab = space.from_XYZ(grays)
        grays_back = space.to_XYZ(lab)
        np.testing.assert_allclose(grays, grays_back, atol=1e-4)

    def test_nc_does_not_change_behavior_when_off(self, params_unequal_gamma):
        """NC=False should give identical results to default constructor."""
        space_default = AnalyticalSpace(params_unequal_gamma)
        space_off = AnalyticalSpace(params_unequal_gamma, neutral_correction=False)
        XYZ = np.array([[0.4, 0.3, 0.2], [0.1, 0.2, 0.5]])
        lab1 = space_default.from_XYZ(XYZ)
        lab2 = space_off.from_XYZ(XYZ)
        np.testing.assert_array_equal(lab1, lab2)

    def test_nc_changes_chromatic_output(self, params_unequal_gamma):
        """NC should shift a,b values for chromatic colors (L-dep shear)."""
        space_off = AnalyticalSpace(params_unequal_gamma, neutral_correction=False)
        space_on = AnalyticalSpace(params_unequal_gamma, neutral_correction=True)
        XYZ = np.array([[0.5, 0.3, 0.1]])
        lab_off = space_off.from_XYZ(XYZ)
        lab_on = space_on.from_XYZ(XYZ)
        # L should be the same (NC doesn't change L)
        assert lab_off[0, 0] == pytest.approx(lab_on[0, 0], abs=1e-10)
        # a,b should differ (shear)
        assert not np.allclose(lab_off[0, 1:], lab_on[0, 1:])

    def test_nc_with_v14_params(self):
        """NC should work with real v14 optimized params."""
        import os
        params_path = "checkpoints/v14_best.json"
        if not os.path.exists(params_path):
            pytest.skip("v14_best.json not found")
        p = AnalyticalParams.load(params_path)
        space = AnalyticalSpace(p, neutral_correction=True)
        D65 = np.array([0.95047, 1.0, 1.08883])
        grays = np.outer(np.linspace(0.01, 1.0, 50), D65)
        lab = space.from_XYZ(grays)
        C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        assert np.max(C) < 1e-3, f"Gray chroma: {np.max(C):.2e}"
        # Round-trip
        grays_back = space.to_XYZ(lab)
        np.testing.assert_allclose(grays, grays_back, atol=1e-3)

    def test_distance_with_nc(self, params_unequal_gamma):
        """Distance computation should work with NC enabled."""
        space = AnalyticalSpace(params_unequal_gamma, neutral_correction=True)
        XYZ_1 = np.array([[0.4, 0.3, 0.2]])
        XYZ_2 = np.array([[0.35, 0.28, 0.22]])
        d = space.distance(XYZ_1, XYZ_2)
        assert np.isfinite(d[0])
        assert d[0] > 0

    def test_ab_rotation_preserves_distance_metric(self, params_unequal_gamma):
        """Rigid a/b rotation should not change the distance metric."""
        space_0 = AnalyticalSpace(params_unequal_gamma, neutral_correction=True, ab_rotate_deg=0.0)
        space_r = AnalyticalSpace(params_unequal_gamma, neutral_correction=True, ab_rotate_deg=21.43)
        rng = np.random.default_rng(0)
        XYZ_1 = rng.uniform(0.05, 0.90, (200, 3))
        XYZ_2 = rng.uniform(0.05, 0.90, (200, 3))
        d0 = space_0.distance(XYZ_1, XYZ_2)
        dr = space_r.distance(XYZ_1, XYZ_2)
        np.testing.assert_allclose(d0, dr, atol=1e-12, rtol=1e-12)

    def test_round_trip_with_nc_and_ab_rotation(self, params_unequal_gamma):
        """Round-trip should be preserved with NC and a/b rotation enabled."""
        space = AnalyticalSpace(params_unequal_gamma, neutral_correction=True, ab_rotate_deg=21.43)
        rng = np.random.default_rng(123)
        XYZ = rng.uniform(0.05, 0.90, (100, 3))
        lab = space.from_XYZ(XYZ)
        XYZ_back = space.to_XYZ(lab)
        np.testing.assert_allclose(XYZ, XYZ_back, atol=1e-4)

    def test_grays_achromatic_with_nc_and_ab_rotation(self, params_unequal_gamma):
        """Grays should map to a=b≈0 with NC even when a/b rotation is enabled."""
        space = AnalyticalSpace(params_unequal_gamma, neutral_correction=True, ab_rotate_deg=21.43)
        D65 = np.array([0.95047, 1.0, 1.08883])
        Y_vals = np.linspace(0.01, 1.0, 20)
        grays = np.outer(Y_vals, D65)
        lab = space.from_XYZ(grays)
        C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        assert np.max(C) < 1e-3, f"Gray chroma too high: {np.max(C):.2e}"
