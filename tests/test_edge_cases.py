"""Comprehensive edge-case tests for MetricSpace and GenSpace.

Covers: black point, near-black, achromatic axis, white point,
NC LUT boundaries, extreme inputs, and round-trip stress tests.
"""

import numpy as np
import pytest

from helmlab.spaces.metric import MetricSpace
from helmlab.spaces.gen import GenSpace
from helmlab.utils.srgb_convert import hex_to_srgb, sRGB_to_XYZ, XYZ_to_sRGB, DisplayP3_to_XYZ

D65 = np.array([0.95047, 1.0, 1.08883])


@pytest.fixture
def ms():
    return MetricSpace(neutral_correction=True, ab_rotate_deg=-28.2)


@pytest.fixture
def gs():
    return GenSpace()


# ── Black point ──────────────────────────────────────────────────────

class TestBlackPoint:
    """Black (XYZ=[0,0,0]) must map to Lab=[0,0,0] exactly."""

    def test_metric_black_is_origin(self, ms):
        lab = ms.from_XYZ(np.array([0.0, 0.0, 0.0]))
        assert lab[0] == pytest.approx(0.0, abs=1e-15), f"L={lab[0]}"
        assert lab[1] == pytest.approx(0.0, abs=1e-15), f"a={lab[1]}"
        assert lab[2] == pytest.approx(0.0, abs=1e-15), f"b={lab[2]}"

    def test_gen_black_is_origin(self, gs):
        lab = gs.from_XYZ(np.array([0.0, 0.0, 0.0]))
        assert lab[0] == pytest.approx(0.0, abs=1e-15), f"L={lab[0]}"
        assert lab[1] == pytest.approx(0.0, abs=1e-15), f"a={lab[1]}"
        assert lab[2] == pytest.approx(0.0, abs=1e-15), f"b={lab[2]}"

    def test_metric_black_roundtrip(self, ms):
        XYZ = np.array([0.0, 0.0, 0.0])
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-14)

    def test_gen_black_roundtrip(self, gs):
        XYZ = np.array([0.0, 0.0, 0.0])
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-14)

    def test_metric_black_hex(self, ms):
        """#000000 should have zero chroma in MetricSpace."""
        srgb = hex_to_srgb("#000000")
        XYZ = sRGB_to_XYZ(srgb)
        lab = ms.from_XYZ(XYZ)
        C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
        assert C < 1e-10, f"#000000 chroma={C:.2e}"

    def test_gen_black_hex(self, gs):
        """#000000 should have zero chroma in GenSpace."""
        srgb = hex_to_srgb("#000000")
        XYZ = sRGB_to_XYZ(srgb)
        lab = gs.from_XYZ(XYZ)
        C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
        assert C < 1e-10, f"#000000 chroma={C:.2e}"

    def test_metric_black_batch(self, ms):
        """Batch of blacks should all map to origin."""
        XYZ = np.zeros((10, 3))
        lab = ms.from_XYZ(XYZ)
        np.testing.assert_allclose(lab, 0.0, atol=1e-14)

    def test_gen_black_batch(self, gs):
        """Batch of blacks should all map to origin."""
        XYZ = np.zeros((10, 3))
        lab = gs.from_XYZ(XYZ)
        np.testing.assert_allclose(lab, 0.0, atol=1e-14)


# ── Near-black ───────────────────────────────────────────────────────

class TestNearBlack:
    """Very dark colors: NC LUT interpolation below first entry."""

    @pytest.mark.parametrize("Y", [1e-8, 1e-6, 1e-4, 0.001, 0.005, 0.01])
    def test_metric_near_black_roundtrip(self, ms, Y):
        XYZ = D65 * Y
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10,
                                   err_msg=f"Y={Y}")

    @pytest.mark.parametrize("Y", [1e-8, 1e-6, 1e-4, 0.001, 0.005, 0.01])
    def test_gen_near_black_roundtrip(self, gs, Y):
        XYZ = D65 * Y
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10,
                                   err_msg=f"Y={Y}")

    @pytest.mark.parametrize("Y", [1e-8, 1e-6, 1e-4, 0.001, 0.005, 0.01])
    def test_metric_near_black_achromatic(self, ms, Y):
        """Near-black grays should have low chroma.

        MetricSpace enrichment stages can introduce small residual chroma
        for very dark colors outside the NC LUT range. Tolerance is relaxed
        for extreme darks (Y < 0.001).
        """
        XYZ = D65 * Y
        lab = ms.from_XYZ(XYZ)
        C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
        tol = 0.05 if Y < 0.001 else 1e-4
        assert C < tol, f"Y={Y}: chroma={C:.2e}"

    @pytest.mark.parametrize("Y", [1e-8, 1e-6, 1e-4, 0.001, 0.005, 0.01])
    def test_gen_near_black_achromatic(self, gs, Y):
        """Near-black grays should have near-zero chroma."""
        XYZ = D65 * Y
        lab = gs.from_XYZ(XYZ)
        C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
        assert C < 1e-4, f"Y={Y}: chroma={C:.2e}"

    @pytest.mark.parametrize("hex_str", [
        "#010101", "#020202", "#050505", "#0a0a0a", "#101010",
    ])
    def test_metric_dark_hex_roundtrip(self, ms, hex_str):
        srgb = hex_to_srgb(hex_str)
        XYZ = sRGB_to_XYZ(srgb)
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10,
                                   err_msg=f"{hex_str}")

    @pytest.mark.parametrize("hex_str", [
        "#010101", "#020202", "#050505", "#0a0a0a", "#101010",
    ])
    def test_gen_dark_hex_roundtrip(self, gs, hex_str):
        srgb = hex_to_srgb(hex_str)
        XYZ = sRGB_to_XYZ(srgb)
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10,
                                   err_msg=f"{hex_str}")


# ── White point ──────────────────────────────────────────────────────

class TestWhitePoint:
    """D65 white and near-white colors."""

    def test_metric_white_achromatic(self, ms):
        lab = ms.from_XYZ(D65)
        C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
        assert C < 1e-4, f"D65 white chroma={C:.2e}"

    def test_gen_white_achromatic(self, gs):
        lab = gs.from_XYZ(D65)
        C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
        assert C < 1e-4, f"D65 white chroma={C:.2e}"

    def test_metric_white_roundtrip(self, ms):
        lab = ms.from_XYZ(D65)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, D65, atol=1e-10)

    def test_gen_white_roundtrip(self, gs):
        lab = gs.from_XYZ(D65)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, D65, atol=1e-10)

    @pytest.mark.parametrize("hex_str", ["#f0f0f0", "#f8f8f8", "#fefefe", "#ffffff"])
    def test_metric_near_white_roundtrip(self, ms, hex_str):
        XYZ = sRGB_to_XYZ(hex_to_srgb(hex_str))
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10,
                                   err_msg=hex_str)


# ── Full achromatic axis ─────────────────────────────────────────────

class TestAchromaticAxis:
    """Full gray ramp: chroma must stay near zero everywhere."""

    def test_metric_achromatic_full_ramp(self, ms):
        """256-step gray ramp: max chroma < 1e-4."""
        Y_vals = np.linspace(0, 1, 256)
        max_C = 0
        for Y in Y_vals:
            XYZ = D65 * Y
            lab = ms.from_XYZ(XYZ)
            C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
            max_C = max(max_C, C)
        assert max_C < 1e-4, f"max achromatic chroma={max_C:.2e}"

    def test_gen_achromatic_full_ramp(self, gs):
        """256-step gray ramp: max chroma < 1e-4."""
        Y_vals = np.linspace(0, 1, 256)
        max_C = 0
        for Y in Y_vals:
            XYZ = D65 * Y
            lab = gs.from_XYZ(XYZ)
            C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
            max_C = max(max_C, C)
        assert max_C < 1e-4, f"max achromatic chroma={max_C:.2e}"

    def test_metric_lightness_monotonic_from_zero(self, ms):
        """Lightness must be monotonically increasing from Y=0."""
        Y_vals = np.concatenate([[0], np.logspace(-6, 0, 100)])
        prev_L = -1
        for Y in Y_vals:
            lab = ms.from_XYZ(D65 * Y)
            L = float(lab[0])
            assert L >= prev_L - 1e-10, (
                f"L monotonicity broken: Y={Y:.2e}, L={L:.6f}, prev={prev_L:.6f}")
            prev_L = L

    def test_gen_lightness_monotonic_from_zero(self, gs):
        """Lightness must be monotonically increasing from Y=0."""
        Y_vals = np.concatenate([[0], np.logspace(-6, 0, 100)])
        prev_L = -1
        for Y in Y_vals:
            lab = gs.from_XYZ(D65 * Y)
            L = float(lab[0])
            assert L >= prev_L - 1e-10, (
                f"L monotonicity broken: Y={Y:.2e}, L={L:.6f}, prev={prev_L:.6f}")
            prev_L = L

    def test_metric_hex_grays_chroma(self, ms):
        """Every hex gray from #000000 to #ffffff (step 17) should be achromatic."""
        max_C = 0
        for v in range(0, 256, 17):
            hex_str = f"#{v:02x}{v:02x}{v:02x}"
            XYZ = sRGB_to_XYZ(hex_to_srgb(hex_str))
            lab = ms.from_XYZ(XYZ)
            C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
            max_C = max(max_C, C)
        assert max_C < 1e-3, f"max hex gray chroma={max_C:.2e}"


# ── NC LUT boundary continuity ───────────────────────────────────────

class TestNCLUTContinuity:
    """NC correction must be continuous — no jumps at LUT boundaries."""

    def test_metric_nc_smooth_at_boundary(self, ms):
        """a,b values should change smoothly across NC LUT lower boundary."""
        # Sample very finely around the dark boundary
        Y_vals = np.logspace(-5, -1, 200)
        labs = []
        for Y in Y_vals:
            lab = ms.from_XYZ(D65 * Y)
            labs.append(lab.copy())
        labs = np.array(labs)
        # Check that consecutive a,b differences are small
        da = np.abs(np.diff(labs[:, 1]))
        db = np.abs(np.diff(labs[:, 2]))
        max_jump_a = np.max(da)
        max_jump_b = np.max(db)
        assert max_jump_a < 0.01, f"NC a-jump={max_jump_a:.4f}"
        assert max_jump_b < 0.01, f"NC b-jump={max_jump_b:.4f}"

    def test_gen_nc_smooth_at_boundary(self, gs):
        """a,b values should change smoothly across NC LUT lower boundary."""
        Y_vals = np.logspace(-5, -1, 200)
        labs = []
        for Y in Y_vals:
            lab = gs.from_XYZ(D65 * Y)
            labs.append(lab.copy())
        labs = np.array(labs)
        da = np.abs(np.diff(labs[:, 1]))
        db = np.abs(np.diff(labs[:, 2]))
        max_jump_a = np.max(da)
        max_jump_b = np.max(db)
        assert max_jump_a < 0.01, f"NC a-jump={max_jump_a:.4f}"
        assert max_jump_b < 0.01, f"NC b-jump={max_jump_b:.4f}"


# ── Extreme / boundary inputs ────────────────────────────────────────

class TestExtremeInputs:
    """Edge cases: negative XYZ, very large values, near-singular."""

    @pytest.mark.parametrize("XYZ", [
        [0.0, 0.0, 0.0],         # black
        [0.95047, 1.0, 1.08883], # D65 white
        [2.0, 2.0, 2.0],         # above white
        [0.0, 0.0, 0.001],       # near-zero single channel
        [0.001, 0.0, 0.0],       # near-zero single channel
    ])
    def test_metric_roundtrip_extremes(self, ms, XYZ):
        XYZ = np.array(XYZ, dtype=np.float64)
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-8,
                                   err_msg=f"XYZ={XYZ}")

    @pytest.mark.parametrize("XYZ", [
        [0.0, 0.0, 0.0],
        [0.95047, 1.0, 1.08883],
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.001],
        [0.001, 0.0, 0.0],
    ])
    def test_gen_roundtrip_extremes(self, gs, XYZ):
        XYZ = np.array(XYZ, dtype=np.float64)
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-8,
                                   err_msg=f"XYZ={XYZ}")

    def test_metric_no_nan_on_zero(self, ms):
        """Pipeline must not produce NaN for zero input."""
        lab = ms.from_XYZ(np.array([0.0, 0.0, 0.0]))
        assert not np.any(np.isnan(lab)), f"NaN in Lab: {lab}"

    def test_gen_no_nan_on_zero(self, gs):
        lab = gs.from_XYZ(np.array([0.0, 0.0, 0.0]))
        assert not np.any(np.isnan(lab)), f"NaN in Lab: {lab}"

    def test_metric_no_nan_on_tiny(self, ms):
        """Pipeline must not produce NaN for tiny inputs."""
        for XYZ in [np.array([1e-20, 1e-20, 1e-20]),
                     np.array([1e-100, 0.0, 0.0])]:
            lab = ms.from_XYZ(XYZ)
            assert not np.any(np.isnan(lab)), f"NaN for XYZ={XYZ}: Lab={lab}"

    def test_gen_no_nan_on_tiny(self, gs):
        for XYZ in [np.array([1e-20, 1e-20, 1e-20]),
                     np.array([1e-100, 0.0, 0.0])]:
            lab = gs.from_XYZ(XYZ)
            assert not np.any(np.isnan(lab)), f"NaN for XYZ={XYZ}: Lab={lab}"


# ── sRGB primaries and secondaries ───────────────────────────────────

class TestSRGBPrimaries:
    """Round-trip for all sRGB primaries and secondaries."""

    COLORS = [
        "#ff0000", "#00ff00", "#0000ff",  # primaries
        "#ffff00", "#ff00ff", "#00ffff",  # secondaries
        "#000000", "#ffffff",             # black, white
        "#808080",                        # mid gray
    ]

    @pytest.mark.parametrize("hex_str", COLORS)
    def test_metric_primary_roundtrip(self, ms, hex_str):
        XYZ = sRGB_to_XYZ(hex_to_srgb(hex_str))
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10,
                                   err_msg=hex_str)

    @pytest.mark.parametrize("hex_str", COLORS)
    def test_gen_primary_roundtrip(self, gs, hex_str):
        XYZ = sRGB_to_XYZ(hex_to_srgb(hex_str))
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10,
                                   err_msg=hex_str)


# ── Web-safe round-trip stress test ──────────────────────────────────

class TestWebSafeStress:
    """All 216 web-safe colors must round-trip accurately."""

    def test_metric_websafe_roundtrip(self, ms):
        max_err = 0
        worst = None
        for r in range(0, 256, 51):
            for g in range(0, 256, 51):
                for b in range(0, 256, 51):
                    srgb = np.array([r, g, b]) / 255.0
                    XYZ = sRGB_to_XYZ(srgb)
                    lab = ms.from_XYZ(XYZ)
                    rec = ms.to_XYZ(lab)
                    err = np.max(np.abs(rec - XYZ))
                    if err > max_err:
                        max_err = err
                        worst = f"#{r:02x}{g:02x}{b:02x}"
        assert max_err < 1e-8, f"worst={worst}, err={max_err:.2e}"

    def test_gen_websafe_roundtrip(self, gs):
        max_err = 0
        worst = None
        for r in range(0, 256, 51):
            for g in range(0, 256, 51):
                for b in range(0, 256, 51):
                    srgb = np.array([r, g, b]) / 255.0
                    XYZ = sRGB_to_XYZ(srgb)
                    lab = gs.from_XYZ(XYZ)
                    rec = gs.to_XYZ(lab)
                    err = np.max(np.abs(rec - XYZ))
                    if err > max_err:
                        max_err = err
                        worst = f"#{r:02x}{g:02x}{b:02x}"
        assert max_err < 1e-8, f"worst={worst}, err={max_err:.2e}"


# ── Distance edge cases ──────────────────────────────────────────────

class TestDistanceEdgeCases:
    """Distance function must handle edge cases correctly."""

    def test_metric_black_to_black_zero(self, ms):
        black = np.array([[0.0, 0.0, 0.0]])
        d = ms.distance(black, black)
        assert d[0] == pytest.approx(0.0, abs=1e-15)

    def test_metric_black_to_white_positive(self, ms):
        black = np.array([[0.0, 0.0, 0.0]])
        white = np.array([D65])
        d = ms.distance(black, white)
        assert d[0] > 0

    def test_gen_black_to_black_zero(self, gs):
        black = np.array([[0.0, 0.0, 0.0]])
        d = gs.distance(black, black)
        assert d[0] == pytest.approx(0.0, abs=1e-15)


# ── refRange validation ──────────────────────────────────────────────

class TestRefRange:
    """Lab ranges must cover D65 white and both sRGB + Display P3 gamuts.

    These tests ensure that the refRange values declared in Color.js
    space definitions are correct. CSS percentage values (100%) map to
    refRange upper bound, so white must be within range.
    """

    # Expected refRange values (must match Color.js space definitions)
    HELMLAB_L_MAX = 1.144
    HELMLAB_AB_MAX = 1.0
    HELMGEN_L_MAX = 1.169
    HELMGEN_AB_MAX = 0.4

    def _gamut_corners(self):
        """All 8 corners of sRGB and Display P3 gamuts."""
        corners = np.array([[r, g, b]
                            for r in [0, 1] for g in [0, 1] for b in [0, 1]],
                           dtype=np.float64)
        srgb_xyz = np.array([sRGB_to_XYZ(c) for c in corners])
        p3_xyz = np.array([DisplayP3_to_XYZ(c) for c in corners])
        return srgb_xyz, p3_xyz

    def test_metric_white_within_L_range(self, ms):
        """D65 white L must be within declared refRange."""
        lab = ms.from_XYZ(D65)
        assert lab[0] <= self.HELMLAB_L_MAX, (
            f"D65 white L={lab[0]:.6f} exceeds refRange max {self.HELMLAB_L_MAX}")
        assert lab[0] > 0

    def test_gen_white_within_L_range(self, gs):
        lab = gs.from_XYZ(D65)
        assert lab[0] <= self.HELMGEN_L_MAX, (
            f"D65 white L={lab[0]:.6f} exceeds refRange max {self.HELMGEN_L_MAX}")
        assert lab[0] > 0

    def test_metric_srgb_within_ab_range(self, ms):
        """All sRGB corners must have a,b within declared refRange."""
        srgb_xyz, _ = self._gamut_corners()
        labs = ms.from_XYZ(srgb_xyz)
        a_max = np.max(np.abs(labs[:, 1]))
        b_max = np.max(np.abs(labs[:, 2]))
        assert a_max <= self.HELMLAB_AB_MAX, (
            f"sRGB |a| max={a_max:.6f} exceeds {self.HELMLAB_AB_MAX}")
        assert b_max <= self.HELMLAB_AB_MAX, (
            f"sRGB |b| max={b_max:.6f} exceeds {self.HELMLAB_AB_MAX}")

    def test_metric_p3_within_ab_range(self, ms):
        """All Display P3 corners must have a,b within declared refRange."""
        _, p3_xyz = self._gamut_corners()
        labs = ms.from_XYZ(p3_xyz)
        a_max = np.max(np.abs(labs[:, 1]))
        b_max = np.max(np.abs(labs[:, 2]))
        assert a_max <= self.HELMLAB_AB_MAX, (
            f"P3 |a| max={a_max:.6f} exceeds {self.HELMLAB_AB_MAX}")
        assert b_max <= self.HELMLAB_AB_MAX, (
            f"P3 |b| max={b_max:.6f} exceeds {self.HELMLAB_AB_MAX}")

    def test_gen_srgb_within_ab_range(self, gs):
        srgb_xyz, _ = self._gamut_corners()
        labs = gs.from_XYZ(srgb_xyz)
        a_max = np.max(np.abs(labs[:, 1]))
        b_max = np.max(np.abs(labs[:, 2]))
        assert a_max <= self.HELMGEN_AB_MAX, (
            f"sRGB |a| max={a_max:.6f} exceeds {self.HELMGEN_AB_MAX}")
        assert b_max <= self.HELMGEN_AB_MAX, (
            f"sRGB |b| max={b_max:.6f} exceeds {self.HELMGEN_AB_MAX}")

    def test_gen_p3_within_ab_range(self, gs):
        _, p3_xyz = self._gamut_corners()
        labs = gs.from_XYZ(p3_xyz)
        a_max = np.max(np.abs(labs[:, 1]))
        b_max = np.max(np.abs(labs[:, 2]))
        assert a_max <= self.HELMGEN_AB_MAX, (
            f"P3 |a| max={a_max:.6f} exceeds {self.HELMGEN_AB_MAX}")
        assert b_max <= self.HELMGEN_AB_MAX, (
            f"P3 |b| max={b_max:.6f} exceeds {self.HELMGEN_AB_MAX}")

    def test_metric_srgb_within_L_range(self, ms):
        """All sRGB corners L must be within [0, L_MAX]."""
        srgb_xyz, _ = self._gamut_corners()
        labs = ms.from_XYZ(srgb_xyz)
        assert np.all(labs[:, 0] >= 0), "Negative L found"
        assert np.all(labs[:, 0] <= self.HELMLAB_L_MAX), (
            f"sRGB L max={labs[:,0].max():.6f} exceeds {self.HELMLAB_L_MAX}")

    def test_gen_srgb_within_L_range(self, gs):
        srgb_xyz, _ = self._gamut_corners()
        labs = gs.from_XYZ(srgb_xyz)
        assert np.all(labs[:, 0] >= 0), "Negative L found"
        assert np.all(labs[:, 0] <= self.HELMGEN_L_MAX), (
            f"sRGB L max={labs[:,0].max():.6f} exceeds {self.HELMGEN_L_MAX}")
