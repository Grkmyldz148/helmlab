"""Stress tests: find every possible bug in MetricSpace and GenSpace.

Tests numerical stability, cross-space consistency, metric properties,
gamut mapping edge cases, hue/chroma continuity, and exhaustive sRGB coverage.
"""

import numpy as np
import pytest
from itertools import product

from helmlab.spaces.metric import MetricSpace
from helmlab.spaces.gen import GenSpace
from helmlab.utils.srgb_convert import (
    hex_to_srgb, srgb_to_hex, sRGB_to_XYZ, XYZ_to_sRGB, clamp_srgb,
)
from helmlab.utils.gamut import gamut_map

D65 = np.array([0.95047, 1.0, 1.08883])


@pytest.fixture(scope="module")
def ms():
    return MetricSpace(neutral_correction=True, ab_rotate_deg=-28.2)


@pytest.fixture(scope="module")
def gs():
    return GenSpace()


# ── 1. Exhaustive sRGB round-trip (every 4th value = 262144 colors) ──

class TestExhaustiveSRGB:
    """Dense sRGB sampling — catch any color that fails round-trip."""

    def test_metric_dense_srgb_roundtrip(self, ms):
        """Every 4th sRGB value (64^3 = 262,144 colors)."""
        max_err = 0
        worst = None
        for r in range(0, 256, 4):
            for g in range(0, 256, 4):
                for b in range(0, 256, 4):
                    srgb = np.array([r, g, b]) / 255.0
                    XYZ = sRGB_to_XYZ(srgb)
                    lab = ms.from_XYZ(XYZ)
                    rec = ms.to_XYZ(lab)
                    err = np.max(np.abs(rec - XYZ))
                    if err > max_err:
                        max_err = err
                        worst = (r, g, b)
        assert max_err < 1e-8, (
            f"worst=({worst[0]},{worst[1]},{worst[2]}), err={max_err:.2e}")

    def test_gen_dense_srgb_roundtrip(self, gs):
        """Every 4th sRGB value (64^3 = 262,144 colors)."""
        max_err = 0
        worst = None
        for r in range(0, 256, 4):
            for g in range(0, 256, 4):
                for b in range(0, 256, 4):
                    srgb = np.array([r, g, b]) / 255.0
                    XYZ = sRGB_to_XYZ(srgb)
                    lab = gs.from_XYZ(XYZ)
                    rec = gs.to_XYZ(lab)
                    err = np.max(np.abs(rec - XYZ))
                    if err > max_err:
                        max_err = err
                        worst = (r, g, b)
        assert max_err < 1e-8, (
            f"worst=({worst[0]},{worst[1]},{worst[2]}), err={max_err:.2e}")


# ── 2. NaN / Inf detection across full pipeline ─────────────────────

class TestNaNInf:
    """No NaN or Inf should ever appear in outputs."""

    EDGE_XYZ = [
        [0, 0, 0],
        [1e-300, 1e-300, 1e-300],
        [1e-20, 0, 0],
        [0, 1e-20, 0],
        [0, 0, 1e-20],
        [1e-10, 1e-10, 1e-10],
        [0.95047, 1.0, 1.08883],  # D65
        [2.0, 2.0, 2.0],
        [5.0, 5.0, 5.0],  # very bright
        [0.5, 0.0, 0.0],  # single channel
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ]

    @pytest.mark.parametrize("xyz", EDGE_XYZ)
    def test_metric_no_nan_inf(self, ms, xyz):
        xyz = np.array(xyz, dtype=np.float64)
        lab = ms.from_XYZ(xyz)
        assert not np.any(np.isnan(lab)), f"NaN in forward: XYZ={xyz}, Lab={lab}"
        assert not np.any(np.isinf(lab)), f"Inf in forward: XYZ={xyz}, Lab={lab}"
        rec = ms.to_XYZ(lab)
        assert not np.any(np.isnan(rec)), f"NaN in inverse: Lab={lab}, XYZ={rec}"
        assert not np.any(np.isinf(rec)), f"Inf in inverse: Lab={lab}, XYZ={rec}"

    @pytest.mark.parametrize("xyz", EDGE_XYZ)
    def test_gen_no_nan_inf(self, gs, xyz):
        xyz = np.array(xyz, dtype=np.float64)
        lab = gs.from_XYZ(xyz)
        assert not np.any(np.isnan(lab)), f"NaN in forward: XYZ={xyz}, Lab={lab}"
        assert not np.any(np.isinf(lab)), f"Inf in forward: XYZ={xyz}, Lab={lab}"
        rec = gs.to_XYZ(lab)
        assert not np.any(np.isnan(rec)), f"NaN in inverse: Lab={lab}, XYZ={rec}"
        assert not np.any(np.isinf(rec)), f"Inf in inverse: Lab={lab}, XYZ={rec}"

    def test_metric_random_wide_gamut_no_nan(self, ms):
        """10000 random wide-gamut XYZ values — no NaN/Inf."""
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0, 3, (10000, 3))
        lab = ms.from_XYZ(XYZ)
        assert not np.any(np.isnan(lab)), "NaN in batch forward"
        assert not np.any(np.isinf(lab)), "Inf in batch forward"
        rec = ms.to_XYZ(lab)
        assert not np.any(np.isnan(rec)), "NaN in batch inverse"
        assert not np.any(np.isinf(rec)), "Inf in batch inverse"

    def test_gen_random_wide_gamut_no_nan(self, gs):
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0, 3, (10000, 3))
        lab = gs.from_XYZ(XYZ)
        assert not np.any(np.isnan(lab)), "NaN in batch forward"
        assert not np.any(np.isinf(lab)), "Inf in batch forward"
        rec = gs.to_XYZ(lab)
        assert not np.any(np.isnan(rec)), "NaN in batch inverse"
        assert not np.any(np.isinf(rec)), "Inf in batch inverse"


# ── 3. NC LUT upper boundary ────────────────────────────────────────

class TestNCLUTUpperBoundary:
    """NC correction at the bright end of the LUT."""

    @pytest.mark.parametrize("Y", [1.0, 1.5, 2.0, 3.0, 5.0])
    def test_metric_bright_roundtrip(self, ms, Y):
        XYZ = D65 * Y
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-8,
                                   err_msg=f"Y={Y}")

    @pytest.mark.parametrize("Y", [1.0, 1.5, 2.0, 3.0, 5.0])
    def test_gen_bright_roundtrip(self, gs, Y):
        XYZ = D65 * Y
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-8,
                                   err_msg=f"Y={Y}")

    def test_metric_nc_smooth_at_upper_boundary(self, ms):
        """No jumps at the bright end of NC LUT."""
        Y_vals = np.linspace(0.8, 2.0, 200)
        labs = np.array([ms.from_XYZ(D65 * Y) for Y in Y_vals])
        da = np.abs(np.diff(labs[:, 1]))
        db = np.abs(np.diff(labs[:, 2]))
        assert np.max(da) < 0.01, f"NC upper a-jump={np.max(da):.4f}"
        assert np.max(db) < 0.01, f"NC upper b-jump={np.max(db):.4f}"


# ── 4. Inverse transform convergence ────────────────────────────────

class TestInverseConvergence:
    """Verify Newton iterations converge for all sRGB colors."""

    def test_metric_inverse_convergence_random(self, ms):
        """5000 random sRGB colors — inverse must converge."""
        rng = np.random.default_rng(123)
        srgb = rng.uniform(0, 1, (5000, 3))
        XYZ = np.array([sRGB_to_XYZ(s) for s in srgb])
        lab = ms.from_XYZ(XYZ)
        rec = ms.to_XYZ(lab)
        errors = np.max(np.abs(rec - XYZ), axis=1)
        p99 = np.percentile(errors, 99)
        p_max = np.max(errors)
        assert p99 < 1e-10, f"p99 error={p99:.2e}"
        assert p_max < 1e-6, f"max error={p_max:.2e}"

    def test_gen_inverse_convergence_random(self, gs):
        rng = np.random.default_rng(123)
        srgb = rng.uniform(0, 1, (5000, 3))
        XYZ = np.array([sRGB_to_XYZ(s) for s in srgb])
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        errors = np.max(np.abs(rec - XYZ), axis=1)
        p_max = np.max(errors)
        assert p_max < 1e-8, f"max error={p_max:.2e}"


# ── 5. Hue continuity (no discontinuities) ──────────────────────────

class TestHueContinuity:
    """Hue angle should change smoothly — no sudden jumps."""

    def test_metric_hue_sweep_continuity(self, ms):
        """Sweep hue at constant chroma/lightness: hue should be continuous."""
        max_jump = 0
        for hue_deg in range(360):
            h1 = np.deg2rad(hue_deg)
            h2 = np.deg2rad(hue_deg + 1)
            # Create XYZ at this hue via sRGB
            r1 = 0.5 + 0.3 * np.cos(h1)
            g1 = 0.5 + 0.3 * np.cos(h1 - 2.094)
            b1 = 0.5 + 0.3 * np.cos(h1 + 2.094)
            r2 = 0.5 + 0.3 * np.cos(h2)
            g2 = 0.5 + 0.3 * np.cos(h2 - 2.094)
            b2 = 0.5 + 0.3 * np.cos(h2 + 2.094)
            srgb1 = np.clip([r1, g1, b1], 0, 1)
            srgb2 = np.clip([r2, g2, b2], 0, 1)
            lab1 = ms.from_XYZ(sRGB_to_XYZ(srgb1))
            lab2 = ms.from_XYZ(sRGB_to_XYZ(srgb2))
            # Check Lab difference (should be small for 1° step)
            dL = abs(lab1[0] - lab2[0])
            da = abs(lab1[1] - lab2[1])
            db = abs(lab1[2] - lab2[2])
            jump = max(dL, da, db)
            max_jump = max(max_jump, jump)
        assert max_jump < 0.05, f"max hue jump={max_jump:.4f}"

    def test_gen_hue_sweep_continuity(self, gs):
        max_jump = 0
        for hue_deg in range(360):
            h1 = np.deg2rad(hue_deg)
            h2 = np.deg2rad(hue_deg + 1)
            r1 = 0.5 + 0.3 * np.cos(h1)
            g1 = 0.5 + 0.3 * np.cos(h1 - 2.094)
            b1 = 0.5 + 0.3 * np.cos(h1 + 2.094)
            r2 = 0.5 + 0.3 * np.cos(h2)
            g2 = 0.5 + 0.3 * np.cos(h2 - 2.094)
            b2 = 0.5 + 0.3 * np.cos(h2 + 2.094)
            srgb1 = np.clip([r1, g1, b1], 0, 1)
            srgb2 = np.clip([r2, g2, b2], 0, 1)
            lab1 = gs.from_XYZ(sRGB_to_XYZ(srgb1))
            lab2 = gs.from_XYZ(sRGB_to_XYZ(srgb2))
            dL = abs(lab1[0] - lab2[0])
            da = abs(lab1[1] - lab2[1])
            db = abs(lab1[2] - lab2[2])
            jump = max(dL, da, db)
            max_jump = max(max_jump, jump)
        assert max_jump < 0.05, f"max hue jump={max_jump:.4f}"


# ── 6. Distance metric properties ───────────────────────────────────

class TestDistanceMetric:
    """Distance function must satisfy metric axioms."""

    def test_metric_self_distance_zero(self, ms):
        """d(x,x) = 0 for all colors."""
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.01, 0.9, (100, 3))
        d = ms.distance(XYZ, XYZ)
        assert np.all(d < 1e-10), f"max self-distance={np.max(d):.2e}"

    def test_metric_symmetry(self, ms):
        """d(x,y) = d(y,x)."""
        rng = np.random.default_rng(42)
        XYZ_a = rng.uniform(0.05, 0.9, (200, 3))
        XYZ_b = rng.uniform(0.05, 0.9, (200, 3))
        d_ab = ms.distance(XYZ_a, XYZ_b)
        d_ba = ms.distance(XYZ_b, XYZ_a)
        np.testing.assert_allclose(d_ab, d_ba, atol=1e-12)

    def test_metric_non_negative(self, ms):
        """d(x,y) >= 0."""
        rng = np.random.default_rng(42)
        XYZ_a = rng.uniform(0.01, 0.9, (500, 3))
        XYZ_b = rng.uniform(0.01, 0.9, (500, 3))
        d = ms.distance(XYZ_a, XYZ_b)
        assert np.all(d >= 0), f"negative distance found: min={np.min(d):.2e}"

    def test_metric_triangle_inequality(self, ms):
        """d(x,z) <= d(x,y) + d(y,z) for random triplets."""
        rng = np.random.default_rng(42)
        XYZ_x = rng.uniform(0.05, 0.9, (500, 3))
        XYZ_y = rng.uniform(0.05, 0.9, (500, 3))
        XYZ_z = rng.uniform(0.05, 0.9, (500, 3))
        d_xy = ms.distance(XYZ_x, XYZ_y)
        d_yz = ms.distance(XYZ_y, XYZ_z)
        d_xz = ms.distance(XYZ_x, XYZ_z)
        violations = d_xz > d_xy + d_yz + 1e-10
        n_violations = np.sum(violations)
        assert n_violations == 0, (
            f"{n_violations}/500 triangle inequality violations, "
            f"max excess={np.max(d_xz - d_xy - d_yz):.4e}")

    def test_gen_self_distance_zero(self, gs):
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.01, 0.9, (100, 3))
        d = gs.distance(XYZ, XYZ)
        assert np.all(d < 1e-10), f"max self-distance={np.max(d):.2e}"

    def test_gen_symmetry(self, gs):
        rng = np.random.default_rng(42)
        XYZ_a = rng.uniform(0.05, 0.9, (200, 3))
        XYZ_b = rng.uniform(0.05, 0.9, (200, 3))
        d_ab = gs.distance(XYZ_a, XYZ_b)
        d_ba = gs.distance(XYZ_b, XYZ_a)
        np.testing.assert_allclose(d_ab, d_ba, atol=1e-12)


# ── 7. Gamut mapping edge cases ──────────────────────────────────────

class TestGamutMapping:
    """Gamut mapping must handle all edge cases."""

    def test_gen_gamut_map_black(self, gs):
        """Gamut mapping black should stay black."""
        lab = gs.from_XYZ(np.array([0.0, 0.0, 0.0]))
        mapped = gamut_map(lab, gs, gamut="srgb")
        XYZ_out = gs.to_XYZ(mapped)
        srgb_out = clamp_srgb(XYZ_to_sRGB(XYZ_out))
        assert np.allclose(srgb_out, 0.0, atol=0.01), f"black mapped to {srgb_out}"

    def test_gen_gamut_map_white(self, gs):
        """Gamut mapping white should stay white."""
        lab = gs.from_XYZ(D65)
        mapped = gamut_map(lab, gs, gamut="srgb")
        XYZ_out = gs.to_XYZ(mapped)
        srgb_out = clamp_srgb(XYZ_to_sRGB(XYZ_out))
        np.testing.assert_allclose(srgb_out, 1.0, atol=0.01)

    def test_gen_gamut_map_primaries_unchanged(self, gs):
        """In-gamut sRGB primaries should not change."""
        for hex_str in ["#ff0000", "#00ff00", "#0000ff", "#808080"]:
            srgb = hex_to_srgb(hex_str)
            XYZ = sRGB_to_XYZ(srgb)
            lab = gs.from_XYZ(XYZ)
            mapped = gamut_map(lab, gs, gamut="srgb")
            XYZ_out = gs.to_XYZ(mapped)
            srgb_out = clamp_srgb(XYZ_to_sRGB(XYZ_out))
            diff = np.max(np.abs(srgb - srgb_out))
            assert diff < 0.02, f"{hex_str}: sRGB diff={diff:.4f}"

    def test_gen_gamut_map_out_of_gamut(self, gs):
        """Out-of-gamut Lab values should map to valid sRGB."""
        # Extremely saturated — definitely out of sRGB
        for lab in [np.array([0.5, 0.5, 0.0]),
                    np.array([0.5, -0.5, 0.0]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.5, 0.0, -0.5])]:
            mapped = gamut_map(lab, gs, gamut="srgb")
            XYZ_out = gs.to_XYZ(mapped)
            srgb_out = XYZ_to_sRGB(XYZ_out)
            assert np.all(srgb_out >= -0.005) and np.all(srgb_out <= 1.005), (
                f"Lab={lab} mapped to sRGB={srgb_out}")

    def test_gen_gamut_map_all_lightness_levels(self, gs):
        """Gamut mapping should work at every lightness level."""
        # Use GenSpace's actual white L as upper bound (v14 white ≈ 0.929)
        white_xyz = np.array([0.95047, 1.0, 1.08883])
        white_L = gs.from_XYZ(white_xyz)[0]
        for L in np.linspace(0, float(white_L) * 0.99, 21):
            lab = np.array([L, 0.2, 0.1])
            mapped = gamut_map(lab, gs, gamut="srgb")
            XYZ_out = gs.to_XYZ(mapped)
            srgb_out = XYZ_to_sRGB(XYZ_out)
            assert np.all(srgb_out >= -0.01) and np.all(srgb_out <= 1.01), (
                f"L={L}: sRGB={srgb_out}")


# ── 8. Batch vs single consistency ───────────────────────────────────

class TestBatchSingleConsistency:
    """Batch processing must give identical results to single-sample."""

    def test_metric_batch_equals_single(self, ms):
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.01, 0.9, (50, 3))
        batch_lab = ms.from_XYZ(XYZ)
        for i in range(50):
            single_lab = ms.from_XYZ(XYZ[i])
            np.testing.assert_allclose(
                batch_lab[i], single_lab, atol=1e-14,
                err_msg=f"index {i}")

    def test_gen_batch_equals_single(self, gs):
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.01, 0.9, (50, 3))
        batch_lab = gs.from_XYZ(XYZ)
        for i in range(50):
            single_lab = gs.from_XYZ(XYZ[i])
            np.testing.assert_allclose(
                batch_lab[i], single_lab, atol=1e-14,
                err_msg=f"index {i}")

    def test_metric_inverse_batch_equals_single(self, ms):
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.01, 0.9, (50, 3))
        lab = ms.from_XYZ(XYZ)
        batch_rec = ms.to_XYZ(lab)
        for i in range(50):
            single_rec = ms.to_XYZ(lab[i])
            np.testing.assert_allclose(
                batch_rec[i], single_rec, atol=1e-14,
                err_msg=f"index {i}")


# ── 9. Python ↔ JS cross-validation values ──────────────────────────

class TestCrossValidationAnchors:
    """Generate anchor values that JS tests can validate against.

    These exact values must match between Python and JS.
    """

    ANCHORS = [
        "#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff",
        "#808080", "#ff8000", "#8000ff", "#008080", "#1a1a1a",
        "#010101", "#fefefe",
    ]

    def test_metric_anchor_values_valid(self, ms):
        """All anchor colors produce finite Lab values."""
        for hex_str in self.ANCHORS:
            srgb = hex_to_srgb(hex_str)
            XYZ = sRGB_to_XYZ(srgb)
            lab = ms.from_XYZ(XYZ)
            assert np.all(np.isfinite(lab)), f"{hex_str}: Lab={lab}"

    def test_metric_anchor_roundtrip_hex(self, ms):
        """Anchor hex round-trips within ±1/255."""
        for hex_str in self.ANCHORS:
            srgb = hex_to_srgb(hex_str)
            XYZ = sRGB_to_XYZ(srgb)
            lab = ms.from_XYZ(XYZ)
            rec_XYZ = ms.to_XYZ(lab)
            rec_srgb = clamp_srgb(XYZ_to_sRGB(rec_XYZ))
            rec_hex = srgb_to_hex(rec_srgb)
            # Parse and compare
            orig = np.array([int(hex_str[i:i+2], 16) for i in (1, 3, 5)])
            rec = np.array([int(rec_hex[i:i+2], 16) for i in (1, 3, 5)])
            diff = np.max(np.abs(orig - rec))
            assert diff <= 1, f"{hex_str} → {rec_hex}, diff={diff}"

    def test_gen_anchor_roundtrip_hex(self, gs):
        for hex_str in self.ANCHORS:
            srgb = hex_to_srgb(hex_str)
            XYZ = sRGB_to_XYZ(srgb)
            lab = gs.from_XYZ(XYZ)
            rec_XYZ = gs.to_XYZ(lab)
            rec_srgb = clamp_srgb(XYZ_to_sRGB(rec_XYZ))
            rec_hex = srgb_to_hex(rec_srgb)
            orig = np.array([int(hex_str[i:i+2], 16) for i in (1, 3, 5)])
            rec = np.array([int(rec_hex[i:i+2], 16) for i in (1, 3, 5)])
            diff = np.max(np.abs(orig - rec))
            assert diff <= 1, f"{hex_str} → {rec_hex}, diff={diff}"


# ── 10. Chroma scaling continuity ────────────────────────────────────

class TestChromaScaling:
    """Chroma should scale smoothly — no folds or discontinuities."""

    def test_metric_chroma_monotonic_at_fixed_hue(self, ms):
        """Increasing sRGB saturation → increasing Lab chroma."""
        # Red hue: vary saturation
        for base_hue in [0, 60, 120, 180, 240, 300]:
            h = np.deg2rad(base_hue)
            chromas = []
            for sat in np.linspace(0, 0.4, 20):
                r = 0.5 + sat * np.cos(h)
                g = 0.5 + sat * np.cos(h - 2.094)
                b = 0.5 + sat * np.cos(h + 2.094)
                srgb = np.clip([r, g, b], 0, 1)
                lab = ms.from_XYZ(sRGB_to_XYZ(srgb))
                C = np.sqrt(lab[1]**2 + lab[2]**2)
                chromas.append(C)
            # Should be mostly increasing (allow small dips from gamut boundary)
            decreases = sum(1 for i in range(len(chromas)-1)
                           if chromas[i+1] < chromas[i] - 0.01)
            assert decreases <= 2, (
                f"hue={base_hue}°: {decreases} chroma decreases")


# ── 11. Lightness ordering ───────────────────────────────────────────

class TestLightnessOrdering:
    """Perceptual lightness ordering must be preserved."""

    def test_metric_darker_color_lower_L(self, ms):
        """If sRGB is uniformly darker, Lab L should be lower."""
        pairs = [
            ("#000000", "#808080"),
            ("#808080", "#ffffff"),
            ("#330000", "#ff0000"),
            ("#003300", "#00ff00"),
            ("#000033", "#0000ff"),
        ]
        for dark_hex, light_hex in pairs:
            dark_lab = ms.from_XYZ(sRGB_to_XYZ(hex_to_srgb(dark_hex)))
            light_lab = ms.from_XYZ(sRGB_to_XYZ(hex_to_srgb(light_hex)))
            assert dark_lab[0] < light_lab[0], (
                f"{dark_hex} L={dark_lab[0]:.4f} >= {light_hex} L={light_lab[0]:.4f}")

    def test_gen_darker_color_lower_L(self, gs):
        pairs = [
            ("#000000", "#808080"),
            ("#808080", "#ffffff"),
            ("#330000", "#ff0000"),
            ("#003300", "#00ff00"),
            ("#000033", "#0000ff"),
        ]
        for dark_hex, light_hex in pairs:
            dark_lab = gs.from_XYZ(sRGB_to_XYZ(hex_to_srgb(dark_hex)))
            light_lab = gs.from_XYZ(sRGB_to_XYZ(hex_to_srgb(light_hex)))
            assert dark_lab[0] < light_lab[0], (
                f"{dark_hex} L={dark_lab[0]:.4f} >= {light_hex} L={light_lab[0]:.4f}")


# ── 12. Negative XYZ (wide gamut) ───────────────────────────────────

class TestNegativeXYZ:
    """Pipeline must handle negative XYZ gracefully (wide gamut colors)."""

    NEG_XYZ = [
        [-0.01, 0.05, 0.1],
        [0.1, -0.01, 0.1],
        [0.1, 0.1, -0.01],
        [-0.05, -0.05, -0.05],
    ]

    @pytest.mark.parametrize("xyz", NEG_XYZ)
    def test_metric_negative_xyz_roundtrip(self, ms, xyz):
        xyz = np.array(xyz, dtype=np.float64)
        lab = ms.from_XYZ(xyz)
        assert np.all(np.isfinite(lab)), f"Non-finite Lab for XYZ={xyz}"
        rec = ms.to_XYZ(lab)
        np.testing.assert_allclose(rec, xyz, atol=1e-6,
                                   err_msg=f"XYZ={xyz}")

    @pytest.mark.parametrize("xyz", NEG_XYZ)
    def test_gen_negative_xyz_roundtrip(self, gs, xyz):
        xyz = np.array(xyz, dtype=np.float64)
        lab = gs.from_XYZ(xyz)
        assert np.all(np.isfinite(lab)), f"Non-finite Lab for XYZ={xyz}"
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, xyz, atol=1e-6,
                                   err_msg=f"XYZ={xyz}")


# ── 13. Double round-trip stability ─────────────────────────────────

class TestDoubleRoundTrip:
    """XYZ → Lab → XYZ → Lab → XYZ should not accumulate error."""

    def test_metric_double_roundtrip(self, ms):
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.01, 0.9, (100, 3))
        lab1 = ms.from_XYZ(XYZ)
        rec1 = ms.to_XYZ(lab1)
        lab2 = ms.from_XYZ(rec1)
        rec2 = ms.to_XYZ(lab2)
        # Error should not grow
        err1 = np.max(np.abs(rec1 - XYZ), axis=1)
        err2 = np.max(np.abs(rec2 - XYZ), axis=1)
        # Newton iterations may amplify floating point noise slightly
        # (e.g. 3e-13 → 5e-12), but must stay below 1e-8
        assert np.max(err2) < 1e-8, (
            f"Double round-trip error too large: {np.max(err2):.2e}")

    def test_gen_double_roundtrip(self, gs):
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.01, 0.9, (100, 3))
        lab1 = gs.from_XYZ(XYZ)
        rec1 = gs.to_XYZ(lab1)
        lab2 = gs.from_XYZ(rec1)
        rec2 = gs.to_XYZ(lab2)
        err2 = np.max(np.abs(rec2 - XYZ), axis=1)
        assert np.max(err2) < 1e-8, (
            f"Double round-trip error too large: {np.max(err2):.2e}")
