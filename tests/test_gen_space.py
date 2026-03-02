"""Tests for GenSpace — generation-optimized color space."""

import numpy as np
import pytest

from helmlab.spaces.gen import GenSpace, GenParams
from helmlab.utils.srgb_convert import hex_to_srgb, srgb_to_hex, sRGB_to_XYZ, XYZ_to_sRGB, clamp_srgb
from helmlab.utils.gamut import gamut_map


class TestGenSpaceRoundtrip:
    """Forward → inverse roundtrip tests."""

    @pytest.fixture
    def gs(self):
        return GenSpace()

    def test_roundtrip_single(self, gs):
        """Single XYZ → Gen Lab → XYZ roundtrip."""
        XYZ = np.array([0.4, 0.3, 0.2])
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-10)

    def test_roundtrip_batch(self, gs):
        """Batch XYZ → Gen Lab → XYZ roundtrip."""
        rng = np.random.default_rng(42)
        XYZ = rng.uniform(0.05, 0.9, (100, 3))
        lab = gs.from_XYZ(XYZ)
        rec = gs.to_XYZ(lab)
        np.testing.assert_allclose(rec, XYZ, atol=1e-8)

    def test_roundtrip_extremes(self, gs):
        """Roundtrip for near-black and near-white."""
        D65 = np.array([0.95047, 1.0, 1.08883])
        for Y in [0.001, 0.01, 0.1, 0.5, 1.0, 1.5]:
            XYZ = D65 * Y
            lab = gs.from_XYZ(XYZ)
            rec = gs.to_XYZ(lab)
            np.testing.assert_allclose(rec, XYZ, atol=1e-8,
                                       err_msg=f"Y={Y} failed roundtrip")

    def test_roundtrip_srgb_colors(self, gs):
        """Roundtrip for named sRGB colors."""
        for hex_str in ["#ff0000", "#00ff00", "#0000ff", "#808080", "#ffffff", "#000000"]:
            srgb = hex_to_srgb(hex_str)
            XYZ = sRGB_to_XYZ(srgb)
            lab = gs.from_XYZ(XYZ)
            XYZ_rec = gs.to_XYZ(lab)
            np.testing.assert_allclose(XYZ_rec, XYZ, atol=1e-8,
                                       err_msg=f"{hex_str} failed roundtrip")


class TestGenSpaceAchromatic:
    """Achromatic axis tests — shared gamma should give perfect grays."""

    @pytest.fixture
    def gs(self):
        return GenSpace()

    def test_grays_zero_chroma(self, gs):
        """Grays should have a=b≈0 (structural guarantee from shared gamma + NC)."""
        D65 = np.array([0.95047, 1.0, 1.08883])
        for Y in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            XYZ = D65 * Y
            lab = gs.from_XYZ(XYZ)
            C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
            assert C < 1e-6, f"Y={Y}: chroma={C:.2e}, expected ≈0"

    def test_hex_grays_zero_chroma(self, gs):
        """Hex grays → Gen Lab should have near-zero chroma."""
        for hex_str in ["#000000", "#333333", "#808080", "#cccccc", "#ffffff"]:
            srgb = hex_to_srgb(hex_str)
            XYZ = sRGB_to_XYZ(srgb)
            lab = gs.from_XYZ(XYZ)
            C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
            assert C < 1e-4, f"{hex_str}: chroma={C:.6f}, expected ≈0"

    def test_lightness_monotonic(self, gs):
        """Lighter grays should have higher L."""
        D65 = np.array([0.95047, 1.0, 1.08883])
        Y_vals = np.linspace(0.01, 1.0, 20)
        Ls = []
        for Y in Y_vals:
            lab = gs.from_XYZ(D65 * Y)
            Ls.append(float(lab[0]))
        for i in range(len(Ls) - 1):
            assert Ls[i] < Ls[i + 1], f"L[{i}]={Ls[i]:.4f} >= L[{i+1}]={Ls[i+1]:.4f}"


class TestGenSpaceGradient:
    """Gradient quality tests — GenSpace should produce good gradients."""

    @pytest.fixture
    def gs(self):
        return GenSpace()

    def test_gradient_lightness_monotonic(self, gs):
        """Gradient from white to black: L should be monotonically decreasing."""
        D65 = np.array([0.95047, 1.0, 1.08883])
        white_lab = gs.from_XYZ(D65)
        black_lab = gs.from_XYZ(np.array([0.0, 0.0, 0.0]))
        steps = 16
        Ls = []
        for i in range(steps):
            t = i / (steps - 1)
            lab = white_lab + t * (black_lab - white_lab)
            Ls.append(float(lab[0]))
        for i in range(len(Ls) - 1):
            assert Ls[i] >= Ls[i + 1] - 0.001

    def test_gradient_no_hue_shift(self, gs):
        """Gradient between same hue: hue angle should be stable."""
        # Red at two lightness levels
        lab_light = gs.from_XYZ(sRGB_to_XYZ(np.array([0.9, 0.3, 0.3])))
        lab_dark = gs.from_XYZ(sRGB_to_XYZ(np.array([0.4, 0.1, 0.1])))
        h_ref = np.arctan2(lab_light[2], lab_light[1])
        steps = 10
        for i in range(steps):
            t = i / (steps - 1)
            lab_i = lab_light + t * (lab_dark - lab_light)
            h_i = np.arctan2(lab_i[2], lab_i[1])
            dh = abs(h_ref - h_i)
            dh = min(dh, 2 * np.pi - dh)
            assert dh < 0.3, f"step {i}: hue shift = {np.degrees(dh):.1f}°"


class TestGenSpaceNoEnrichment:
    """With default params (all enrichment=0), GenSpace should be Oklab-like."""

    @pytest.fixture
    def gs(self):
        return GenSpace()

    def test_no_brightness_fold(self, gs):
        """Orange palette should not have brightness fold (key GenSpace advantage)."""
        srgb = hex_to_srgb("#ff6b00")
        XYZ = sRGB_to_XYZ(srgb)
        lab = gs.from_XYZ(XYZ)
        # Sweep lightness
        Ls_out = []
        for L_target in np.linspace(0.7, 0.1, 10):
            sample = np.array([L_target, lab[1], lab[2]])
            mapped = gamut_map(sample, gs, gamut="srgb")
            XYZ_out = gs.to_XYZ(mapped)
            srgb_out = clamp_srgb(XYZ_to_sRGB(XYZ_out))
            lab_check = gs.from_XYZ(sRGB_to_XYZ(srgb_out))
            Ls_out.append(float(lab_check[0]))
        # Should be monotonically decreasing
        for i in range(len(Ls_out) - 1):
            assert Ls_out[i] >= Ls_out[i + 1] - 0.02, (
                f"Brightness fold at step {i}: L={Ls_out[i]:.3f} < L={Ls_out[i+1]:.3f}")


class TestGenParams:
    """Tests for GenParams serialization."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save → load preserves all params."""
        p = GenParams()
        p.hue_cos1 = 0.123
        p.lp_dark = 0.456
        path = tmp_path / "test_gen.json"
        p.save(path)
        p2 = GenParams.load(path)
        assert p2.hue_cos1 == pytest.approx(0.123)
        assert p2.lp_dark == pytest.approx(0.456)
        np.testing.assert_array_equal(p2.M1, p.M1)

    def test_from_dict_defaults(self):
        """from_dict with missing keys uses defaults."""
        d = {"M1": np.eye(3).tolist(), "gamma": [1/3, 1/3, 1/3], "M2": np.eye(3).tolist()}
        p = GenParams.from_dict(d)
        assert p.hue_cos1 == 0.0
        assert p.lp_dark == 0.0
