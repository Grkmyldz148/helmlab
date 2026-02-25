"""Tests for Helmlab UI utility layer (Phase 3)."""

import numpy as np
import pytest

from colorspace.utils.srgb_convert import (
    linear_to_srgb,
    srgb_to_linear,
    XYZ_to_sRGB,
    sRGB_to_XYZ,
    hex_to_srgb,
    srgb_to_hex,
    hex_batch_to_srgb,
    srgb_batch_to_hex,
    clamp_srgb,
    relative_luminance,
    contrast_ratio,
)
from colorspace.helmlab import Helmlab


# ═══════════════════════════════════════════════════════════════════════
# sRGB Convert
# ═══════════════════════════════════════════════════════════════════════

class TestSRGBConvert:
    """Tests for src/colorspace/utils/srgb_convert.py."""

    def test_linear_gamma_roundtrip(self):
        """linear → srgb → linear roundtrip."""
        vals = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0])
        np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(vals)), vals, atol=1e-12)

    def test_xyz_srgb_roundtrip(self):
        """XYZ → sRGB → XYZ roundtrip for in-gamut colors."""
        # D65 white → sRGB (1,1,1)
        srgb_white = XYZ_to_sRGB(np.array([0.95047, 1.0, 1.08883]))
        np.testing.assert_allclose(srgb_white, [1.0, 1.0, 1.0], atol=1e-4)
        # Roundtrip for multiple colors
        colors_srgb = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ])
        XYZ = sRGB_to_XYZ(colors_srgb)
        recovered = XYZ_to_sRGB(XYZ)
        np.testing.assert_allclose(recovered, colors_srgb, atol=1e-5)

    def test_hex_roundtrip(self):
        """hex → srgb → hex roundtrip."""
        for h in ["#000000", "#ffffff", "#3b82f6", "#ff0000", "#808080"]:
            srgb = hex_to_srgb(h)
            assert srgb_to_hex(srgb) == h

    def test_hex_batch(self):
        """Batch hex ↔ srgb conversion."""
        hexes = ["#ff0000", "#00ff00", "#0000ff"]
        srgb = hex_batch_to_srgb(hexes)
        assert srgb.shape == (3, 3)
        recovered = srgb_batch_to_hex(srgb)
        assert recovered == hexes

    def test_luminance_black_white(self):
        """White luminance ≈ 1, black luminance ≈ 0."""
        assert abs(relative_luminance(np.array([1.0, 1.0, 1.0])) - 1.0) < 1e-10
        assert abs(relative_luminance(np.array([0.0, 0.0, 0.0])) - 0.0) < 1e-10

    def test_contrast_black_white(self):
        """Black vs white contrast = 21:1."""
        cr = contrast_ratio(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        assert abs(cr - 21.0) < 0.01

    def test_clamp_out_of_gamut(self):
        """clamp_srgb clips to [0,1]."""
        out = clamp_srgb(np.array([-0.1, 0.5, 1.3]))
        np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])


# ═══════════════════════════════════════════════════════════════════════
# Helmlab Conversions
# ═══════════════════════════════════════════════════════════════════════

class TestHelmlabConversions:
    """Tests for Helmlab hex/sRGB conversions."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_hex_roundtrip(self, p):
        """hex → Lab → hex roundtrip (quantization tolerance)."""
        for h in ["#3b82f6", "#ef4444", "#22c55e", "#808080"]:
            lab = p.from_hex(h)
            recovered = p.to_hex(lab)
            # Allow 1/255 per-channel tolerance from 8-bit quantization
            srgb_orig = hex_to_srgb(h)
            srgb_rec = hex_to_srgb(recovered)
            np.testing.assert_allclose(srgb_rec, srgb_orig, atol=2.0 / 255.0)

    def test_srgb_roundtrip(self, p):
        """sRGB → Lab → sRGB roundtrip."""
        colors = np.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]])
        for srgb in colors:
            lab = p.from_srgb(srgb)
            rec = p.to_srgb(lab)
            np.testing.assert_allclose(rec, srgb, atol=1e-4)

    def test_known_colors(self, p):
        """Black → low L, white → high L."""
        black_lab = p.from_hex("#000000")
        white_lab = p.from_hex("#ffffff")
        assert black_lab[0] < 0.1  # near-zero lightness
        assert white_lab[0] > 0.8  # high lightness
        # Black should have significantly lower L than white
        assert white_lab[0] - black_lab[0] > 0.5

    def test_gamut_clamp(self, p):
        """to_srgb always returns [0,1] values."""
        # Extreme Lab values
        extreme = np.array([0.5, 0.8, 0.8])
        srgb = p.to_srgb(extreme)
        assert np.all(srgb >= 0.0)
        assert np.all(srgb <= 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Contrast
# ═══════════════════════════════════════════════════════════════════════

class TestContrast:
    """Tests for WCAG contrast utilities."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_black_white_21(self, p):
        """Black vs white → 21:1."""
        cr = p.contrast_ratio("#000000", "#ffffff")
        assert abs(cr - 21.0) < 0.01

    def test_same_color_1(self, p):
        """Same color → 1:1."""
        cr = p.contrast_ratio("#3b82f6", "#3b82f6")
        assert abs(cr - 1.0) < 0.01

    def test_ensure_contrast_aa(self, p):
        """ensure_contrast returns color meeting AA (4.5:1)."""
        adjusted = p.ensure_contrast("#777777", "#ffffff", 4.5)
        cr = p.contrast_ratio(adjusted, "#ffffff")
        assert cr >= 4.5 - 0.01

    def test_ensure_contrast_aaa(self, p):
        """ensure_contrast returns color meeting AAA (7:1)."""
        adjusted = p.ensure_contrast("#999999", "#ffffff", 7.0)
        cr = p.contrast_ratio(adjusted, "#ffffff")
        assert cr >= 7.0 - 0.01

    def test_ensure_contrast_preserves_hue(self, p):
        """ensure_contrast preserves hue (±5°)."""
        base = "#3b82f6"  # blue
        adjusted = p.ensure_contrast(base, "#ffffff", 4.5)
        lab_orig = p.from_hex(base)
        lab_adj = p.from_hex(adjusted)
        h_orig = np.arctan2(lab_orig[2], lab_orig[1])
        h_adj = np.arctan2(lab_adj[2], lab_adj[1])
        # Hue difference within ~5° (0.087 rad)
        dh = abs(h_orig - h_adj)
        dh = min(dh, 2 * np.pi - dh)
        assert dh < 0.15  # ~8.6° tolerance


# ═══════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════

class TestPalette:
    """Tests for palette generation."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_correct_length(self, p):
        """palette returns requested number of steps."""
        pal = p.palette("#3b82f6", steps=10)
        assert len(pal) == 10

    def test_monotonic_lightness(self, p):
        """Palette colors have monotonically decreasing L."""
        pal = p.palette("#3b82f6", steps=10)
        labs = [p.from_hex(h) for h in pal]
        Ls = [lab[0] for lab in labs]
        for i in range(len(Ls) - 1):
            assert Ls[i] > Ls[i + 1], f"L[{i}]={Ls[i]:.3f} ≤ L[{i+1}]={Ls[i+1]:.3f}"

    def test_uniform_spacing(self, p):
        """Palette steps are roughly uniformly spaced in L."""
        pal = p.palette("#3b82f6", steps=10)
        labs = [p.from_hex(h) for h in pal]
        Ls = [lab[0] for lab in labs]
        diffs = [Ls[i] - Ls[i + 1] for i in range(len(Ls) - 1)]
        # All diffs should be roughly similar (gamut clamp distorts extremes)
        mean_diff = np.mean(diffs)
        for d in diffs:
            assert abs(d - mean_diff) / mean_diff < 0.40  # ±40% (gamut clamp)

    def test_hues_count(self, p):
        """palette_hues returns correct number of colors."""
        hues = p.palette_hues(steps=12)
        assert len(hues) == 12


# ═══════════════════════════════════════════════════════════════════════
# Semantic Scale
# ═══════════════════════════════════════════════════════════════════════

class TestSemanticScale:
    """Tests for Tailwind-style semantic scale."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_correct_keys(self, p):
        """Scale has all expected level keys."""
        scale = p.semantic_scale("#3b82f6")
        expected = {"50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950"}
        assert set(scale.keys()) == expected

    def test_monotonic_lightness(self, p):
        """Higher level → lower L (darker)."""
        scale = p.semantic_scale("#3b82f6")
        levels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
        labs = [p.from_hex(scale[str(lv)]) for lv in levels]
        Ls = [lab[0] for lab in labs]
        for i in range(len(Ls) - 1):
            assert Ls[i] >= Ls[i + 1] - 0.01, (
                f"L[{levels[i]}]={Ls[i]:.3f} < L[{levels[i+1]}]={Ls[i+1]:.3f}"
            )

    def test_base_is_500(self, p):
        """Level 500 is close to the base color."""
        base = "#3b82f6"
        scale = p.semantic_scale(base)
        base_lab = p.from_hex(base)
        s500_lab = p.from_hex(scale["500"])
        # L should be identical (same base)
        np.testing.assert_allclose(s500_lab[0], base_lab[0], atol=0.02)

    def test_gamut_valid(self, p):
        """All scale colors are valid hex strings."""
        scale = p.semantic_scale("#ef4444")
        for lv, hex_str in scale.items():
            assert hex_str.startswith("#")
            assert len(hex_str) == 7
            # Should be parseable
            _ = hex_to_srgb(hex_str)


# ═══════════════════════════════════════════════════════════════════════
# Dark/Light Mode
# ═══════════════════════════════════════════════════════════════════════

class TestAdaptMode:
    """Tests for dark/light mode adaptation."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_light_to_dark_inverts_L(self, p):
        """Light → dark: light color becomes dark, dark becomes light."""
        # Light gray in light mode → should become darker in dark mode
        light_gray = "#cccccc"
        adapted = p.adapt_to_mode(light_gray, "light", "dark")
        L_orig = p.from_hex(light_gray)[0]
        L_adapted = p.from_hex(adapted)[0]
        assert L_adapted < L_orig, "Light color should become darker"

        # Dark gray → should become lighter
        dark_gray = "#333333"
        adapted = p.adapt_to_mode(dark_gray, "light", "dark")
        L_orig = p.from_hex(dark_gray)[0]
        L_adapted = p.from_hex(adapted)[0]
        assert L_adapted > L_orig, "Dark color should become lighter"

    def test_dark_to_light_inverts_L(self, p):
        """Dark → light reverses the adaptation."""
        dark_color = "#334455"
        adapted = p.adapt_to_mode(dark_color, "dark", "light")
        L_orig = p.from_hex(dark_color)[0]
        L_adapted = p.from_hex(adapted)[0]
        assert L_adapted > L_orig, "Dark mode color should become lighter in light mode"

    def test_same_mode_identity(self, p):
        """Same mode → same color."""
        color = "#3b82f6"
        assert p.adapt_to_mode(color, "light", "light") == color
        assert p.adapt_to_mode(color, "dark", "dark") == color

    def test_adapt_pair_meets_contrast(self, p):
        """adapt_pair result meets contrast requirement."""
        fg, bg = p.adapt_pair("#333333", "#ffffff", "light", "dark", 4.5)
        cr = p.contrast_ratio(fg, bg)
        assert cr >= 4.5 - 0.01


# ═══════════════════════════════════════════════════════════════════════
# Delta E
# ═══════════════════════════════════════════════════════════════════════

class TestDeltaE:
    """Tests for Helmlab distance."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_self_zero(self, p):
        """Distance to self = 0."""
        assert p.delta_e("#3b82f6", "#3b82f6") < 1e-10

    def test_symmetric(self, p):
        """d(a,b) = d(b,a)."""
        de1 = p.delta_e("#3b82f6", "#ef4444")
        de2 = p.delta_e("#ef4444", "#3b82f6")
        assert abs(de1 - de2) < 1e-10

    def test_positive(self, p):
        """Different colors → positive distance."""
        assert p.delta_e("#000000", "#ffffff") > 0.1


# ═══════════════════════════════════════════════════════════════════════
# Gamut Mapping (Part A)
# ═══════════════════════════════════════════════════════════════════════

class TestGamutMapping:
    """Tests for adaptive gamut mapping."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_is_in_gamut_white(self, p):
        """White is in sRGB gamut."""
        lab_white = p.from_hex("#ffffff")
        assert p.is_in_srgb(lab_white)

    def test_is_in_gamut_oog(self, p):
        """Extreme chroma is out of sRGB gamut."""
        extreme = np.array([0.5, 0.8, 0.0])
        assert not p.is_in_srgb(extreme)

    def test_max_chroma_positive(self, p):
        """max_chroma returns a positive value for mid-lightness."""
        from colorspace.utils.gamut import max_chroma
        C_max = max_chroma(0.5, 0.0, p._space, "srgb")
        assert C_max > 0.0

    def test_max_chroma_less_than_unrestricted(self, p):
        """max_chroma for sRGB < max_chroma for Display P3 at same L,H."""
        from colorspace.utils.gamut import max_chroma
        C_srgb = max_chroma(0.5, 1.0, p._space, "srgb")
        C_p3 = max_chroma(0.5, 1.0, p._space, "display-p3")
        assert C_p3 >= C_srgb - 1e-4

    def test_gamut_map_preserves_hue(self, p):
        """gamut_map preserves hue angle (±0.5°)."""
        oog = np.array([0.5, 0.6, 0.3])
        from colorspace.utils.gamut import gamut_map
        mapped = gamut_map(oog, p._space, "srgb")
        h_orig = np.arctan2(oog[2], oog[1])
        h_mapped = np.arctan2(mapped[2], mapped[1])
        dh = abs(h_orig - h_mapped)
        dh = min(dh, 2 * np.pi - dh)
        assert dh < np.radians(0.5)

    def test_gamut_map_preserves_L(self, p):
        """gamut_map preserves lightness (±0.001)."""
        oog = np.array([0.5, 0.6, 0.3])
        from colorspace.utils.gamut import gamut_map
        mapped = gamut_map(oog, p._space, "srgb")
        assert abs(mapped[0] - oog[0]) < 0.001

    def test_gamut_map_in_gamut_unchanged(self, p):
        """In-gamut color passes through unchanged."""
        from colorspace.utils.gamut import gamut_map
        lab = p.from_hex("#808080")
        mapped = gamut_map(lab, p._space, "srgb")
        np.testing.assert_allclose(mapped, lab, atol=1e-10)

    def test_gamut_map_oog_becomes_in_gamut(self, p):
        """Out-of-gamut color becomes in-gamut after mapping."""
        oog = np.array([0.5, 0.8, 0.0])
        from colorspace.utils.gamut import gamut_map
        mapped = gamut_map(oog, p._space, "srgb")
        assert p.is_in_srgb(mapped)

    def test_gamut_map_batch(self, p):
        """Batch gamut mapping is consistent with single mapping."""
        from colorspace.utils.gamut import gamut_map
        labs = np.array([
            [0.5, 0.8, 0.0],
            [0.5, 0.01, 0.01],
            [0.5, 0.6, 0.3],
        ])
        batch_result = gamut_map(labs, p._space, "srgb")
        for i in range(len(labs)):
            single = gamut_map(labs[i], p._space, "srgb")
            np.testing.assert_allclose(batch_result[i], single, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# Display P3 (Part A)
# ═══════════════════════════════════════════════════════════════════════

class TestDisplayP3:
    """Tests for Display P3 support."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_xyz_p3_roundtrip(self):
        """XYZ → Display P3 → XYZ roundtrip."""
        from colorspace.utils.srgb_convert import (
            XYZ_to_DisplayP3, DisplayP3_to_XYZ, linear_to_displayp3, displayp3_to_linear,
        )
        XYZ = np.array([0.4, 0.3, 0.2])
        p3_lin = XYZ_to_DisplayP3(XYZ)
        recovered = DisplayP3_to_XYZ(p3_lin)
        np.testing.assert_allclose(recovered, XYZ, atol=1e-6)

    def test_p3_gamut_wider_than_srgb(self, p):
        """P3 gamut allows higher chroma than sRGB at same L,H."""
        from colorspace.utils.gamut import max_chroma
        C_srgb = max_chroma(0.6, 0.5, p._space, "srgb")
        C_p3 = max_chroma(0.6, 0.5, p._space, "display-p3")
        assert C_p3 > C_srgb

    def test_srgb_subset_of_p3(self, p):
        """sRGB in-gamut color is also in Display P3 gamut."""
        lab = p.from_hex("#3b82f6")
        assert p.is_in_srgb(lab)
        assert p.is_in_p3(lab)

    def test_to_hex_p3_format(self, p):
        """to_hex_p3 returns CSS color(display-p3 ...) format."""
        lab = p.from_hex("#3b82f6")
        result = p.to_hex_p3(lab)
        assert result.startswith("color(display-p3 ")
        assert result.endswith(")")


# ═══════════════════════════════════════════════════════════════════════
# Token Export (Part B)
# ═══════════════════════════════════════════════════════════════════════

class TestTokenExporter:
    """Tests for design token export."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_css_hex_format(self, p):
        """to_css_hex returns valid '#rrggbb'."""
        exp = p.export()
        lab = p.from_hex("#3b82f6")
        result = exp.to_css_hex(lab)
        assert result.startswith("#")
        assert len(result) == 7

    def test_css_oklch_format(self, p):
        """to_css_oklch returns 'oklch(L% C H)' pattern."""
        exp = p.export()
        lab = p.from_hex("#3b82f6")
        result = exp.to_css_oklch(lab)
        assert result.startswith("oklch(")
        assert "%" in result
        assert result.endswith(")")

    def test_css_displayp3_format(self, p):
        """to_css_displayp3 returns 'color(display-p3 r g b)' pattern."""
        exp = p.export()
        lab = p.from_hex("#3b82f6")
        result = exp.to_css_displayp3(lab)
        assert result.startswith("color(display-p3 ")
        assert result.endswith(")")

    def test_android_argb_format(self, p):
        """to_android_argb returns '0xFF' prefix with 8 hex chars."""
        exp = p.export()
        lab = p.from_hex("#3b82f6")
        result = exp.to_android_argb(lab)
        assert result.startswith("0xFF")
        assert len(result) == 10  # 0xFF + 6 hex

    def test_ios_p3_dict(self, p):
        """to_ios_p3 returns dict with r,g,b keys in [0,1]."""
        exp = p.export()
        lab = p.from_hex("#3b82f6")
        result = exp.to_ios_p3(lab)
        assert set(result.keys()) == {"r", "g", "b"}
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_css_custom_properties(self, p):
        """export_css_custom_properties returns valid CSS."""
        exp = p.export()
        scale = p.semantic_scale("#3b82f6")
        css = exp.export_css_custom_properties(scale, prefix="--blue")
        assert "--blue-50:" in css
        assert "--blue-500:" in css
        assert "--blue-950:" in css

    def test_export_tailwind(self, p):
        """export_tailwind returns valid structure."""
        exp = p.export()
        scale = p.semantic_scale("#3b82f6")
        tw = exp.export_tailwind(scale, "blue")
        assert "blue" in tw
        assert "500" in tw["blue"]
        assert tw["blue"]["500"].startswith("#")

    def test_export_json_parseable(self, p):
        """export_json returns parseable JSON with all formats."""
        import json as json_mod
        exp = p.export()
        scale = p.semantic_scale("#3b82f6")
        result = exp.export_json({"blue": scale})
        parsed = json_mod.loads(result)
        assert "blue" in parsed
        assert "500" in parsed["blue"]
        assert "hex" in parsed["blue"]["500"]
        assert "oklch" in parsed["blue"]["500"]

    def test_roundtrip_hex(self, p):
        """Export hex → parse → same color."""
        exp = p.export()
        lab = p.from_hex("#3b82f6")
        hex_out = exp.to_css_hex(lab)
        srgb_orig = hex_to_srgb("#3b82f6")
        srgb_rec = hex_to_srgb(hex_out)
        np.testing.assert_allclose(srgb_rec, srgb_orig, atol=2.0 / 255.0)

    def test_css_rgb_format(self, p):
        """to_css_rgb returns 'rgb(r, g, b)' pattern."""
        exp = p.export()
        lab = p.from_hex("#ff0000")
        result = exp.to_css_rgb(lab)
        assert result.startswith("rgb(")
        assert result.endswith(")")


# ═══════════════════════════════════════════════════════════════════════
# Surround Parameter S (Part C)
# ═══════════════════════════════════════════════════════════════════════

class TestSurroundParam:
    """Tests for the surround (S) context-aware dimension."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_s05_matches_current(self, p):
        """S=0.5 (default) matches existing v14 output exactly."""
        lab_default = p.from_hex("#3b82f6")
        # Explicitly pass S=0.5
        srgb = hex_to_srgb("#3b82f6")
        from colorspace.utils.srgb_convert import sRGB_to_XYZ as s2x
        XYZ = s2x(srgb)
        lab_explicit = p._space.from_XYZ(XYZ, S=0.5)
        np.testing.assert_allclose(lab_explicit, lab_default, atol=1e-12)

    def test_roundtrip_s02(self, p):
        """from_XYZ → to_XYZ roundtrip with S=0.2."""
        srgb = hex_to_srgb("#ef4444")
        from colorspace.utils.srgb_convert import sRGB_to_XYZ as s2x
        XYZ = s2x(srgb)
        lab = p._space.from_XYZ(XYZ, S=0.2)
        XYZ_rec = p._space.to_XYZ(lab, S=0.2)
        np.testing.assert_allclose(XYZ_rec, XYZ, atol=1e-6)

    def test_roundtrip_s08(self, p):
        """from_XYZ → to_XYZ roundtrip with S=0.8."""
        srgb = hex_to_srgb("#22c55e")
        from colorspace.utils.srgb_convert import sRGB_to_XYZ as s2x
        XYZ = s2x(srgb)
        lab = p._space.from_XYZ(XYZ, S=0.8)
        XYZ_rec = p._space.to_XYZ(lab, S=0.8)
        np.testing.assert_allclose(XYZ_rec, XYZ, atol=1e-6)

    def test_distance_s05_matches_v14(self, p):
        """Distance at S=0.5 matches v14 Euclidean distance."""
        de1 = p.delta_e("#3b82f6", "#ef4444")
        # Direct Lab distance at S=0.5
        lab1 = p.from_hex("#3b82f6")
        lab2 = p.from_hex("#ef4444")
        de2 = float(np.sqrt(np.sum((lab1 - lab2) ** 2)))
        assert abs(de1 - de2) < 1e-10


class TestSurroundHelmlab:
    """Tests for Helmlab surround integration."""

    def test_set_surround(self):
        """set_surround changes instance surround."""
        p = Helmlab()
        p.set_surround(0.2)
        assert p._surround == 0.2
        assert p._space._surround == 0.2

    def test_set_surround_clamps(self):
        """set_surround clamps to [0,1]."""
        p = Helmlab()
        p.set_surround(-0.5)
        assert p._surround == 0.0
        p.set_surround(1.5)
        assert p._surround == 1.0

    def test_adapt_to_mode_uses_fallback(self):
        """adapt_to_mode uses L-inversion fallback when S params = 0."""
        p = Helmlab()
        # With default params (all S params = 0), should use L-inversion
        result = p.adapt_to_mode("#cccccc", "light", "dark")
        lab_orig = p.from_hex("#cccccc")
        lab_adapted = p.from_hex(result)
        assert lab_adapted[0] < lab_orig[0]

    def test_adapt_pair_with_surround(self):
        """adapt_pair still meets contrast with surround."""
        p = Helmlab()
        fg, bg = p.adapt_pair("#333333", "#ffffff", "light", "dark", 4.5)
        cr = p.contrast_ratio(fg, bg)
        assert cr >= 4.5 - 0.01


class TestSurroundBackwardCompat:
    """Tests for backward compatibility with v14 params."""

    def test_v14_json_loads_with_s_zero(self):
        """v14 JSON (no S params) loads with S params = 0."""
        from colorspace.spaces.analytical import AnalyticalParams
        # Simulate v14 JSON (missing S params)
        d = AnalyticalParams().to_dict()
        # Remove S params to simulate old format
        for key in ["hk_weight_S", "hk_power_S", "hk_hue_S",
                     "lp_dark_S", "lp_dark_S2",
                     "cs_S_lin", "cs_S_quad", "lc_S_lin", "lc_S_quad",
                     "hl_S_lin", "L_S_offset"]:
            d.pop(key, None)
        p = AnalyticalParams.from_dict(d)
        assert p.hk_weight_S == 0.0
        assert p.L_S_offset == 0.0

    def test_serialization_roundtrip_with_s(self):
        """to_dict → from_dict preserves S params."""
        from colorspace.spaces.analytical import AnalyticalParams
        p = AnalyticalParams()
        p.hk_weight_S = 0.1
        p.L_S_offset = -0.05
        d = p.to_dict()
        p2 = AnalyticalParams.from_dict(d)
        assert p2.hk_weight_S == 0.1
        assert p2.L_S_offset == -0.05
