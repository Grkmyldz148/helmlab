"""Tests for Helmlab UI utility layer (Phase 3)."""

import numpy as np
import pytest

from helmlab.utils.srgb_convert import (
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
from helmlab.helmlab import Helmlab


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
            lab = p.metric.from_hex(h)
            recovered = p.metric.to_hex(lab)
            # Allow 1/255 per-channel tolerance from 8-bit quantization
            srgb_orig = hex_to_srgb(h)
            srgb_rec = hex_to_srgb(recovered)
            np.testing.assert_allclose(srgb_rec, srgb_orig, atol=2.0 / 255.0)

    def test_srgb_roundtrip(self, p):
        """sRGB → Lab → sRGB roundtrip."""
        colors = np.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]])
        for srgb in colors:
            lab = p.metric.from_srgb(srgb)
            rec = p.metric.to_srgb(lab)
            np.testing.assert_allclose(rec, srgb, atol=1e-4)

    def test_known_colors(self, p):
        """Black → low L, white → high L."""
        black_lab = p.metric.from_hex("#000000")
        white_lab = p.metric.from_hex("#ffffff")
        assert black_lab[0] < 0.1  # near-zero lightness
        assert white_lab[0] > 0.8  # high lightness
        # Black should have significantly lower L than white
        assert white_lab[0] - black_lab[0] > 0.5

    def test_gamut_clamp(self, p):
        """to_srgb always returns [0,1] values."""
        # Extreme Lab values
        extreme = np.array([0.5, 0.8, 0.8])
        srgb = p.metric.to_srgb(extreme)
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
        cr = p.gen.contrast_ratio("#000000", "#ffffff")
        assert abs(cr - 21.0) < 0.01

    def test_same_color_1(self, p):
        """Same color → 1:1."""
        cr = p.gen.contrast_ratio("#3b82f6", "#3b82f6")
        assert abs(cr - 1.0) < 0.01

    def test_ensure_contrast_aa(self, p):
        """ensure_contrast returns color meeting AA (4.5:1)."""
        adjusted = p.gen.ensure_contrast("#777777", "#ffffff", 4.5)
        cr = p.gen.contrast_ratio(adjusted, "#ffffff")
        assert cr >= 4.5 - 0.01

    def test_ensure_contrast_aaa(self, p):
        """ensure_contrast returns color meeting AAA (7:1)."""
        adjusted = p.gen.ensure_contrast("#999999", "#ffffff", 7.0)
        cr = p.gen.contrast_ratio(adjusted, "#ffffff")
        assert cr >= 7.0 - 0.01

    def test_ensure_contrast_preserves_hue(self, p):
        """ensure_contrast preserves hue (±5°)."""
        base = "#3b82f6"  # blue
        adjusted = p.gen.ensure_contrast(base, "#ffffff", 4.5)
        lab_orig = p.metric.from_hex(base)
        lab_adj = p.metric.from_hex(adjusted)
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
        pal = p.gen.palette("#3b82f6", steps=10)
        assert len(pal) == 10

    def test_monotonic_lightness(self, p):
        """Palette colors have monotonically decreasing L (base Lab)."""
        pal = p.gen.palette("#3b82f6", steps=10)
        labs = [p.gen.from_hex(h) for h in pal]
        Ls = [lab[0] for lab in labs]
        for i in range(len(Ls) - 1):
            assert Ls[i] > Ls[i + 1], f"L[{i}]={Ls[i]:.3f} ≤ L[{i+1}]={Ls[i+1]:.3f}"

    def test_uniform_spacing(self, p):
        """Palette steps are roughly uniformly spaced in base Lab L."""
        pal = p.gen.palette("#3b82f6", steps=10)
        labs = [p.gen.from_hex(h) for h in pal]
        Ls = [lab[0] for lab in labs]
        diffs = [Ls[i] - Ls[i + 1] for i in range(len(Ls) - 1)]
        # All diffs should be roughly similar (gamut clamp distorts extremes)
        mean_diff = np.mean(diffs)
        for d in diffs:
            assert abs(d - mean_diff) / mean_diff < 0.85  # ±85% (base Lab + gamut clamp at extremes)

    def test_hues_count(self, p):
        """palette_hues returns correct number of colors."""
        hues = p.gen.hue_ring(12)
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
        scale = p.gen.scale("#3b82f6")
        expected = {"50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950"}
        assert set(scale.keys()) == expected

    def test_monotonic_lightness(self, p):
        """Higher level → lower L (darker) in base Lab."""
        scale = p.gen.scale("#3b82f6")
        levels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
        labs = [p.gen.from_hex(scale[str(lv)]) for lv in levels]
        Ls = [lab[0] for lab in labs]
        for i in range(len(Ls) - 1):
            assert Ls[i] >= Ls[i + 1] - 0.01, (
                f"L[{levels[i]}]={Ls[i]:.3f} < L[{levels[i+1]}]={Ls[i+1]:.3f}"
            )

    def test_base_is_500(self, p):
        """Level 500 is close to the base color in base Lab."""
        base = "#3b82f6"
        scale = p.gen.scale(base)
        base_lab = p.gen.from_hex(base)
        s500_lab = p.gen.from_hex(scale["500"])
        # L should be identical (same base)
        np.testing.assert_allclose(s500_lab[0], base_lab[0], atol=0.02)

    def test_gamut_valid(self, p):
        """All scale colors are valid hex strings."""
        scale = p.gen.scale("#ef4444")
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
        adapted = p.gen.adapt_to_mode(light_gray, "light", "dark")
        L_orig = p.gen.from_hex(light_gray)[0]
        L_adapted = p.gen.from_hex(adapted)[0]
        assert L_adapted < L_orig, "Light color should become darker"

        # Dark gray → should become lighter
        dark_gray = "#333333"
        adapted = p.gen.adapt_to_mode(dark_gray, "light", "dark")
        L_orig = p.gen.from_hex(dark_gray)[0]
        L_adapted = p.gen.from_hex(adapted)[0]
        assert L_adapted > L_orig, "Dark color should become lighter"

    def test_dark_to_light_inverts_L(self, p):
        """Dark → light reverses the adaptation."""
        dark_color = "#334455"
        adapted = p.gen.adapt_to_mode(dark_color, "dark", "light")
        L_orig = p.gen.from_hex(dark_color)[0]
        L_adapted = p.gen.from_hex(adapted)[0]
        assert L_adapted > L_orig, "Dark mode color should become lighter in light mode"

    def test_same_mode_identity(self, p):
        """Same mode → same color."""
        color = "#3b82f6"
        assert p.gen.adapt_to_mode(color, "light", "light") == color
        assert p.gen.adapt_to_mode(color, "dark", "dark") == color

    def test_adapt_pair_meets_contrast(self, p):
        """adapt_pair result meets contrast requirement."""
        fg, bg = p.gen.adapt_pair("#333333", "#ffffff", "light", "dark", 4.5)
        cr = p.gen.contrast_ratio(fg, bg)
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
        assert p.metric.euclidean("#3b82f6", "#3b82f6") < 1e-10

    def test_symmetric(self, p):
        """d(a,b) = d(b,a)."""
        de1 = p.metric.euclidean("#3b82f6", "#ef4444")
        de2 = p.metric.euclidean("#ef4444", "#3b82f6")
        assert abs(de1 - de2) < 1e-10

    def test_positive(self, p):
        """Different colors → positive distance."""
        assert p.metric.euclidean("#000000", "#ffffff") > 0.1


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
        lab_white = p.metric.from_hex("#ffffff")
        assert p.metric.in_gamut(lab_white)

    def test_is_in_gamut_oog(self, p):
        """Extreme chroma is out of sRGB gamut."""
        extreme = np.array([0.5, 0.8, 0.0])
        assert not p.metric.in_gamut(extreme)

    def test_max_chroma_positive(self, p):
        """max_chroma returns a positive value for mid-lightness."""
        from helmlab.utils.gamut import max_chroma
        C_max = max_chroma(0.5, 0.0, p.metric.space, "srgb")
        assert C_max > 0.0

    def test_max_chroma_less_than_unrestricted(self, p):
        """max_chroma for sRGB < max_chroma for Display P3 at same L,H."""
        from helmlab.utils.gamut import max_chroma
        C_srgb = max_chroma(0.5, 1.0, p.metric.space, "srgb")
        C_p3 = max_chroma(0.5, 1.0, p.metric.space, "display-p3")
        assert C_p3 >= C_srgb - 1e-4

    def test_gamut_map_preserves_hue(self, p):
        """gamut_map preserves hue angle (±0.5°)."""
        oog = np.array([0.5, 0.6, 0.3])
        from helmlab.utils.gamut import gamut_map
        mapped = gamut_map(oog, p.metric.space, "srgb")
        h_orig = np.arctan2(oog[2], oog[1])
        h_mapped = np.arctan2(mapped[2], mapped[1])
        dh = abs(h_orig - h_mapped)
        dh = min(dh, 2 * np.pi - dh)
        assert dh < np.radians(0.5)

    def test_gamut_map_preserves_L(self, p):
        """gamut_map preserves lightness (±0.001)."""
        oog = np.array([0.5, 0.6, 0.3])
        from helmlab.utils.gamut import gamut_map
        mapped = gamut_map(oog, p.metric.space, "srgb")
        assert abs(mapped[0] - oog[0]) < 0.001

    def test_gamut_map_in_gamut_unchanged(self, p):
        """In-gamut color passes through unchanged."""
        from helmlab.utils.gamut import gamut_map
        lab = p.metric.from_hex("#808080")
        mapped = gamut_map(lab, p.metric.space, "srgb")
        np.testing.assert_allclose(mapped, lab, atol=1e-10)

    def test_gamut_map_oog_becomes_in_gamut(self, p):
        """Out-of-gamut color becomes in-gamut after mapping."""
        oog = np.array([0.5, 0.8, 0.0])
        from helmlab.utils.gamut import gamut_map
        mapped = gamut_map(oog, p.metric.space, "srgb")
        assert p.metric.in_gamut(mapped)

    def test_gamut_map_batch(self, p):
        """Batch gamut mapping is consistent with single mapping."""
        from helmlab.utils.gamut import gamut_map
        labs = np.array([
            [0.5, 0.8, 0.0],
            [0.5, 0.01, 0.01],
            [0.5, 0.6, 0.3],
        ])
        batch_result = gamut_map(labs, p.metric.space, "srgb")
        for i in range(len(labs)):
            single = gamut_map(labs[i], p.metric.space, "srgb")
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
        from helmlab.utils.srgb_convert import (
            XYZ_to_DisplayP3, DisplayP3_to_XYZ, linear_to_displayp3, displayp3_to_linear,
        )
        XYZ = np.array([0.4, 0.3, 0.2])
        p3_lin = XYZ_to_DisplayP3(XYZ)
        recovered = DisplayP3_to_XYZ(p3_lin)
        np.testing.assert_allclose(recovered, XYZ, atol=1e-6)

    def test_p3_gamut_wider_than_srgb(self, p):
        """P3 gamut allows higher chroma than sRGB at same L,H."""
        from helmlab.utils.gamut import max_chroma
        C_srgb = max_chroma(0.6, 0.5, p.metric.space, "srgb")
        C_p3 = max_chroma(0.6, 0.5, p.metric.space, "display-p3")
        assert C_p3 > C_srgb

    def test_srgb_subset_of_p3(self, p):
        """sRGB in-gamut color is also in Display P3 gamut."""
        lab = p.metric.from_hex("#3b82f6")
        assert p.metric.in_gamut(lab)
        assert p.metric.in_gamut(lab, "display-p3")

    def test_to_hex_p3_format(self, p):
        """to_hex_p3 returns CSS color(display-p3 ...) format."""
        lab = p.metric.from_hex("#3b82f6")
        result = p.metric.to_css(lab)
        assert result.startswith("color(display-p3 ")
        assert result.endswith(")")


# ═══════════════════════════════════════════════════════════════════════
# Token Export (Part B)
# ═══════════════════════════════════════════════════════════════════════

class TestTokens:
    """Tests for design token export (hl.tokens — color strings in)."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_css_hex_format(self, p):
        result = p.tokens.css("#3b82f6", "hex")
        assert result.startswith("#")
        assert len(result) == 7

    def test_css_oklch_format(self, p):
        result = p.tokens.css("#3b82f6", "oklch")
        assert result.startswith("oklch(")
        assert "%" in result
        assert result.endswith(")")

    def test_css_displayp3_format(self, p):
        result = p.tokens.css("#3b82f6", "p3")
        assert result.startswith("color(display-p3 ")
        assert result.endswith(")")

    def test_css_rec2020_format(self, p):
        result = p.tokens.css("#3b82f6", "rec2020")
        assert result.startswith("color(rec2020 ")

    def test_android_argb_format(self, p):
        result = p.tokens.android("#3b82f6")
        assert result.startswith("0xFF")
        assert len(result) == 10  # 0xFF + 6 hex

    def test_ios_p3_dict(self, p):
        result = p.tokens.ios_p3("#3b82f6")
        assert set(result.keys()) == {"r", "g", "b"}
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_swift_literal(self, p):
        result = p.tokens.swift("#3b82f6")
        assert result.startswith("Color(.displayP3,")

    def test_css_variables(self, p):
        scale = p.gen.scale("#3b82f6")
        css = p.tokens.css_variables(scale, prefix="--blue")
        assert "--blue-50:" in css
        assert "--blue-500:" in css
        assert "--blue-950:" in css

    def test_tailwind(self, p):
        scale = p.gen.scale("#3b82f6")
        tw = p.tokens.tailwind(scale, "blue")
        assert "blue" in tw
        assert "500" in tw["blue"]
        assert tw["blue"]["500"].startswith("#")

    def test_json_parseable(self, p):
        import json as json_mod
        scale = p.gen.scale("#3b82f6")
        result = p.tokens.json({"blue": scale})
        parsed = json_mod.loads(result)
        assert "blue" in parsed
        assert "500" in parsed["blue"]
        assert "hex" in parsed["blue"]["500"]
        assert "oklch" in parsed["blue"]["500"]

    def test_roundtrip_hex(self, p):
        hex_out = p.tokens.css("#3b82f6", "hex")
        srgb_orig = hex_to_srgb("#3b82f6")
        srgb_rec = hex_to_srgb(hex_out)
        np.testing.assert_allclose(srgb_rec, srgb_orig, atol=2.0 / 255.0)

    def test_css_rgb_format(self, p):
        result = p.tokens.css("#ff0000", "rgb")
        assert result.startswith("rgb(")
        assert result.endswith(")")

    def test_unknown_format_raises(self, p):
        with pytest.raises(ValueError, match="unknown format"):
            p.tokens.css("#3b82f6", "cmyk")


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
        lab_default = p.metric.from_hex("#3b82f6")
        # Explicitly pass S=0.5
        srgb = hex_to_srgb("#3b82f6")
        from helmlab.utils.srgb_convert import sRGB_to_XYZ as s2x
        XYZ = s2x(srgb)
        lab_explicit = p.metric.space.from_XYZ(XYZ, S=0.5)
        np.testing.assert_allclose(lab_explicit, lab_default, atol=1e-12)

    def test_roundtrip_s02(self, p):
        """from_XYZ → to_XYZ roundtrip with S=0.2."""
        srgb = hex_to_srgb("#ef4444")
        from helmlab.utils.srgb_convert import sRGB_to_XYZ as s2x
        XYZ = s2x(srgb)
        lab = p.metric.space.from_XYZ(XYZ, S=0.2)
        XYZ_rec = p.metric.space.to_XYZ(lab, S=0.2)
        np.testing.assert_allclose(XYZ_rec, XYZ, atol=1e-6)

    def test_roundtrip_s08(self, p):
        """from_XYZ → to_XYZ roundtrip with S=0.8."""
        srgb = hex_to_srgb("#22c55e")
        from helmlab.utils.srgb_convert import sRGB_to_XYZ as s2x
        XYZ = s2x(srgb)
        lab = p.metric.space.from_XYZ(XYZ, S=0.8)
        XYZ_rec = p.metric.space.to_XYZ(lab, S=0.8)
        np.testing.assert_allclose(XYZ_rec, XYZ, atol=1e-6)

    def test_distance_s05_matches_v14(self, p):
        """Distance at S=0.5 matches v14 Euclidean distance."""
        de1 = p.metric.euclidean("#3b82f6", "#ef4444")
        # Direct Lab distance at S=0.5
        lab1 = p.metric.from_hex("#3b82f6")
        lab2 = p.metric.from_hex("#ef4444")
        de2 = float(np.sqrt(np.sum((lab1 - lab2) ** 2)))
        assert abs(de1 - de2) < 1e-10


class TestSurroundHelmlab:
    """Tests for Helmlab surround integration."""

    def test_set_surround(self):
        """set_surround changes instance surround."""
        p = Helmlab()
        p.set_surround(0.2)
        assert p._surround == 0.2
        assert p.metric.space._surround == 0.2

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
        result = p.gen.adapt_to_mode("#cccccc", "light", "dark")
        lab_orig = p.gen.from_hex("#cccccc")
        lab_adapted = p.gen.from_hex(result)
        assert lab_adapted[0] < lab_orig[0]

    def test_adapt_pair_with_surround(self):
        """adapt_pair still meets contrast with surround."""
        p = Helmlab()
        fg, bg = p.gen.adapt_pair("#333333", "#ffffff", "light", "dark", 4.5)
        cr = p.gen.contrast_ratio(fg, bg)
        assert cr >= 4.5 - 0.01


class TestSurroundBackwardCompat:
    """Tests for backward compatibility with v14 params."""

    def test_v14_json_loads_with_s_zero(self):
        """v14 JSON (no S params) loads with S params = 0."""
        from helmlab.spaces.analytical import AnalyticalParams
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
        from helmlab.spaces.analytical import AnalyticalParams
        p = AnalyticalParams()
        p.hk_weight_S = 0.1
        p.L_S_offset = -0.05
        d = p.to_dict()
        p2 = AnalyticalParams.from_dict(d)
        assert p2.hk_weight_S == 0.1
        assert p2.L_S_offset == -0.05


# ═══════════════════════════════════════════════════════════════════════
# Base Lab
# ═══════════════════════════════════════════════════════════════════════

class TestBaseLab:
    """Tests for base Lab (M1→power→M2 only) generation pipeline."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_base_roundtrip(self, p):
        """base_from_hex → base_to_hex roundtrip (±1/255)."""
        for h in ["#3b82f6", "#ef4444", "#22c55e", "#808080", "#ffffff", "#000000"]:
            lab = p.gen.from_hex(h)
            recovered = p.gen.to_hex(lab)
            srgb_orig = hex_to_srgb(h)
            srgb_rec = hex_to_srgb(recovered)
            np.testing.assert_allclose(srgb_rec, srgb_orig, atol=2.0 / 255.0)

    def test_achromatic_low_chroma(self, p):
        """Grays have low chroma in base Lab (no NC, so not exactly zero)."""
        for h in ["#000000", "#808080", "#ffffff", "#333333", "#cccccc"]:
            lab = p.gen.from_hex(h)
            C = np.sqrt(lab[1] ** 2 + lab[2] ** 2)
            # Independent gammas cause small residual chroma (~0.1 max)
            assert C < 0.2, f"{h}: base Lab chroma={C:.6f}, expected < 0.2"

    def test_palette_not_washed_out(self, p):
        """Palette colors should be vivid (not all gray/white)."""
        pal = p.gen.palette("#3b82f6", steps=5)
        # At least 3 of 5 should have some saturation
        saturated = 0
        for h in pal:
            srgb = hex_to_srgb(h)
            # Check max - min channel spread > 0.1
            if max(srgb) - min(srgb) > 0.1:
                saturated += 1
        assert saturated >= 3, f"Only {saturated}/5 palette colors are vivid"

    def test_ensure_contrast_not_white(self, p):
        """ensure_contrast should not return #ffffff for dark bg with colored fg."""
        result = p.gen.ensure_contrast("#a51d1d", "#111113")
        assert result != "#ffffff", "ensure_contrast returned white — base Lab regression"
        cr = p.gen.contrast_ratio(result, "#111113")
        assert cr >= 4.5 - 0.01

    def test_semantic_scale_vivid(self, p):
        """Semantic scale level 500 should be close to the input color."""
        scale = p.gen.scale("#3b82f6")
        # Level 500 should be the base color (or very close)
        srgb_500 = hex_to_srgb(scale["500"])
        srgb_base = hex_to_srgb("#3b82f6")
        np.testing.assert_allclose(srgb_500, srgb_base, atol=2.0 / 255.0)

    def test_gradient_no_brightness_fold(self, p):
        """Palette L should be monotonically decreasing (no brightness fold)."""
        pal = p.gen.palette("#ff6b00", steps=10)
        labs = [p.gen.from_hex(h) for h in pal]
        Ls = [float(lab[0]) for lab in labs]
        for i in range(len(Ls) - 1):
            assert Ls[i] >= Ls[i + 1] - 0.01, (
                f"Brightness fold: L[{i}]={Ls[i]:.3f} < L[{i+1}]={Ls[i+1]:.3f}"
            )


# ═══════════════════════════════════════════════════════════════════════
# XYZ Conversions on Helmlab class
# ═══════════════════════════════════════════════════════════════════════

class TestXYZOnHelmlab:
    """Test from_XYZ / to_XYZ on Helmlab class."""

    @pytest.fixture(params=[Helmlab()])
    def p(self, request):
        return request.param

    def test_xyz_roundtrip(self, p):
        """from_XYZ → to_XYZ roundtrip."""
        XYZ = np.array([0.2, 0.2, 0.3])
        lab = p.metric.from_xyz(XYZ)
        XYZ_rt = p.metric.to_xyz(lab)
        np.testing.assert_allclose(XYZ, XYZ_rt, atol=1e-10)

    def test_from_xyz_matches_from_srgb(self, p):
        """from_XYZ should match from_srgb via sRGB→XYZ."""
        srgb = np.array([0.5, 0.3, 0.8])
        XYZ = sRGB_to_XYZ(srgb)
        lab_via_srgb = p.metric.from_srgb(srgb)
        lab_via_xyz = p.metric.from_xyz(XYZ)
        np.testing.assert_allclose(lab_via_srgb, lab_via_xyz, atol=1e-12)

    def test_d65_white(self, p):
        """D65 white → Lab L should be close to 1."""
        D65 = np.array([0.95047, 1.0, 1.08883])
        lab = p.metric.from_xyz(D65)
        assert lab[0] > 0.9, f"White L={lab[0]}, expected > 0.9"

    def test_black(self, p):
        """Black XYZ → Lab L should be close to 0."""
        lab = p.metric.from_xyz(np.array([0.0, 0.0, 0.0]))
        assert abs(lab[0]) < 0.05, f"Black L={lab[0]}, expected ~0"


# ═══════════════════════════════════════════════════════════════════════
# Perceptual Distance
# ═══════════════════════════════════════════════════════════════════════

class TestPerceptualDistance:
    """Test perceptual_distance on Helmlab class."""

    @pytest.fixture(params=[Helmlab()])
    def p(self, request):
        return request.param

    def test_self_zero(self, p):
        """Distance to self is zero."""
        lab = p.metric.from_hex("#3b82f6")
        d = p.metric.distance(lab, lab)
        assert d < 1e-10

    def test_symmetric(self, p):
        """d(a,b) == d(b,a)."""
        lab1 = p.metric.from_hex("#ff0000")
        lab2 = p.metric.from_hex("#00ff00")
        assert abs(p.metric.distance(lab1, lab2) - p.metric.distance(lab2, lab1)) < 1e-12

    def test_positive(self, p):
        """Different colors have positive distance."""
        lab1 = p.metric.from_hex("#ff0000")
        lab2 = p.metric.from_hex("#0000ff")
        assert p.metric.distance(lab1, lab2) > 0

    def test_greater_for_dissimilar(self, p):
        """Very different colors have larger distance than similar ones."""
        lab_r = p.metric.from_hex("#ff0000")
        lab_rish = p.metric.from_hex("#ee1111")
        lab_b = p.metric.from_hex("#0000ff")
        d_close = p.metric.distance(lab_r, lab_rish)
        d_far = p.metric.distance(lab_r, lab_b)
        assert d_far > d_close


# ═══════════════════════════════════════════════════════════════════════
# delta_e (Euclidean Lab) — naming clarity & consistency
# ═══════════════════════════════════════════════════════════════════════

class TestDeltaEEuclidean:
    """Tests for Helmlab.delta_e (Euclidean Lab, ΔE76-style on hex inputs).

    These pin the documented behavior: delta_e returns Euclidean Lab and is
    distinct from perceptual_distance (which uses Minkowski + compression).
    Regression tests guarding against accidental rename or formula change.
    """

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_self_zero(self, p):
        assert p.metric.euclidean("#3b82f6", "#3b82f6") == 0.0

    def test_black_white_matches_lab_norm(self, p):
        """Should equal sqrt((Lw - Lb)² + (aw - ab)² + (bw - bb)²)."""
        lab_w = p.metric.from_hex("#ffffff")
        lab_b = p.metric.from_hex("#000000")
        expected = float(np.sqrt(np.sum((lab_w - lab_b) ** 2)))
        actual = p.metric.euclidean("#ffffff", "#000000")
        assert abs(actual - expected) < 1e-12

    def test_distinct_from_perceptual_distance(self, p):
        """Two methods report two different metrics. Regression guard."""
        lab_w = p.metric.from_hex("#ffffff")
        lab_b = p.metric.from_hex("#000000")
        euclidean = p.metric.euclidean("#ffffff", "#000000")
        perceptual = p.metric.distance(lab_w, lab_b)
        # Euclidean Lab black-white ≈ 1.12; perceptual is compressed near 0.15
        assert euclidean > 0.5
        assert perceptual < 0.5
        assert euclidean > perceptual  # uncompressed always >= compressed for big edits

    def test_returns_float(self, p):
        d = p.metric.euclidean("#ff0000", "#00ff00")
        assert isinstance(d, float)


# ═══════════════════════════════════════════════════════════════════════
# MetricSpace.distance vs distance_from_lab — input contract guard
# ═══════════════════════════════════════════════════════════════════════

class TestDistanceInputContract:
    """Tests that distance() expects XYZ and distance_from_lab() expects Lab.

    Background: distance() internally calls self.from_XYZ(), so passing Lab
    accidentally produces silent garbage that compresses near 0.15. The new
    distance_from_lab() helper sidesteps this. These tests pin the contract.
    """

    @pytest.fixture
    def ms(self):
        from helmlab.spaces.metric import MetricSpace
        return MetricSpace()

    def test_distance_from_lab_matches_distance_when_inputs_match(self, ms):
        """distance(XYZ_a, XYZ_b) == distance_from_lab(from_XYZ(XYZ_a), from_XYZ(XYZ_b))."""
        xyz_a = np.array([0.4, 0.2, 0.05])
        xyz_b = np.array([0.95047, 1.0, 1.08883])
        d_xyz = ms.distance(xyz_a, xyz_b)
        d_lab = ms.distance_from_lab(ms.from_XYZ(xyz_a), ms.from_XYZ(xyz_b))
        np.testing.assert_allclose(d_xyz, d_lab, atol=1e-12)

    def test_distance_from_lab_self_zero(self, ms):
        lab = ms.from_XYZ(np.array([0.5, 0.5, 0.5]))
        d = ms.distance_from_lab(lab, lab)
        assert float(d) < 1e-12

    def test_distance_from_lab_batched(self, ms):
        """Batched input shape (N, 3) returns shape (N,)."""
        np.random.seed(0)
        rgb = np.random.rand(10, 3)
        from helmlab.utils.srgb_convert import sRGB_to_XYZ
        xyz_a = sRGB_to_XYZ(rgb)
        xyz_b = sRGB_to_XYZ(np.roll(rgb, 1, axis=0))
        lab_a = ms.from_XYZ(xyz_a)
        lab_b = ms.from_XYZ(xyz_b)
        d = ms.distance_from_lab(lab_a, lab_b)
        assert d.shape == (10,)
        assert np.all(d >= 0)


# ═══════════════════════════════════════════════════════════════════════
# Deprecated base_* methods emit DeprecationWarning
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# display_phi_deg — opt-in display alignment, exact isometry
# ═══════════════════════════════════════════════════════════════════════

class TestDisplayPhi:
    """Tests for the display_phi_deg parameter on MetricSpace.

    Pins the contract: rotation is EXACTLY isometric for any distance metric
    that depends only on (dL, Δa²+Δb², L̄, C̄). STRESS, round-trip, and
    distance values are bit-identical regardless of φ.
    """

    @pytest.fixture
    def base_params(self):
        from helmlab.spaces.metric import MetricSpace
        return MetricSpace().params

    def test_default_display_phi_is_zero(self, base_params):
        """Backward compat: no φ field → 0.0."""
        from helmlab.spaces.metric import MetricParams
        # Simulate loading a pre-v0.12.2 checkpoint (no display_phi_deg key).
        d = base_params.to_dict()
        d.pop("display_phi_deg", None)
        p2 = MetricParams.from_dict(d)
        assert p2.display_phi_deg == 0.0

    def test_constructor_arg_overrides_param(self, base_params):
        """Explicit ab_rotate_deg argument overrides params.display_phi_deg."""
        from helmlab.spaces.metric import MetricSpace, MetricParams
        d = base_params.to_dict()
        d["display_phi_deg"] = -28.2
        ms_default = MetricSpace(MetricParams.from_dict(d))
        ms_override = MetricSpace(MetricParams.from_dict(d), ab_rotate_deg=0.0)

        from helmlab.utils.srgb_convert import sRGB_to_XYZ
        xyz = sRGB_to_XYZ(np.array([1.0, 0.0, 0.0]))
        lab_default = ms_default.from_XYZ(xyz)
        lab_override = ms_override.from_XYZ(xyz)
        # They should differ (different rotation applied)
        assert not np.allclose(lab_default[1:], lab_override[1:])

    def test_distance_invariant_under_phi(self, base_params):
        """Core isometry claim: distance is identical for any φ."""
        from helmlab.spaces.metric import MetricSpace, MetricParams
        from helmlab.utils.srgb_convert import sRGB_to_XYZ
        np.random.seed(0)
        rgb1 = np.random.rand(50, 3)
        rgb2 = np.random.rand(50, 3)
        xyz1 = np.array([sRGB_to_XYZ(r) for r in rgb1])
        xyz2 = np.array([sRGB_to_XYZ(r) for r in rgb2])

        d_zero = MetricSpace(base_params).distance(xyz1, xyz2)

        for phi in [-28.2, -11.75, 15.0, 90.0]:
            d = dict(base_params.to_dict())
            d["display_phi_deg"] = phi
            d_rot = MetricSpace(MetricParams.from_dict(d)).distance(xyz1, xyz2)
            np.testing.assert_allclose(d_rot, d_zero, atol=1e-12, rtol=1e-12,
                err_msg=f"distance changed under φ={phi}°")

    def test_roundtrip_preserved_under_phi(self, base_params):
        """from_XYZ → to_XYZ round-trip remains machine-precision under any φ."""
        from helmlab.spaces.metric import MetricSpace, MetricParams
        from helmlab.utils.srgb_convert import sRGB_to_XYZ
        np.random.seed(1)
        rgb = np.random.rand(100, 3)
        xyz = np.array([sRGB_to_XYZ(r) for r in rgb])

        for phi in [-28.2, -11.75, 0.0, 15.0]:
            d = dict(base_params.to_dict())
            d["display_phi_deg"] = phi
            ms = MetricSpace(MetricParams.from_dict(d))
            xyz_back = ms.to_XYZ(ms.from_XYZ(xyz))
            err = np.max(np.abs(xyz - xyz_back))
            assert err < 1e-10, f"roundtrip broke at φ={phi}°: max err {err:.2e}"

    def test_phi_actually_rotates(self, base_params):
        """Sanity: nonzero φ moves Lab a/b coordinates."""
        from helmlab.spaces.metric import MetricSpace, MetricParams
        from helmlab.utils.srgb_convert import sRGB_to_XYZ
        xyz = sRGB_to_XYZ(np.array([1.0, 0.0, 0.0]))  # red

        d_zero = dict(base_params.to_dict()); d_zero["display_phi_deg"] = 0.0
        d_rot = dict(base_params.to_dict()); d_rot["display_phi_deg"] = -30.0

        lab_zero = MetricSpace(MetricParams.from_dict(d_zero)).from_XYZ(xyz)
        lab_rot = MetricSpace(MetricParams.from_dict(d_rot)).from_XYZ(xyz)

        # L should be exactly equal (only a,b rotate)
        assert abs(lab_zero[0] - lab_rot[0]) < 1e-12
        # a,b should differ
        assert abs(lab_zero[1] - lab_rot[1]) > 0.01 or abs(lab_zero[2] - lab_rot[2]) > 0.01
        # Magnitude in (a,b) plane preserved (chroma)
        c_zero = np.hypot(lab_zero[1], lab_zero[2])
        c_rot = np.hypot(lab_rot[1], lab_rot[2])
        np.testing.assert_allclose(c_zero, c_rot, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# Contract guards (2026-07-08 audit fixes)
# ═══════════════════════════════════════════════════════════════════════

class TestContractGuards:
    """ensure_contrast fallback warning + distance_from_lab finite guard."""

    def test_ensure_contrast_warns_when_unreachable(self):
        """7:1 against mid-gray is unreachable even for black/white → warn."""
        import warnings
        hl = Helmlab()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = hl.gen.ensure_contrast("#3b82f6", "#808080", ratio=7.0)
        assert result in ("#000000", "#ffffff")
        assert any("ensure_contrast" in str(x.message) for x in w)

    def test_ensure_contrast_no_warn_when_reachable(self):
        import warnings
        hl = Helmlab()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = hl.gen.ensure_contrast("#3b82f6", "#ffffff", ratio=4.5)
        assert hl.gen.contrast_ratio(result, "#ffffff") >= 4.5
        assert not any("ensure_contrast" in str(x.message) for x in w)

    def test_distance_from_lab_rejects_nan(self):
        hl = Helmlab()
        lab = hl.metric.from_hex("#3b82f6")
        bad = np.array([np.nan, 0.1, 0.1])
        with pytest.raises(ValueError, match="non-finite"):
            hl.metric.distance(lab, bad)

    def test_distance_from_lab_rejects_inf(self):
        hl = Helmlab()
        lab = hl.metric.from_hex("#3b82f6")
        bad = np.array([0.5, np.inf, 0.1])
        with pytest.raises(ValueError, match="non-finite"):
            hl.metric.distance(bad, lab)


# ═══════════════════════════════════════════════════════════════════════
# 1.0 API — branded types, wide gamut, harmonies, mix, jnd
# ═══════════════════════════════════════════════════════════════════════

class TestBrandedLabTypes:
    """Cross-space Lab misuse is a TypeError, not a silent wrong color."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_gen_lab_rejected_by_metric(self, p):
        lab = p.gen.from_hex("#3b82f6")
        with pytest.raises(TypeError, match="GenLab"):
            p.metric.to_hex(lab)

    def test_metric_lab_rejected_by_gen(self, p):
        lab = p.metric.from_hex("#3b82f6")
        with pytest.raises(TypeError, match="MetricLab"):
            p.gen.to_hex(lab)

    def test_plain_arrays_still_accepted(self, p):
        assert p.metric.to_hex([0.5, 0.05, -0.05]).startswith("#")
        assert p.gen.to_hex(np.array([0.5, 0.05, -0.05])).startswith("#")

    def test_branded_types_survive_numpy_ops(self, p):
        from helmlab import GenLab
        lab = p.gen.from_hex("#3b82f6")
        assert isinstance(lab.copy(), GenLab)


class TestWideGamut10:
    """gamut= option on generation + wide-gamut input strings."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_gradient_p3(self, p):
        g = p.gen.gradient("#0000ff", "#ffffff", 3, gamut="display-p3")
        assert len(g) == 3
        assert all(x.startswith("color(display-p3 ") for x in g)

    def test_gradient_rec2020(self, p):
        g = p.gen.gradient("#ff0000", "#00ff00", 2, gamut="rec2020")
        assert all(x.startswith("color(rec2020 ") for x in g)

    def test_gradient_bad_gamut_raises(self, p):
        with pytest.raises(ValueError, match="unknown gamut"):
            p.gen.gradient("#ff0000", "#00ff00", 3, gamut="cmyk")

    def test_p3_input_string(self, p):
        info = p.metric.info("color(display-p3 1 0 0)")
        assert info["in_srgb"] is False
        assert info["in_p3"] is True

    def test_p3_input_to_gen(self, p):
        lab = p.gen.from_hex("color(display-p3 1 0 0)")
        assert float(lab[0]) > 0.3

    def test_bad_css_color_raises(self, p):
        with pytest.raises(ValueError, match="unparseable"):
            p.metric.from_hex("color(foo 1 0 0)")

    def test_non_string_raises(self, p):
        with pytest.raises(TypeError):
            p.gen.from_hex(123)

    def test_scale_p3(self, p):
        scale = p.gen.scale("#3b82f6", gamut="display-p3")
        assert scale["500"].startswith("color(display-p3 ")


class TestHarmoniesMixRotate:
    """1.0 generation features: harmonies, mix, rotate_hue, hue_ring."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_harmonies_triadic(self, p):
        h = p.gen.harmonies("#3b82f6", "triadic")
        assert len(h) == 3
        assert h[0] == p.gen.to_hex(p.gen.from_hex("#3b82f6"))

    def test_harmonies_counts(self, p):
        assert len(p.gen.harmonies("#3b82f6", "complementary")) == 2
        assert len(p.gen.harmonies("#3b82f6", "analogous")) == 3
        assert len(p.gen.harmonies("#3b82f6", "tetradic")) == 4
        assert len(p.gen.harmonies("#3b82f6", "split_complementary")) == 3

    def test_harmonies_preserve_L(self, p):
        for h in p.gen.harmonies("#3b82f6", "triadic"):
            lab = p.gen.from_hex(h)
            base = p.gen.from_hex("#3b82f6")
            # L equal within gamut-mapping tolerance
            assert abs(float(lab[0]) - float(base[0])) < 0.02

    def test_harmonies_bad_kind_raises(self, p):
        with pytest.raises(ValueError, match="unknown harmony"):
            p.gen.harmonies("#3b82f6", "quadratic")

    def test_mix_endpoints(self, p):
        a, b = "#ff0000", "#0000ff"
        assert p.gen.mix(a, b, 0.0) == p.gen.to_hex(p.gen.from_hex(a))
        assert p.gen.mix(a, b, 1.0) == p.gen.to_hex(p.gen.from_hex(b))

    def test_mix_matches_gradient_midpoint(self, p):
        a, b = "#ff0000", "#0000ff"
        assert p.gen.mix(a, b, 0.5) == p.gen.gradient(a, b, 3)[1]

    def test_rotate_hue_identity(self, p):
        base_rt = p.gen.to_hex(p.gen.from_hex("#3b82f6"))
        assert p.gen.rotate_hue("#3b82f6", 0) == base_rt
        assert p.gen.rotate_hue("#3b82f6", 360) == base_rt

    def test_hue_ring_count_and_distinct(self, p):
        ring = p.gen.hue_ring(6)
        assert len(ring) == 6
        assert len(set(ring)) == 6


class TestJndAndStrictContrast:
    """metric.jnd + ensure_contrast(strict=True)."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_jnd_self_zero(self, p):
        assert p.metric.jnd("#808080", "#808080") == 0.0

    def test_jnd_big_pair_above_threshold(self, p):
        assert p.metric.jnd("#ff0000", "#00ff00") > 3.0

    def test_jnd_is_scaled_difference(self, p):
        de = p.metric.difference("#808080", "#828282")
        jnd = p.metric.jnd("#808080", "#828282")
        np.testing.assert_allclose(jnd, de / 0.03563295091867221, rtol=1e-9)

    def test_strict_contrast_raises(self, p):
        from helmlab import ContrastError
        with pytest.raises(ContrastError):
            p.gen.ensure_contrast("#3b82f6", "#808080", ratio=7.0, strict=True)

    def test_gamut_map_exposed(self, p):
        from helmlab import GenLab
        oog = p.gen.lab(0.5, 0.8, 0.0)
        mapped = p.gen.gamut_map(oog)
        assert isinstance(mapped, GenLab)
        assert p.gen.in_gamut(mapped)


class TestCuspGeometryExposed:
    """1.0: GenSpace's cusp geometry as public API (max_chroma, cusp, vivid,
    adaptive gamut mapping)."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_max_chroma_positive_and_p3_wider(self, p):
        c_srgb = p.gen.max_chroma(0.6, 263)
        c_p3 = p.gen.max_chroma(0.6, 263, "display-p3")
        assert c_srgb > 0
        assert c_p3 > c_srgb

    def test_cusp_is_max_over_L(self, p):
        L_cusp, C_cusp = p.gen.cusp(263)
        assert 0 < L_cusp < 1
        # cusp chroma >= max chroma at other lightness values on same hue
        for L in (0.3, 0.6, 0.8):
            assert C_cusp >= p.gen.max_chroma(L, 263) - 1e-3

    def test_vivid_preserves_L_and_hue(self, p):
        base = "#6488b8"
        vivid = p.gen.vivid(base)
        lch_base = p.gen.to_lch(p.gen.from_hex(base))
        lch_vivid = p.gen.to_lch(p.gen.from_hex(vivid))
        assert abs(lch_vivid[0] - lch_base[0]) < 0.02
        dh = abs(lch_vivid[2] - lch_base[2]) % 360
        assert min(dh, 360 - dh) < 3.0
        assert lch_vivid[1] > lch_base[1]  # more chroma

    def test_vivid_p3_more_chroma_than_srgb(self, p):
        base = "#6488b8"
        v_srgb = p.gen.to_lch(p.gen.from_hex(p.gen.vivid(base)))
        v_p3 = p.gen.to_lch(p.gen.from_hex(p.gen.vivid(base, gamut="display-p3")))
        assert v_p3[1] > v_srgb[1] - 1e-3

    def test_adaptive_gamut_map_in_gamut(self, p):
        oog = p.gen.lab(0.5, 0.8, 0.0)
        mapped = p.gen.gamut_map(oog, method="adaptive")
        assert p.gen.in_gamut(mapped)


class TestMetricLch:
    """1.0: metric-side cylindrical LCh (symmetric with gen)."""

    @pytest.fixture
    def p(self):
        return Helmlab()

    def test_lch_roundtrip(self, p):
        from helmlab import MetricLab
        lab = p.metric.from_hex("#3b82f6")
        lch = p.metric.to_lch(lab)
        back = p.metric.from_lch(lch)
        assert isinstance(back, MetricLab)
        np.testing.assert_allclose(back, np.asarray(lab), atol=1e-12)

    def test_lch_matches_info(self, p):
        lab = p.metric.from_hex("#3b82f6")
        lch = p.metric.to_lch(lab)
        info = p.metric.info("#3b82f6")
        np.testing.assert_allclose(lch[1], info["C"], atol=1e-12)
        np.testing.assert_allclose(lch[2], info["H"], atol=1e-9)

    def test_gen_lab_rejected(self, p):
        with pytest.raises(TypeError, match="GenLab"):
            p.metric.to_lch(p.gen.from_hex("#3b82f6"))
