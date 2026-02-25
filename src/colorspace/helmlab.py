"""Helmlab — UI design system utility layer.

Built on the Helmlab analytical color space (v14 balanced).
Provides hex/sRGB conversions, WCAG contrast utilities, palette generation,
semantic scales, and dark/light mode adaptation.
"""

import numpy as np

from colorspace.spaces.analytical import AnalyticalSpace, AnalyticalParams
from colorspace.utils.srgb_convert import (
    hex_to_srgb,
    srgb_to_hex,
    sRGB_to_XYZ,
    XYZ_to_sRGB,
    XYZ_to_DisplayP3,
    linear_to_displayp3,
    clamp_srgb,
    relative_luminance,
    contrast_ratio as _wcag_cr,
)
from colorspace.utils.gamut import gamut_map, is_in_gamut


class Helmlab:
    """UI design system utility layer built on Helmlab color space."""

    def __init__(self, params_path: str | None = None, surround: float = 0.5,
                 neutral_correction: bool = True, ab_rotate_deg: float = -28.2):
        self._surround = float(np.clip(surround, 0.0, 1.0))
        if params_path is not None:
            params = AnalyticalParams.load(params_path)
            self._space = AnalyticalSpace(params, surround=self._surround,
                                          neutral_correction=neutral_correction,
                                          ab_rotate_deg=ab_rotate_deg)
        else:
            self._space = AnalyticalSpace(surround=self._surround,
                                          neutral_correction=neutral_correction,
                                          ab_rotate_deg=ab_rotate_deg)

    def set_surround(self, S: float):
        """Change viewing context. 0=dark, 0.5=normal, 1=bright."""
        self._surround = float(np.clip(S, 0.0, 1.0))
        self._space._surround = self._surround

    # ── Conversions ──────────────────────────────────────────────────

    def from_hex(self, hex_str: str) -> np.ndarray:
        """Hex '#rrggbb' → Helmlab Lab [L, a, b]."""
        srgb = hex_to_srgb(hex_str)
        return self.from_srgb(srgb)

    def to_hex(self, lab: np.ndarray) -> str:
        """Helmlab Lab [L, a, b] → hex '#rrggbb' (clamped to sRGB)."""
        srgb = self.to_srgb(lab)
        return srgb_to_hex(srgb)

    def from_srgb(self, srgb: np.ndarray) -> np.ndarray:
        """sRGB [0,1] → Helmlab Lab [L, a, b]."""
        XYZ = sRGB_to_XYZ(np.asarray(srgb, dtype=np.float64))
        return self._space.from_XYZ(XYZ)

    def to_srgb(self, lab: np.ndarray) -> np.ndarray:
        """Helmlab Lab [L, a, b] → sRGB [0,1] (gamut mapped)."""
        lab = np.asarray(lab, dtype=np.float64)
        mapped = gamut_map(lab, self._space, gamut="srgb")
        XYZ = self._space.to_XYZ(mapped)
        return clamp_srgb(XYZ_to_sRGB(XYZ))

    def to_displayp3(self, lab: np.ndarray) -> np.ndarray:
        """Helmlab Lab [L, a, b] → Display P3 [0,1] (gamut mapped, gamma-encoded)."""
        lab = np.asarray(lab, dtype=np.float64)
        mapped = gamut_map(lab, self._space, gamut="display-p3")
        XYZ = self._space.to_XYZ(mapped)
        linear = XYZ_to_DisplayP3(XYZ)
        return clamp_srgb(linear_to_displayp3(linear))

    def to_hex_p3(self, lab: np.ndarray) -> str:
        """Helmlab Lab → CSS color(display-p3 r g b) string."""
        p3 = self.to_displayp3(lab)
        return f"color(display-p3 {p3[0]:.4f} {p3[1]:.4f} {p3[2]:.4f})"

    def is_in_srgb(self, lab: np.ndarray) -> bool:
        """Check if Lab coordinates are within sRGB gamut."""
        return bool(is_in_gamut(np.asarray(lab, dtype=np.float64), self._space, "srgb"))

    def is_in_p3(self, lab: np.ndarray) -> bool:
        """Check if Lab coordinates are within Display P3 gamut."""
        return bool(is_in_gamut(np.asarray(lab, dtype=np.float64), self._space, "display-p3"))

    # ── Contrast ─────────────────────────────────────────────────────

    def contrast_ratio(self, fg_hex: str, bg_hex: str) -> float:
        """WCAG contrast ratio between two hex colors (1.0 – 21.0)."""
        return float(_wcag_cr(hex_to_srgb(fg_hex), hex_to_srgb(bg_hex)))

    def ensure_contrast(
        self, fg_hex: str, bg_hex: str, min_ratio: float = 4.5
    ) -> str:
        """Adjust fg lightness to meet min_ratio against bg.

        Binary search on Helmlab L axis. Hue and chroma are preserved.
        Returns adjusted fg as hex.
        """
        current = self.contrast_ratio(fg_hex, bg_hex)
        if current >= min_ratio:
            return fg_hex

        fg_lab = self.from_hex(fg_hex)
        bg_srgb = hex_to_srgb(bg_hex)

        # Try both directions, pick the closest L that meets target
        best_hex = fg_hex
        best_ratio = current
        best_L = float(fg_lab[0])
        orig_L = float(fg_lab[0])

        for direction in ("darken", "lighten"):
            if direction == "darken":
                lo, hi = 0.0, orig_L
            else:
                lo, hi = orig_L, 1.5  # allow overshoot, clamp handles it

            candidate_lab = fg_lab.copy()
            for _ in range(40):
                mid = (lo + hi) / 2.0
                candidate_lab[0] = mid
                candidate_srgb = self.to_srgb(candidate_lab)
                # Check ratio AFTER hex quantization to avoid rounding issues
                hex_quantized = hex_to_srgb(srgb_to_hex(candidate_srgb))
                ratio = float(_wcag_cr(hex_quantized, bg_srgb))

                if direction == "darken":
                    if ratio >= min_ratio:
                        lo = mid  # try less extreme
                    else:
                        hi = mid  # need darker
                else:
                    if ratio >= min_ratio:
                        hi = mid  # try less extreme
                    else:
                        lo = mid  # need lighter

            # Use the endpoint known to meet the ratio, not the midpoint.
            # darken: lo is the last L that met min_ratio
            # lighten: hi is the last L that met min_ratio
            safe_L = lo if direction == "darken" else hi
            candidate_lab[0] = safe_L
            candidate_srgb = self.to_srgb(candidate_lab)
            candidate_hex = srgb_to_hex(candidate_srgb)
            hex_quantized = hex_to_srgb(candidate_hex)
            ratio = float(_wcag_cr(hex_quantized, bg_srgb))

            if ratio >= min_ratio:
                dist = abs(safe_L - orig_L)
                best_dist = abs(best_L - orig_L)
                if best_ratio < min_ratio or dist < best_dist:
                    best_hex = candidate_hex
                    best_ratio = ratio
                    best_L = safe_L

        # If neither direction worked (extremely rare), return darkest or lightest
        if best_ratio < min_ratio:
            # Try pure black and pure white
            for fallback in ("#000000", "#ffffff"):
                r = self.contrast_ratio(fallback, bg_hex)
                if r >= min_ratio:
                    return fallback
            # Last resort: return whichever has more contrast
            r_black = self.contrast_ratio("#000000", bg_hex)
            r_white = self.contrast_ratio("#ffffff", bg_hex)
            return "#000000" if r_black > r_white else "#ffffff"

        return best_hex

    def meets_contrast(
        self, fg_hex: str, bg_hex: str, level: str = "AA"
    ) -> bool:
        """Check if fg/bg pair meets WCAG contrast level.

        AA: 4.5:1, AAA: 7:1
        """
        thresholds = {"AA": 4.5, "AAA": 7.0}
        threshold = thresholds.get(level.upper(), 4.5)
        return self.contrast_ratio(fg_hex, bg_hex) >= threshold

    # ── Palette Generation ───────────────────────────────────────────

    def palette(self, base_hex: str, steps: int = 10) -> list[str]:
        """Generate lightness palette from a base color.

        Evenly spaced L from 0.95 (lightest) to 0.15 (darkest).
        Hue and chroma preserved, gamut clamped.
        """
        lab = self.from_hex(base_hex)
        L_values = np.linspace(0.95, 0.15, steps)
        result = []
        for L in L_values:
            sample = lab.copy()
            sample[0] = L
            result.append(self.to_hex(sample))
        return result

    def palette_hues(
        self, lightness: float = 0.6, chroma: float = 0.15, steps: int = 12
    ) -> list[str]:
        """Generate hue ring at fixed lightness and chroma.

        Hue angles evenly distributed 0–360°.
        """
        hues = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        result = []
        for h in hues:
            a = chroma * np.cos(h)
            b = chroma * np.sin(h)
            lab = np.array([lightness, a, b])
            result.append(self.to_hex(lab))
        return result

    # ── Semantic Scale ───────────────────────────────────────────────

    def semantic_scale(
        self,
        base_hex: str,
        levels: list[int] | None = None,
    ) -> dict[str, str]:
        """Generate Tailwind-style semantic scale (50–950).

        base_hex maps to level 500. Lightness distributed with smooth
        mapping: lower levels → lighter, higher levels → darker.
        """
        if levels is None:
            levels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]

        lab = self.from_hex(base_hex)
        base_L = float(lab[0])

        # Map levels to L: 50→0.97, 500→base_L, 950→0.10
        # Use piecewise linear interpolation
        L_light = 0.97   # L at level 50
        L_dark = 0.10    # L at level 950

        result = {}
        for level in levels:
            if level <= 500:
                # Interpolate from L_light (50) to base_L (500)
                t = level / 500.0
                L = L_light + t * (base_L - L_light)
            else:
                # Interpolate from base_L (500) to L_dark (950)
                t = (level - 500) / 450.0
                L = base_L + t * (L_dark - base_L)
            sample = lab.copy()
            sample[0] = L
            result[str(level)] = self.to_hex(sample)
        return result

    # ── Dark/Light Mode ──────────────────────────────────────────────

    def adapt_to_mode(
        self,
        color_hex: str,
        from_mode: str = "light",
        to_mode: str = "dark",
    ) -> str:
        """Adapt color between light and dark mode.

        Uses surround parameter S when S-dependent params are active.
        Falls back to soft L-inversion otherwise.
        Light range: [0.10, 0.95], Dark range: [0.15, 0.90].
        Hue and chroma preserved.
        """
        if from_mode == to_mode:
            return color_hex

        S_MAP = {"light": 0.7, "dark": 0.2}

        # If surround params are active, use S-based adaptation
        if self._space._has_surround:
            S_src = S_MAP.get(from_mode, 0.5)
            S_dst = S_MAP.get(to_mode, 0.5)
            srgb = hex_to_srgb(color_hex)
            XYZ = sRGB_to_XYZ(srgb)
            lab_src = self._space.from_XYZ(XYZ, S=S_src)
            XYZ_dst = self._space.to_XYZ(lab_src, S=S_dst)
            srgb_dst = clamp_srgb(XYZ_to_sRGB(XYZ_dst))
            return srgb_to_hex(srgb_dst)

        # Fallback: soft L-inversion
        LIGHT_LO, LIGHT_HI = 0.10, 0.95
        DARK_LO, DARK_HI = 0.15, 0.90

        lab = self.from_hex(color_hex)
        L = float(lab[0])

        if from_mode == "light":
            src_lo, src_hi = LIGHT_LO, LIGHT_HI
            dst_lo, dst_hi = DARK_LO, DARK_HI
        else:
            src_lo, src_hi = DARK_LO, DARK_HI
            dst_lo, dst_hi = LIGHT_LO, LIGHT_HI

        # Normalize to [0, 1] within source range
        t = np.clip((L - src_lo) / (src_hi - src_lo), 0.0, 1.0)
        # Invert and map to destination range
        L_new = dst_hi - t * (dst_hi - dst_lo)

        lab[0] = L_new
        return self.to_hex(lab)

    def adapt_pair(
        self,
        fg_hex: str,
        bg_hex: str,
        from_mode: str = "light",
        to_mode: str = "dark",
        min_ratio: float = 4.5,
    ) -> tuple[str, str]:
        """Adapt fg/bg pair to target mode, ensuring contrast.

        Both colors are mode-adapted, then contrast is enforced.
        """
        new_fg = self.adapt_to_mode(fg_hex, from_mode, to_mode)
        new_bg = self.adapt_to_mode(bg_hex, from_mode, to_mode)
        new_fg = self.ensure_contrast(new_fg, new_bg, min_ratio)
        return new_fg, new_bg

    # ── Info ─────────────────────────────────────────────────────────

    def delta_e(self, color1_hex: str, color2_hex: str) -> float:
        """Helmlab distance (Euclidean in Lab) between two hex colors."""
        lab1 = self.from_hex(color1_hex)
        lab2 = self.from_hex(color2_hex)
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))

    def export(self):
        """Return a TokenExporter for this Helmlab instance."""
        from colorspace.export import TokenExporter
        return TokenExporter(self)

    def info(self, color_hex: str) -> dict:
        """Return color information dict."""
        srgb = hex_to_srgb(color_hex)
        lab = self.from_srgb(srgb)
        XYZ = sRGB_to_XYZ(srgb)
        C = float(np.sqrt(lab[1] ** 2 + lab[2] ** 2))
        H_deg = float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)
        return {
            "hex": color_hex,
            "srgb": srgb.tolist(),
            "xyz": XYZ.tolist(),
            "lab": lab.tolist(),
            "L": float(lab[0]),
            "C": C,
            "H": H_deg,
            "luminance": float(relative_luminance(srgb)),
        }
