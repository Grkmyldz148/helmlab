"""Helmlab — UI design system utility layer.

Composes two purpose-built color spaces:
    MetricSpace — full 72-param enriched pipeline (distance, deltaE)
    GenSpace    — generation-optimized pipeline (palette, gradient, gamut map)

Public API routing:
    Distance/measurement → MetricSpace: delta_e(), from_hex(), to_hex(), info()
    Generation/creation  → GenSpace:    palette(), gradient(), semantic_scale(),
                                        palette_hues(), ensure_contrast(), adapt_to_mode()
"""

import numpy as np

from helmlab.spaces.metric import MetricSpace, MetricParams
from helmlab.spaces.gen import GenSpace, GenParams
from helmlab.utils.srgb_convert import (
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
from helmlab.utils.gamut import gamut_map, is_in_gamut


class Helmlab:
    """UI design system utility layer built on Helmlab color space family.

    Composes MetricSpace (distance) + GenSpace (generation).
    """

    def __init__(self, params_path: str | None = None, surround: float = 0.5,
                 neutral_correction: bool = True, ab_rotate_deg: float = -28.2,
                 gen_params_path: str | None = None):
        self._surround = float(np.clip(surround, 0.0, 1.0))

        # MetricSpace for distance/measurement
        if params_path is not None:
            params = MetricParams.load(params_path)
            self._metric = MetricSpace(params, surround=self._surround,
                                       neutral_correction=neutral_correction,
                                       ab_rotate_deg=ab_rotate_deg)
        else:
            self._metric = MetricSpace(surround=self._surround,
                                       neutral_correction=neutral_correction,
                                       ab_rotate_deg=ab_rotate_deg)

        # GenSpace for generation (palette, gradient, gamut map)
        if gen_params_path is not None:
            gen_p = GenParams.load(gen_params_path)
            self._gen = GenSpace(gen_p, neutral_correction=True)
        else:
            self._gen = GenSpace(neutral_correction=True)

        # Backward compat: expose _space as metric
        self._space = self._metric

        # Cache GenSpace white L for palette/scale range
        self._gen_white_L = float(self._gen.from_XYZ(np.array([0.95047, 1.0, 1.08883]))[0])

    def set_surround(self, S: float):
        """Change viewing context. 0=dark, 0.5=normal, 1=bright."""
        self._surround = float(np.clip(S, 0.0, 1.0))
        self._metric._surround = self._surround

    # ── Full-pipeline conversions (MetricSpace — public API) ────────

    def from_hex(self, hex_str: str) -> np.ndarray:
        """Hex '#rrggbb' → Helmlab Lab [L, a, b] (metric pipeline)."""
        srgb = hex_to_srgb(hex_str)
        return self.from_srgb(srgb)

    def to_hex(self, lab: np.ndarray) -> str:
        """Helmlab Lab [L, a, b] → hex '#rrggbb' (metric pipeline, clamped to sRGB)."""
        srgb = self.to_srgb(lab)
        return srgb_to_hex(srgb)

    def from_srgb(self, srgb: np.ndarray) -> np.ndarray:
        """sRGB [0,1] → Helmlab Lab [L, a, b] (metric pipeline)."""
        XYZ = sRGB_to_XYZ(np.asarray(srgb, dtype=np.float64))
        return self._metric.from_XYZ(XYZ)

    def to_srgb(self, lab: np.ndarray) -> np.ndarray:
        """Helmlab Lab [L, a, b] → sRGB [0,1] (metric pipeline, gamut mapped)."""
        lab = np.asarray(lab, dtype=np.float64)
        mapped = gamut_map(lab, self._metric, gamut="srgb")
        XYZ = self._metric.to_XYZ(mapped)
        return clamp_srgb(XYZ_to_sRGB(XYZ))

    def to_displayp3(self, lab: np.ndarray) -> np.ndarray:
        """Helmlab Lab [L, a, b] → Display P3 [0,1] (gamut mapped, gamma-encoded)."""
        lab = np.asarray(lab, dtype=np.float64)
        mapped = gamut_map(lab, self._metric, gamut="display-p3")
        XYZ = self._metric.to_XYZ(mapped)
        linear = XYZ_to_DisplayP3(XYZ)
        return clamp_srgb(linear_to_displayp3(linear))

    def to_hex_p3(self, lab: np.ndarray) -> str:
        """Helmlab Lab → CSS color(display-p3 r g b) string."""
        p3 = self.to_displayp3(lab)
        return f"color(display-p3 {p3[0]:.4f} {p3[1]:.4f} {p3[2]:.4f})"

    def is_in_srgb(self, lab: np.ndarray) -> bool:
        """Check if Lab coordinates are within sRGB gamut (metric space)."""
        return bool(is_in_gamut(np.asarray(lab, dtype=np.float64), self._metric, "srgb"))

    def is_in_p3(self, lab: np.ndarray) -> bool:
        """Check if Lab coordinates are within Display P3 gamut (metric space)."""
        return bool(is_in_gamut(np.asarray(lab, dtype=np.float64), self._metric, "display-p3"))

    # ── GenSpace conversions (for generation) ──────────────────────

    def gen_from_hex(self, hex_str: str) -> np.ndarray:
        """Hex '#rrggbb' → Gen Lab [L, a, b] (generation pipeline)."""
        XYZ = sRGB_to_XYZ(hex_to_srgb(hex_str))
        return self._gen.from_XYZ(XYZ)

    def gen_to_hex(self, lab: np.ndarray) -> str:
        """Gen Lab [L, a, b] → hex '#rrggbb' (generation pipeline, gamut mapped)."""
        srgb = self.gen_to_srgb(lab)
        return srgb_to_hex(srgb)

    def gen_from_srgb(self, srgb: np.ndarray) -> np.ndarray:
        """sRGB [0,1] → Gen Lab [L, a, b]."""
        XYZ = sRGB_to_XYZ(np.asarray(srgb, dtype=np.float64))
        return self._gen.from_XYZ(XYZ)

    def gen_to_srgb(self, lab: np.ndarray) -> np.ndarray:
        """Gen Lab [L, a, b] → sRGB [0,1] (gamut mapped, clamped)."""
        lab = np.asarray(lab, dtype=np.float64)
        mapped = gamut_map(lab, self._gen, gamut="srgb")
        XYZ = self._gen.to_XYZ(mapped)
        return clamp_srgb(XYZ_to_sRGB(XYZ))

    # ── Deprecated base_* aliases → gen_* ──────────────────────────

    def base_from_hex(self, hex_str: str) -> np.ndarray:
        """Deprecated: use gen_from_hex(). Hex → Gen Lab."""
        return self.gen_from_hex(hex_str)

    def base_to_hex(self, lab: np.ndarray) -> str:
        """Deprecated: use gen_to_hex(). Gen Lab → hex."""
        return self.gen_to_hex(lab)

    def base_from_srgb(self, srgb: np.ndarray) -> np.ndarray:
        """Deprecated: use gen_from_srgb(). sRGB → Gen Lab."""
        return self.gen_from_srgb(srgb)

    def base_to_srgb(self, lab: np.ndarray) -> np.ndarray:
        """Deprecated: use gen_to_srgb(). Gen Lab → sRGB."""
        return self.gen_to_srgb(lab)

    # ── Contrast ─────────────────────────────────────────────────────

    def contrast_ratio(self, fg_hex: str, bg_hex: str) -> float:
        """WCAG contrast ratio between two hex colors (1.0 – 21.0)."""
        return float(_wcag_cr(hex_to_srgb(fg_hex), hex_to_srgb(bg_hex)))

    def ensure_contrast(
        self, fg_hex: str, bg_hex: str, min_ratio: float = 4.5
    ) -> str:
        """Adjust fg lightness to meet min_ratio against bg.

        Binary search on Gen Lab L axis. Hue and chroma are preserved.
        Returns adjusted fg as hex.
        """
        current = self.contrast_ratio(fg_hex, bg_hex)
        if current >= min_ratio:
            return fg_hex

        fg_lab = self.gen_from_hex(fg_hex)
        bg_srgb = hex_to_srgb(bg_hex)

        best_hex = fg_hex
        best_ratio = current
        best_L = float(fg_lab[0])
        orig_L = float(fg_lab[0])

        for direction in ("darken", "lighten"):
            if direction == "darken":
                lo, hi = 0.0, orig_L
            else:
                lo, hi = orig_L, 1.5

            candidate_lab = fg_lab.copy()
            for _ in range(40):
                mid = (lo + hi) / 2.0
                candidate_lab[0] = mid
                candidate_srgb = self.gen_to_srgb(candidate_lab)
                hex_quantized = hex_to_srgb(srgb_to_hex(candidate_srgb))
                ratio = float(_wcag_cr(hex_quantized, bg_srgb))

                if direction == "darken":
                    if ratio >= min_ratio:
                        lo = mid
                    else:
                        hi = mid
                else:
                    if ratio >= min_ratio:
                        hi = mid
                    else:
                        lo = mid

            safe_L = lo if direction == "darken" else hi
            if direction == "darken":
                safe_L = max(0.0, safe_L - 0.003)
            else:
                safe_L = min(1.5, safe_L + 0.003)
            candidate_lab[0] = safe_L
            candidate_srgb = self.gen_to_srgb(candidate_lab)
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

        if best_ratio < min_ratio:
            for fallback in ("#000000", "#ffffff"):
                r = self.contrast_ratio(fallback, bg_hex)
                if r >= min_ratio:
                    return fallback
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

    # ── Gradient (GenSpace + arc-length) ────────────────────────────

    @staticmethod
    def _srgb_to_cielab(rgb):
        """sRGB [0,1] → CIE Lab."""
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        lr = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        lg = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        lb = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047
        y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb
        z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883
        def f(t):
            return t ** (1/3) if t > 0.008856 else 7.787 * t + 16 / 116
        fx, fy, fz = f(x), f(y), f(z)
        return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])

    @staticmethod
    def _ciede2000(lab1, lab2):
        """CIEDE2000 color difference."""
        L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
        L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])
        C1 = np.sqrt(a1*a1 + b1*b1)
        C2 = np.sqrt(a2*a2 + b2*b2)
        Cab = (C1 + C2) / 2
        Cab7 = Cab ** 7
        p257 = 25.0 ** 7
        G = 0.5 * (1 - np.sqrt(Cab7 / (Cab7 + p257)))
        ap1 = (1 + G) * a1
        ap2 = (1 + G) * a2
        Cp1 = np.sqrt(ap1*ap1 + b1*b1)
        Cp2 = np.sqrt(ap2*ap2 + b2*b2)
        hp1 = np.arctan2(b1, ap1)
        hp2 = np.arctan2(b2, ap2)
        if hp1 < 0: hp1 += 2 * np.pi
        if hp2 < 0: hp2 += 2 * np.pi
        dLp = L2 - L1
        dCp = Cp2 - Cp1
        if Cp1 * Cp2 == 0:
            dhp = 0.0
        else:
            dhp = hp2 - hp1
            if dhp > np.pi: dhp -= 2 * np.pi
            if dhp < -np.pi: dhp += 2 * np.pi
        dHp = 2 * np.sqrt(Cp1 * Cp2) * np.sin(dhp / 2)
        Lp = (L1 + L2) / 2
        Cp = (Cp1 + Cp2) / 2
        if Cp1 * Cp2 == 0:
            hp = hp1 + hp2
        else:
            hp = (hp1 + hp2) / 2
            if abs(hp1 - hp2) > np.pi:
                hp += np.pi if hp < np.pi else -np.pi
        T = (1 - 0.17 * np.cos(hp - np.pi/6) + 0.24 * np.cos(2*hp)
             + 0.32 * np.cos(3*hp + np.pi/30) - 0.20 * np.cos(4*hp - 63*np.pi/180))
        Lp50 = Lp - 50
        SL = 1 + 0.015 * Lp50 * Lp50 / np.sqrt(20 + Lp50 * Lp50)
        SC = 1 + 0.045 * Cp
        SH = 1 + 0.015 * Cp * T
        Cp7 = Cp ** 7
        RC = 2 * np.sqrt(Cp7 / (Cp7 + p257))
        hpDeg = hp * 180 / np.pi
        dth = 30 * np.exp(-((hpDeg - 275) / 25) ** 2)
        RT = -np.sin(2 * dth * np.pi / 180) * RC
        return np.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2
                       + RT * (dCp/SC) * (dHp/SH))

    def gradient(self, start_hex: str, end_hex: str, steps: int = 16) -> list[str]:
        """Generate a perceptually uniform gradient between two hex colors.

        Uses GenSpace Lab path with CIEDE2000 arc-length reparameterization
        for equal perceptual step sizes on any color pair.
        """
        if steps == 1:
            return [start_hex]

        lab1 = self.gen_from_hex(start_hex)
        lab2 = self.gen_from_hex(end_hex)
        dlab = lab2 - lab1

        # Fine-sample the GenSpace Lab line and build cumulative CIEDE2000 arc length
        N = 256
        cum_dist = np.zeros(N + 1)
        prev_cie = self._srgb_to_cielab(self.gen_to_srgb(lab1))
        for i in range(1, N + 1):
            t = i / N
            srgb = self.gen_to_srgb(lab1 + dlab * t)
            cie = self._srgb_to_cielab(srgb)
            cum_dist[i] = cum_dist[i - 1] + self._ciede2000(prev_cie, cie)
            prev_cie = cie
        total_dist = cum_dist[N]

        # Binary search for t values that produce equal cumulative distances
        result = []
        for s in range(steps):
            target = (s / (steps - 1)) * total_dist
            lo, hi = 0, N
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if cum_dist[mid] < target:
                    lo = mid
                else:
                    hi = mid
            denom = cum_dist[hi] - cum_dist[lo]
            frac = (target - cum_dist[lo]) / denom if denom > 1e-12 else 0.0
            t_new = (lo + frac) / N
            result.append(self.gen_to_hex(lab1 + dlab * t_new))
        return result

    # ── Palette Generation (GenSpace) ──────────────────────────────

    def palette(self, base_hex: str, steps: int = 10) -> list[str]:
        """Generate lightness palette from a base color.

        Evenly spaced L from near-white to near-black (GenSpace L range).
        Hue and chroma preserved, gamut clamped. Uses GenSpace.
        """
        lab = self.gen_from_hex(base_hex)
        L_hi = self._gen_white_L - 0.01
        L_lo = 0.05
        L_values = np.linspace(L_hi, L_lo, steps)
        result = []
        for L in L_values:
            sample = lab.copy()
            sample[0] = L
            result.append(self.gen_to_hex(sample))
        return result

    def palette_hues(
        self, lightness: float = 0.6, chroma: float = 0.15, steps: int = 12
    ) -> list[str]:
        """Generate hue ring at fixed lightness and chroma.

        Hue angles evenly distributed 0–360°. Uses GenSpace.
        """
        hues = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        result = []
        for h in hues:
            a = chroma * np.cos(h)
            b = chroma * np.sin(h)
            lab = np.array([lightness, a, b])
            result.append(self.gen_to_hex(lab))
        return result

    # ── Semantic Scale (GenSpace) ──────────────────────────────────

    def semantic_scale(
        self,
        base_hex: str,
        levels: list[int] | None = None,
    ) -> dict[str, str]:
        """Generate Tailwind-style semantic scale (50–950).

        base_hex maps to level 500. Lightness distributed with smooth
        mapping: lower levels → lighter, higher levels → darker.
        Uses GenSpace.
        """
        if levels is None:
            levels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]

        lab = self.gen_from_hex(base_hex)
        base_L = float(lab[0])

        L_light = self._gen_white_L - 0.01
        L_dark = 0.05

        result = {}
        for level in levels:
            if level <= 500:
                t = level / 500.0
                L = L_light + t * (base_L - L_light)
            else:
                t = (level - 500) / 450.0
                L = base_L + t * (L_dark - base_L)
            sample = lab.copy()
            sample[0] = L
            result[str(level)] = self.gen_to_hex(sample)
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
        Falls back to soft L-inversion via GenSpace otherwise.
        """
        if from_mode == to_mode:
            return color_hex

        S_MAP = {"light": 0.7, "dark": 0.2}

        if self._metric._has_surround:
            S_src = S_MAP.get(from_mode, 0.5)
            S_dst = S_MAP.get(to_mode, 0.5)
            srgb = hex_to_srgb(color_hex)
            XYZ = sRGB_to_XYZ(srgb)
            lab_src = self._metric.from_XYZ(XYZ, S=S_src)
            XYZ_dst = self._metric.to_XYZ(lab_src, S=S_dst)
            srgb_dst = clamp_srgb(XYZ_to_sRGB(XYZ_dst))
            return srgb_to_hex(srgb_dst)

        # Fallback: soft L-inversion via GenSpace
        L_max = self._gen_white_L - 0.02
        LIGHT_LO, LIGHT_HI = 0.05, L_max
        DARK_LO, DARK_HI = 0.08, L_max - 0.05

        lab = self.gen_from_hex(color_hex)
        L = float(lab[0])

        if from_mode == "light":
            src_lo, src_hi = LIGHT_LO, LIGHT_HI
            dst_lo, dst_hi = DARK_LO, DARK_HI
        else:
            src_lo, src_hi = DARK_LO, DARK_HI
            dst_lo, dst_hi = LIGHT_LO, LIGHT_HI

        t = np.clip((L - src_lo) / (src_hi - src_lo), 0.0, 1.0)
        L_new = dst_hi - t * (dst_hi - dst_lo)

        lab[0] = L_new
        return self.gen_to_hex(lab)

    def adapt_pair(
        self,
        fg_hex: str,
        bg_hex: str,
        from_mode: str = "light",
        to_mode: str = "dark",
        min_ratio: float = 4.5,
    ) -> tuple[str, str]:
        """Adapt fg/bg pair to target mode, ensuring contrast."""
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
        from helmlab.export import TokenExporter
        return TokenExporter(self)

    def info(self, color_hex: str) -> dict:
        """Return color information dict (metric pipeline)."""
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
