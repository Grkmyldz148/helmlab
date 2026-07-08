"""Helmlab 1.0 — two-space color API.

One `Helmlab` instance exposes three namespaces:

    hl.gen     — CREATE colors (GenSpace): gradient, palette, scale,
                 harmonies, contrast, dark mode. Wide-gamut output via the
                 `gamut` option on every generation function.
    hl.metric  — MEASURE colors (MetricSpace): difference (trained metric),
                 euclidean, ciede2000, jnd, confidence, nearest, info.
    hl.tokens  — EXPORT design tokens (hex/CSS strings in, strings out).

The two spaces have branded Lab types (`GenLab` / `MetricLab`); passing one
space's Lab into the other's API raises `TypeError`. Color-string parameters
accept '#rrggbb', '#rgb', 'color(display-p3 r g b)' and
'color(rec2020 r g b)' everywhere.

See API.md for the full specification.
"""

import re
import warnings

import numpy as np

from helmlab.spaces.metric import MetricSpace, MetricParams
from helmlab.spaces.gen import GenSpace, GenParams
from helmlab.utils.srgb_convert import (
    hex_to_srgb,
    srgb_to_hex,
    sRGB_to_XYZ,
    XYZ_to_sRGB,
    XYZ_to_DisplayP3,
    DisplayP3_to_XYZ,
    linear_to_displayp3,
    displayp3_to_linear,
    XYZ_to_Rec2020,
    Rec2020_to_XYZ,
    linear_to_rec2020,
    rec2020_to_linear,
    clamp_srgb,
    relative_luminance,
    contrast_ratio as _wcag_cr,
)
from helmlab.utils.gamut import gamut_map, is_in_gamut, max_chroma as _max_chroma, find_cusp as _find_cusp


# ═══════════════════════════════════════════════════════════════════════
# Branded Lab types + errors
# ═══════════════════════════════════════════════════════════════════════

class GenLab(np.ndarray):
    """Lab coordinates in the GENERATION space (`hl.gen`).

    A plain ``np.ndarray`` subclass — indexing, ``.copy()``, arithmetic all
    work — that carries its space identity so it cannot be silently fed to
    the measurement API. Construct via ``hl.gen.from_hex`` / ``hl.gen.lab``.
    """
    __slots__ = ()


class MetricLab(np.ndarray):
    """Lab coordinates in the MEASUREMENT space (`hl.metric`).

    See :class:`GenLab`; construct via ``hl.metric.from_hex`` /
    ``hl.metric.lab``.
    """
    __slots__ = ()


class ContrastError(ValueError):
    """Raised by ``ensure_contrast(strict=True)`` when the requested ratio is
    unreachable against the given background (even with pure black/white)."""


def _brand(arr, cls):
    return np.asarray(arr, dtype=np.float64).view(cls)


def _accept_lab(lab, forbidden_cls, this_ns, that_ns):
    """Runtime space check: reject the OTHER space's branded Lab, accept the
    own brand and plain arrays/lists (interop escape hatch)."""
    if isinstance(lab, forbidden_cls):
        raise TypeError(
            f"hl.{this_ns} got a {forbidden_cls.__name__} — that Lab belongs "
            f"to hl.{that_ns}. The two spaces have different coordinates; "
            f"convert the original color with hl.{this_ns}.from_hex() instead."
        )
    return np.asarray(lab, dtype=np.float64).view(np.ndarray)


# ═══════════════════════════════════════════════════════════════════════
# Color-string parsing (shared boundary)
# ═══════════════════════════════════════════════════════════════════════

_CSS_COLOR_RE = re.compile(
    r"^\s*color\(\s*(display-p3|rec2020)\s+"
    r"([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*\)\s*$"
)

_VALID_GAMUTS = ("srgb", "display-p3", "rec2020")


def parse_color(color: str) -> np.ndarray:
    """Any supported color string → CIE XYZ (D65).

    Accepts '#rrggbb', '#rgb', 'color(display-p3 r g b)',
    'color(rec2020 r g b)'. Raises ``ValueError`` on anything else.
    """
    if not isinstance(color, str):
        raise TypeError(
            f"expected a color string, got {type(color).__name__} — Lab "
            "arrays go to the lab-typed methods (to_hex, distance, ...)"
        )
    s = color.strip()
    if s.startswith("color("):
        m = _CSS_COLOR_RE.match(s)
        if not m:
            raise ValueError(
                f"unparseable CSS color() string: {color!r} — supported: "
                "'color(display-p3 r g b)', 'color(rec2020 r g b)'"
            )
        comps = np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        if m.group(1) == "display-p3":
            return DisplayP3_to_XYZ(displayp3_to_linear(comps))
        return Rec2020_to_XYZ(rec2020_to_linear(comps))
    return sRGB_to_XYZ(hex_to_srgb(s))


def _color_to_srgb(color: str) -> np.ndarray:
    """Color string → sRGB [0,1] (wide-gamut inputs are clipped — used only
    for sRGB-defined operations like WCAG luminance)."""
    s = color.strip()
    if s.startswith("color("):
        return clamp_srgb(XYZ_to_sRGB(parse_color(s)))
    return hex_to_srgb(s)


def _check_gamut(gamut: str) -> None:
    if gamut not in _VALID_GAMUTS:
        raise ValueError(f"unknown gamut {gamut!r}; use one of {_VALID_GAMUTS}")


# ═══════════════════════════════════════════════════════════════════════
# CIELAB / CIEDE2000 (reference formulas, module-level)
# ═══════════════════════════════════════════════════════════════════════

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
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def _xyz_to_cielab(XYZ):
    """CIE XYZ (D65) → CIE Lab. Same white constants as :func:`_srgb_to_cielab`."""
    x = float(XYZ[0]) / 0.95047
    y = float(XYZ[1])
    z = float(XYZ[2]) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def _ciede2000(lab1, lab2):
    """CIEDE2000 color difference (full CIE 224:2017 formula)."""
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])
    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    Cab = (C1 + C2) / 2
    Cab7 = Cab ** 7
    p257 = 25.0 ** 7
    G = 0.5 * (1 - np.sqrt(Cab7 / (Cab7 + p257)))
    ap1 = (1 + G) * a1
    ap2 = (1 + G) * a2
    Cp1 = np.sqrt(ap1 * ap1 + b1 * b1)
    Cp2 = np.sqrt(ap2 * ap2 + b2 * b2)
    hp1 = np.arctan2(b1, ap1)
    hp2 = np.arctan2(b2, ap2)
    if hp1 < 0:
        hp1 += 2 * np.pi
    if hp2 < 0:
        hp2 += 2 * np.pi
    dLp = L2 - L1
    dCp = Cp2 - Cp1
    if Cp1 * Cp2 == 0:
        dhp = 0.0
    else:
        dhp = hp2 - hp1
        if dhp > np.pi:
            dhp -= 2 * np.pi
        if dhp < -np.pi:
            dhp += 2 * np.pi
    dHp = 2 * np.sqrt(Cp1 * Cp2) * np.sin(dhp / 2)
    Lp = (L1 + L2) / 2
    Cp = (Cp1 + Cp2) / 2
    if Cp1 * Cp2 == 0:
        hp = hp1 + hp2
    else:
        hp = (hp1 + hp2) / 2
        if abs(hp1 - hp2) > np.pi:
            hp += np.pi if hp < np.pi else -np.pi
    T = (1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp)
         + 0.32 * np.cos(3 * hp + np.pi / 30) - 0.20 * np.cos(4 * hp - 63 * np.pi / 180))
    Lp50 = Lp - 50
    SL = 1 + 0.015 * Lp50 * Lp50 / np.sqrt(20 + Lp50 * Lp50)
    SC = 1 + 0.045 * Cp
    SH = 1 + 0.015 * Cp * T
    Cp7 = Cp ** 7
    RC = 2 * np.sqrt(Cp7 / (Cp7 + p257))
    hpDeg = hp * 180 / np.pi
    dth = 30 * np.exp(-((hpDeg - 275) / 25) ** 2)
    RT = -np.sin(2 * dth * np.pi / 180) * RC
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                   + RT * (dCp / SC) * (dHp / SH))


# ═══════════════════════════════════════════════════════════════════════
# hl.gen — generation namespace
# ═══════════════════════════════════════════════════════════════════════

_HARMONY_OFFSETS = {
    "complementary": (0, 180),
    "analogous": (0, -30, 30),
    "triadic": (0, 120, 240),
    "tetradic": (0, 90, 180, 270),
    "split_complementary": (0, 150, 210),
}

_DEFAULT_SCALE_LEVELS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]


class Gen:
    """Generation namespace — everything that CREATES colors (GenSpace).

    All generation functions take ``gamut='srgb' | 'display-p3' | 'rec2020'``
    and return '#rrggbb' for sRGB or 'color(<gamut> r g b)' CSS strings for
    the wide gamuts. Color-string inputs accept the same forms everywhere.
    """

    def __init__(self, space: GenSpace, metric_space: MetricSpace, owner: "Helmlab"):
        self.space = space          # raw GenSpace (advanced access)
        self._metric = metric_space  # used by adapt_to_mode surround path
        self._owner = owner
        self._white_L = float(space.from_XYZ(np.array([0.95047, 1.0, 1.08883]))[0])

    # ── Conversions ─────────────────────────────────────────────────

    def from_hex(self, color: str) -> GenLab:
        """Color string → GenLab. Accepts '#rrggbb', '#rgb', 'color(display-p3 …)', 'color(rec2020 …)'."""
        return _brand(self.space.from_XYZ(parse_color(color)), GenLab)

    def to_hex(self, lab) -> str:
        """GenLab → '#rrggbb' (gamut-mapped to sRGB)."""
        return self._emit(self._accept(lab), "srgb")

    def to_css(self, lab, gamut: str = "display-p3") -> str:
        """GenLab → CSS color string in the requested gamut (gamut-mapped)."""
        return self._emit(self._accept(lab), gamut)

    def from_srgb(self, srgb) -> GenLab:
        """sRGB [0,1] → GenLab."""
        return _brand(self.space.from_XYZ(sRGB_to_XYZ(np.asarray(srgb, dtype=np.float64))), GenLab)

    def to_srgb(self, lab) -> np.ndarray:
        """GenLab → sRGB [0,1] (gamut-mapped, clamped)."""
        return self._to_srgb_nd(self._accept(lab))

    def lab(self, L: float, a: float, b: float) -> GenLab:
        """Branded GenLab constructor from raw coordinates."""
        return _brand([L, a, b], GenLab)

    def lch(self, L: float, C: float, h_deg: float) -> GenLab:
        """Cylindrical constructor → GenLab. ⚠ L and C are 0–1-scale, not 0–100."""
        rad = float(np.radians(h_deg))
        return _brand([L, C * np.cos(rad), C * np.sin(rad)], GenLab)

    def to_lch(self, lab) -> np.ndarray:
        """GenLab → [L, C, h°] (plain array); h ∈ [0, 360).

        Same coordinates as the ``helmgenlch`` space registered on Color.js.
        For hue rotations prefer :meth:`rotate_hue` / :meth:`harmonies`.
        """
        nd = self._accept(lab)
        C = float(np.hypot(nd[1], nd[2]))
        h = float(np.degrees(np.arctan2(nd[2], nd[1]))) % 360.0
        return np.array([nd[0], C, h])

    def from_lch(self, lch) -> GenLab:
        """[L, C, h°] → GenLab."""
        lch = np.asarray(lch, dtype=np.float64)
        return self.lch(float(lch[0]), float(lch[1]), float(lch[2]))

    def gamut_map(self, lab, gamut: str = "srgb", method: str = "chroma") -> GenLab:
        """Map a GenLab into the requested gamut.

        ``method='chroma'`` (default) reduces chroma at constant L and hue —
        predictable, hue-exact. ``method='adaptive'`` is the Ottosson-style
        projection toward the cusp that trades a little L for more chroma —
        better for saturated photographic content.
        """
        _check_gamut(gamut)
        return _brand(gamut_map(self._accept(lab), self.space, gamut=gamut, method=method), GenLab)

    def max_chroma(self, lightness: float, hue_deg: float, gamut: str = "srgb") -> float:
        """Maximum in-gamut chroma at fixed lightness and hue.

        The gamut-boundary query behind :meth:`vivid` — use it to build
        saturation ramps or to know how much chroma headroom a wide gamut
        adds (``max_chroma(L, h, 'display-p3') / max_chroma(L, h, 'srgb')``).
        """
        _check_gamut(gamut)
        return float(_max_chroma(float(lightness), float(np.radians(hue_deg)), self.space, gamut))

    def cusp(self, hue_deg: float, gamut: str = "srgb") -> tuple[float, float]:
        """The (L, C) cusp of a hue — the single most colorful point of that
        hue leaf in the requested gamut. GenSpace resolves a clean cusp at
        all 360 hues in sRGB, P3 and Rec2020 (the ColorBench 360/360/360
        result); this exposes that geometry directly."""
        _check_gamut(gamut)
        L, C = _find_cusp(float(np.radians(hue_deg)), self.space, gamut)
        return float(L), float(C)

    def vivid(self, color: str, *, gamut: str = "srgb") -> str:
        """The most saturated version of a color: same lightness, same hue,
        chroma pushed to the gamut boundary. With ``gamut='display-p3'`` this
        is the honest "P3 makes this color pop" upgrade — hue and lightness
        stay put, only colorfulness grows."""
        _check_gamut(gamut)
        lch = self.to_lch(self.from_hex(color))
        C_max = self.max_chroma(lch[0], lch[2], gamut)
        return self._emit(self.lch(float(lch[0]), C_max, float(lch[2])).view(np.ndarray), gamut)

    def in_gamut(self, lab_or_color, gamut: str = "srgb") -> bool:
        """Whether a GenLab or color string is inside the requested gamut."""
        _check_gamut(gamut)
        if isinstance(lab_or_color, str):
            nd = self.space.from_XYZ(parse_color(lab_or_color)).view(np.ndarray)
        else:
            nd = self._accept(lab_or_color)
        return bool(is_in_gamut(nd, self.space, gamut))

    # ── Generation ──────────────────────────────────────────────────

    def gradient(self, start: str, end: str, steps: int = 16, *, gamut: str = "srgb") -> list[str]:
        """Perceptually uniform gradient between two colors.

        Equal perceptual step sizes on any color pair (GenSpace Lab path,
        CIEDE2000 arc-length reparameterization). Step spacing is measured on
        the sRGB projection of the path; with a wide ``gamut`` the emitted
        stops keep the extra chroma the target gamut allows.

        Edge cases: ``steps=1`` → ``[start]`` (end not included);
        ``steps < 1`` → ``[]``.
        """
        _check_gamut(gamut)
        if steps < 1:
            return []
        lab1 = self.from_hex(start).view(np.ndarray)
        if steps == 1:
            return [self._emit(lab1, gamut)]
        lab2 = self.from_hex(end).view(np.ndarray)
        fractions = [s / (steps - 1) for s in range(steps)]
        return [self._emit(p, gamut) for p in self._path_points(lab1, lab2, fractions)]

    def mix(self, a: str, b: str, t: float = 0.5, *, gamut: str = "srgb") -> str:
        """The color at fraction ``t`` (0 → a, 1 → b) along the SAME
        perceptual arc-length path :meth:`gradient` uses — ``mix(a, b, 0.5)``
        is the visual midpoint, not the coordinate average."""
        _check_gamut(gamut)
        t = float(np.clip(t, 0.0, 1.0))
        lab1 = self.from_hex(a).view(np.ndarray)
        lab2 = self.from_hex(b).view(np.ndarray)
        point = self._path_points(lab1, lab2, [t])[0]
        return self._emit(point, gamut)

    def palette(self, base: str, steps: int = 10, *, gamut: str = "srgb") -> list[str]:
        """Lightness ramp from a base color, light → dark.

        Evenly spaced GenSpace L; hue and chroma preserved, gamut mapped.
        The base color is only approximate inside the result (its L is
        re-spaced) — for a scale where the input is returned exactly at
        level 500 use :meth:`scale`. ``steps < 1`` returns ``[]``.
        """
        _check_gamut(gamut)
        if steps < 1:
            return []
        lab = self.from_hex(base).view(np.ndarray)
        if steps == 1:
            return [self._emit(lab, gamut)]
        L_hi = self._white_L - 0.01
        L_lo = 0.05
        result = []
        for L in np.linspace(L_hi, L_lo, steps):
            sample = lab.copy()
            sample[0] = L
            result.append(self._emit(sample, gamut))
        return result

    def scale(self, base: str, levels: list[int] | None = None, *, gamut: str = "srgb") -> dict[str, str]:
        """Tailwind-style semantic scale (50–950). ``base`` maps to level 500
        EXACTLY; lower levels are lighter, higher levels darker."""
        _check_gamut(gamut)
        if levels is None:
            levels = _DEFAULT_SCALE_LEVELS
        lab = self.from_hex(base).view(np.ndarray)
        base_L = float(lab[0])
        L_light = self._white_L - 0.01
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
            result[str(level)] = self._emit(sample, gamut)
        return result

    def hue_ring(self, count: int = 12, *, lightness: float = 0.6,
                 chroma: float = 0.15, gamut: str = "srgb") -> list[str]:
        """Categorical palette: ``count`` evenly spaced hues at fixed
        lightness and chroma (equal perceptual weight per category)."""
        _check_gamut(gamut)
        result = []
        for h in np.linspace(0, 2 * np.pi, count, endpoint=False):
            lab = np.array([lightness, chroma * np.cos(h), chroma * np.sin(h)])
            result.append(self._emit(lab, gamut))
        return result

    def harmonies(self, base: str, kind: str = "complementary", *, gamut: str = "srgb") -> list[str]:
        """Classic hue harmonies with the base color first.

        ``kind`` ∈ complementary [0°, 180°] · analogous [0°, ±30°] ·
        triadic [0°, 120°, 240°] · tetradic [0°, 90°, 180°, 270°] ·
        split_complementary [0°, 150°, 210°]. Rotations happen in GenSpace
        LCh at constant L and C — perceptual lightness and colorfulness stay
        matched across the set (the property hue harmonies assume but sRGB
        HSL rotations don't deliver).
        """
        _check_gamut(gamut)
        if kind not in _HARMONY_OFFSETS:
            raise ValueError(
                f"unknown harmony kind {kind!r}; use one of {sorted(_HARMONY_OFFSETS)}"
            )
        lch = self.to_lch(self.from_hex(base))
        return [
            self._emit(self.lch(lch[0], lch[1], (lch[2] + off) % 360.0).view(np.ndarray), gamut)
            for off in _HARMONY_OFFSETS[kind]
        ]

    def rotate_hue(self, color: str, degrees: float, *, gamut: str = "srgb") -> str:
        """Rotate a color's hue by ``degrees`` at constant L and C (GenLCh)."""
        _check_gamut(gamut)
        lch = self.to_lch(self.from_hex(color))
        return self._emit(self.lch(lch[0], lch[1], (lch[2] + degrees) % 360.0).view(np.ndarray), gamut)

    # ── Contrast ────────────────────────────────────────────────────

    def contrast_ratio(self, fg: str, bg: str) -> float:
        """WCAG 2.1 contrast ratio between two colors (1.0 – 21.0)."""
        return float(_wcag_cr(_color_to_srgb(fg), _color_to_srgb(bg)))

    def meets_contrast(self, fg: str, bg: str, level: str = "AA") -> bool:
        """Whether fg/bg meets WCAG level AA (4.5:1) or AAA (7:1)."""
        thresholds = {"AA": 4.5, "AAA": 7.0}
        return self.contrast_ratio(fg, bg) >= thresholds.get(level.upper(), 4.5)

    def ensure_contrast(self, fg: str, bg: str, ratio: float = 4.5, *, strict: bool = False) -> str:
        """Return a variant of ``fg`` meeting ``ratio`` against ``bg``.

        Binary search on the GenSpace L axis — hue and chroma are preserved
        while a lightness-only adjustment can reach the ratio. If no
        lightness of this hue reaches it, falls back to pure #000000/#ffffff
        (hue lost). If even black/white cannot reach the ratio (possible for
        mid-gray backgrounds with high ratios, e.g. 7:1 vs #808080):
        ``strict=True`` raises :class:`ContrastError`; ``strict=False``
        (default) warns and returns the best effort BELOW the requested
        ratio. Check with :meth:`meets_contrast` for hard requirements.
        """
        current = self.contrast_ratio(fg, bg)
        if current >= ratio:
            return srgb_to_hex(_color_to_srgb(fg))

        fg_lab = self.from_hex(fg).view(np.ndarray)
        bg_srgb = _color_to_srgb(bg)

        best_hex = srgb_to_hex(_color_to_srgb(fg))
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
                candidate_srgb = self._to_srgb_nd(candidate_lab)
                hex_quantized = hex_to_srgb(srgb_to_hex(candidate_srgb))
                r = float(_wcag_cr(hex_quantized, bg_srgb))
                if direction == "darken":
                    if r >= ratio:
                        lo = mid
                    else:
                        hi = mid
                else:
                    if r >= ratio:
                        hi = mid
                    else:
                        lo = mid

            safe_L = lo if direction == "darken" else hi
            if direction == "darken":
                safe_L = max(0.0, safe_L - 0.003)
            else:
                safe_L = min(1.5, safe_L + 0.003)
            candidate_lab[0] = safe_L
            candidate_hex = srgb_to_hex(self._to_srgb_nd(candidate_lab))
            r = float(_wcag_cr(hex_to_srgb(candidate_hex), bg_srgb))

            if r >= ratio:
                dist = abs(safe_L - orig_L)
                best_dist = abs(best_L - orig_L)
                if best_ratio < ratio or dist < best_dist:
                    best_hex = candidate_hex
                    best_ratio = r
                    best_L = safe_L

        if best_ratio < ratio:
            for fallback in ("#000000", "#ffffff"):
                if self.contrast_ratio(fallback, bg) >= ratio:
                    return fallback
            r_black = self.contrast_ratio("#000000", bg)
            r_white = self.contrast_ratio("#ffffff", bg)
            best = "#000000" if r_black > r_white else "#ffffff"
            msg = (
                f"ensure_contrast: no color reaches {ratio}:1 against {bg} "
                f"(best achievable: {max(r_black, r_white):.2f}:1 with {best})."
            )
            if strict:
                raise ContrastError(msg)
            warnings.warn(msg + " Returning best effort below the requested ratio.",
                          stacklevel=2)
            return best

        return best_hex

    # ── Dark/light mode ─────────────────────────────────────────────

    def adapt_to_mode(self, color: str, from_mode: str = "light", to_mode: str = "dark") -> str:
        """Adapt a color between light and dark mode.

        Uses the metric surround pipeline when S-dependent params are
        active; otherwise a soft L-inversion in GenSpace.
        """
        if from_mode == to_mode:
            return srgb_to_hex(_color_to_srgb(color))

        S_MAP = {"light": 0.7, "dark": 0.2}

        if self._metric._has_surround:
            S_src = S_MAP.get(from_mode, 0.5)
            S_dst = S_MAP.get(to_mode, 0.5)
            XYZ = parse_color(color)
            lab_src = self._metric.from_XYZ(XYZ, S=S_src)
            XYZ_dst = self._metric.to_XYZ(lab_src, S=S_dst)
            return srgb_to_hex(clamp_srgb(XYZ_to_sRGB(XYZ_dst)))

        L_max = self._white_L - 0.02
        LIGHT_LO, LIGHT_HI = 0.05, L_max
        DARK_LO, DARK_HI = 0.08, L_max - 0.05

        lab = self.from_hex(color).view(np.ndarray)
        L = float(lab[0])

        if from_mode == "light":
            src_lo, src_hi = LIGHT_LO, LIGHT_HI
            dst_lo, dst_hi = DARK_LO, DARK_HI
        else:
            src_lo, src_hi = DARK_LO, DARK_HI
            dst_lo, dst_hi = LIGHT_LO, LIGHT_HI

        t = np.clip((L - src_lo) / (src_hi - src_lo), 0.0, 1.0)
        lab[0] = dst_hi - t * (dst_hi - dst_lo)
        return self._emit(lab, "srgb")

    def adapt_pair(self, fg: str, bg: str, from_mode: str = "light",
                   to_mode: str = "dark", ratio: float = 4.5) -> tuple[str, str]:
        """Adapt an fg/bg pair to the target mode, then re-ensure contrast."""
        new_fg = self.adapt_to_mode(fg, from_mode, to_mode)
        new_bg = self.adapt_to_mode(bg, from_mode, to_mode)
        new_fg = self.ensure_contrast(new_fg, new_bg, ratio)
        return new_fg, new_bg

    # ── Internals ───────────────────────────────────────────────────

    def _accept(self, lab) -> np.ndarray:
        return _accept_lab(lab, MetricLab, "gen", "metric")

    def _to_srgb_nd(self, lab_nd: np.ndarray) -> np.ndarray:
        mapped = gamut_map(lab_nd, self.space, gamut="srgb")
        return clamp_srgb(XYZ_to_sRGB(self.space.to_XYZ(mapped)))

    def _emit(self, lab_nd: np.ndarray, gamut: str) -> str:
        """Gamut-map in GenSpace and emit the color string for the gamut."""
        if gamut == "srgb":
            return srgb_to_hex(self._to_srgb_nd(lab_nd))
        mapped = gamut_map(lab_nd, self.space, gamut=gamut)
        XYZ = self.space.to_XYZ(mapped)
        if gamut == "display-p3":
            c = clamp_srgb(linear_to_displayp3(XYZ_to_DisplayP3(XYZ)))
            return f"color(display-p3 {c[0]:.4f} {c[1]:.4f} {c[2]:.4f})"
        c = clamp_srgb(linear_to_rec2020(XYZ_to_Rec2020(XYZ)))
        return f"color(rec2020 {c[0]:.4f} {c[1]:.4f} {c[2]:.4f})"

    def _path_points(self, lab1: np.ndarray, lab2: np.ndarray, fractions) -> list[np.ndarray]:
        """Points along the CIEDE2000 arc-length-parameterized GenLab line at
        the given arc fractions (0..1). Shared by gradient() and mix()."""
        dlab = lab2 - lab1
        N = 256
        cum = np.zeros(N + 1)
        prev = _srgb_to_cielab(self._to_srgb_nd(lab1))
        for i in range(1, N + 1):
            cie = _srgb_to_cielab(self._to_srgb_nd(lab1 + dlab * (i / N)))
            cum[i] = cum[i - 1] + _ciede2000(prev, cie)
            prev = cie
        total = cum[N]

        points = []
        for f in fractions:
            target = f * total
            lo, hi = 0, N
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if cum[mid] < target:
                    lo = mid
                else:
                    hi = mid
            denom = cum[hi] - cum[lo]
            frac = (target - cum[lo]) / denom if denom > 1e-12 else 0.0
            points.append(lab1 + dlab * ((lo + frac) / N))
        return points


# ═══════════════════════════════════════════════════════════════════════
# hl.metric — measurement namespace
# ═══════════════════════════════════════════════════════════════════════

class Metric:
    """Measurement namespace — everything that MEASURES colors (MetricSpace).

    Four clearly-named difference metrics:
    :meth:`difference` (trained, saturates ~0.15), :meth:`euclidean`
    (unbounded ΔE76-analogue), :meth:`ciede2000` (industry standard),
    :meth:`jnd` (difference in just-noticeable-difference units).
    """

    def __init__(self, space: MetricSpace, owner: "Helmlab"):
        self.space = space  # raw MetricSpace (advanced access; XYZ-in distance lives here)
        self._owner = owner
        self._jnd_de = None       # lazy: tau[1]/mu_scale from confidence params
        self._confidence = None   # lazy ConfidenceModel

    # ── Conversions ─────────────────────────────────────────────────

    def from_hex(self, color: str) -> MetricLab:
        """Color string → MetricLab. Accepts '#rrggbb', '#rgb', 'color(display-p3 …)', 'color(rec2020 …)'."""
        return _brand(self.space.from_XYZ(parse_color(color)), MetricLab)

    def to_hex(self, lab) -> str:
        """MetricLab → '#rrggbb' (gamut-mapped to sRGB)."""
        return srgb_to_hex(self._to_srgb_nd(self._accept(lab)))

    def to_css(self, lab, gamut: str = "display-p3") -> str:
        """MetricLab → CSS color string in the requested gamut (gamut-mapped)."""
        _check_gamut(gamut)
        nd = self._accept(lab)
        if gamut == "srgb":
            return self.to_hex(nd)
        mapped = gamut_map(nd, self.space, gamut=gamut)
        XYZ = self.space.to_XYZ(mapped)
        if gamut == "display-p3":
            c = clamp_srgb(linear_to_displayp3(XYZ_to_DisplayP3(XYZ)))
            return f"color(display-p3 {c[0]:.4f} {c[1]:.4f} {c[2]:.4f})"
        c = clamp_srgb(linear_to_rec2020(XYZ_to_Rec2020(XYZ)))
        return f"color(rec2020 {c[0]:.4f} {c[1]:.4f} {c[2]:.4f})"

    def from_srgb(self, srgb) -> MetricLab:
        """sRGB [0,1] → MetricLab."""
        return _brand(self.space.from_XYZ(sRGB_to_XYZ(np.asarray(srgb, dtype=np.float64))), MetricLab)

    def to_srgb(self, lab) -> np.ndarray:
        """MetricLab → sRGB [0,1] (gamut-mapped, clamped)."""
        return self._to_srgb_nd(self._accept(lab))

    def from_xyz(self, XYZ) -> MetricLab:
        """CIE XYZ (D65) → MetricLab."""
        return _brand(self.space.from_XYZ(np.asarray(XYZ, dtype=np.float64)), MetricLab)

    def to_xyz(self, lab) -> np.ndarray:
        """MetricLab → CIE XYZ (D65)."""
        return self.space.to_XYZ(self._accept(lab))

    def lab(self, L: float, a: float, b: float) -> MetricLab:
        """Branded MetricLab constructor from raw coordinates."""
        return _brand([L, a, b], MetricLab)

    def lch(self, L: float, C: float, h_deg: float) -> MetricLab:
        """Cylindrical constructor → MetricLab. ⚠ L and C are 0–1-scale, not 0–100."""
        rad = float(np.radians(h_deg))
        return _brand([L, C * np.cos(rad), C * np.sin(rad)], MetricLab)

    def to_lch(self, lab) -> np.ndarray:
        """MetricLab → [L, C, h°] (plain array); h ∈ [0, 360).

        Cylindrical view of Metric Lab — the same C and H that
        :meth:`info` reports. For hue manipulation prefer the GENERATION
        space (``hl.gen.rotate_hue`` / ``harmonies``): Metric Lab is tuned
        for difference prediction, not for producing colors.
        """
        nd = self._accept(lab)
        C = float(np.hypot(nd[1], nd[2]))
        h = float(np.degrees(np.arctan2(nd[2], nd[1]))) % 360.0
        return np.array([nd[0], C, h])

    def from_lch(self, lch) -> MetricLab:
        """[L, C, h°] → MetricLab."""
        lch = np.asarray(lch, dtype=np.float64)
        return self.lch(float(lch[0]), float(lch[1]), float(lch[2]))

    def in_gamut(self, lab_or_color, gamut: str = "srgb") -> bool:
        """Whether a MetricLab or color string is inside the requested gamut."""
        _check_gamut(gamut)
        if isinstance(lab_or_color, str):
            nd = self.space.from_XYZ(parse_color(lab_or_color)).view(np.ndarray)
        else:
            nd = self._accept(lab_or_color)
        return bool(is_in_gamut(nd, self.space, gamut))

    # ── The four difference metrics ─────────────────────────────────

    def difference(self, a: str, b: str) -> float:
        """THE trained perceptual difference between two colors (v21 metric,
        COMBVD STRESS 22.48 — 23% better than CIEDE2000's 29.20).

        Best for near-threshold judgments. Monotonic compression saturates
        near ~0.15 for very different colors: rank is preserved but absolute
        values plateau — for an unbounded number use :meth:`euclidean`, for
        threshold units use :meth:`jnd`.
        """
        return float(self.space.distance(parse_color(a), parse_color(b)))

    def euclidean(self, a: str, b: str) -> float:
        """Uncompressed Euclidean distance in Metric Lab (ΔE76-analogue).

        Unbounded companion to :meth:`difference`: range ≈ 0–1.6 across sRGB
        (#000000 ↔ #ffffff ≈ 1.12, #ff0000 ↔ #00ff00 ≈ 1.62).
        """
        lab1 = self.from_hex(a).view(np.ndarray)
        lab2 = self.from_hex(b).view(np.ndarray)
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))

    def ciede2000(self, a: str, b: str) -> float:
        """CIEDE2000 ΔE between two colors (CIELAB scale, ~0–100).

        The industry-standard formula, self-contained. Recommended for
        fixed-catalog nearest-color matching: measurably more stable under
        tiny input perturbations than the trained metric (7 vs 16 selection
        flips per 100 targets at ±2/255), because :meth:`difference` is fit
        for near-threshold judgments, not suprathreshold ranking.
        Sanity anchor: ``ciede2000('#ff0000', '#00ff00') ≈ 86.6``.
        """
        return float(_ciede2000(self._cielab(a), self._cielab(b)))

    def jnd(self, a: str, b: str) -> float:
        """Difference in just-noticeable-difference units.

        ``jnd = difference / 0.0356`` where 0.0356 de is the point at which
        the median observer of the v2 ordinal confidence model rates a pair
        at least "moderately different" (``tau[1] / mu_scale``, read from the
        bundled confidence params). Interpretation: < 1 likely unnoticed,
        1–2 subtle, > 2 clearly visible. Because :meth:`difference` saturates
        near ~0.15, values above ≈ 4.2 mean "far above threshold", not a
        precise multiple.
        """
        if self._jnd_de is None:
            m = self._confidence_model()
            self._jnd_de = float(m.p["tau"][1] / m.p["mu_scale"])
        return self.difference(a, b) / self._jnd_de

    # ── Lab-level distance (identical contract in both languages) ───

    def distance(self, lab_a, lab_b):
        """Trained perceptual distance between two MetricLab values.

        Identical contract in Python and JS: MetricLab in, distance out (the
        0.x Python-XYZ/JS-Lab asymmetry is gone; the XYZ-in variant lives
        only on the raw ``MetricSpace`` class). Supports batching (..., 3).
        Same saturation behavior as :meth:`difference`.
        """
        a = self._accept(lab_a)
        b = self._accept(lab_b)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("distance got non-finite (NaN/Inf) Lab input.")
        if max(np.max(np.abs(a[..., 0])), np.max(np.abs(b[..., 0]))) > 3.0:
            raise ValueError(
                "distance expects HELMLAB Lab (L≈0..1), but got L>3 — this "
                "looks like CIELAB (L*=0..100). Convert via from_hex/from_xyz first."
            )
        res = self.space.distance(self.space.to_XYZ(a), self.space.to_XYZ(b))
        return float(res) if np.ndim(res) == 0 else res

    # ── Confidence / catalog / info ─────────────────────────────────

    def confidence(self, a: str, b: str) -> dict:
        """Perceptual difference plus how much observers will disagree.

        EXPERIMENTAL (v2 ordinal model, n=47 pairs — see
        :mod:`helmlab.metrics.confidence` for provenance and limits).
        Returns ``de``, ``latent`` (mu), ``disagreement`` (sigma),
        ``reliability`` ∈ [0, 1), ``p_noticeable`` (chance a random observer
        rates the pair at least moderately different), ``reliable`` and
        ``extrapolated`` (True when de is beyond the training range — treat
        the numbers as indicative only).
        """
        XYZ1, XYZ2 = parse_color(a), parse_color(b)
        lab1 = self.space.from_XYZ(XYZ1).view(np.ndarray)
        lab2 = self.space.from_XYZ(XYZ2).view(np.ndarray)
        de = float(self.space.distance(XYZ1, XYZ2))
        chroma = 0.5 * (float(np.hypot(lab1[1], lab1[2])) + float(np.hypot(lab2[1], lab2[2])))
        out = self._confidence_model().assess(de, chroma)
        out["de"] = de
        return out

    def nearest(self, target: str, palette: list[str], metric: str = "ciede2000") -> dict:
        """Pick the perceptually nearest color to ``target`` from ``palette``.

        ``metric``: ``'ciede2000'`` (default — most perturbation-stable for
        catalog argmax), ``'difference'`` (trained, near-threshold),
        ``'euclidean'`` (fast unbounded). Returns ``hex``, ``index``,
        ``distance``, ``runner_up``, ``runner_up_distance`` and ``margin``
        (relative gap to the runner-up — small margins mean a fragile pick).
        """
        if not palette:
            raise ValueError("palette is empty")
        fns = {
            "ciede2000": self.ciede2000,
            "difference": self.difference,
            "euclidean": self.euclidean,
        }
        if metric not in fns:
            raise ValueError(f"unknown metric {metric!r}; use one of {sorted(fns)}")
        fn = fns[metric]
        dists = [float(fn(target, c)) for c in palette]
        order = sorted(range(len(palette)), key=lambda i: dists[i])
        best, second = order[0], (order[1] if len(order) > 1 else order[0])
        d0, d1 = dists[best], dists[second]
        return {
            "hex": palette[best],
            "index": best,
            "distance": d0,
            "runner_up": palette[second],
            "runner_up_distance": d1,
            "margin": (d1 - d0) / d0 if d0 > 0 else float("inf"),
            "metric": metric,
        }

    def info(self, color: str) -> dict:
        """Descriptive dict for a color: hex, srgb, xyz, lab, L, C, H,
        luminance, in_srgb, in_p3, in_rec2020."""
        XYZ = parse_color(color)
        srgb = clamp_srgb(XYZ_to_sRGB(XYZ))
        lab = self.space.from_XYZ(XYZ).view(np.ndarray)
        C = float(np.sqrt(lab[1] ** 2 + lab[2] ** 2))
        H_deg = float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)
        return {
            "hex": srgb_to_hex(srgb),
            "srgb": srgb.tolist(),
            "xyz": XYZ.tolist(),
            "lab": lab.tolist(),
            "L": float(lab[0]),
            "C": C,
            "H": H_deg,
            "luminance": float(relative_luminance(srgb)),
            "in_srgb": bool(is_in_gamut(lab, self.space, "srgb")),
            "in_p3": bool(is_in_gamut(lab, self.space, "display-p3")),
            "in_rec2020": bool(is_in_gamut(lab, self.space, "rec2020")),
        }

    # ── Internals ───────────────────────────────────────────────────

    def _accept(self, lab) -> np.ndarray:
        return _accept_lab(lab, GenLab, "metric", "gen")

    def _to_srgb_nd(self, lab_nd: np.ndarray) -> np.ndarray:
        mapped = gamut_map(lab_nd, self.space, gamut="srgb")
        return clamp_srgb(XYZ_to_sRGB(self.space.to_XYZ(mapped)))

    def _cielab(self, color: str) -> np.ndarray:
        s = color.strip()
        if s.startswith("color("):
            return _xyz_to_cielab(parse_color(s))
        return _srgb_to_cielab(hex_to_srgb(s))

    def _confidence_model(self):
        if self._confidence is None:
            from helmlab.metrics.confidence import ConfidenceModel
            self._confidence = ConfidenceModel()
        return self._confidence


# ═══════════════════════════════════════════════════════════════════════
# Helmlab — the entry point
# ═══════════════════════════════════════════════════════════════════════

class Helmlab:
    """Two purpose-built color spaces behind three namespaces.

    ``hl.gen`` creates colors (GenSpace — ColorBench 62-9-19 vs OKLab),
    ``hl.metric`` measures them (MetricSpace — COMBVD STRESS 22.48),
    ``hl.tokens`` exports design tokens. See API.md.
    """

    def __init__(self, metric_params: str | None = None, gen_params: str | None = None,
                 surround: float = 0.5, neutral_correction: bool = True,
                 ab_rotate_deg: float = -28.2):
        self._surround = float(np.clip(surround, 0.0, 1.0))

        if metric_params is not None:
            mp = MetricParams.load(metric_params)
            metric_space = MetricSpace(mp, surround=self._surround,
                                       neutral_correction=neutral_correction,
                                       ab_rotate_deg=ab_rotate_deg)
        else:
            metric_space = MetricSpace(surround=self._surround,
                                       neutral_correction=neutral_correction,
                                       ab_rotate_deg=ab_rotate_deg)

        if gen_params is not None:
            gen_space = GenSpace(GenParams.load(gen_params))
        else:
            gen_space = GenSpace()

        self.gen = Gen(gen_space, metric_space, self)
        self.metric = Metric(metric_space, self)

        from helmlab.tokens import Tokens
        self.tokens = Tokens(self.metric)

    def set_surround(self, S: float):
        """Change viewing context. 0 = dark, 0.5 = normal, 1 = bright."""
        self._surround = float(np.clip(S, 0.0, 1.0))
        self.metric.space._surround = self._surround
