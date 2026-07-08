"""Design-token export — CSS, Android, iOS, Tailwind formats.

1.0 replacement for the 0.x ``TokenExporter``: every method takes color
STRINGS ('#rrggbb', 'color(display-p3 …)', …), never Lab arrays — the
Gen-Lab-into-the-exporter footgun is structurally gone. Access via
``hl.tokens``.
"""

import json as _json

import numpy as np

from helmlab.utils.srgb_convert import srgb_to_hex

# ── Oklab matrices (Björn Ottosson) — for CSS oklch() output ────────
_M1_OKLAB = np.array([
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715,  0.0361456387],
    [0.0482003018, 0.2643662691,  0.6338517070],
])

_M2_OKLAB = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])


def _XYZ_to_oklab(XYZ: np.ndarray) -> np.ndarray:
    """CIE XYZ (D65) → Oklab [L, a, b]."""
    XYZ = np.asarray(XYZ, dtype=np.float64)
    LMS = XYZ @ _M1_OKLAB.T
    LMS_c = np.sign(LMS) * np.abs(LMS) ** (1.0 / 3.0)
    return LMS_c @ _M2_OKLAB.T


def _oklab_to_oklch(lab: np.ndarray) -> tuple[float, float, float]:
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    C = np.sqrt(a ** 2 + b ** 2)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    return L, float(C), float(H)


class Tokens:
    """Design-token export namespace (``hl.tokens``). Color strings in,
    platform token strings out."""

    _CSS_FORMATS = ("hex", "rgb", "hsl", "oklch", "p3", "rec2020")

    def __init__(self, metric):
        self._metric = metric

    # ── Single-color formats ─────────────────────────────────────────

    def css(self, color: str, format: str = "oklch") -> str:
        """One CSS value for a color. ``format`` ∈ hex, rgb, hsl, oklch,
        p3 ('color(display-p3 …)'), rec2020."""
        if format not in self._CSS_FORMATS:
            raise ValueError(f"unknown format {format!r}; use one of {self._CSS_FORMATS}")
        m = self._metric
        lab = m.from_hex(color)
        if format == "hex":
            return m.to_hex(lab)
        if format == "rgb":
            r, g, b = (int(round(v * 255)) for v in m.to_srgb(lab))
            return f"rgb({r}, {g}, {b})"
        if format == "hsl":
            return self._hsl(m.to_srgb(lab))
        if format == "oklch":
            L, C, H = _oklab_to_oklch(_XYZ_to_oklab(m.to_xyz(lab)))
            return f"oklch({L * 100:.1f}% {C:.4f} {H:.1f})"
        if format == "p3":
            return m.to_css(lab, "display-p3")
        return m.to_css(lab, "rec2020")

    def android(self, color: str) -> str:
        """Color → '0xFFrrggbb' (Android ARGB int literal)."""
        r, g, b = (int(round(v * 255)) for v in self._metric.to_srgb(self._metric.from_hex(color)))
        return f"0xFF{r:02x}{g:02x}{b:02x}"

    def ios_p3(self, color: str) -> dict:
        """Color → {'r','g','b'} Display P3 floats (UIColor displayP3)."""
        css = self._metric.to_css(self._metric.from_hex(color), "display-p3")
        vals = css[len("color(display-p3 "):-1].split()
        return {"r": round(float(vals[0]), 4), "g": round(float(vals[1]), 4), "b": round(float(vals[2]), 4)}

    def swift(self, color: str) -> str:
        """Color → Swift ``Color(.displayP3, …)`` literal."""
        p3 = self.ios_p3(color)
        return f"Color(.displayP3, red: {p3['r']:.4f}, green: {p3['g']:.4f}, blue: {p3['b']:.4f})"

    # ── Scale exports ────────────────────────────────────────────────

    def css_variables(self, scale: dict, prefix: str = "--color") -> str:
        """Scale (level → color string) → CSS custom-properties block.
        Include the ``--`` in the prefix yourself."""
        lines = []
        for level, value in sorted(scale.items(), key=lambda x: int(x[0])):
            lines.append(f"  {prefix}-{level}: {value};")
        return "\n".join(lines)

    def tailwind(self, scale: dict, name: str) -> dict:
        """Scale → Tailwind config-compatible dict ``{name: {level: value}}``."""
        return {name: dict(scale)}

    def multi_format(self, scale: dict, name: str, formats: list[str] | None = None) -> dict:
        """Scale → nested dict ``{name: {level: {format: value}}}``.
        Default formats: hex, oklch, p3."""
        if formats is None:
            formats = ["hex", "oklch", "p3"]
        fns = {
            "hex": lambda c: self.css(c, "hex"),
            "rgb": lambda c: self.css(c, "rgb"),
            "hsl": lambda c: self.css(c, "hsl"),
            "oklch": lambda c: self.css(c, "oklch"),
            "p3": lambda c: self.css(c, "p3"),
            "rec2020": lambda c: self.css(c, "rec2020"),
            "android": self.android,
        }
        result = {}
        for level, value in scale.items():
            result[level] = {fmt: fns[fmt](value) for fmt in formats if fmt in fns}
        return {name: result}

    def json(self, scales: dict[str, dict]) -> str:
        """Multiple scales (name → scale dict) → multi-format JSON string."""
        result = {}
        for name, scale in scales.items():
            result.update(self.multi_format(scale, name))
        return _json.dumps(result, indent=2)

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _hsl(srgb) -> str:
        r, g, b = float(srgb[0]), float(srgb[1]), float(srgb[2])
        cmax, cmin = max(r, g, b), min(r, g, b)
        delta = cmax - cmin
        l = (cmax + cmin) / 2.0
        if delta < 1e-10:
            h, s = 0.0, 0.0
        else:
            s = delta / (1.0 - abs(2.0 * l - 1.0)) if abs(2.0 * l - 1.0) < 1.0 else 1.0
            if cmax == r:
                h = 60.0 * (((g - b) / delta) % 6.0)
            elif cmax == g:
                h = 60.0 * (((b - r) / delta) + 2.0)
            else:
                h = 60.0 * (((r - g) / delta) + 4.0)
            h = h % 360.0
        return f"hsl({h:.0f}, {s * 100:.0f}%, {l * 100:.0f}%)"
