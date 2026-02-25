"""Design token export — CSS, Android, iOS, Tailwind formats.

Converts Helmlab Lab colors to platform-specific token strings.
Includes Oklab/oklch conversion for CSS oklch() output.
"""

import json

import numpy as np

from colorspace.utils.srgb_convert import (
    sRGB_to_XYZ,
    XYZ_to_sRGB,
    XYZ_to_DisplayP3,
    clamp_srgb,
    linear_to_srgb,
    linear_to_displayp3,
    srgb_to_hex,
    hex_to_srgb,
)

# ── Oklab matrices (Björn Ottosson) ─────────────────────────────────
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
    """Oklab [L, a, b] → (L, C, H_deg)."""
    L = float(lab[0])
    a = float(lab[1])
    b = float(lab[2])
    C = np.sqrt(a ** 2 + b ** 2)
    H = np.degrees(np.arctan2(b, a)) % 360.0
    return L, float(C), float(H)


class TokenExporter:
    """Export Helmlab colors to various platform token formats."""

    def __init__(self, helmlab):
        self._helmlab = helmlab

    def _to_XYZ(self, lab: np.ndarray) -> np.ndarray:
        return self._helmlab._space.to_XYZ(np.asarray(lab, dtype=np.float64))

    # ── Single color formats ─────────────────────────────────────────

    def to_css_hex(self, lab: np.ndarray) -> str:
        """Helmlab Lab → '#rrggbb'."""
        return self._helmlab.to_hex(lab)

    def to_css_rgb(self, lab: np.ndarray) -> str:
        """Helmlab Lab → 'rgb(r, g, b)'."""
        srgb = self._helmlab.to_srgb(lab)
        r, g, b = (int(round(v * 255)) for v in srgb)
        return f"rgb({r}, {g}, {b})"

    def to_css_oklch(self, lab: np.ndarray) -> str:
        """Helmlab Lab → 'oklch(L% C H)' via XYZ → Oklab → oklch."""
        XYZ = self._to_XYZ(lab)
        oklab = _XYZ_to_oklab(XYZ)
        L, C, H = _oklab_to_oklch(oklab)
        return f"oklch({L * 100:.1f}% {C:.4f} {H:.1f})"

    def to_css_displayp3(self, lab: np.ndarray) -> str:
        """Helmlab Lab → 'color(display-p3 r g b)'."""
        return self._helmlab.to_hex_p3(lab)

    def to_css_hsl(self, lab: np.ndarray) -> str:
        """Helmlab Lab → 'hsl(H, S%, L%)'."""
        srgb = self._helmlab.to_srgb(lab)
        r, g, b = float(srgb[0]), float(srgb[1]), float(srgb[2])
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin
        # Lightness
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

    # ── Platform-specific ────────────────────────────────────────────

    def to_android_argb(self, lab: np.ndarray) -> str:
        """Helmlab Lab → '0xFFrrggbb' (Android ARGB int)."""
        srgb = self._helmlab.to_srgb(lab)
        r, g, b = (int(round(v * 255)) for v in srgb)
        return f"0xFF{r:02x}{g:02x}{b:02x}"

    def to_ios_p3(self, lab: np.ndarray) -> dict:
        """Helmlab Lab → {"r": float, "g": float, "b": float} (UIColor Display P3)."""
        p3 = self._helmlab.to_displayp3(lab)
        return {"r": round(float(p3[0]), 4), "g": round(float(p3[1]), 4), "b": round(float(p3[2]), 4)}

    def to_swift_literal(self, lab: np.ndarray) -> str:
        """Helmlab Lab → Swift Color literal with Display P3."""
        p3 = self._helmlab.to_displayp3(lab)
        return f"Color(.displayP3, red: {p3[0]:.4f}, green: {p3[1]:.4f}, blue: {p3[2]:.4f})"

    # ── Scale/palette export ─────────────────────────────────────────

    def export_scale(self, scale: dict, name: str, formats: list[str] | None = None) -> dict:
        """Export a semantic scale to multiple formats.

        Parameters
        ----------
        scale : dict mapping level str → hex str (e.g. {"50": "#eff6ff", ...})
        name : scale name (e.g. "blue")
        formats : list of format names. Default: ["hex", "oklch", "p3"]

        Returns
        -------
        dict: {name: {level: {format: value}}}
        """
        if formats is None:
            formats = ["hex", "oklch", "p3"]

        _format_fns = {
            "hex": self.to_css_hex,
            "rgb": self.to_css_rgb,
            "oklch": self.to_css_oklch,
            "p3": self.to_css_displayp3,
            "hsl": self.to_css_hsl,
            "android": self.to_android_argb,
        }

        result = {}
        for level, hex_str in scale.items():
            lab = self._helmlab.from_hex(hex_str)
            level_data = {}
            for fmt in formats:
                if fmt in _format_fns:
                    level_data[fmt] = _format_fns[fmt](lab)
            result[level] = level_data

        return {name: result}

    def export_css_custom_properties(self, scale: dict, prefix: str = "--color") -> str:
        """Export scale as CSS custom properties.

        Returns
        -------
        str: CSS custom properties block
        """
        lines = []
        for level, hex_str in sorted(scale.items(), key=lambda x: int(x[0])):
            lines.append(f"  {prefix}-{level}: {hex_str};")
        return "\n".join(lines)

    def export_tailwind(self, scale: dict, name: str) -> dict:
        """Export scale as Tailwind config-compatible dict.

        Returns
        -------
        dict: {name: {level: hex_str}}
        """
        return {name: {level: hex_str for level, hex_str in scale.items()}}

    def export_json(self, scales: dict[str, dict]) -> str:
        """Export multiple scales as multi-format JSON.

        Parameters
        ----------
        scales : dict mapping name → scale dict (level → hex)

        Returns
        -------
        str: JSON string with all formats
        """
        result = {}
        for name, scale in scales.items():
            exported = self.export_scale(scale, name)
            result.update(exported)
        return json.dumps(result, indent=2)
