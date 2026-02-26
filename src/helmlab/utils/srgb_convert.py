"""Pure numpy sRGB conversions (no colour-science dependency).

IEC 61966-2-1 sRGB ↔ XYZ (D65), gamma transfer, hex, WCAG utilities.
"""

import numpy as np

# ── IEC 61966-2-1 matrices (D65) ────────────────────────────────────────

# XYZ → linear sRGB (row-major: srgb = XYZ @ M_XYZ_TO_SRGB.T)
M_XYZ_TO_SRGB = np.array([
    [ 3.2404541621141054, -1.5371385940306089, -0.4985314095560162],
    [-0.9692660305051868,  1.8760108454466942,  0.0415560175303498],
    [ 0.0556434309591147, -0.2040259135167538,  1.0572251882231791],
])

# linear sRGB → XYZ (row-major: XYZ = srgb @ M_SRGB_TO_XYZ.T)
M_SRGB_TO_XYZ = np.array([
    [0.4124564390896922, 0.3575760776439511, 0.1804374832663989],
    [0.2126728514056226, 0.7151521552878178, 0.0721750036064596],
    [0.0193338955823293, 0.1191920258813418, 0.9503040785363679],
])


# ── Gamma transfer ──────────────────────────────────────────────────────

def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear [0,1] → sRGB gamma-encoded [0,1]."""
    linear = np.asarray(linear, dtype=np.float64)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB gamma-encoded [0,1] → linear [0,1]."""
    srgb = np.asarray(srgb, dtype=np.float64)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )


# ── XYZ ↔ sRGB ──────────────────────────────────────────────────────────

def XYZ_to_sRGB(XYZ: np.ndarray) -> np.ndarray:
    """CIE XYZ (D65) → sRGB [0,1] (gamma-encoded, unclamped)."""
    XYZ = np.asarray(XYZ, dtype=np.float64)
    linear = XYZ @ M_XYZ_TO_SRGB.T
    return linear_to_srgb(linear)


def sRGB_to_XYZ(srgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] (gamma-encoded) → CIE XYZ (D65)."""
    srgb = np.asarray(srgb, dtype=np.float64)
    linear = srgb_to_linear(srgb)
    return linear @ M_SRGB_TO_XYZ.T


# ── Hex ↔ sRGB ──────────────────────────────────────────────────────────

def hex_to_srgb(hex_str: str) -> np.ndarray:
    """'#rrggbb' → ndarray [R, G, B] in [0, 1]."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-char hex, got '{hex_str}'")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([r, g, b], dtype=np.float64) / 255.0


def srgb_to_hex(srgb: np.ndarray) -> str:
    """ndarray [R, G, B] in [0, 1] → '#rrggbb'."""
    rgb8 = np.clip(np.round(np.asarray(srgb) * 255.0), 0, 255).astype(int)
    return f"#{rgb8[0]:02x}{rgb8[1]:02x}{rgb8[2]:02x}"


def hex_batch_to_srgb(hex_list: list[str]) -> np.ndarray:
    """List of '#rrggbb' → ndarray (N, 3) in [0, 1]."""
    return np.array([hex_to_srgb(h) for h in hex_list])


def srgb_batch_to_hex(srgb: np.ndarray) -> list[str]:
    """ndarray (N, 3) in [0, 1] → list of '#rrggbb'."""
    srgb = np.asarray(srgb)
    return [srgb_to_hex(srgb[i]) for i in range(srgb.shape[0])]


# ── Gamut clamp ──────────────────────────────────────────────────────────

def clamp_srgb(srgb: np.ndarray) -> np.ndarray:
    """Clamp to [0, 1] gamut."""
    return np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0)


# ── WCAG utilities ───────────────────────────────────────────────────────

def relative_luminance(srgb: np.ndarray) -> np.ndarray:
    """WCAG 2.x relative luminance from sRGB [0,1].

    Y = 0.2126*R_lin + 0.7152*G_lin + 0.0722*B_lin
    """
    srgb = np.asarray(srgb, dtype=np.float64)
    linear = srgb_to_linear(srgb)
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


# ── Display P3 matrices (D65 white) ────────────────────────────────────
# Derived from Display P3 chromaticities (r=0.680,0.320 g=0.265,0.690 b=0.150,0.060)
# Using Bradford adaptation to D65 — same TRC as sRGB (IEC 61966-2-1)

M_XYZ_TO_DISPLAYP3 = np.array([
    [ 2.4934969119, -0.9313836179, -0.4027107845],
    [-0.8294889696,  1.7626640603,  0.0236246858],
    [ 0.0358458302, -0.0761723893,  0.9568845240],
])

M_DISPLAYP3_TO_XYZ = np.array([
    [0.4865709486, 0.2656676932, 0.1982172852],
    [0.2289745641, 0.6917385218, 0.0792869141],
    [0.0000000000, 0.0451133819, 1.0439443689],
])


def XYZ_to_DisplayP3(XYZ: np.ndarray) -> np.ndarray:
    """CIE XYZ (D65) → Display P3 linear [unclamped]."""
    return np.asarray(XYZ, dtype=np.float64) @ M_XYZ_TO_DISPLAYP3.T


def DisplayP3_to_XYZ(p3: np.ndarray) -> np.ndarray:
    """Display P3 linear → CIE XYZ (D65)."""
    return np.asarray(p3, dtype=np.float64) @ M_DISPLAYP3_TO_XYZ.T


def linear_to_displayp3(linear: np.ndarray) -> np.ndarray:
    """Linear [0,1] → Display P3 gamma-encoded [0,1]. Same TRC as sRGB."""
    return linear_to_srgb(linear)


def displayp3_to_linear(p3: np.ndarray) -> np.ndarray:
    """Display P3 gamma-encoded [0,1] → linear [0,1]. Same TRC as sRGB."""
    return srgb_to_linear(p3)


def contrast_ratio(srgb_fg: np.ndarray, srgb_bg: np.ndarray) -> np.ndarray:
    """WCAG contrast ratio (1:1 to 21:1).

    CR = (L_lighter + 0.05) / (L_darker + 0.05)
    """
    L1 = relative_luminance(srgb_fg)
    L2 = relative_luminance(srgb_bg)
    lighter = np.maximum(L1, L2)
    darker = np.minimum(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)
