"""GenSpace — generation-optimized color space for palette, gradient, gamut map.

Pipeline (subset of MetricSpace, ~35 params):
    XYZ → M1 → γ^(1/3) shared → M2 → Lab_raw
    → [hue correction δ(h)]
    → [cubic L correction]
    → [dark L compression]
    → [L-dependent chroma scaling]
    → neutral correction (NC)
    → Lab_final

Key differences from MetricSpace:
    - Shared gamma (1/3) guarantees structural achromatic axis (grays → a=b≈0)
    - No H-K, chroma power, HLC, hue-lightness — these cause brightness fold
    - No hue-dep chroma scaling — causes distortion in gradients
    - NC cleans up any residual achromatic error

All stages are exactly invertible.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from helmlab.spaces.base import ColorSpace

_PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "gen_params.json"

# D65 white point (Y=1)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883])


@dataclass
class GenParams:
    """Parameters for the generation color space.

    Core (21 params): M1(9), gamma(3), M2(9)
    Enrichment (~14 params): hue_correction(8), cubic_L(3), dark_L(3), L_chroma(2)
    Total: ~35 params (vs 72 for MetricSpace).
    """

    M1: np.ndarray = field(default_factory=lambda: np.eye(3))
    gamma: np.ndarray = field(default_factory=lambda: np.array([1/3, 1/3, 1/3]))
    M2: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Hue correction (8 params, 4 harmonics)
    hue_cos1: float = 0.0
    hue_sin1: float = 0.0
    hue_cos2: float = 0.0
    hue_sin2: float = 0.0
    hue_cos3: float = 0.0
    hue_sin3: float = 0.0
    hue_cos4: float = 0.0
    hue_sin4: float = 0.0

    # Cubic L correction (3 params)
    L_corr_p1: float = 0.0
    L_corr_p2: float = 0.0
    L_corr_p3: float = 0.0

    # Dark L compression (3 params)
    lp_dark: float = 0.0
    lp_dark_hcos: float = 0.0
    lp_dark_hsin: float = 0.0

    # L-dependent chroma scaling (2 params)
    lc1: float = 0.0
    lc2: float = 0.0

    def to_dict(self) -> dict:
        return {
            "M1": self.M1.tolist(),
            "gamma": self.gamma.tolist(),
            "M2": self.M2.tolist(),
            "hue_cos1": self.hue_cos1, "hue_sin1": self.hue_sin1,
            "hue_cos2": self.hue_cos2, "hue_sin2": self.hue_sin2,
            "hue_cos3": self.hue_cos3, "hue_sin3": self.hue_sin3,
            "hue_cos4": self.hue_cos4, "hue_sin4": self.hue_sin4,
            "L_corr_p1": self.L_corr_p1,
            "L_corr_p2": self.L_corr_p2,
            "L_corr_p3": self.L_corr_p3,
            "lp_dark": self.lp_dark,
            "lp_dark_hcos": self.lp_dark_hcos,
            "lp_dark_hsin": self.lp_dark_hsin,
            "lc1": self.lc1,
            "lc2": self.lc2,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GenParams":
        return cls(
            M1=np.array(d["M1"]),
            gamma=np.array(d["gamma"]),
            M2=np.array(d["M2"]),
            hue_cos1=d.get("hue_cos1", 0.0), hue_sin1=d.get("hue_sin1", 0.0),
            hue_cos2=d.get("hue_cos2", 0.0), hue_sin2=d.get("hue_sin2", 0.0),
            hue_cos3=d.get("hue_cos3", 0.0), hue_sin3=d.get("hue_sin3", 0.0),
            hue_cos4=d.get("hue_cos4", 0.0), hue_sin4=d.get("hue_sin4", 0.0),
            L_corr_p1=d.get("L_corr_p1", 0.0),
            L_corr_p2=d.get("L_corr_p2", 0.0),
            L_corr_p3=d.get("L_corr_p3", 0.0),
            lp_dark=d.get("lp_dark", 0.0),
            lp_dark_hcos=d.get("lp_dark_hcos", 0.0),
            lp_dark_hsin=d.get("lp_dark_hsin", 0.0),
            lc1=d.get("lc1", 0.0),
            lc2=d.get("lc2", 0.0),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GenParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class GenSpace(ColorSpace):
    """Generation-optimized color space for palette, gradient, gamut mapping.

    Forward pipeline:
        1.   XYZ → M1 → LMS
        2.   LMS → signed power compression (shared γ=1/3) → LMS_c
        3.   LMS_c → M2 → Lab_raw
        3.5  Hue correction: rotate (a,b) by δ(h)
        4.   Cubic L correction
        4.5  Dark L compression
        6.   L-dependent chroma scaling
        10.  Neutral correction (NC): a -= a_err(L), b -= b_err(L)

    All stages are exactly invertible.
    """

    name = "Gen"

    def __init__(self, params: GenParams | None = None,
                 neutral_correction: bool = True):
        if params is not None:
            self.params = params
        elif _PARAMS_PATH.exists():
            self.params = GenParams.load(_PARAMS_PATH)
        else:
            self.params = GenParams()

        self._neutral_correction = neutral_correction
        self._M1_inv = np.linalg.inv(self.params.M1)
        self._M2_inv = np.linalg.inv(self.params.M2)

        p = self.params
        self._has_hue_correction = any(v != 0.0 for v in [
            p.hue_cos1, p.hue_sin1, p.hue_cos2, p.hue_sin2,
            p.hue_cos3, p.hue_sin3, p.hue_cos4, p.hue_sin4])
        self._has_L_corr = (p.L_corr_p1 != 0.0 or p.L_corr_p2 != 0.0 or p.L_corr_p3 != 0.0)
        self._has_dark_L = (p.lp_dark != 0.0 or p.lp_dark_hcos != 0.0 or p.lp_dark_hsin != 0.0)
        self._has_dark_L_hue = (p.lp_dark_hcos != 0.0 or p.lp_dark_hsin != 0.0)
        self._has_L_chroma = (p.lc1 != 0.0 or p.lc2 != 0.0)

        # NC LUT (lazy)
        self._nc_lut_built = False

    # ── Hue correction helpers ────────────────────────────────────

    def _hue_delta(self, h: np.ndarray) -> np.ndarray:
        p = self.params
        return (
            p.hue_cos1 * np.cos(h) + p.hue_sin1 * np.sin(h) +
            p.hue_cos2 * np.cos(2.0 * h) + p.hue_sin2 * np.sin(2.0 * h) +
            p.hue_cos3 * np.cos(3.0 * h) + p.hue_sin3 * np.sin(3.0 * h) +
            p.hue_cos4 * np.cos(4.0 * h) + p.hue_sin4 * np.sin(4.0 * h)
        )

    def _hue_delta_deriv(self, h: np.ndarray) -> np.ndarray:
        p = self.params
        return (
            -p.hue_cos1 * np.sin(h) + p.hue_sin1 * np.cos(h) +
            -2.0 * p.hue_cos2 * np.sin(2.0 * h) + 2.0 * p.hue_sin2 * np.cos(2.0 * h) +
            -3.0 * p.hue_cos3 * np.sin(3.0 * h) + 3.0 * p.hue_sin3 * np.cos(3.0 * h) +
            -4.0 * p.hue_cos4 * np.sin(4.0 * h) + 4.0 * p.hue_sin4 * np.cos(4.0 * h)
        )

    def _apply_hue_correction(self, a, b):
        h = np.arctan2(b, a)
        delta = self._hue_delta(h)
        h_new = h + delta
        C = np.sqrt(a ** 2 + b ** 2)
        return C * np.cos(h_new), C * np.sin(h_new)

    def _undo_hue_correction(self, a, b):
        h_out = np.arctan2(b, a)
        C = np.sqrt(a ** 2 + b ** 2)
        h_raw = h_out.copy()
        for _ in range(8):
            f = h_raw + self._hue_delta(h_raw) - h_out
            fp = 1.0 + self._hue_delta_deriv(h_raw)
            fp = np.where(np.abs(fp) < 1e-10, 1.0, fp)
            h_raw = h_raw - f / fp
        return C * np.cos(h_raw), C * np.sin(h_raw)

    # ── Cubic L correction ─────────────────────────────────────────

    def _L_correct(self, L_raw):
        p = self.params
        t = L_raw * (1.0 - L_raw)
        return L_raw + p.L_corr_p1 * t + p.L_corr_p2 * t * (0.5 - L_raw) + p.L_corr_p3 * t * t

    def _L_correct_inv(self, L1):
        p = self.params
        L = L1.copy()
        for _ in range(15):
            t = L * (1.0 - L)
            dt = 1.0 - 2.0 * L
            f = L + p.L_corr_p1 * t + p.L_corr_p2 * t * (0.5 - L) + p.L_corr_p3 * t * t - L1
            dfdL = (1.0 + p.L_corr_p1 * dt +
                    p.L_corr_p2 * (dt * (0.5 - L) - t) +
                    p.L_corr_p3 * 2.0 * t * dt)
            dfdL = np.where(np.abs(dfdL) < 1e-10, 1.0, dfdL)
            L = L - f / dfdL
        return L

    # ── Dark L compression ─────────────────────────────────────────

    def _dark_L_compress(self, L, h=None):
        coeff = self.params.lp_dark
        if self._has_dark_L_hue and h is not None:
            coeff = coeff + self.params.lp_dark_hcos * np.cos(h) + self.params.lp_dark_hsin * np.sin(h)
        g = coeff * L * (1.0 - L) ** 2
        return L * np.exp(np.clip(g, -30.0, 30.0))

    def _dark_L_compress_inv(self, L_new, h=None):
        coeff = self.params.lp_dark
        if self._has_dark_L_hue and h is not None:
            coeff = coeff + self.params.lp_dark_hcos * np.cos(h) + self.params.lp_dark_hsin * np.sin(h)
        L = L_new.copy()
        for _ in range(12):
            oml = 1.0 - L
            g = coeff * L * oml ** 2
            eg = np.exp(np.clip(g, -30.0, 30.0))
            f = L * eg - L_new
            gp = coeff * oml * (1.0 - 3.0 * L)
            fp = eg * (1.0 + L * gp)
            fp = np.where(np.abs(fp) < 1e-10, 1.0, fp)
            L = L - f / fp
        return L

    # ── L-dependent chroma scaling ─────────────────────────────────

    def _L_chroma_scale(self, L):
        p = self.params
        dL = L - 0.5
        arg = p.lc1 * dL + p.lc2 * dL ** 2
        return np.exp(np.clip(arg, -30.0, 30.0))

    # ── Neutral correction (NC) ────────────────────────────────────

    def _build_neutral_lut(self, n_samples: int = 256):
        from scipy.interpolate import PchipInterpolator

        Y_vals = np.concatenate([
            [0.0],  # black point anchor: NC error is zero at L=0
            np.linspace(0.001, 0.01, 10),
            np.linspace(0.01, 0.1, 20),
            np.linspace(0.1, 2.0, n_samples - 30),
        ])
        gray_XYZ = np.outer(Y_vals, _D65_WHITE)

        old_nc = self._neutral_correction
        self._neutral_correction = False
        Lab_gray = self.from_XYZ(gray_XYZ)
        self._neutral_correction = old_nc

        L_gray = Lab_gray[:, 0]
        a_gray = Lab_gray[:, 1]
        b_gray = Lab_gray[:, 2]

        order = np.argsort(L_gray)
        L_sorted = L_gray[order]
        a_sorted = a_gray[order]
        b_sorted = b_gray[order]

        mask = np.diff(L_sorted, prepend=-np.inf) > 1e-12
        L_sorted = L_sorted[mask]
        a_sorted = a_sorted[mask]
        b_sorted = b_sorted[mask]

        self._nc_a_interp = PchipInterpolator(L_sorted, a_sorted, extrapolate=True)
        self._nc_b_interp = PchipInterpolator(L_sorted, b_sorted, extrapolate=True)
        self._nc_lut_built = True

    def _neutral_error(self, L):
        if not self._nc_lut_built:
            self._build_neutral_lut()
        return self._nc_a_interp(L), self._nc_b_interp(L)

    # ── Forward transform ──────────────────────────────────────────

    def from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → Gen Lab (generation-optimized pipeline)."""
        XYZ = np.asarray(XYZ, dtype=np.float64)

        # 1. XYZ → LMS
        LMS = XYZ @ self.params.M1.T

        # 2. Shared power compression (γ = 1/3 for all channels)
        LMS_c = np.sign(LMS) * np.abs(LMS) ** self.params.gamma

        # 3. LMS_c → Lab_raw
        Lab = LMS_c @ self.params.M2.T
        L = Lab[..., 0]
        a = Lab[..., 1]
        b = Lab[..., 2]

        # 3.5 Hue correction
        if self._has_hue_correction:
            a, b = self._apply_hue_correction(a, b)

        # 4. Cubic L correction
        if self._has_L_corr:
            L = self._L_correct(L)

        # 4.5 Dark L compression
        if self._has_dark_L:
            h = np.arctan2(b, a) if self._has_dark_L_hue else None
            L = self._dark_L_compress(L, h)

        # 6. L-dependent chroma scaling
        if self._has_L_chroma:
            T = self._L_chroma_scale(L)
            a = a * T
            b = b * T

        # 10. Neutral correction
        if self._neutral_correction:
            a_err, b_err = self._neutral_error(L)
            a = a - a_err
            b = b - b_err

        return np.stack([L, a, b], axis=-1)

    # ── Inverse transform ──────────────────────────────────────────

    def to_XYZ(self, coords: np.ndarray) -> np.ndarray:
        """Gen Lab → XYZ (exact inverse)."""
        coords = np.asarray(coords, dtype=np.float64)
        L = coords[..., 0]
        a = coords[..., 1]
        b = coords[..., 2]

        # 10. Undo NC
        if self._neutral_correction:
            a_err, b_err = self._neutral_error(L)
            a = a + a_err
            b = b + b_err

        # 6. Undo L-dep chroma scaling
        if self._has_L_chroma:
            T = self._L_chroma_scale(L)
            a = a / T
            b = b / T

        # 4.5 Undo dark L
        if self._has_dark_L:
            h = np.arctan2(b, a) if self._has_dark_L_hue else None
            L = self._dark_L_compress_inv(L, h)

        # 4. Undo cubic L
        if self._has_L_corr:
            L = self._L_correct_inv(L)

        # 3.5 Undo hue correction
        if self._has_hue_correction:
            a, b = self._undo_hue_correction(a, b)

        # 3. Lab → LMS_c
        Lab = np.stack([L, a, b], axis=-1)
        LMS_c = Lab @ self._M2_inv.T

        # 2. Undo power
        inv_gamma = 1.0 / self.params.gamma
        LMS = np.sign(LMS_c) * np.abs(LMS_c) ** inv_gamma

        # 1. LMS → XYZ
        return LMS @ self._M1_inv.T
