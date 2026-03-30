"""GenSpace — generation-optimized color space for palette, gradient, gamut map.

Pipeline:
    XYZ → M1 → transfer → M2 → [enrichment] → Lab

    Transfer functions:
    - "cbrt": standard cube root x^(1/3) (OKLab-compatible)
    - "softcbrt": softened cube root (|x|+ε)^(1/3) - ε^(1/3)
      → finite derivative at zero → 360/360 cusps
      → exact analytical inverse: (|y|+ε^(1/3))^3 - ε

    Optional enrichment stages:
    → [hue_L_correction] — hue-dependent L compression
    → [hue correction δ(h)]
    → [PW L correction] — piecewise-linear, analytically invertible
    → [cubic L correction]
    → [dark L compression]
    → [L-dependent chroma scaling]
    → [neutral correction (NC)]

Key differences from MetricSpace:
    - Shared transfer guarantees structural achromatic axis (grays → a=b≈0)
    - No H-K, chroma power, HLC, hue-lightness — these cause brightness fold

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

    # Chroma power (1 param) — sublinear chroma compression
    chroma_power: float = 1.0  # 1.0 = no effect, <1 = compression

    # Hue-dependent L correction (4 params) — v31 yellow cusp fix
    hue_L_amp: float = 0.0
    hue_L_center: float = 0.0  # radians
    hue_L_width: float = 1.0   # radians
    hue_L_knee: float = 1.0    # L threshold

    # Transfer function type: "cbrt" (default) or "softcbrt"
    transfer: str = "cbrt"
    softcbrt_eps: float = 0.001

    # Piecewise-linear L correction (analytically invertible, replaces cubic when non-empty)
    L_corr_pw: list = field(default_factory=list)
    L_corr_pw_step: float = 0.05

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
            "chroma_power": self.chroma_power,
            "hue_L_amp": self.hue_L_amp,
            "hue_L_center": self.hue_L_center,
            "hue_L_width": self.hue_L_width,
            "hue_L_knee": self.hue_L_knee,
            "transfer": self.transfer,
            "softcbrt_eps": self.softcbrt_eps,
            "L_corr_pw": self.L_corr_pw,
            "L_corr_pw_step": self.L_corr_pw_step,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GenParams":
        # Support compact array format for hue_correction
        hc = d.get("hue_correction", None)
        if isinstance(hc, list):
            hc1 = hc[0] if len(hc) > 0 else 0.0
            hs1 = hc[1] if len(hc) > 1 else 0.0
            hc2 = hc[2] if len(hc) > 2 else 0.0
            hs2 = hc[3] if len(hc) > 3 else 0.0
        else:
            hc1 = d.get("hue_cos1", 0.0)
            hs1 = d.get("hue_sin1", 0.0)
            hc2 = d.get("hue_cos2", 0.0)
            hs2 = d.get("hue_sin2", 0.0)
        return cls(
            M1=np.array(d["M1"]),
            gamma=np.array(d["gamma"]),
            M2=np.array(d["M2"]),
            hue_cos1=hc1, hue_sin1=hs1,
            hue_cos2=hc2, hue_sin2=hs2,
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
            chroma_power=d.get("chroma_power", 1.0),
            hue_L_amp=d.get("hue_L_amp", 0.0),
            hue_L_center=d.get("hue_L_center", 0.0),
            hue_L_width=d.get("hue_L_width", 1.0),
            hue_L_knee=d.get("hue_L_knee", 1.0),
            transfer=d.get("transfer", "cbrt"),
            softcbrt_eps=d.get("softcbrt_eps", 0.001),
            L_corr_pw=d.get("L_corr_pw", []),
            L_corr_pw_step=d.get("L_corr_pw_step", 0.05),
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
                 neutral_correction: bool = False):
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
        self._has_chroma_power = (p.chroma_power != 1.0)
        self._has_hue_L = (p.hue_L_amp != 0.0)
        self._is_softcbrt = (p.transfer == "softcbrt")

        # Piecewise-linear L correction setup
        self._has_pw_L = len(p.L_corr_pw) > 0
        if self._has_pw_L:
            n = len(p.L_corr_pw)
            step = p.L_corr_pw_step
            shifts = [0.0] + list(p.L_corr_pw) + [0.0]
            breakpoints = [i * step for i in range(n + 2)]
            breakpoints[-1] = 1.0
            self._pw_L_in = np.array(breakpoints)
            self._pw_L_out = np.array([bp + s for bp, s in zip(breakpoints, shifts)])

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

    # ── Softened cbrt transfer ───────────────────────────────────────

    def _softcbrt(self, x):
        eps = self.params.softcbrt_eps
        ax = np.abs(x)
        return np.sign(x) * ((ax + eps) ** (1.0 / 3.0) - eps ** (1.0 / 3.0))

    def _softcbrt_inv(self, y):
        eps = self.params.softcbrt_eps
        eps_cbrt = eps ** (1.0 / 3.0)
        ay = np.abs(y)
        return np.sign(y) * ((ay + eps_cbrt) ** 3.0 - eps)

    # ── Piecewise-linear L correction ─────────────────────────────

    def _pw_L_forward(self, L):
        L_arr = np.asarray(L)
        result = np.empty_like(L_arr)
        inside = (L_arr >= 0.0) & (L_arr <= 1.0)
        # Inside [0,1]: piecewise-linear interpolation
        L_c = np.clip(L_arr, 0.0, 1.0)
        idx = np.searchsorted(self._pw_L_in, L_c, side='right') - 1
        idx = np.clip(idx, 0, len(self._pw_L_in) - 2)
        L_lo = self._pw_L_in[idx]
        L_hi = self._pw_L_in[idx + 1]
        t = np.clip((L_c - L_lo) / np.maximum(L_hi - L_lo, 1e-30), 0.0, 1.0)
        pw_result = self._pw_L_out[idx] + t * (self._pw_L_out[idx + 1] - self._pw_L_out[idx])
        # Outside [0,1]: linear extrapolation (identity + endpoint slope)
        result = np.where(inside, pw_result, L_arr)
        return result

    def _pw_L_inverse(self, L_target):
        L_arr = np.asarray(L_target)
        result = np.empty_like(L_arr)
        inside = (L_arr >= self._pw_L_out[0]) & (L_arr <= self._pw_L_out[-1])
        L_c = np.clip(L_arr, self._pw_L_out[0], self._pw_L_out[-1])
        idx = np.searchsorted(self._pw_L_out, L_c, side='right') - 1
        idx = np.clip(idx, 0, len(self._pw_L_out) - 2)
        Lo_lo = self._pw_L_out[idx]
        Lo_hi = self._pw_L_out[idx + 1]
        t = np.clip((L_c - Lo_lo) / np.maximum(Lo_hi - Lo_lo, 1e-30), 0.0, 1.0)
        pw_result = self._pw_L_in[idx] + t * (self._pw_L_in[idx + 1] - self._pw_L_in[idx])
        # Outside range: linear extrapolation (identity)
        result = np.where(inside, pw_result, L_arr)
        return result

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

    # ── Hue-dependent L correction (v31 yellow cusp fix) ──────────

    def _hue_L_weight(self, h, C):
        """Gaussian hue weight × chroma gate (zero at achromatic)."""
        p = self.params
        dh = np.arctan2(np.sin(h - p.hue_L_center), np.cos(h - p.hue_L_center))
        w = np.exp(-(dh / p.hue_L_width) ** 2) * C / (C + 0.01)
        return w

    def _apply_hue_L(self, L, a, b):
        """Forward: compress high-L at target hue region."""
        C = np.sqrt(a ** 2 + b ** 2)
        h = np.arctan2(b, a)
        w = self._hue_L_weight(h, C)
        excess = np.maximum(0.0, L - self.params.hue_L_knee)
        return L - self.params.hue_L_amp * w * excess

    def _undo_hue_L(self, L, a, b):
        """Inverse: analytical — L_in = (L_out - aw*knee) / (1 - aw)."""
        C = np.sqrt(a ** 2 + b ** 2)
        h = np.arctan2(b, a)
        w = self._hue_L_weight(h, C)
        aw = np.minimum(self.params.hue_L_amp * w, 0.99)
        L_candidate = (L - aw * self.params.hue_L_knee) / (1.0 - aw)
        return np.where(L_candidate > self.params.hue_L_knee, L_candidate, L)

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

        # 1. XYZ → LMS (clamp: cone responses are physically non-negative)
        LMS = np.maximum(XYZ @ self.params.M1.T, 0.0)

        # 2. Transfer function
        if self._is_softcbrt:
            LMS_c = self._softcbrt(LMS)
        else:
            LMS_c = LMS ** self.params.gamma

        # 3. LMS_c → Lab_raw
        Lab = LMS_c @ self.params.M2.T
        L = Lab[..., 0]
        a = Lab[..., 1]
        b = Lab[..., 2]

        # 3.25 Hue-dependent L correction (yellow cusp fix)
        if self._has_hue_L:
            L = self._apply_hue_L(L, a, b)

        # 3.5 Hue correction
        if self._has_hue_correction:
            a, b = self._apply_hue_correction(a, b)

        # 4. L correction (PW takes priority over cubic)
        if self._has_pw_L:
            L = self._pw_L_forward(L)
        elif self._has_L_corr:
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

        # 7. Chroma power (sublinear compression)
        if self._has_chroma_power:
            C = np.sqrt(a ** 2 + b ** 2 + 1e-30)
            scale = C ** (self.params.chroma_power - 1.0)
            a = a * scale
            b = b * scale

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

        # 7. Undo chroma power
        if self._has_chroma_power:
            C = np.sqrt(a ** 2 + b ** 2 + 1e-30)
            inv_cp = 1.0 / self.params.chroma_power
            C_raw = C ** inv_cp
            scale = C_raw / C
            a = a * scale
            b = b * scale

        # 6. Undo L-dep chroma scaling
        if self._has_L_chroma:
            T = self._L_chroma_scale(L)
            a = a / T
            b = b / T

        # 4.5 Undo dark L
        if self._has_dark_L:
            h = np.arctan2(b, a) if self._has_dark_L_hue else None
            L = self._dark_L_compress_inv(L, h)

        # 4. Undo L correction (PW takes priority over cubic)
        if self._has_pw_L:
            L = self._pw_L_inverse(L)
        elif self._has_L_corr:
            L = self._L_correct_inv(L)

        # 3.5 Undo hue correction
        if self._has_hue_correction:
            a, b = self._undo_hue_correction(a, b)

        # 3.25 Undo hue-dependent L correction
        if self._has_hue_L:
            L = self._undo_hue_L(L, a, b)

        # 3. Lab → LMS_c
        Lab = np.stack([L, a, b], axis=-1)
        LMS_c = Lab @ self._M2_inv.T

        # 2. Undo transfer function
        if self._is_softcbrt:
            LMS = self._softcbrt_inv(LMS_c)
        else:
            inv_gamma = 1.0 / self.params.gamma
            LMS = np.maximum(LMS_c, 0.0) ** inv_gamma

        # 1. LMS → XYZ
        return LMS @ self._M1_inv.T
