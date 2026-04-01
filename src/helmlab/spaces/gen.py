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

    # Transfer function type: "cbrt" (default), "softcbrt", or "depcubic"
    transfer: str = "cbrt"
    softcbrt_eps: float = 0.001
    depcubic_alpha: float = 0.020

    # Piecewise-linear L correction (analytically invertible, replaces cubic when non-empty)
    L_corr_pw: list = field(default_factory=list)
    L_corr_pw_step: float = 0.05

    # L-gated hue enrichment (post-M2, pre-PW)
    enrichment_type: str = ""   # "L_gated_hue" or ""
    enrichment_amp: float = 0.0
    enrichment_center_deg: float = 240.0
    enrichment_sigma: float = 0.7
    enrichment_L_lo: float = 0.37
    enrichment_L_hi: float = 1.0

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
            "depcubic_alpha": self.depcubic_alpha,
            "L_corr_pw": self.L_corr_pw,
            "L_corr_pw_step": self.L_corr_pw_step,
            **({"enrichment": {
                "type": self.enrichment_type,
                "amp": self.enrichment_amp,
                "center_deg": self.enrichment_center_deg,
                "sigma": self.enrichment_sigma,
                "L_lo": self.enrichment_L_lo,
                "L_hi": self.enrichment_L_hi,
            }} if self.enrichment_type else {}),
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
            gamma=np.array(d.get("gamma", [1/3, 1/3, 1/3])),
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
            depcubic_alpha=d.get("depcubic_alpha", 0.020),
            L_corr_pw=d.get("L_corr_pw", []),
            L_corr_pw_step=d.get("L_corr_pw_step", 0.05),
            enrichment_type=d.get("enrichment", {}).get("type", ""),
            enrichment_amp=d.get("enrichment", {}).get("amp", 0.0),
            enrichment_center_deg=d.get("enrichment", {}).get("center_deg", 240.0),
            enrichment_sigma=d.get("enrichment", {}).get("sigma", 0.7),
            enrichment_L_lo=d.get("enrichment", {}).get("L_lo", 0.37),
            enrichment_L_hi=d.get("enrichment", {}).get("L_hi", 1.0),
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
        self._is_depcubic = (p.transfer == "depcubic")
        self._has_enrichment = (p.enrichment_type == "L_gated_hue" and abs(p.enrichment_amp) > 1e-10)
        if self._has_enrichment:
            self._enr_center = np.radians(p.enrichment_center_deg)

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

    # ── Depressed cubic transfer ──────────────────────────────────

    def _depcubic_fwd(self, x):
        """Forward: solve y³ + αy = x via sinh/asinh + Halley."""
        alpha = self.params.depcubic_alpha
        s = np.sqrt(alpha / 3)
        t = x / (2 * s ** 3)
        y = 2 * s * np.sinh(np.arcsinh(t) / 3)
        # Halley refinement
        f = y ** 3 + alpha * y - x
        fp = 3 * y ** 2 + alpha
        fpp = 6 * y
        denom = 2 * fp * fp - f * fpp
        safe = np.abs(denom) > 1e-30
        y = np.where(safe, y - 2 * f * fp / np.where(safe, denom, 1.0), y)
        return y

    def _depcubic_inv(self, y):
        """Inverse: exact."""
        return y ** 3 + self.params.depcubic_alpha * y

    # ── L-gated hue enrichment ──────────────────────────────────

    def _enrichment_gate(self, L):
        """sin²(π(L-L_lo)/(L_hi-L_lo)), 0 outside [L_lo, L_hi]."""
        p = self.params
        t = np.clip((L - p.enrichment_L_lo) / (p.enrichment_L_hi - p.enrichment_L_lo), 0.0, 1.0)
        return np.sin(np.pi * t) ** 2

    def _apply_enrichment(self, L, a, b):
        """Forward: h' = h + amp * gate(L) * gauss(h - center)."""
        p = self.params
        C = np.sqrt(a ** 2 + b ** 2)
        is_achromatic = C < 1e-12
        h = np.arctan2(b, a)
        gate = self._enrichment_gate(L)
        dh = h - self._enr_center
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        gauss = np.exp(-0.5 * (dh / p.enrichment_sigma) ** 2)
        rotation = p.enrichment_amp * gate * gauss
        h_new = h + rotation
        a_new = np.where(is_achromatic, a, C * np.cos(h_new))
        b_new = np.where(is_achromatic, b, C * np.sin(h_new))
        return a_new, b_new

    def _undo_enrichment(self, L, a, b):
        """Inverse: Halley iteration for h (cubic convergence)."""
        p = self.params
        C = np.sqrt(a ** 2 + b ** 2)
        is_achromatic = C < 1e-12
        h_target = np.arctan2(b, a)
        gate = self._enrichment_gate(L)
        sig2 = p.enrichment_sigma ** 2
        h = h_target.copy() if isinstance(h_target, np.ndarray) else h_target
        for _ in range(8):  # Halley converges cubically
            dh = h - self._enr_center
            dh = (dh + np.pi) % (2 * np.pi) - np.pi
            gauss = np.exp(-0.5 * (dh / p.enrichment_sigma) ** 2)
            ag = p.enrichment_amp * gate
            F = h + ag * gauss - h_target
            dg = gauss * (-dh / sig2)
            Fp = 1.0 + ag * dg
            ddg = gauss * (-1.0 / sig2 + dh * dh / (sig2 * sig2))
            Fpp = ag * ddg
            denom = 2.0 * Fp * Fp - F * Fpp
            denom = np.where(np.abs(denom) < 1e-30, np.ones_like(denom), denom)
            h = h - 2.0 * F * Fp / denom
        a_new = np.where(is_achromatic, a, C * np.cos(h))
        b_new = np.where(is_achromatic, b, C * np.sin(h))
        return a_new, b_new

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
        if self._is_depcubic:
            LMS_c = self._depcubic_fwd(LMS)
        elif self._is_softcbrt:
            LMS_c = self._softcbrt(LMS)
        else:
            LMS_c = LMS ** self.params.gamma

        # 2.5 Smooth neutral blend: C∞ correction for sRGB matrix rounding
        if self._is_depcubic:
            lms_mean = np.mean(LMS_c, axis=-1, keepdims=True)
            lms_spread = (np.max(LMS_c, axis=-1) - np.min(LMS_c, axis=-1)) / np.maximum(np.abs(lms_mean.squeeze()), 1e-30)
            blend_w = np.exp(-(lms_spread / 1e-5) ** 2)
            if LMS_c.ndim == 1:
                LMS_c = LMS_c + blend_w * (lms_mean.squeeze() - LMS_c)
            else:
                LMS_c = LMS_c + blend_w[..., None] * (np.broadcast_to(lms_mean, LMS_c.shape) - LMS_c)

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

        # 4.25 L-gated hue enrichment (after PW L correction)
        if self._has_enrichment:
            a, b = self._apply_enrichment(L, a, b)

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

        # 4.25 Undo L-gated hue enrichment (before PW undo)
        if self._has_enrichment:
            a, b = self._undo_enrichment(L, a, b)

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

        # 2.5 Smooth neutral blend (matching forward — C∞, branchless)
        if self._is_depcubic:
            lms_mean = np.mean(LMS_c, axis=-1, keepdims=True)
            lms_spread = (np.max(LMS_c, axis=-1) - np.min(LMS_c, axis=-1)) / np.maximum(np.abs(lms_mean.squeeze()), 1e-30)
            blend_w = np.exp(-(lms_spread / 1e-5) ** 2)
            if LMS_c.ndim == 1:
                LMS_c = LMS_c + blend_w * (lms_mean.squeeze() - LMS_c)
            else:
                LMS_c = LMS_c + blend_w[..., None] * (np.broadcast_to(lms_mean, LMS_c.shape) - LMS_c)

        # 2. Undo transfer function
        if self._is_depcubic:
            LMS = self._depcubic_inv(LMS_c)
        elif self._is_softcbrt:
            LMS = self._softcbrt_inv(LMS_c)
        else:
            inv_gamma = 1.0 / self.params.gamma
            LMS = np.maximum(LMS_c, 0.0) ** inv_gamma

        # 1. LMS → XYZ
        return LMS @ self._M1_inv.T
