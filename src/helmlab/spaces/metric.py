"""MetricSpace — full enriched analytical color space for perceptual distance.

Renamed from MetricSpace in v0.4.0. This is the distance-optimized space
with 72 parameters. For generation (palette, gradient), use GenSpace instead.

Original docstring:
Analytical parametric color space distilled from neural model.

Pipeline (v14, embedded chroma-dependent lightness + hue-dep dark L):
    XYZ → M1 → LMS → power compression → M2 → Lab_raw
    → hue correction δ(h)
    → embedded H-K (chroma-dependent lightness, uses raw C)  [Phase 2]
    → cubic L correction
    → dark L compression (v6, v13: hue-dependent coefficient)
    → hue-dependent chroma scaling
    → nonlinear chroma power (v6)
    → L-dependent chroma scaling
    → Lab_final

Distance: DE = ((dL/SL)² + wC*(da/SC)² + wC*(db/SC)²)^(p/2)
    v14c: SL = 1 + sl*(Lavg-0.5)², SC = 1 + sc*Cavg  — pair-dep weights
    v15: sl = dist_sl + F_sl(h), sc = dist_sc + F_sc(h) — hue-modulated SL/SC
    v12: DE_final = DE / (1 + c * DE)  — monotonic compression
    v14: DE_final = DE * (1 + α*c*DE) / (1 + c*DE)  — reduces S-curve bias

All stages are exactly invertible (Newton iteration for cubic L, hue correction,
and dark L compression). H-K at step 3.7 uses exact subtraction in inverse.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from helmlab.spaces.base import ColorSpace

# Default params JSON location
_PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "metric_params.json"
# Fallback to old name for backward compat
if not _PARAMS_PATH.exists():
    _PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "analytical_params.json"


@dataclass
class MetricParams:
    """Parameters for the analytical color space transform.

    Core pipeline (24 params):
        M1 (9), gamma (3), M2 (9), H-K base (3)

    Enrichment stages (21 params):
        Cubic L correction (2), hue-dep chroma scaling (6, 3 harmonics),
        L-dep chroma scaling (2), enhanced H-K harmonics (3),
        hue correction (6, 3 harmonics)

    Total: 45 free parameters.
    """

    # ── Core (24 params) ──────────────────────────────────────────
    # M1: XYZ -> LMS cone response (3x3 = 9 params)
    M1: np.ndarray = field(default_factory=lambda: np.eye(3))
    # Power compression exponents per channel (3 params)
    gamma: np.ndarray = field(default_factory=lambda: np.array([1 / 3, 1 / 3, 1 / 3]))
    # M2: compressed LMS -> Lab opponent channels (3x3 = 9 params)
    M2: np.ndarray = field(default_factory=lambda: np.eye(3))
    # H-K base: L += hk_weight * C^hk_power * (1 + hk_hue_mod*cos(h) + ...)
    hk_weight: float = 0.0
    hk_power: float = 0.5
    hk_hue_mod: float = 0.0

    # ── Cubic L correction (3 params) ─────────────────────────────
    # L1 = L_raw + p1*t + p2*t*(0.5-L) + p3*t^2, where t = L*(1-L)
    L_corr_p1: float = 0.0
    L_corr_p2: float = 0.0
    L_corr_p3: float = 0.0  # quartic term for dark/light extremes

    # ── Hue-dependent chroma scaling (6 params) ──────────────────
    # S(h) = exp(Fourier series up to 3rd harmonic)
    # a,b multiplied by S(h) — hue preserved, exp guarantees S>0
    cs_cos1: float = 0.0
    cs_sin1: float = 0.0
    cs_cos2: float = 0.0
    cs_sin2: float = 0.0
    cs_cos3: float = 0.0
    cs_sin3: float = 0.0

    # ── L-dependent chroma scaling (2 params) ────────────────────
    # T(L) = exp(lc1*(L-0.5) + lc2*(L-0.5)^2)
    lc1: float = 0.0
    lc2: float = 0.0

    # ── Enhanced H-K harmonics (3 params) ─────────────────────────
    # Additional Fourier terms: sin(h), cos(2h), sin(2h)
    hk_sin1: float = 0.0
    hk_cos2: float = 0.0
    hk_sin2: float = 0.0

    # ── Hue correction (6 params) ──────────────────────────────────
    # δ(h) = Fourier series up to 3rd harmonic
    # Applied after M2 → Lab_raw, before cubic L correction
    hue_cos1: float = 0.0
    hue_sin1: float = 0.0
    hue_cos2: float = 0.0
    hue_sin2: float = 0.0
    hue_cos3: float = 0.0
    hue_sin3: float = 0.0

    # ── Hue×Lightness chroma interaction (4 params) ────────────────
    # Chroma scale depends on both hue and lightness:
    # S(h,L) includes (L-0.5) * Fourier(h) interaction term
    hlc_cos1: float = 0.0
    hlc_sin1: float = 0.0
    hlc_cos2: float = 0.0
    hlc_sin2: float = 0.0

    # ── Hue-dependent lightness scaling (4 params) ─────────────────
    # L *= exp(Fourier(h)) — pure hue → lightness effect, independent of chroma
    # Different from H-K (which depends on C). Captures: blue appears darker,
    # yellow appears lighter at same luminance.
    hl_cos1: float = 0.0
    hl_sin1: float = 0.0
    hl_cos2: float = 0.0
    hl_sin2: float = 0.0

    # ── Nonlinear chroma power (4 params, v6) ────────────────────
    # C_new = C^(1 + Fourier(h, 2 harmonics))
    # Power > 1 expands high-C distances, < 1 contracts
    cp_cos1: float = 0.0
    cp_sin1: float = 0.0
    cp_cos2: float = 0.0
    cp_sin2: float = 0.0

    # ── Adaptive dark L compression (3 params, v6+v13) ───────────
    # L_new = L * exp(g), g = L*(1-L)² * (lp_dark + lp_dark_hcos*cos(h) + lp_dark_hsin*sin(h))
    # Peaks near L≈1/3, targets dark region errors
    # v13: hue-dependent coefficient — dark blues vs dark yellows need different treatment
    lp_dark: float = 0.0
    lp_dark_hcos: float = 0.0
    lp_dark_hsin: float = 0.0

    # ── Distance metric parameters (v7) ─────────────────────────
    # DE = (dL² + dist_wC * (da² + db²))^(dist_power/2)
    # dist_power=1.0 → Euclidean, >1 amplifies large distances
    dist_power: float = 1.0
    dist_wC: float = 1.0

    # ── Hue correction 4th harmonic (v7) ─────────────────────────
    hue_cos4: float = 0.0
    hue_sin4: float = 0.0

    # ── Hue-dependent L correction (2 params, v8) ───────────────
    # L correction gets a hue-dependent additive term: t * (Lh_cos1*cos(h) + Lh_sin1*sin(h))
    # Targets L×H interaction sign flip (Dark+CynBlu vs Mid+CynBlu)
    Lh_cos1: float = 0.0
    Lh_sin1: float = 0.0

    # ── Chroma scaling 4th harmonic (2 params, v8) ──────────────
    cs_cos4: float = 0.0
    cs_sin4: float = 0.0

    # ── Nonlinear distance correction ──────────────────────────────
    # v12: DE_final = DE / (1 + dist_compress * DE)
    #   - Always positive and monotonic for dist_compress >= 0
    #   - No ceiling in theory (approaches 1/dist_compress asymptotically)
    #   - Concave: compresses large DE more than small DE
    #   - Small DE: ≈ DE * (1 - dist_compress * DE)
    # Legacy: dist_nl (v9/v10), dist_sat (v11) — kept for backward compat
    dist_nl: float = 0.0
    dist_sat: float = 0.0
    dist_compress: float = 0.0  # v12: monotonic compression. 0=identity, >0=compression

    # ── Linear asymptote for distance (v14) ────────────────────────
    # DE_final = DE * (1 + α*c*DE) / (1 + c*DE)
    # α=0: pure compression (v12 behavior), α=1: identity (no compression)
    # 0<α<1: reduces S-curve bias by preserving large DE better
    # Always monotonic for α∈[0,1], c≥0
    dist_linear: float = 0.0  # v14: linear asymptote. 0=v12 compat

    # ── Post-compression power (v14b) ─────────────────────────────
    # DE_final = DE_compressed^q
    # q=1.0: identity (current behavior)
    # q>1.0: expands large DE, reduces S-curve bias
    # q<1.0: further compresses (not useful)
    # Scan shows: q=1.1 → |r|=0.24, q=1.2 → |r|=0.11, q=1.3 → |r|≈0
    dist_post_power: float = 1.0  # v14b: post-compress power. 1.0=identity

    # ── Pair-dependent distance weights (v14c) ─────────────────────
    # CIEDE2000-style SL/SC: weight L and C contributions by pair average
    # SL = 1 + dist_sl * (L_avg - 0.5)²  — adjusts L weight for dark/light pairs
    # SC = 1 + dist_sc * C_avg            — adjusts C weight for chromatic pairs
    # DE = ((dL/SL)² + wC*(da/SC)² + wC*(db/SC)²)^(p/2)
    dist_sl: float = 0.0  # v14c: L-dep lightness weight. 0=no effect
    dist_sc: float = 0.0  # v14c: C-dep chroma weight. 0=no effect

    # ── Hue-dependent SL/SC (v15) ─────────────────────────────────────
    # Fourier modulation of SL/SC coefficients: sl_coeff = dist_sl + F_sl(h)
    # Allows pair-weighting to be active in some hue regions, inactive in others
    # 2 harmonics per channel (4 params each, 8 total)
    dist_sl_hcos1: float = 0.0  # v15: SL 1st harmonic cos
    dist_sl_hsin1: float = 0.0  # v15: SL 1st harmonic sin
    dist_sl_hcos2: float = 0.0  # v15: SL 2nd harmonic cos
    dist_sl_hsin2: float = 0.0  # v15: SL 2nd harmonic sin
    dist_sc_hcos1: float = 0.0  # v15: SC 1st harmonic cos
    dist_sc_hsin1: float = 0.0  # v15: SC 1st harmonic sin
    dist_sc_hcos2: float = 0.0  # v15: SC 2nd harmonic cos
    dist_sc_hsin2: float = 0.0  # v15: SC 2nd harmonic sin

    # ── Surround-dependent parameters (v16) ──────────────────────────
    # All centered at S=0.5 — default 0.0 preserves v14 behavior exactly.
    # Surround S ∈ [0, 1]: 0=dark, 0.5=average, 1=bright.

    # H-K modulation by surround (3 params)
    hk_weight_S: float = 0.0   # hk_weight_eff = hk_weight + hk_weight_S * (S - 0.5)
    hk_power_S: float = 0.0    # hk_power_eff = hk_power + hk_power_S * (S - 0.5)
    hk_hue_S: float = 0.0      # hue mod strength varies with S

    # Dark L compression surround (2 params)
    lp_dark_S: float = 0.0     # lp_dark_eff = lp_dark + lp_dark_S * (S - 0.5)
    lp_dark_S2: float = 0.0    # quadratic: + lp_dark_S2 * (S - 0.5)²

    # Chroma scaling surround (4 params)
    cs_S_lin: float = 0.0      # chroma_scale *= exp(cs_S_lin * (S-0.5))
    cs_S_quad: float = 0.0     # + cs_S_quad * (S-0.5)²
    lc_S_lin: float = 0.0      # L-dep chroma surround
    lc_S_quad: float = 0.0

    # Lightness scaling surround (2 params)
    hl_S_lin: float = 0.0      # hue-lightness interaction varies with S
    L_S_offset: float = 0.0    # global L shift: L += L_S_offset * (S - 0.5)

    def to_dict(self) -> dict:
        return {
            "M1": self.M1.tolist(),
            "gamma": self.gamma.tolist(),
            "M2": self.M2.tolist(),
            "hk_weight": self.hk_weight,
            "hk_power": self.hk_power,
            "hk_hue_mod": self.hk_hue_mod,
            "L_corr_p1": self.L_corr_p1,
            "L_corr_p2": self.L_corr_p2,
            "L_corr_p3": self.L_corr_p3,
            "cs_cos1": self.cs_cos1,
            "cs_sin1": self.cs_sin1,
            "cs_cos2": self.cs_cos2,
            "cs_sin2": self.cs_sin2,
            "cs_cos3": self.cs_cos3,
            "cs_sin3": self.cs_sin3,
            "lc1": self.lc1,
            "lc2": self.lc2,
            "hk_sin1": self.hk_sin1,
            "hk_cos2": self.hk_cos2,
            "hk_sin2": self.hk_sin2,
            "hue_cos1": self.hue_cos1,
            "hue_sin1": self.hue_sin1,
            "hue_cos2": self.hue_cos2,
            "hue_sin2": self.hue_sin2,
            "hue_cos3": self.hue_cos3,
            "hue_sin3": self.hue_sin3,
            "hlc_cos1": self.hlc_cos1,
            "hlc_sin1": self.hlc_sin1,
            "hlc_cos2": self.hlc_cos2,
            "hlc_sin2": self.hlc_sin2,
            "hl_cos1": self.hl_cos1,
            "hl_sin1": self.hl_sin1,
            "hl_cos2": self.hl_cos2,
            "hl_sin2": self.hl_sin2,
            "cp_cos1": self.cp_cos1,
            "cp_sin1": self.cp_sin1,
            "cp_cos2": self.cp_cos2,
            "cp_sin2": self.cp_sin2,
            "lp_dark": self.lp_dark,
            "lp_dark_hcos": self.lp_dark_hcos,
            "lp_dark_hsin": self.lp_dark_hsin,
            "dist_power": self.dist_power,
            "dist_wC": self.dist_wC,
            "hue_cos4": self.hue_cos4,
            "hue_sin4": self.hue_sin4,
            "Lh_cos1": self.Lh_cos1,
            "Lh_sin1": self.Lh_sin1,
            "cs_cos4": self.cs_cos4,
            "cs_sin4": self.cs_sin4,
            "dist_nl": self.dist_nl,
            "dist_sat": self.dist_sat,
            "dist_compress": self.dist_compress,
            "dist_linear": self.dist_linear,
            "dist_post_power": self.dist_post_power,
            "dist_sl": self.dist_sl,
            "dist_sc": self.dist_sc,
            "dist_sl_hcos1": self.dist_sl_hcos1,
            "dist_sl_hsin1": self.dist_sl_hsin1,
            "dist_sl_hcos2": self.dist_sl_hcos2,
            "dist_sl_hsin2": self.dist_sl_hsin2,
            "dist_sc_hcos1": self.dist_sc_hcos1,
            "dist_sc_hsin1": self.dist_sc_hsin1,
            "dist_sc_hcos2": self.dist_sc_hcos2,
            "dist_sc_hsin2": self.dist_sc_hsin2,
            # v16 surround
            "hk_weight_S": self.hk_weight_S,
            "hk_power_S": self.hk_power_S,
            "hk_hue_S": self.hk_hue_S,
            "lp_dark_S": self.lp_dark_S,
            "lp_dark_S2": self.lp_dark_S2,
            "cs_S_lin": self.cs_S_lin,
            "cs_S_quad": self.cs_S_quad,
            "lc_S_lin": self.lc_S_lin,
            "lc_S_quad": self.lc_S_quad,
            "hl_S_lin": self.hl_S_lin,
            "L_S_offset": self.L_S_offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetricParams":
        return cls(
            M1=np.array(d["M1"]),
            gamma=np.array(d["gamma"]),
            M2=np.array(d["M2"]),
            hk_weight=d.get("hk_weight", 0.0),
            hk_power=d.get("hk_power", 0.5),
            hk_hue_mod=d.get("hk_hue_mod", 0.0),
            L_corr_p1=d.get("L_corr_p1", 0.0),
            L_corr_p2=d.get("L_corr_p2", 0.0),
            L_corr_p3=d.get("L_corr_p3", 0.0),
            cs_cos1=d.get("cs_cos1", 0.0),
            cs_sin1=d.get("cs_sin1", 0.0),
            cs_cos2=d.get("cs_cos2", 0.0),
            cs_sin2=d.get("cs_sin2", 0.0),
            cs_cos3=d.get("cs_cos3", 0.0),
            cs_sin3=d.get("cs_sin3", 0.0),
            lc1=d.get("lc1", 0.0),
            lc2=d.get("lc2", 0.0),
            hk_sin1=d.get("hk_sin1", 0.0),
            hk_cos2=d.get("hk_cos2", 0.0),
            hk_sin2=d.get("hk_sin2", 0.0),
            hue_cos1=d.get("hue_cos1", 0.0),
            hue_sin1=d.get("hue_sin1", 0.0),
            hue_cos2=d.get("hue_cos2", 0.0),
            hue_sin2=d.get("hue_sin2", 0.0),
            hue_cos3=d.get("hue_cos3", 0.0),
            hue_sin3=d.get("hue_sin3", 0.0),
            hlc_cos1=d.get("hlc_cos1", 0.0),
            hlc_sin1=d.get("hlc_sin1", 0.0),
            hlc_cos2=d.get("hlc_cos2", 0.0),
            hlc_sin2=d.get("hlc_sin2", 0.0),
            hl_cos1=d.get("hl_cos1", 0.0),
            hl_sin1=d.get("hl_sin1", 0.0),
            hl_cos2=d.get("hl_cos2", 0.0),
            hl_sin2=d.get("hl_sin2", 0.0),
            cp_cos1=d.get("cp_cos1", 0.0),
            cp_sin1=d.get("cp_sin1", 0.0),
            cp_cos2=d.get("cp_cos2", 0.0),
            cp_sin2=d.get("cp_sin2", 0.0),
            lp_dark=d.get("lp_dark", 0.0),
            lp_dark_hcos=d.get("lp_dark_hcos", 0.0),
            lp_dark_hsin=d.get("lp_dark_hsin", 0.0),
            dist_power=d.get("dist_power", 1.0),
            dist_wC=d.get("dist_wC", 1.0),
            hue_cos4=d.get("hue_cos4", 0.0),
            hue_sin4=d.get("hue_sin4", 0.0),
            Lh_cos1=d.get("Lh_cos1", 0.0),
            Lh_sin1=d.get("Lh_sin1", 0.0),
            cs_cos4=d.get("cs_cos4", 0.0),
            cs_sin4=d.get("cs_sin4", 0.0),
            dist_nl=d.get("dist_nl", 0.0),
            dist_sat=d.get("dist_sat", 0.0),
            dist_compress=d.get("dist_compress", 0.0),
            dist_linear=d.get("dist_linear", 0.0),
            dist_post_power=d.get("dist_post_power", 1.0),
            dist_sl=d.get("dist_sl", 0.0),
            dist_sc=d.get("dist_sc", 0.0),
            dist_sl_hcos1=d.get("dist_sl_hcos1", 0.0),
            dist_sl_hsin1=d.get("dist_sl_hsin1", 0.0),
            dist_sl_hcos2=d.get("dist_sl_hcos2", 0.0),
            dist_sl_hsin2=d.get("dist_sl_hsin2", 0.0),
            dist_sc_hcos1=d.get("dist_sc_hcos1", 0.0),
            dist_sc_hsin1=d.get("dist_sc_hsin1", 0.0),
            dist_sc_hcos2=d.get("dist_sc_hcos2", 0.0),
            dist_sc_hsin2=d.get("dist_sc_hsin2", 0.0),
            # v16 surround
            hk_weight_S=d.get("hk_weight_S", 0.0),
            hk_power_S=d.get("hk_power_S", 0.0),
            hk_hue_S=d.get("hk_hue_S", 0.0),
            lp_dark_S=d.get("lp_dark_S", 0.0),
            lp_dark_S2=d.get("lp_dark_S2", 0.0),
            cs_S_lin=d.get("cs_S_lin", 0.0),
            cs_S_quad=d.get("cs_S_quad", 0.0),
            lc_S_lin=d.get("lc_S_lin", 0.0),
            lc_S_quad=d.get("lc_S_quad", 0.0),
            hl_S_lin=d.get("hl_S_lin", 0.0),
            L_S_offset=d.get("L_S_offset", 0.0),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MetricParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def oklab_params() -> MetricParams:
    """Return Oklab-equivalent initial parameters."""
    return MetricParams(
        M1=np.array([
            [0.8189330101, 0.3618667424, -0.1288597137],
            [0.0329845436, 0.9293118715, 0.0361456387],
            [0.0482003018, 0.2643662691, 0.6338517070],
        ]),
        gamma=np.array([1 / 3, 1 / 3, 1 / 3]),
        M2=np.array([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]),
        hk_weight=0.0,
        hk_power=0.5,
        hk_hue_mod=0.0,
    )


class MetricSpace(ColorSpace):
    """Parametric color space with embedded chroma-dependent lightness (v10).

    Forward pipeline:
        1.   XYZ → M1 → LMS
        2.   LMS → signed power compression → LMS_c
        3.   LMS_c → M2 → Lab_raw (L_raw, a_raw, b_raw)
        3.1  Neutral axis correction (v19, optional): a -= a_err(L), b -= b_err(L)
             Guarantees grays → a=b=0 even with independent gammas per channel.
        3.5  Hue correction: rotate (a,b) by δ(h)
        3.7  Embedded H-K: L_raw += hk(C_raw, h_raw) — uses raw chroma [Phase 2]
        4.   Cubic L correction: L1 = f(L_raw)
        4.5  Dark L compression: L1 = L1 * exp(lp_dark * L1 * (1-L1)²)
        5.   Hue-dependent chroma scaling: a,b *= exp(Fourier(h))
        5.5  Nonlinear chroma power: C = C^(1+Fourier(h))
        6.   L-dependent chroma scaling: a,b *= exp(poly(L))
        6.5  HLC interaction
        8.   Hue-dep lightness scaling

    All stages are exactly invertible. H-K uses exact subtraction in inverse.
    """

    name = "Metric"

    def __init__(self, params: MetricParams | None = None, surround: float = 0.5,
                 neutral_correction: bool = False, ab_rotate_deg: float = 0.0):
        if params is not None:
            self.params = params
        elif _PARAMS_PATH.exists():
            self.params = MetricParams.load(_PARAMS_PATH)
        else:
            self.params = oklab_params()

        self._surround = surround
        self._neutral_correction = neutral_correction
        self._ab_rotate_rad = float(np.deg2rad(ab_rotate_deg))
        self._ab_rot_cos = float(np.cos(self._ab_rotate_rad))
        self._ab_rot_sin = float(np.sin(self._ab_rotate_rad))

        # Pre-compute inverse matrices
        self._M1_inv = np.linalg.inv(self.params.M1)
        self._M2_inv = np.linalg.inv(self.params.M2)

        # Check if enrichment stages are active (for fast path)
        p = self.params
        self._has_L_corr = (p.L_corr_p1 != 0.0 or p.L_corr_p2 != 0.0 or p.L_corr_p3 != 0.0)
        self._has_chroma_scale = (
            p.cs_cos1 != 0.0 or p.cs_sin1 != 0.0 or
            p.cs_cos2 != 0.0 or p.cs_sin2 != 0.0 or
            p.cs_cos3 != 0.0 or p.cs_sin3 != 0.0 or
            p.cs_cos4 != 0.0 or p.cs_sin4 != 0.0
        )
        self._has_L_chroma = (p.lc1 != 0.0 or p.lc2 != 0.0)
        self._has_hk = (p.hk_weight != 0.0)
        self._has_hk_harmonics = (
            p.hk_hue_mod != 0.0 or p.hk_sin1 != 0.0 or
            p.hk_cos2 != 0.0 or p.hk_sin2 != 0.0
        )
        self._has_hue_correction = (
            p.hue_cos1 != 0.0 or p.hue_sin1 != 0.0 or
            p.hue_cos2 != 0.0 or p.hue_sin2 != 0.0 or
            p.hue_cos3 != 0.0 or p.hue_sin3 != 0.0 or
            p.hue_cos4 != 0.0 or p.hue_sin4 != 0.0
        )
        self._has_hlc = (
            p.hlc_cos1 != 0.0 or p.hlc_sin1 != 0.0 or
            p.hlc_cos2 != 0.0 or p.hlc_sin2 != 0.0
        )
        self._has_hue_lightness = (
            p.hl_cos1 != 0.0 or p.hl_sin1 != 0.0 or
            p.hl_cos2 != 0.0 or p.hl_sin2 != 0.0
        )
        self._has_chroma_power = (
            p.cp_cos1 != 0.0 or p.cp_sin1 != 0.0 or
            p.cp_cos2 != 0.0 or p.cp_sin2 != 0.0
        )
        self._has_dark_L = (p.lp_dark != 0.0 or p.lp_dark_hcos != 0.0 or p.lp_dark_hsin != 0.0)
        self._has_dark_L_hue = (p.lp_dark_hcos != 0.0 or p.lp_dark_hsin != 0.0)
        self._has_dist_metric = (p.dist_power != 1.0 or p.dist_wC != 1.0)
        self._has_dist_nl = (p.dist_nl != 0.0)
        self._has_dist_sat = (p.dist_sat != 0.0)
        self._has_dist_compress = (p.dist_compress != 0.0)
        self._has_dist_linear = (p.dist_linear != 0.0)
        self._has_dist_post_power = (p.dist_post_power != 1.0)
        self._has_dist_sl = (p.dist_sl != 0.0)
        self._has_dist_sc = (p.dist_sc != 0.0)
        self._has_dist_sl_hue = any(v != 0.0 for v in [
            p.dist_sl_hcos1, p.dist_sl_hsin1, p.dist_sl_hcos2, p.dist_sl_hsin2])
        self._has_dist_sc_hue = any(v != 0.0 for v in [
            p.dist_sc_hcos1, p.dist_sc_hsin1, p.dist_sc_hcos2, p.dist_sc_hsin2])
        self._has_pair_weights = (
            self._has_dist_sl or self._has_dist_sc or
            self._has_dist_sl_hue or self._has_dist_sc_hue)
        self._has_Lh = (p.Lh_cos1 != 0.0 or p.Lh_sin1 != 0.0)
        self._has_surround = any(v != 0.0 for v in [
            p.hk_weight_S, p.hk_power_S, p.hk_hue_S,
            p.lp_dark_S, p.lp_dark_S2,
            p.cs_S_lin, p.cs_S_quad, p.lc_S_lin, p.lc_S_quad,
            p.hl_S_lin, p.L_S_offset])

    # ── Distance metric override (v7) ──────────────────────────────

    def distance(self, XYZ_1: np.ndarray, XYZ_2: np.ndarray) -> np.ndarray:
        """Weighted Minkowski distance with optional monotonic compression.

        DE_raw = (dL² + wC*(da²+db²))^(p/2)
        v14b: DE_compressed^q  (q=dist_post_power, q>1 reduces S-curve bias)
        v14: DE * (1 + α*c*DE) / (1 + c*DE)  (α=dist_linear)
        v12: DE_raw / (1 + dist_compress * DE_raw)
        Legacy (v9/v10): DE * exp(dist_nl * DE)
        """
        c1 = self.from_XYZ(XYZ_1)
        c2 = self.from_XYZ(XYZ_2)
        no_correction = (
            not self._has_dist_nl and
            not self._has_dist_compress
        )
        if not self._has_dist_metric and no_correction and not self._has_pair_weights:
            return np.sqrt(np.sum((c1 - c2) ** 2, axis=-1))
        d = c1 - c2
        dL2 = d[..., 0] ** 2
        dab2 = d[..., 1] ** 2 + d[..., 2] ** 2
        # v14c/v15: pair-dependent SL/SC weighting (v15: hue-modulated)
        if self._has_pair_weights:
            # Compute average hue once (needed for hue-dep SL/SC, v15)
            if self._has_dist_sl_hue or self._has_dist_sc_hue:
                a_avg = (c1[..., 1] + c2[..., 1]) * 0.5
                b_avg = (c1[..., 2] + c2[..., 2]) * 0.5
                h_avg = np.arctan2(b_avg, a_avg)

            if self._has_dist_sl or self._has_dist_sl_hue:
                L_avg = (c1[..., 0] + c2[..., 0]) * 0.5
                sl_coeff = self.params.dist_sl
                if self._has_dist_sl_hue:
                    sl_coeff = sl_coeff + (
                        self.params.dist_sl_hcos1 * np.cos(h_avg) +
                        self.params.dist_sl_hsin1 * np.sin(h_avg) +
                        self.params.dist_sl_hcos2 * np.cos(2.0 * h_avg) +
                        self.params.dist_sl_hsin2 * np.sin(2.0 * h_avg))
                SL = 1.0 + sl_coeff * (L_avg - 0.5) ** 2
                dL2 = dL2 / (SL ** 2)

            if self._has_dist_sc or self._has_dist_sc_hue:
                C1 = np.sqrt(c1[..., 1] ** 2 + c1[..., 2] ** 2)
                C2 = np.sqrt(c2[..., 1] ** 2 + c2[..., 2] ** 2)
                C_avg = (C1 + C2) * 0.5
                sc_coeff = self.params.dist_sc
                if self._has_dist_sc_hue:
                    sc_coeff = sc_coeff + (
                        self.params.dist_sc_hcos1 * np.cos(h_avg) +
                        self.params.dist_sc_hsin1 * np.sin(h_avg) +
                        self.params.dist_sc_hcos2 * np.cos(2.0 * h_avg) +
                        self.params.dist_sc_hsin2 * np.sin(2.0 * h_avg))
                SC = 1.0 + sc_coeff * C_avg
                dab2 = dab2 / (SC ** 2)
        if self._has_dist_metric:
            sum_sq = dL2 + self.params.dist_wC * dab2
            DE = sum_sq ** (self.params.dist_power / 2.0)
        else:
            DE = np.sqrt(dL2 + dab2)
        if self._has_dist_compress:
            c = self.params.dist_compress
            if self._has_dist_linear:
                # v14: DE * (1 + α*c*DE) / (1 + c*DE) — reduces S-curve bias
                alpha = self.params.dist_linear
                DE = DE * (1.0 + alpha * c * DE) / (1.0 + c * DE)
            else:
                # v12: monotonic compression — always positive, no ceiling issues
                DE = DE / (1.0 + c * DE)
        elif self._has_dist_nl:
            # Legacy v9/v10/v11: exp-based (non-monotonic, kept for compat)
            if self._has_dist_sat:
                arg = self.params.dist_nl * DE / (1.0 + self.params.dist_sat * DE)
            else:
                arg = self.params.dist_nl * DE
            DE = DE * np.exp(arg)
        if self._has_dist_post_power:
            # v14b: post-compress power — expands large DE to reduce S-curve bias
            DE = DE ** self.params.dist_post_power
        return DE

    # ── Helper: hue correction δ(h) ────────────────────────────────

    def _hue_delta(self, h: np.ndarray) -> np.ndarray:
        """δ(h) = Fourier series for hue rotation (up to 4th harmonic)."""
        p = self.params
        return (
            p.hue_cos1 * np.cos(h) +
            p.hue_sin1 * np.sin(h) +
            p.hue_cos2 * np.cos(2.0 * h) +
            p.hue_sin2 * np.sin(2.0 * h) +
            p.hue_cos3 * np.cos(3.0 * h) +
            p.hue_sin3 * np.sin(3.0 * h) +
            p.hue_cos4 * np.cos(4.0 * h) +
            p.hue_sin4 * np.sin(4.0 * h)
        )

    def _hue_delta_deriv(self, h: np.ndarray) -> np.ndarray:
        """d/dh of δ(h), needed for Newton iteration in inverse."""
        p = self.params
        return (
            -p.hue_cos1 * np.sin(h) +
            p.hue_sin1 * np.cos(h) +
            -2.0 * p.hue_cos2 * np.sin(2.0 * h) +
            2.0 * p.hue_sin2 * np.cos(2.0 * h) +
            -3.0 * p.hue_cos3 * np.sin(3.0 * h) +
            3.0 * p.hue_sin3 * np.cos(3.0 * h) +
            -4.0 * p.hue_cos4 * np.sin(4.0 * h) +
            4.0 * p.hue_sin4 * np.cos(4.0 * h)
        )

    def _apply_hue_correction(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward: rotate (a, b) by δ(h) where h = atan2(b, a)."""
        h = np.arctan2(b, a)
        delta = self._hue_delta(h)
        h_new = h + delta
        C = np.sqrt(a ** 2 + b ** 2)
        return C * np.cos(h_new), C * np.sin(h_new)

    def _undo_hue_correction(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse: recover original (a, b) from hue-corrected values.

        Solve h_out = h_raw + δ(h_raw) for h_raw via Newton iteration.
        """
        h_out = np.arctan2(b, a)
        C = np.sqrt(a ** 2 + b ** 2)
        # Newton iteration: find h_raw such that h_raw + δ(h_raw) = h_out
        h_raw = h_out.copy()
        for _ in range(8):
            f = h_raw + self._hue_delta(h_raw) - h_out
            fp = 1.0 + self._hue_delta_deriv(h_raw)
            fp = np.where(np.abs(fp) < 1e-10, 1.0, fp)
            h_raw = h_raw - f / fp
        return C * np.cos(h_raw), C * np.sin(h_raw)

    # ── Helper: H-K hue modulation factor ──────────────────────────

    def _hk_hue_factor(self, h: np.ndarray) -> np.ndarray:
        """Compute 1 + Fourier hue modulation for H-K."""
        p = self.params
        factor = np.ones_like(h)
        if self._has_hk_harmonics:
            factor += (
                p.hk_hue_mod * np.cos(h) +
                p.hk_sin1 * np.sin(h) +
                p.hk_cos2 * np.cos(2.0 * h) +
                p.hk_sin2 * np.sin(2.0 * h)
            )
        return factor

    # ── Helper: hue-dependent chroma scale ─────────────────────────

    def _chroma_scale(self, h: np.ndarray) -> np.ndarray:
        """S(h) = exp(Fourier series up to 4th harmonic). Always > 0."""
        p = self.params
        log_s = (
            p.cs_cos1 * np.cos(h) +
            p.cs_sin1 * np.sin(h) +
            p.cs_cos2 * np.cos(2.0 * h) +
            p.cs_sin2 * np.sin(2.0 * h) +
            p.cs_cos3 * np.cos(3.0 * h) +
            p.cs_sin3 * np.sin(3.0 * h) +
            p.cs_cos4 * np.cos(4.0 * h) +
            p.cs_sin4 * np.sin(4.0 * h)
        )
        return np.exp(log_s)

    # ── Helper: L-dependent chroma scale ───────────────────────────

    def _L_chroma_scale(self, L: np.ndarray) -> np.ndarray:
        """T(L) = exp(polynomial). Always > 0. Clips exponent for extreme L."""
        p = self.params
        dL = L - 0.5
        arg = p.lc1 * dL + p.lc2 * dL ** 2
        return np.exp(np.clip(arg, -30.0, 30.0))

    # ── Helper: hue×lightness chroma interaction ────────────────────

    def _hlc_scale(self, h: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Hue×Lightness chroma interaction: exp((L-0.5) * Fourier(h)). Always > 0."""
        p = self.params
        hue_factor = (
            p.hlc_cos1 * np.cos(h) +
            p.hlc_sin1 * np.sin(h) +
            p.hlc_cos2 * np.cos(2.0 * h) +
            p.hlc_sin2 * np.sin(2.0 * h)
        )
        arg = (L - 0.5) * hue_factor
        return np.exp(np.clip(arg, -30.0, 30.0))

    # ── Helper: hue-dependent lightness scaling ─────────────────────

    def _hue_lightness_scale(self, h: np.ndarray) -> np.ndarray:
        """exp(Fourier(h)) — pure hue → lightness modulation. Always > 0."""
        p = self.params
        log_s = (
            p.hl_cos1 * np.cos(h) +
            p.hl_sin1 * np.sin(h) +
            p.hl_cos2 * np.cos(2.0 * h) +
            p.hl_sin2 * np.sin(2.0 * h)
        )
        return np.exp(log_s)

    # ── Helper: nonlinear chroma power (v6) ───────────────────────

    def _chroma_power(self, h: np.ndarray) -> np.ndarray:
        """1 + Fourier(h, 2 harmonics) — exponent for chroma power compression."""
        p = self.params
        return 1.0 + (
            p.cp_cos1 * np.cos(h) +
            p.cp_sin1 * np.sin(h) +
            p.cp_cos2 * np.cos(2.0 * h) +
            p.cp_sin2 * np.sin(2.0 * h)
        )

    # ── Helper: adaptive dark L compression (v6, v13 hue-dep) ─────

    def _dark_L_coeff(self, h: np.ndarray | None = None) -> float | np.ndarray:
        """Dark L coefficient: lp_dark + hue-dependent terms (v13)."""
        coeff = self.params.lp_dark
        if self._has_dark_L_hue and h is not None:
            coeff = coeff + self.params.lp_dark_hcos * np.cos(h) + self.params.lp_dark_hsin * np.sin(h)
        return coeff

    def _dark_L_compress(self, L: np.ndarray, h: np.ndarray | None = None, dS: float = 0.0) -> np.ndarray:
        """L_new = L * exp(g), g = L*(1-L)² * coeff(h, S). Targets dark region.

        v13: coeff is hue-dependent when lp_dark_hcos/hsin != 0.
        v16: coeff is surround-dependent when lp_dark_S/S2 != 0.
        """
        coeff = self._dark_L_coeff(h)
        coeff = coeff + self.params.lp_dark_S * dS + self.params.lp_dark_S2 * dS ** 2
        g = coeff * L * (1.0 - L) ** 2
        return L * np.exp(np.clip(g, -30.0, 30.0))

    def _dark_L_compress_inv(self, L_new: np.ndarray, h: np.ndarray | None = None, dS: float = 0.0) -> np.ndarray:
        """Invert dark L compression via Newton iteration.

        v13: coeff is hue-dependent when lp_dark_hcos/hsin != 0.
        v16: coeff is surround-dependent when lp_dark_S/S2 != 0.
        """
        coeff = self._dark_L_coeff(h)
        coeff = coeff + self.params.lp_dark_S * dS + self.params.lp_dark_S2 * dS ** 2
        L = L_new.copy()
        for _ in range(12):
            oml = 1.0 - L
            g = coeff * L * oml ** 2
            eg = np.exp(np.clip(g, -30.0, 30.0))
            f = L * eg - L_new
            # f'(L) = exp(g) * (1 + L * g'(L))
            # g'(L) = coeff * (1-L)(1-3L)
            gp = coeff * oml * (1.0 - 3.0 * L)
            fp = eg * (1.0 + L * gp)
            fp = np.where(np.abs(fp) < 1e-10, 1.0, fp)
            L = L - f / fp
        return L

    # ── Helper: quartic L correction ───────────────────────────────

    def _L_correct(self, L_raw: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """L1 = L_raw + p1*t + p2*t*(0.5-L) + p3*t^2 [+ t*Lh(h)], t = L*(1-L)."""
        p = self.params
        t = L_raw * (1.0 - L_raw)
        result = L_raw + p.L_corr_p1 * t + p.L_corr_p2 * t * (0.5 - L_raw) + p.L_corr_p3 * t * t
        if self._has_Lh and h is not None:
            Lh = p.Lh_cos1 * np.cos(h) + p.Lh_sin1 * np.sin(h)
            result = result + t * Lh
        return result

    def _L_correct_inv(self, L1: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Invert L correction via Newton iteration."""
        p = self.params
        # Precompute hue-dependent term (constant w.r.t. L)
        Lh = 0.0
        if self._has_Lh and h is not None:
            Lh = p.Lh_cos1 * np.cos(h) + p.Lh_sin1 * np.sin(h)
        L = L1.copy()  # initial guess
        for _ in range(15):
            t = L * (1.0 - L)
            dt = 1.0 - 2.0 * L
            f = (L + p.L_corr_p1 * t + p.L_corr_p2 * t * (0.5 - L)
                 + p.L_corr_p3 * t * t + t * Lh - L1)
            dfdL = (
                1.0 +
                (p.L_corr_p1 + Lh) * dt +
                p.L_corr_p2 * (dt * (0.5 - L) - t) +
                p.L_corr_p3 * 2.0 * t * dt
            )
            dfdL = np.where(np.abs(dfdL) < 1e-10, 1.0, dfdL)
            L = L - f / dfdL
        return L

    # ── Helper: neutral axis correction (v19) ────────────────────

    # D65 white point (Y=1)
    _D65_WHITE = np.array([0.95047, 1.0, 1.08883])

    def _build_neutral_lut(self, n_samples: int = 256) -> None:
        """Build lookup table for end-of-pipeline neutral correction.

        Samples grays through the full pipeline (without NC) to find
        the achromatic error at each lightness level. Uses PCHIP
        interpolation for smooth a_err(L), b_err(L) functions.
        """
        from scipy.interpolate import PchipInterpolator

        # Sample Y values (log-spaced for better coverage of darks)
        Y_vals = np.concatenate([
            [0.0],  # black point anchor: NC error is zero at L=0
            np.linspace(0.001, 0.01, 10),
            np.linspace(0.01, 0.1, 20),
            np.linspace(0.1, 2.0, n_samples - 30),
        ])
        gray_XYZ = np.outer(Y_vals, self._D65_WHITE)

        # Run through the full pipeline WITHOUT neutral correction
        old_nc = self._neutral_correction
        old_rot_rad = self._ab_rotate_rad
        old_rot_cos = self._ab_rot_cos
        old_rot_sin = self._ab_rot_sin
        self._neutral_correction = False
        # Neutral LUT must be built in the *pre-rotation* coordinate system,
        # since the correction is applied before any post-pipeline rotation.
        self._ab_rotate_rad = 0.0
        self._ab_rot_cos = 1.0
        self._ab_rot_sin = 0.0
        Lab_gray = self.from_XYZ(gray_XYZ)
        self._neutral_correction = old_nc
        self._ab_rotate_rad = old_rot_rad
        self._ab_rot_cos = old_rot_cos
        self._ab_rot_sin = old_rot_sin

        L_gray = Lab_gray[:, 0]
        a_gray = Lab_gray[:, 1]
        b_gray = Lab_gray[:, 2]

        # Sort by L for monotone interpolation
        order = np.argsort(L_gray)
        L_sorted = L_gray[order]
        a_sorted = a_gray[order]
        b_sorted = b_gray[order]

        # Remove duplicates (needed for PCHIP)
        mask = np.diff(L_sorted, prepend=-np.inf) > 1e-12
        L_sorted = L_sorted[mask]
        a_sorted = a_sorted[mask]
        b_sorted = b_sorted[mask]

        self._nc_L_range = (L_sorted[0], L_sorted[-1])
        self._nc_a_interp = PchipInterpolator(L_sorted, a_sorted, extrapolate=True)
        self._nc_b_interp = PchipInterpolator(L_sorted, b_sorted, extrapolate=True)
        self._nc_lut_built = True

    def _neutral_error(self, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get achromatic error (a_err, b_err) at given final L values.

        Uses precomputed LUT through the full pipeline.
        """
        if not getattr(self, '_nc_lut_built', False):
            self._build_neutral_lut()

        a_err = self._nc_a_interp(L)
        b_err = self._nc_b_interp(L)
        return a_err, b_err

    # ── Base Lab (M1 → power → M2 only, no enrichment) ─────────────

    def base_from_XYZ(self, XYZ: np.ndarray) -> np.ndarray:
        """XYZ → base Lab (M1 → power → M2 only, no enrichment stages).

        Structurally identical to Oklab — good for color generation
        (palettes, gradients, gamut mapping) without enrichment distortions.
        """
        XYZ = np.asarray(XYZ, dtype=np.float64)
        LMS = XYZ @ self.params.M1.T
        LMS_c = np.sign(LMS) * np.abs(LMS) ** self.params.gamma
        return LMS_c @ self.params.M2.T

    def base_to_XYZ(self, lab: np.ndarray) -> np.ndarray:
        """Base Lab → XYZ (M2_inv → inv_power → M1_inv).

        Inverse of base_from_XYZ.
        """
        lab = np.asarray(lab, dtype=np.float64)
        LMS_c = lab @ self._M2_inv.T
        inv_gamma = 1.0 / self.params.gamma
        LMS = np.sign(LMS_c) * np.abs(LMS_c) ** inv_gamma
        return LMS @ self._M1_inv.T

    # ── Forward transform ──────────────────────────────────────────

    def from_XYZ(self, XYZ: np.ndarray, S: float | None = None) -> np.ndarray:
        """XYZ -> Analytical Lab (enriched pipeline).

        Parameters
        ----------
        S : surround luminance (0=dark, 0.5=average, 1=bright).
            If None, uses instance default (self._surround).
        """
        XYZ = np.asarray(XYZ, dtype=np.float64)
        if S is None:
            S = self._surround
        dS = S - 0.5  # centered — all S params multiply by dS

        # 1. XYZ -> LMS
        LMS = XYZ @ self.params.M1.T

        # 2. Power compression (signed)
        LMS_c = np.sign(LMS) * np.abs(LMS) ** self.params.gamma

        # 3. LMS_c -> Lab_raw
        Lab = LMS_c @ self.params.M2.T
        L_raw = Lab[..., 0]
        a = Lab[..., 1]
        b = Lab[..., 2]

        # 3.5. Hue correction (rotate chromatic plane)
        if self._has_hue_correction:
            a, b = self._apply_hue_correction(a, b)

        # 3.7. Embedded H-K correction (Phase 2: chroma-dependent lightness)
        # v16: H-K weight and power modulated by surround
        if self._has_hk:
            C_raw = np.sqrt(a ** 2 + b ** 2)
            hk_w = self.params.hk_weight + self.params.hk_weight_S * dS
            hk_p = self.params.hk_power + self.params.hk_power_S * dS
            hk_boost = hk_w * C_raw ** np.clip(hk_p, 0.01, 10.0)
            if self._has_hk_harmonics or self.params.hk_hue_S != 0.0:
                h_raw = np.arctan2(b, a)
                hue_factor = self._hk_hue_factor(h_raw)
                if self.params.hk_hue_S != 0.0:
                    hue_factor = hue_factor + self.params.hk_hue_S * dS * np.cos(h_raw)
                hk_boost = hk_boost * hue_factor
            L_raw = L_raw + hk_boost

        # 4. Cubic L correction (with optional hue-dependent term, v8)
        if self._has_L_corr or self._has_Lh:
            h_L = np.arctan2(b, a) if self._has_Lh else None
            L1 = self._L_correct(L_raw, h_L)
        else:
            L1 = L_raw

        # 4.5. Dark L compression (v6, v13 hue-dependent, v16 surround)
        if self._has_dark_L or self.params.lp_dark_S != 0.0 or self.params.lp_dark_S2 != 0.0:
            h_dark = np.arctan2(b, a) if self._has_dark_L_hue else None
            L1 = self._dark_L_compress(L1, h_dark, dS=dS)

        # 5. Hue-dependent chroma scaling
        # v16: surround modulates global chroma scale
        if self._has_chroma_scale or self.params.cs_S_lin != 0.0 or self.params.cs_S_quad != 0.0:
            h = np.arctan2(b, a)
            cs = self._chroma_scale(h) if self._has_chroma_scale else np.ones_like(h)
            if self.params.cs_S_lin != 0.0 or self.params.cs_S_quad != 0.0:
                cs = cs * np.exp(self.params.cs_S_lin * dS + self.params.cs_S_quad * dS ** 2)
            a = a * cs
            b = b * cs

        # 5.5. Nonlinear chroma power (v6)
        if self._has_chroma_power:
            h_cp = np.arctan2(b, a)
            C = np.sqrt(a ** 2 + b ** 2)
            power = self._chroma_power(h_cp)
            C_new = np.where(C > 0, C ** power, 0.0)
            a = C_new * np.cos(h_cp)
            b = C_new * np.sin(h_cp)

        # 6. L-dependent chroma scaling
        # v16: surround modulates L-dep chroma
        if self._has_L_chroma or self.params.lc_S_lin != 0.0 or self.params.lc_S_quad != 0.0:
            T = self._L_chroma_scale(L1) if self._has_L_chroma else np.ones_like(L1)
            if self.params.lc_S_lin != 0.0 or self.params.lc_S_quad != 0.0:
                T = T * np.exp(self.params.lc_S_lin * dS + self.params.lc_S_quad * dS ** 2)
            a = a * T
            b = b * T

        # 6.5. Hue×Lightness chroma interaction
        if self._has_hlc:
            h_hlc = np.arctan2(b, a)
            HLC = self._hlc_scale(h_hlc, L1)
            a = a * HLC
            b = b * HLC

        # (7. removed — H-K moved to step 3.7)

        # 8. Hue-dependent lightness scaling
        # v16: surround modulates hue-lightness
        if self._has_hue_lightness or self.params.hl_S_lin != 0.0:
            h_hl = np.arctan2(b, a)
            hl_scale = self._hue_lightness_scale(h_hl) if self._has_hue_lightness else np.ones_like(h_hl)
            if self.params.hl_S_lin != 0.0:
                hl_scale = hl_scale * np.exp(self.params.hl_S_lin * dS)
            L1 = L1 * hl_scale

        # 9. Global L offset from surround (v16)
        if self.params.L_S_offset != 0.0:
            L1 = L1 + self.params.L_S_offset * dS

        # 10. Neutral axis correction (v19): subtract achromatic error at pipeline end
        # Applied AFTER all enrichment stages to avoid hue interaction issues
        if self._neutral_correction:
            a_err, b_err = self._neutral_error(L1)
            a = a - a_err
            b = b - b_err

        # 11. Optional rigid rotation in a/b plane (post-NC).
        # This is a pure isometry: preserves C=sqrt(a^2+b^2) and any distance
        # metric that depends only on (dL, da^2+db^2, L_avg, C_avg).
        if self._ab_rotate_rad != 0.0:
            a_rot = a * self._ab_rot_cos - b * self._ab_rot_sin
            b_rot = a * self._ab_rot_sin + b * self._ab_rot_cos
            a, b = a_rot, b_rot

        return np.stack([L1, a, b], axis=-1)

    # ── Inverse transform ──────────────────────────────────────────

    def to_XYZ(self, coords: np.ndarray, S: float | None = None) -> np.ndarray:
        """Analytical Lab (enriched pipeline) -> XYZ.

        Parameters
        ----------
        S : surround luminance (0=dark, 0.5=average, 1=bright).
            If None, uses instance default (self._surround).
        """
        coords = np.asarray(coords, dtype=np.float64)
        if S is None:
            S = self._surround
        dS = S - 0.5

        L2 = coords[..., 0]
        a2 = coords[..., 1]
        b2 = coords[..., 2]

        # 11. Undo rigid rotation in a/b plane (pre-NC).
        if self._ab_rotate_rad != 0.0:
            a_unrot = a2 * self._ab_rot_cos + b2 * self._ab_rot_sin
            b_unrot = -a2 * self._ab_rot_sin + b2 * self._ab_rot_cos
            a2, b2 = a_unrot, b_unrot

        # 10. Undo neutral axis correction (v19): add back achromatic error
        if self._neutral_correction:
            a_err, b_err = self._neutral_error(L2)
            a2 = a2 + a_err
            b2 = b2 + b_err

        # 9. Undo global L offset from surround (v16)
        if self.params.L_S_offset != 0.0:
            L2 = L2 - self.params.L_S_offset * dS

        # 8. Undo hue-dependent lightness scaling (v16: surround)
        if self._has_hue_lightness or self.params.hl_S_lin != 0.0:
            h_hl = np.arctan2(b2, a2)
            hl_scale = self._hue_lightness_scale(h_hl) if self._has_hue_lightness else np.ones_like(h_hl)
            if self.params.hl_S_lin != 0.0:
                hl_scale = hl_scale * np.exp(self.params.hl_S_lin * dS)
            L_pre_hl = L2 / hl_scale
        else:
            L_pre_hl = L2

        # (7. removed — H-K moved to step 3.7)
        L1 = L_pre_hl

        # 6.5. Undo hue×lightness chroma interaction
        if self._has_hlc:
            h_hlc = np.arctan2(b2, a2)
            HLC = self._hlc_scale(h_hlc, L1)
            a2 = a2 / HLC
            b2 = b2 / HLC

        # 6. Undo L-dependent chroma scaling (v16: surround)
        if self._has_L_chroma or self.params.lc_S_lin != 0.0 or self.params.lc_S_quad != 0.0:
            T = self._L_chroma_scale(L1) if self._has_L_chroma else np.ones_like(L1)
            if self.params.lc_S_lin != 0.0 or self.params.lc_S_quad != 0.0:
                T = T * np.exp(self.params.lc_S_lin * dS + self.params.lc_S_quad * dS ** 2)
            a1 = a2 / T
            b1 = b2 / T
        else:
            a1, b1 = a2, b2

        # 5.5. Undo nonlinear chroma power (v6)
        if self._has_chroma_power:
            h_cp = np.arctan2(b1, a1)
            C = np.sqrt(a1 ** 2 + b1 ** 2)
            power = self._chroma_power(h_cp)
            C_orig = np.where(C > 0, C ** (1.0 / power), 0.0)
            a1 = C_orig * np.cos(h_cp)
            b1 = C_orig * np.sin(h_cp)

        # 5. Undo hue-dependent chroma scaling (v16: surround)
        if self._has_chroma_scale or self.params.cs_S_lin != 0.0 or self.params.cs_S_quad != 0.0:
            h = np.arctan2(b1, a1)
            cs = self._chroma_scale(h) if self._has_chroma_scale else np.ones_like(h)
            if self.params.cs_S_lin != 0.0 or self.params.cs_S_quad != 0.0:
                cs = cs * np.exp(self.params.cs_S_lin * dS + self.params.cs_S_quad * dS ** 2)
            a_raw = a1 / cs
            b_raw = b1 / cs
        else:
            a_raw, b_raw = a1, b1

        # 4.5. Undo dark L compression (v6, v13, v16 surround)
        if self._has_dark_L or self.params.lp_dark_S != 0.0 or self.params.lp_dark_S2 != 0.0:
            h_dark = np.arctan2(b_raw, a_raw) if self._has_dark_L_hue else None
            L1 = self._dark_L_compress_inv(L1, h_dark, dS=dS)

        # 4. Undo cubic L correction (with optional hue-dependent term, v8)
        if self._has_L_corr or self._has_Lh:
            h_L = np.arctan2(b_raw, a_raw) if self._has_Lh else None
            L_raw = self._L_correct_inv(L1, h_L)
        else:
            L_raw = L1

        # 3.7. Undo embedded H-K (v16: surround-modulated)
        if self._has_hk:
            C_raw = np.sqrt(a_raw ** 2 + b_raw ** 2)
            hk_w = self.params.hk_weight + self.params.hk_weight_S * dS
            hk_p = self.params.hk_power + self.params.hk_power_S * dS
            hk_boost = hk_w * C_raw ** np.clip(hk_p, 0.01, 10.0)
            if self._has_hk_harmonics or self.params.hk_hue_S != 0.0:
                h_raw = np.arctan2(b_raw, a_raw)
                hue_factor = self._hk_hue_factor(h_raw)
                if self.params.hk_hue_S != 0.0:
                    hue_factor = hue_factor + self.params.hk_hue_S * dS * np.cos(h_raw)
                hk_boost = hk_boost * hue_factor
            L_raw = L_raw - hk_boost

        # 3.5. Undo hue correction
        if self._has_hue_correction:
            a_raw, b_raw = self._undo_hue_correction(a_raw, b_raw)

        # 3. Lab_raw -> LMS_c
        Lab = np.stack([L_raw, a_raw, b_raw], axis=-1)
        LMS_c = Lab @ self._M2_inv.T

        # 2. Undo power compression
        inv_gamma = 1.0 / self.params.gamma
        LMS = np.sign(LMS_c) * np.abs(LMS_c) ** inv_gamma

        # 1. LMS -> XYZ
        return LMS @ self._M1_inv.T
