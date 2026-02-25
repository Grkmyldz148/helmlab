/**
 * Analytical parametric color space — 13-stage forward/inverse transform.
 *
 * Pipeline (v19):
 *   XYZ → M1 → LMS → power compression → M2 → Lab_raw
 *   → hue correction → embedded H-K → cubic L → dark L
 *   → hue-dep chroma scale → chroma power → L-dep chroma scale
 *   → HLC interaction → hue-dep lightness → neutral correction → rotation
 */

import type { Lab, XYZ } from '../types.js';
import { mat3TMulVec, signedPow, clamp } from '../utils/math.js';
import type { CompiledParams } from './params.js';
import { neutralError } from './neutral-lut.js';

const { cos, sin, sqrt, atan2, exp, abs, pow, PI } = Math;

export interface AnalyticalOptions {
  neutralCorrection?: boolean;
  abRotateDeg?: number;
}

export class AnalyticalSpace {
  private readonly p: CompiledParams;
  private readonly nc: boolean;
  private readonly rotCos: number;
  private readonly rotSin: number;
  private readonly hasRot: boolean;

  constructor(params: CompiledParams, opts: AnalyticalOptions = {}) {
    this.p = params;
    this.nc = opts.neutralCorrection ?? true;
    const rad = ((opts.abRotateDeg ?? -28.2) * PI) / 180;
    this.rotCos = cos(rad);
    this.rotSin = sin(rad);
    this.hasRot = abs(rad) > 1e-12;
  }

  // ── Forward transform ──────────────────────────────────────────

  fromXYZ(xyz: XYZ): Lab {
    const r = this.p.raw;
    const M1 = this.p.M1;
    const M2 = this.p.M2;
    const g = this.p.gamma;

    // 1. XYZ → LMS (via M1^T multiply: LMS_i = sum_j M1[i,j]*XYZ[j])
    const lms0 = M1[0] * xyz[0] + M1[1] * xyz[1] + M1[2] * xyz[2];
    const lms1 = M1[3] * xyz[0] + M1[4] * xyz[1] + M1[5] * xyz[2];
    const lms2 = M1[6] * xyz[0] + M1[7] * xyz[1] + M1[8] * xyz[2];

    // 2. Power compression (signed)
    const c0 = signedPow(lms0, g[0]);
    const c1 = signedPow(lms1, g[1]);
    const c2 = signedPow(lms2, g[2]);

    // 3. LMS_c → Lab_raw (via M2)
    let L = M2[0] * c0 + M2[1] * c1 + M2[2] * c2;
    let a = M2[3] * c0 + M2[4] * c1 + M2[5] * c2;
    let b = M2[6] * c0 + M2[7] * c1 + M2[8] * c2;

    // 3.5 Hue correction
    if (r.hue_cos1 !== 0 || r.hue_sin1 !== 0 || r.hue_cos2 !== 0 ||
        r.hue_sin2 !== 0 || r.hue_cos3 !== 0 || r.hue_sin3 !== 0 ||
        r.hue_cos4 !== 0 || r.hue_sin4 !== 0) {
      const h = atan2(b, a);
      const C = sqrt(a * a + b * b);
      const delta = hueDelta(r, h);
      const hNew = h + delta;
      a = C * cos(hNew);
      b = C * sin(hNew);
    }

    // 3.7 Embedded H-K correction
    if (r.hk_weight !== 0) {
      const Cr = sqrt(a * a + b * b);
      let hkBoost = r.hk_weight * pow(Cr, clamp(r.hk_power, 0.01, 10));
      if (r.hk_hue_mod !== 0 || r.hk_sin1 !== 0 || r.hk_cos2 !== 0 || r.hk_sin2 !== 0) {
        const hr = atan2(b, a);
        const factor = 1 + r.hk_hue_mod * cos(hr) + r.hk_sin1 * sin(hr) +
                        r.hk_cos2 * cos(2 * hr) + r.hk_sin2 * sin(2 * hr);
        hkBoost *= factor;
      }
      L += hkBoost;
    }

    // 4. Cubic L correction
    if (r.L_corr_p1 !== 0 || r.L_corr_p2 !== 0 || r.L_corr_p3 !== 0 ||
        r.Lh_cos1 !== 0 || r.Lh_sin1 !== 0) {
      const h = (r.Lh_cos1 !== 0 || r.Lh_sin1 !== 0) ? atan2(b, a) : 0;
      L = lCorrect(r, L, h);
    }

    // 4.5 Dark L compression
    if (r.lp_dark !== 0 || r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) {
      const h = (r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) ? atan2(b, a) : 0;
      L = darkLCompress(r, L, h);
    }

    // 5. Hue-dependent chroma scaling
    if (r.cs_cos1 !== 0 || r.cs_sin1 !== 0 || r.cs_cos2 !== 0 || r.cs_sin2 !== 0 ||
        r.cs_cos3 !== 0 || r.cs_sin3 !== 0 || r.cs_cos4 !== 0 || r.cs_sin4 !== 0) {
      const h = atan2(b, a);
      const cs = chromaScale(r, h);
      a *= cs;
      b *= cs;
    }

    // 5.5 Nonlinear chroma power
    if (r.cp_cos1 !== 0 || r.cp_sin1 !== 0 || r.cp_cos2 !== 0 || r.cp_sin2 !== 0) {
      const h = atan2(b, a);
      const C = sqrt(a * a + b * b);
      const p = chromaPower(r, h);
      const Cn = C > 0 ? pow(C, p) : 0;
      a = Cn * cos(h);
      b = Cn * sin(h);
    }

    // 6. L-dependent chroma scaling
    if (r.lc1 !== 0 || r.lc2 !== 0) {
      const T = lChromaScale(r, L);
      a *= T;
      b *= T;
    }

    // 6.5 HLC interaction
    if (r.hlc_cos1 !== 0 || r.hlc_sin1 !== 0 || r.hlc_cos2 !== 0 || r.hlc_sin2 !== 0) {
      const h = atan2(b, a);
      const hlc = hlcScale(r, h, L);
      a *= hlc;
      b *= hlc;
    }

    // 8. Hue-dependent lightness scaling
    if (r.hl_cos1 !== 0 || r.hl_sin1 !== 0 || r.hl_cos2 !== 0 || r.hl_sin2 !== 0) {
      const h = atan2(b, a);
      L *= hueLightnessScale(r, h);
    }

    // 10. Neutral correction
    if (this.nc) {
      const [aErr, bErr] = neutralError(L);
      a -= aErr;
      b -= bErr;
    }

    // 11. Rigid rotation
    if (this.hasRot) {
      const aRot = a * this.rotCos - b * this.rotSin;
      const bRot = a * this.rotSin + b * this.rotCos;
      a = aRot;
      b = bRot;
    }

    return [L, a, b];
  }

  // ── Inverse transform ──────────────────────────────────────────

  toXYZ(lab: Lab): XYZ {
    const r = this.p.raw;
    let [L, a, b] = lab;

    // 11. Undo rotation
    if (this.hasRot) {
      const aUn = a * this.rotCos + b * this.rotSin;
      const bUn = -a * this.rotSin + b * this.rotCos;
      a = aUn;
      b = bUn;
    }

    // 10. Undo neutral correction
    if (this.nc) {
      const [aErr, bErr] = neutralError(L);
      a += aErr;
      b += bErr;
    }

    // 8. Undo hue-dep lightness
    if (r.hl_cos1 !== 0 || r.hl_sin1 !== 0 || r.hl_cos2 !== 0 || r.hl_sin2 !== 0) {
      const h = atan2(b, a);
      L /= hueLightnessScale(r, h);
    }

    // 6.5 Undo HLC
    if (r.hlc_cos1 !== 0 || r.hlc_sin1 !== 0 || r.hlc_cos2 !== 0 || r.hlc_sin2 !== 0) {
      const h = atan2(b, a);
      const hlc = hlcScale(r, h, L);
      a /= hlc;
      b /= hlc;
    }

    // 6. Undo L-dep chroma
    if (r.lc1 !== 0 || r.lc2 !== 0) {
      const T = lChromaScale(r, L);
      a /= T;
      b /= T;
    }

    // 5.5 Undo chroma power
    if (r.cp_cos1 !== 0 || r.cp_sin1 !== 0 || r.cp_cos2 !== 0 || r.cp_sin2 !== 0) {
      const h = atan2(b, a);
      const C = sqrt(a * a + b * b);
      const p = chromaPower(r, h);
      const Co = C > 0 ? pow(C, 1 / p) : 0;
      a = Co * cos(h);
      b = Co * sin(h);
    }

    // 5. Undo chroma scaling
    if (r.cs_cos1 !== 0 || r.cs_sin1 !== 0 || r.cs_cos2 !== 0 || r.cs_sin2 !== 0 ||
        r.cs_cos3 !== 0 || r.cs_sin3 !== 0 || r.cs_cos4 !== 0 || r.cs_sin4 !== 0) {
      const h = atan2(b, a);
      const cs = chromaScale(r, h);
      a /= cs;
      b /= cs;
    }

    // 4.5 Undo dark L
    if (r.lp_dark !== 0 || r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) {
      const h = (r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) ? atan2(b, a) : 0;
      L = darkLCompressInv(r, L, h);
    }

    // 4. Undo cubic L
    if (r.L_corr_p1 !== 0 || r.L_corr_p2 !== 0 || r.L_corr_p3 !== 0 ||
        r.Lh_cos1 !== 0 || r.Lh_sin1 !== 0) {
      const h = (r.Lh_cos1 !== 0 || r.Lh_sin1 !== 0) ? atan2(b, a) : 0;
      L = lCorrectInv(r, L, h);
    }

    // 3.7 Undo H-K
    if (r.hk_weight !== 0) {
      const Cr = sqrt(a * a + b * b);
      let hkBoost = r.hk_weight * pow(Cr, clamp(r.hk_power, 0.01, 10));
      if (r.hk_hue_mod !== 0 || r.hk_sin1 !== 0 || r.hk_cos2 !== 0 || r.hk_sin2 !== 0) {
        const hr = atan2(b, a);
        const factor = 1 + r.hk_hue_mod * cos(hr) + r.hk_sin1 * sin(hr) +
                        r.hk_cos2 * cos(2 * hr) + r.hk_sin2 * sin(2 * hr);
        hkBoost *= factor;
      }
      L -= hkBoost;
    }

    // 3.5 Undo hue correction
    if (r.hue_cos1 !== 0 || r.hue_sin1 !== 0 || r.hue_cos2 !== 0 ||
        r.hue_sin2 !== 0 || r.hue_cos3 !== 0 || r.hue_sin3 !== 0 ||
        r.hue_cos4 !== 0 || r.hue_sin4 !== 0) {
      const hOut = atan2(b, a);
      const C = sqrt(a * a + b * b);
      // Newton iteration: find h_raw s.t. h_raw + delta(h_raw) = h_out
      let hRaw = hOut;
      for (let i = 0; i < 8; i++) {
        const f = hRaw + hueDelta(r, hRaw) - hOut;
        let fp = 1 + hueDeltaDeriv(r, hRaw);
        if (abs(fp) < 1e-10) fp = 1;
        hRaw -= f / fp;
      }
      a = C * cos(hRaw);
      b = C * sin(hRaw);
    }

    // 3. Lab → LMS_c (via M2_inv)
    const M2i = this.p.M2_inv;
    const lc0 = M2i[0] * L + M2i[1] * a + M2i[2] * b;
    const lc1 = M2i[3] * L + M2i[4] * a + M2i[5] * b;
    const lc2 = M2i[6] * L + M2i[7] * a + M2i[8] * b;

    // 2. Undo power compression
    const ig = this.p.inv_gamma;
    const l0 = signedPow(lc0, ig[0]);
    const l1 = signedPow(lc1, ig[1]);
    const l2 = signedPow(lc2, ig[2]);

    // 1. LMS → XYZ (via M1_inv)
    const M1i = this.p.M1_inv;
    return [
      M1i[0] * l0 + M1i[1] * l1 + M1i[2] * l2,
      M1i[3] * l0 + M1i[4] * l1 + M1i[5] * l2,
      M1i[6] * l0 + M1i[7] * l1 + M1i[8] * l2,
    ];
  }

  // ── Distance ───────────────────────────────────────────────────

  distance(lab1: Lab, lab2: Lab): number {
    const r = this.p.raw;
    let dL2 = (lab1[0] - lab2[0]) ** 2;
    let dab2 = (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2;

    // Pair-dependent SL/SC
    if (r.dist_sl !== 0) {
      const Lavg = (lab1[0] + lab2[0]) * 0.5;
      const SL = 1 + r.dist_sl * (Lavg - 0.5) ** 2;
      dL2 /= SL ** 2;
    }
    if (r.dist_sc !== 0) {
      const C1 = sqrt(lab1[1] ** 2 + lab1[2] ** 2);
      const C2 = sqrt(lab2[1] ** 2 + lab2[2] ** 2);
      const SC = 1 + r.dist_sc * (C1 + C2) * 0.5;
      dab2 /= SC ** 2;
    }

    // Weighted Minkowski
    let DE: number;
    if (r.dist_power !== 1 || r.dist_wC !== 1) {
      DE = (dL2 + r.dist_wC * dab2) ** (r.dist_power / 2);
    } else {
      DE = sqrt(dL2 + dab2);
    }

    // Monotonic compression
    if (r.dist_compress !== 0) {
      const c = r.dist_compress;
      if (r.dist_linear !== 0) {
        DE = DE * (1 + r.dist_linear * c * DE) / (1 + c * DE);
      } else {
        DE /= 1 + c * DE;
      }
    }

    // Post-power
    if (r.dist_post_power !== 1) {
      DE = DE ** r.dist_post_power;
    }

    return DE;
  }
}

// ── Helper functions (module-private) ────────────────────────────

interface R {
  hue_cos1: number; hue_sin1: number;
  hue_cos2: number; hue_sin2: number;
  hue_cos3: number; hue_sin3: number;
  hue_cos4: number; hue_sin4: number;
  cs_cos1: number; cs_sin1: number;
  cs_cos2: number; cs_sin2: number;
  cs_cos3: number; cs_sin3: number;
  cs_cos4: number; cs_sin4: number;
  lc1: number; lc2: number;
  hlc_cos1: number; hlc_sin1: number;
  hlc_cos2: number; hlc_sin2: number;
  hl_cos1: number; hl_sin1: number;
  hl_cos2: number; hl_sin2: number;
  cp_cos1: number; cp_sin1: number;
  cp_cos2: number; cp_sin2: number;
  lp_dark: number; lp_dark_hcos: number; lp_dark_hsin: number;
  L_corr_p1: number; L_corr_p2: number; L_corr_p3: number;
  Lh_cos1: number; Lh_sin1: number;
  hk_weight: number; hk_power: number; hk_hue_mod: number;
  hk_sin1: number; hk_cos2: number; hk_sin2: number;
}

function hueDelta(r: R, h: number): number {
  return r.hue_cos1 * cos(h) + r.hue_sin1 * sin(h) +
         r.hue_cos2 * cos(2 * h) + r.hue_sin2 * sin(2 * h) +
         r.hue_cos3 * cos(3 * h) + r.hue_sin3 * sin(3 * h) +
         r.hue_cos4 * cos(4 * h) + r.hue_sin4 * sin(4 * h);
}

function hueDeltaDeriv(r: R, h: number): number {
  return -r.hue_cos1 * sin(h) + r.hue_sin1 * cos(h) +
         -2 * r.hue_cos2 * sin(2 * h) + 2 * r.hue_sin2 * cos(2 * h) +
         -3 * r.hue_cos3 * sin(3 * h) + 3 * r.hue_sin3 * cos(3 * h) +
         -4 * r.hue_cos4 * sin(4 * h) + 4 * r.hue_sin4 * cos(4 * h);
}

function chromaScale(r: R, h: number): number {
  const logS = r.cs_cos1 * cos(h) + r.cs_sin1 * sin(h) +
               r.cs_cos2 * cos(2 * h) + r.cs_sin2 * sin(2 * h) +
               r.cs_cos3 * cos(3 * h) + r.cs_sin3 * sin(3 * h) +
               r.cs_cos4 * cos(4 * h) + r.cs_sin4 * sin(4 * h);
  return exp(logS);
}

function lChromaScale(r: R, L: number): number {
  const dL = L - 0.5;
  const arg = r.lc1 * dL + r.lc2 * dL * dL;
  return exp(clamp(arg, -30, 30));
}

function hlcScale(r: R, h: number, L: number): number {
  const hueFactor = r.hlc_cos1 * cos(h) + r.hlc_sin1 * sin(h) +
                    r.hlc_cos2 * cos(2 * h) + r.hlc_sin2 * sin(2 * h);
  const arg = (L - 0.5) * hueFactor;
  return exp(clamp(arg, -30, 30));
}

function hueLightnessScale(r: R, h: number): number {
  const logS = r.hl_cos1 * cos(h) + r.hl_sin1 * sin(h) +
               r.hl_cos2 * cos(2 * h) + r.hl_sin2 * sin(2 * h);
  return exp(logS);
}

function chromaPower(r: R, h: number): number {
  return 1 + r.cp_cos1 * cos(h) + r.cp_sin1 * sin(h) +
         r.cp_cos2 * cos(2 * h) + r.cp_sin2 * sin(2 * h);
}

function lCorrect(r: R, L: number, h: number): number {
  const t = L * (1 - L);
  let result = L + r.L_corr_p1 * t + r.L_corr_p2 * t * (0.5 - L) + r.L_corr_p3 * t * t;
  if (r.Lh_cos1 !== 0 || r.Lh_sin1 !== 0) {
    result += t * (r.Lh_cos1 * cos(h) + r.Lh_sin1 * sin(h));
  }
  return result;
}

function lCorrectInv(r: R, L1: number, h: number): number {
  const Lh = (r.Lh_cos1 !== 0 || r.Lh_sin1 !== 0)
    ? r.Lh_cos1 * cos(h) + r.Lh_sin1 * sin(h) : 0;
  let L = L1;
  for (let i = 0; i < 15; i++) {
    const t = L * (1 - L);
    const dt = 1 - 2 * L;
    const f = L + (r.L_corr_p1 + Lh) * t + r.L_corr_p2 * t * (0.5 - L) +
              r.L_corr_p3 * t * t - L1;
    let dfdL = 1 + (r.L_corr_p1 + Lh) * dt +
               r.L_corr_p2 * (dt * (0.5 - L) - t) +
               r.L_corr_p3 * 2 * t * dt;
    if (abs(dfdL) < 1e-10) dfdL = 1;
    L -= f / dfdL;
  }
  return L;
}

function darkLCompress(r: R, L: number, h: number): number {
  let coeff = r.lp_dark;
  if (r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) {
    coeff += r.lp_dark_hcos * cos(h) + r.lp_dark_hsin * sin(h);
  }
  const g = coeff * L * (1 - L) ** 2;
  return L * exp(clamp(g, -30, 30));
}

function darkLCompressInv(r: R, Ln: number, h: number): number {
  let coeff = r.lp_dark;
  if (r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) {
    coeff += r.lp_dark_hcos * cos(h) + r.lp_dark_hsin * sin(h);
  }
  let L = Ln;
  for (let i = 0; i < 12; i++) {
    const oml = 1 - L;
    const g = coeff * L * oml * oml;
    const eg = exp(clamp(g, -30, 30));
    const f = L * eg - Ln;
    const gp = coeff * oml * (1 - 3 * L);
    let fp = eg * (1 + L * gp);
    if (abs(fp) < 1e-10) fp = 1;
    L -= f / fp;
  }
  return L;
}
