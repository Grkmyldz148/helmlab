/**
 * GenSpace — generation-optimized color space for palette, gradient, gamut map.
 *
 * Pipeline (subset, ~35 params):
 *   XYZ → M1 → γ^(1/3) shared → M2 → Lab_raw
 *   → [hue correction] → [cubic L] → [dark L] → [L-dep chroma] → NC → Lab
 *
 * Key differences from MetricSpace:
 *   - Shared gamma (1/3) guarantees grays → a=b=0
 *   - No H-K, chroma power, HLC, hue-lightness, hue-dep chroma scaling
 */

import type { Lab, XYZ } from '../types.js';
import { signedPow, clamp } from '../utils/math.js';
import type { CompiledGenParams, GenParams } from '../core/params.js';

const { cos, sin, sqrt, atan2, exp, abs, PI } = Math;

const D65_X = 0.95047, D65_Y = 1.0, D65_Z = 1.08883;

export interface GenSpaceOptions {
  neutralCorrection?: boolean;
}

export class GenSpace {
  private readonly p: CompiledGenParams;
  private nc: boolean;

  // NC LUT (lazy)
  private ncL: Float64Array | null = null;
  private ncA: Float64Array | null = null;
  private ncB: Float64Array | null = null;

  constructor(params: CompiledGenParams, opts: GenSpaceOptions = {}) {
    this.p = params;
    this.nc = opts.neutralCorrection ?? true;
  }

  // ── Forward transform ──────────────────────────────────────────

  fromXYZ(xyz: XYZ): Lab {
    const r = this.p.raw;
    const M1 = this.p.M1;
    const M2 = this.p.M2;
    const g = this.p.gamma;

    // 1. XYZ → LMS
    const lms0 = M1[0] * xyz[0] + M1[1] * xyz[1] + M1[2] * xyz[2];
    const lms1 = M1[3] * xyz[0] + M1[4] * xyz[1] + M1[5] * xyz[2];
    const lms2 = M1[6] * xyz[0] + M1[7] * xyz[1] + M1[8] * xyz[2];

    // 2. Shared power compression
    const c0 = signedPow(lms0, g[0]);
    const c1 = signedPow(lms1, g[1]);
    const c2 = signedPow(lms2, g[2]);

    // 3. LMS_c → Lab_raw
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

    // 4. Cubic L correction
    if (r.L_corr_p1 !== 0 || r.L_corr_p2 !== 0 || r.L_corr_p3 !== 0) {
      L = lCorrect(r, L);
    }

    // 4.5 Dark L compression
    if (r.lp_dark !== 0 || r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) {
      const h = (r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) ? atan2(b, a) : 0;
      L = darkLCompress(r, L, h);
    }

    // 6. L-dependent chroma scaling
    if (r.lc1 !== 0 || r.lc2 !== 0) {
      const T = lChromaScale(r, L);
      a *= T;
      b *= T;
    }

    // 10. Neutral correction
    if (this.nc) {
      const [aErr, bErr] = this.neutralError(L);
      a -= aErr;
      b -= bErr;
    }

    return [L, a, b];
  }

  // ── Inverse transform ──────────────────────────────────────────

  toXYZ(lab: Lab): XYZ {
    const r = this.p.raw;
    let [L, a, b] = lab;

    // 10. Undo NC
    if (this.nc) {
      const [aErr, bErr] = this.neutralError(L);
      a += aErr;
      b += bErr;
    }

    // 6. Undo L-dep chroma
    if (r.lc1 !== 0 || r.lc2 !== 0) {
      const T = lChromaScale(r, L);
      a /= T;
      b /= T;
    }

    // 4.5 Undo dark L
    if (r.lp_dark !== 0 || r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) {
      const h = (r.lp_dark_hcos !== 0 || r.lp_dark_hsin !== 0) ? atan2(b, a) : 0;
      L = darkLCompressInv(r, L, h);
    }

    // 4. Undo cubic L
    if (r.L_corr_p1 !== 0 || r.L_corr_p2 !== 0 || r.L_corr_p3 !== 0) {
      L = lCorrectInv(r, L);
    }

    // 3.5 Undo hue correction
    if (r.hue_cos1 !== 0 || r.hue_sin1 !== 0 || r.hue_cos2 !== 0 ||
        r.hue_sin2 !== 0 || r.hue_cos3 !== 0 || r.hue_sin3 !== 0 ||
        r.hue_cos4 !== 0 || r.hue_sin4 !== 0) {
      const hOut = atan2(b, a);
      const C = sqrt(a * a + b * b);
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

    // 3. Lab → LMS_c
    const M2i = this.p.M2_inv;
    const lc0 = M2i[0] * L + M2i[1] * a + M2i[2] * b;
    const lc1 = M2i[3] * L + M2i[4] * a + M2i[5] * b;
    const lc2 = M2i[6] * L + M2i[7] * a + M2i[8] * b;

    // 2. Undo power
    const ig = this.p.inv_gamma;
    const l0 = signedPow(lc0, ig[0]);
    const l1 = signedPow(lc1, ig[1]);
    const l2 = signedPow(lc2, ig[2]);

    // 1. LMS → XYZ
    const M1i = this.p.M1_inv;
    return [
      M1i[0] * l0 + M1i[1] * l1 + M1i[2] * l2,
      M1i[3] * l0 + M1i[4] * l1 + M1i[5] * l2,
      M1i[6] * l0 + M1i[7] * l1 + M1i[8] * l2,
    ];
  }

  // ── NC LUT ──────────────────────────────────────────────────────

  private buildNcLut(): void {
    const N = 256;
    const Ls = new Float64Array(N);
    const As = new Float64Array(N);
    const Bs = new Float64Array(N);

    // Temporarily disable NC to measure pipeline achromatic error
    const oldNc = this.nc;
    this.nc = false;

    for (let i = 0; i < N; i++) {
      const Y = i / (N - 1);
      const lab = this.fromXYZ([Y * D65_X, Y * D65_Y, Y * D65_Z]);
      Ls[i] = lab[0];
      As[i] = lab[1];
      Bs[i] = lab[2];
    }

    this.nc = oldNc;

    this.ncL = Ls;
    this.ncA = As;
    this.ncB = Bs;
  }

  private neutralError(L: number): [number, number] {
    if (!this.ncL) this.buildNcLut();
    const Ls = this.ncL!;
    const As = this.ncA!;
    const Bs = this.ncB!;
    const N = Ls.length;
    if (L <= 0) return [0, 0];
    if (L < Ls[0]) {
      const t = L / Ls[0];
      return [As[0] * t, Bs[0] * t];
    }
    if (L >= Ls[N - 1]) return [As[N - 1], Bs[N - 1]];
    let lo = 0, hi = N - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (Ls[mid] <= L) lo = mid; else hi = mid;
    }
    const t = (L - Ls[lo]) / (Ls[lo + 1] - Ls[lo]);
    return [
      As[lo] + t * (As[lo + 1] - As[lo]),
      Bs[lo] + t * (Bs[lo + 1] - Bs[lo]),
    ];
  }
}

// ── Helper functions (module-private) ────────────────────────────

interface R {
  hue_cos1: number; hue_sin1: number;
  hue_cos2: number; hue_sin2: number;
  hue_cos3: number; hue_sin3: number;
  hue_cos4: number; hue_sin4: number;
  L_corr_p1: number; L_corr_p2: number; L_corr_p3: number;
  lp_dark: number; lp_dark_hcos: number; lp_dark_hsin: number;
  lc1: number; lc2: number;
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

function lCorrect(r: R, L: number): number {
  const t = L * (1 - L);
  return L + r.L_corr_p1 * t + r.L_corr_p2 * t * (0.5 - L) + r.L_corr_p3 * t * t;
}

function lCorrectInv(r: R, L1: number): number {
  let L = L1;
  for (let i = 0; i < 15; i++) {
    const t = L * (1 - L);
    const dt = 1 - 2 * L;
    const f = L + r.L_corr_p1 * t + r.L_corr_p2 * t * (0.5 - L) +
              r.L_corr_p3 * t * t - L1;
    let dfdL = 1 + r.L_corr_p1 * dt +
               r.L_corr_p2 * (dt * (0.5 - L) - t) +
               r.L_corr_p3 * 2 * t * dt;
    if (abs(dfdL) < 1e-10) dfdL = 1;
    L -= f / dfdL;
  }
  return L;
}

function lChromaScale(r: R, L: number): number {
  const dL = L - 0.5;
  const arg = r.lc1 * dL + r.lc2 * dL * dL;
  return exp(clamp(arg, -30, 30));
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
