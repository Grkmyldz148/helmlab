/** Gamut mapping — binary search chroma reduction preserving L and hue. */

import type { Lab, XYZ } from '../types.js';
import { M_XYZ_TO_SRGB, M_XYZ_TO_DISPLAYP3, M_XYZ_TO_REC2020 } from './srgb.js';

const { cos, sin, sqrt, atan2, min } = Math;

/** Minimal interface for gamut mapping — any object with toXYZ. */
export interface SpaceLike {
  toXYZ(lab: Lab): XYZ;
}

export type Gamut = 'srgb' | 'display-p3' | 'rec2020';

function getMatrix(gamut: Gamut): Float64Array {
  if (gamut === 'srgb') return M_XYZ_TO_SRGB;
  if (gamut === 'display-p3') return M_XYZ_TO_DISPLAYP3;
  return M_XYZ_TO_REC2020;
}

/** Check if XYZ is in the given RGB gamut (linear check, no gamma). */
function xyzInGamut(M: Float64Array, x: number, y: number, z: number, tol: number): boolean {
  const r = M[0] * x + M[1] * y + M[2] * z;
  const g = M[3] * x + M[4] * y + M[5] * z;
  const b = M[6] * x + M[7] * y + M[8] * z;
  return r >= -tol && r <= 1 + tol &&
         g >= -tol && g <= 1 + tol &&
         b >= -tol && b <= 1 + tol;
}

/** Check if Lab coordinates are in the specified gamut. */
export function isInGamut(lab: Lab, space: SpaceLike, gamut: Gamut = 'srgb', tol = 1e-4): boolean {
  const xyz = space.toXYZ(lab);
  return xyzInGamut(getMatrix(gamut), xyz[0], xyz[1], xyz[2], tol);
}

/** Binary search for maximum in-gamut chroma at fixed L and hue. */
export function maxChroma(L: number, hRad: number, space: SpaceLike, gamut: Gamut = 'srgb', tol = 1e-4): number {
  const cosH = cos(hRad);
  const sinH = sin(hRad);
  const M = getMatrix(gamut);

  let lo = 0, hi = 1;

  // Expand hi until out of gamut
  while (true) {
    const xyz = space.toXYZ([L, hi * cosH, hi * sinH]);
    if (!xyzInGamut(M, xyz[0], xyz[1], xyz[2], tol)) break;
    hi *= 2;
    if (hi > 100) return hi;
  }

  // Binary search
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) * 0.5;
    const xyz = space.toXYZ([L, mid * cosH, mid * sinH]);
    if (xyzInGamut(M, xyz[0], xyz[1], xyz[2], tol)) lo = mid; else hi = mid;
    if (hi - lo < tol) break;
  }

  return lo;
}

/** Find the cusp (L, C) at a given hue angle — maximum chroma over all L values. */
export function findCusp(hRad: number, space: SpaceLike, gamut: Gamut = 'srgb', nScan = 64, tol = 1e-4): [number, number] {
  const cosH = cos(hRad);
  const sinH = sin(hRad);
  let bestL = 0.5, bestC = 0;

  // Coarse scan
  for (let i = 0; i <= nScan; i++) {
    const L = i / nScan;
    const C = maxChroma(L, hRad, space, gamut, tol);
    if (C > bestC) { bestC = C; bestL = L; }
  }

  // Fine refine around bestL
  const step = 1 / nScan;
  let lo = Math.max(0, bestL - step), hi = Math.min(1, bestL + step);
  for (let i = 0; i < 20; i++) {
    const m1 = lo + (hi - lo) / 3;
    const m2 = hi - (hi - lo) / 3;
    const c1 = maxChroma(m1, hRad, space, gamut, tol);
    const c2 = maxChroma(m2, hRad, space, gamut, tol);
    if (c1 < c2) lo = m1; else hi = m2;
  }
  bestL = (lo + hi) / 2;
  bestC = maxChroma(bestL, hRad, space, gamut, tol);

  return [bestL, bestC];
}

/** Gamut-map a single Lab by reducing chroma (preserving L and hue). */
function gamutMapSingle(lab: Lab, space: SpaceLike, gamut: Gamut): Lab {
  if (isInGamut(lab, space, gamut)) return [...lab];

  const [L, a, b] = lab;
  const C = sqrt(a * a + b * b);
  const H = atan2(b, a);

  if (C < 1e-10) return [L, 0, 0];

  const Cmax = maxChroma(L, H, space, gamut);
  const Cn = min(C, Cmax);
  return [L, Cn * cos(H), Cn * sin(H)];
}

/** Adaptive gamut clipping (Ottosson-style) — mirrors the Python sibling's
 * `_gamut_clip_adaptive_single`. Instead of only reducing chroma, shifts
 * both L and C toward a cusp-derived projection target: important for
 * hues like yellow whose cusp sits at very high L, where pure chroma
 * reduction collapses to near-gray (the cliff problem). */
function gamutClipAdaptiveSingle(lab: Lab, space: SpaceLike, gamut: Gamut, alpha: number): Lab {
  if (isInGamut(lab, space, gamut)) return [...lab];

  const [L, a, b] = lab;
  const C = sqrt(a * a + b * b);
  const H = atan2(b, a);

  if (C < 1e-10) return [L, 0, 0];

  const [Lcusp] = findCusp(H, space, gamut);

  const Ld = L - Lcusp;
  let k = Ld >= 0 ? 2 * (1 - Lcusp) : 2 * Lcusp;
  k = Math.max(k, 1e-6);

  const e1 = k / 2 + Math.abs(Ld) + (alpha * C) / k;
  const discriminant = e1 * e1 - 2 * k * Math.abs(Ld);

  let L0: number;
  if (discriminant < 0) {
    L0 = Lcusp;
  } else {
    const sgn = Ld >= 0 ? 1 : -1;
    L0 = Lcusp + (sgn * (e1 - sqrt(discriminant))) / 2;
  }
  L0 = min(Math.max(L0, 0), 1);

  const cosH = cos(H);
  const sinH = sin(H);

  let loT = 0;
  let hiT = 1;
  for (let i = 0; i < 50; i++) {
    const t = (loT + hiT) * 0.5;
    const Ltest = L0 * (1 - t) + t * L;
    const Ctest = t * C;
    if (isInGamut([Ltest, Ctest * cosH, Ctest * sinH], space, gamut)) {
      loT = t;
    } else {
      hiT = t;
    }
    if (hiT - loT < 1e-6) break;
  }

  const t = loT;
  const Lclip = L0 * (1 - t) + t * L;
  const Cclip = t * C;
  return [Lclip, Cclip * cosH, Cclip * sinH];
}

export type GamutMapMethod = 'chroma' | 'adaptive';

/** Gamut-map Lab coordinates.
 * `method='chroma'` (default) reduces chroma at constant L and hue;
 * `method='adaptive'` is the Ottosson-style cusp projection. */
export function gamutMap(lab: Lab, space: SpaceLike, gamut: Gamut = 'srgb',
                         method: GamutMapMethod = 'chroma', alpha = 0.05): Lab {
  if (method === 'adaptive') return gamutClipAdaptiveSingle(lab, space, gamut, alpha);
  return gamutMapSingle(lab, space, gamut);
}
