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

/** Gamut-map Lab coordinates. Handles single or batch. */
export function gamutMap(lab: Lab, space: SpaceLike, gamut: Gamut = 'srgb'): Lab {
  return gamutMapSingle(lab, space, gamut);
}
