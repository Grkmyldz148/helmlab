/** Pre-computed neutral-correction LUT with linear interpolation.
 *
 * The LUT stores achromatic error (a_err, b_err) at each L level.
 * Forward: a -= a_err(L), b -= b_err(L)
 * Inverse: a += a_err(L), b += b_err(L)
 */

import lutData from '../data/neutral-lut.json';

const L_LUT = new Float64Array(lutData.L);
const A_LUT = new Float64Array(lutData.a_err);
const B_LUT = new Float64Array(lutData.b_err);
const N = L_LUT.length;

/** Binary search: find index i such that L_LUT[i] <= L < L_LUT[i+1]. */
function findIndex(L: number): number {
  if (L <= L_LUT[0]) return 0;
  if (L >= L_LUT[N - 1]) return N - 2;
  let lo = 0, hi = N - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (L_LUT[mid] <= L) lo = mid; else hi = mid;
  }
  return lo;
}

/** Get achromatic error (a_err, b_err) at given L via linear interpolation. */
export function neutralError(L: number): [number, number] {
  if (L <= 0) return [0, 0];
  if (L < L_LUT[0]) {
    const t = L / L_LUT[0];
    return [A_LUT[0] * t, B_LUT[0] * t];
  }
  const i = findIndex(L);
  if (i >= N - 1) return [A_LUT[N - 1], B_LUT[N - 1]];
  const t = (L - L_LUT[i]) / (L_LUT[i + 1] - L_LUT[i]);
  return [
    A_LUT[i] + t * (A_LUT[i + 1] - A_LUT[i]),
    B_LUT[i] + t * (B_LUT[i + 1] - B_LUT[i]),
  ];
}
