/** Pre-computed neutral-correction LUT with PCHIP interpolation.
 *
 * The LUT stores achromatic error (a_err, b_err) at each L level.
 * Forward: a -= a_err(L), b -= b_err(L)
 * Inverse: a += a_err(L), b += b_err(L)
 *
 * Interpolation is monotone cubic (PCHIP, Fritsch–Carlson derivatives),
 * implemented to match scipy.interpolate.PchipInterpolator exactly — the
 * Python sibling uses that interpolator, and linear interpolation here left
 * a systematic ~1e-5 gray-axis gap between the two languages. Out-of-range
 * L is clamped to the LUT boundaries (Python clamps the same way).
 */

import lutData from '../data/neutral-lut.json';

const L_LUT = new Float64Array(lutData.L);
const A_LUT = new Float64Array(lutData.a_err);
const B_LUT = new Float64Array(lutData.b_err);
const N = L_LUT.length;

/** scipy PchipInterpolator edge derivative (one-sided three-point estimate). */
function edgeCase(h0: number, h1: number, m0: number, m1: number): number {
  let d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
  if (Math.sign(d) !== Math.sign(m0)) {
    d = 0;
  } else if (Math.sign(m0) !== Math.sign(m1) && Math.abs(d) > 3 * Math.abs(m0)) {
    d = 3 * m0;
  }
  return d;
}

/** Fritsch–Carlson monotone derivatives, matching scipy's PCHIP exactly. */
function pchipDerivatives(x: Float64Array, y: Float64Array): Float64Array {
  const n = x.length;
  const h = new Float64Array(n - 1);
  const m = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) {
    h[i] = x[i + 1] - x[i];
    m[i] = (y[i + 1] - y[i]) / h[i];
  }
  const d = new Float64Array(n);
  if (n === 2) {
    d[0] = m[0];
    d[1] = m[0];
    return d;
  }
  for (let i = 1; i < n - 1; i++) {
    const mk0 = m[i - 1], mk1 = m[i];
    if (Math.sign(mk0) !== Math.sign(mk1) || mk0 === 0 || mk1 === 0) {
      d[i] = 0;
    } else {
      // weighted harmonic mean (scipy: 1/whmean = (w1/mk0 + w2/mk1)/(w1+w2))
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      d[i] = (w1 + w2) / (w1 / mk0 + w2 / mk1);
    }
  }
  d[0] = edgeCase(h[0], h[1], m[0], m[1]);
  d[n - 1] = edgeCase(h[n - 2], h[n - 3], m[n - 2], m[n - 3]);
  return d;
}

const A_DERIV = pchipDerivatives(L_LUT, A_LUT);
const B_DERIV = pchipDerivatives(L_LUT, B_LUT);

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

/** Cubic Hermite evaluation on interval i. */
function hermite(L: number, i: number, y: Float64Array, d: Float64Array): number {
  const h = L_LUT[i + 1] - L_LUT[i];
  const t = (L - L_LUT[i]) / h;
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    y[i] * (2 * t3 - 3 * t2 + 1) +
    h * d[i] * (t3 - 2 * t2 + t) +
    y[i + 1] * (-2 * t3 + 3 * t2) +
    h * d[i + 1] * (t3 - t2)
  );
}

/** Get achromatic error (a_err, b_err) at given L via PCHIP interpolation. */
export function neutralError(L: number): [number, number] {
  // Clamp into LUT range — matches Python's np.clip before evaluating.
  if (L <= L_LUT[0]) return [A_LUT[0], B_LUT[0]];
  if (L >= L_LUT[N - 1]) return [A_LUT[N - 1], B_LUT[N - 1]];
  const i = findIndex(L);
  return [hermite(L, i, A_LUT, A_DERIV), hermite(L, i, B_LUT, B_DERIV)];
}
