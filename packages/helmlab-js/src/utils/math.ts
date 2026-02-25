/** Minimal math utilities: 3x3 matrix ops, Fourier, signedPow. */

/** 3x3 matrix as flat row-major Float64Array (9 elements). */
export type Mat3 = Float64Array;

/** Create Mat3 from nested array [[a,b,c],[d,e,f],[g,h,i]]. */
export function mat3(rows: number[][]): Mat3 {
  const m = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      m[r * 3 + c] = rows[r][c];
  return m;
}

/** Multiply Mat3 × vec3 (row-major: result[i] = sum_j M[i,j]*v[j]). */
export function mat3MulVec(M: Mat3, v: Float64Array): Float64Array {
  const out = new Float64Array(3);
  out[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
  out[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
  out[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
  return out;
}

/** Transpose-multiply: result[i] = sum_j M[j,i]*v[j] (== M^T × v). */
export function mat3TMulVec(M: Mat3, v: Float64Array): Float64Array {
  const out = new Float64Array(3);
  out[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
  out[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
  out[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
  return out;
}

/** Invert a 3x3 matrix (Cramer's rule). Throws if singular. */
export function mat3Inv(M: Mat3): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = M;
  const A = e * i - f * h;
  const B = f * g - d * i;
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-15) throw new Error('Singular matrix');
  const inv = 1 / det;
  return new Float64Array([
    A * inv, (c * h - b * i) * inv, (b * f - c * e) * inv,
    B * inv, (a * i - c * g) * inv, (c * d - a * f) * inv,
    C * inv, (b * g - a * h) * inv, (a * e - b * d) * inv,
  ]);
}

/** sign(x) * |x|^p. Handles negative values correctly. */
export function signedPow(x: number, p: number): number {
  return x >= 0 ? x ** p : -((-x) ** p);
}

/** Evaluate Fourier series: sum(cos_k * cos(k*h) + sin_k * sin(k*h)). */
export function fourier(h: number, coeffs: readonly number[]): number {
  let sum = 0;
  for (let k = 0; k < coeffs.length; k += 2) {
    const harmonic = (k / 2) + 1;
    sum += coeffs[k] * Math.cos(harmonic * h) + coeffs[k + 1] * Math.sin(harmonic * h);
  }
  return sum;
}

/** Clamp value to [lo, hi]. */
export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
