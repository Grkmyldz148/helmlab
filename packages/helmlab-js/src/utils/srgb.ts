/** sRGB/Display P3/Rec2020 ↔ XYZ conversions, hex, gamma transfer. */

import type { XYZ, RGB, Hex } from '../types.js';

// ── IEC 61966-2-1 matrices (D65) ────────────────────────────────

/** XYZ → linear sRGB (row-major). */
export const M_XYZ_TO_SRGB = new Float64Array([
   3.2404541621141054, -1.5371385940306089, -0.4985314095560162,
  -0.9692660305051868,  1.8760108454466942,  0.0415560175303498,
   0.0556434309591147, -0.2040259135167538,  1.0572251882231791,
]);

/** Linear sRGB → XYZ (row-major). */
export const M_SRGB_TO_XYZ = new Float64Array([
  0.4124564390896922, 0.3575760776439511, 0.1804374832663989,
  0.2126728514056226, 0.7151521552878178, 0.0721750036064596,
  0.0193338955823293, 0.1191920258813418, 0.9503040785363679,
]);

/** XYZ → linear Display P3 (row-major). */
export const M_XYZ_TO_DISPLAYP3 = new Float64Array([
   2.4934969119, -0.9313836179, -0.4027107845,
  -0.8294889696,  1.7626640603,  0.0236246858,
   0.0358458302, -0.0761723893,  0.9568845240,
]);

/** Linear Display P3 → XYZ (row-major). */
export const M_DISPLAYP3_TO_XYZ = new Float64Array([
  0.4865709486, 0.2656676932, 0.1982172852,
  0.2289745641, 0.6917385218, 0.0792869141,
  0.0000000000, 0.0451133819, 1.0439443689,
]);

/** XYZ → linear Rec2020 / BT.2020 (row-major). */
export const M_XYZ_TO_REC2020 = new Float64Array([
   1.7166511880, -0.3556707838, -0.2533662814,
  -0.6666843518,  1.6164812366,  0.0157685458,
   0.0176398574, -0.0427706133,  0.9421031212,
]);

/** Linear Rec2020 / BT.2020 → XYZ (row-major). */
export const M_REC2020_TO_XYZ = new Float64Array([
  0.6369580483, 0.1446169036, 0.1688809752,
  0.2627002120, 0.6779980715, 0.0593017165,
  0.0000000000, 0.0280726930, 1.0609850577,
]);

// ── Gamma transfer ──────────────────────────────────────────────

/** Linear [0,1] → sRGB gamma-encoded [0,1]. */
export function linearToSrgb(c: number): number {
  return c <= 0.0031308
    ? 12.92 * c
    : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055;
}

/** sRGB gamma-encoded [0,1] → linear [0,1]. */
export function srgbToLinear(c: number): number {
  return c <= 0.04045
    ? c / 12.92
    : Math.pow((c + 0.055) / 1.055, 2.4);
}

// BT.2020 OETF constants (12-bit precision per ITU-R BT.2020-2 / CSS Color 4).
const BT2020_ALPHA = 1.09929682680944;
const BT2020_BETA = 0.018053968510807;
const BT2020_THRESH_ENC = 4.5 * BT2020_BETA; // ≈ 0.0812428

/** Linear [0,1] → Rec2020 / BT.2020 gamma-encoded [0,1]. */
export function linearToRec2020(c: number): number {
  return c < BT2020_BETA
    ? 4.5 * c
    : BT2020_ALPHA * Math.pow(Math.max(c, 0), 0.45) - (BT2020_ALPHA - 1);
}

/** Rec2020 / BT.2020 gamma-encoded [0,1] → linear [0,1]. */
export function rec2020ToLinear(c: number): number {
  return c < BT2020_THRESH_ENC
    ? c / 4.5
    : Math.pow((c + (BT2020_ALPHA - 1)) / BT2020_ALPHA, 1 / 0.45);
}

// ── XYZ ↔ sRGB ──────────────────────────────────────────────────

/** Mat3 (flat row-major) × vec3. */
function m3v(M: Float64Array, x: number, y: number, z: number): [number, number, number] {
  return [
    M[0] * x + M[1] * y + M[2] * z,
    M[3] * x + M[4] * y + M[5] * z,
    M[6] * x + M[7] * y + M[8] * z,
  ];
}

/** CIE XYZ (D65) → sRGB [0,1] (gamma-encoded, unclamped). */
export function xyzToSrgb(xyz: XYZ): RGB {
  const [lr, lg, lb] = m3v(M_XYZ_TO_SRGB, xyz[0], xyz[1], xyz[2]);
  return [linearToSrgb(lr), linearToSrgb(lg), linearToSrgb(lb)];
}

/** sRGB [0,1] → CIE XYZ (D65). */
export function srgbToXyz(rgb: RGB): XYZ {
  const lr = srgbToLinear(rgb[0]);
  const lg = srgbToLinear(rgb[1]);
  const lb = srgbToLinear(rgb[2]);
  return m3v(M_SRGB_TO_XYZ, lr, lg, lb) as XYZ;
}

/** CIE XYZ → Display P3 gamma-encoded [0,1]. */
export function xyzToDisplayP3(xyz: XYZ): RGB {
  const [lr, lg, lb] = m3v(M_XYZ_TO_DISPLAYP3, xyz[0], xyz[1], xyz[2]);
  return [linearToSrgb(lr), linearToSrgb(lg), linearToSrgb(lb)];
}

/** Display P3 gamma-encoded → CIE XYZ. */
export function displayP3ToXyz(rgb: RGB): XYZ {
  const lr = srgbToLinear(rgb[0]);
  const lg = srgbToLinear(rgb[1]);
  const lb = srgbToLinear(rgb[2]);
  return m3v(M_DISPLAYP3_TO_XYZ, lr, lg, lb) as XYZ;
}

/** CIE XYZ → Rec2020 / BT.2020 gamma-encoded [0,1]. */
export function xyzToRec2020(xyz: XYZ): RGB {
  const [lr, lg, lb] = m3v(M_XYZ_TO_REC2020, xyz[0], xyz[1], xyz[2]);
  return [linearToRec2020(lr), linearToRec2020(lg), linearToRec2020(lb)];
}

/** Rec2020 / BT.2020 gamma-encoded → CIE XYZ. */
export function rec2020ToXyz(rgb: RGB): XYZ {
  const lr = rec2020ToLinear(rgb[0]);
  const lg = rec2020ToLinear(rgb[1]);
  const lb = rec2020ToLinear(rgb[2]);
  return m3v(M_REC2020_TO_XYZ, lr, lg, lb) as XYZ;
}

// ── Hex ↔ sRGB ──────────────────────────────────────────────────

/** '#rrggbb' → [R, G, B] in [0, 1]. */
export function hexToSrgb(hex: Hex): RGB {
  let h = hex.startsWith('#') ? hex.slice(1) : hex;
  if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
  if (h.length !== 6) throw new Error(`Expected 3 or 6-char hex, got '${hex}'`);
  // Guard invalid hex characters: parseInt() silently yields NaN otherwise,
  // which then propagates through the whole pipeline (e.g. 'red', '#xyzxyz').
  if (!/^[0-9a-fA-F]{6}$/.test(h)) throw new Error(`Invalid hex color: '${hex}'`);
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

/** [R, G, B] in [0, 1] → '#rrggbb'. */
export function srgbToHex(rgb: RGB): Hex {
  const r = Math.round(Math.min(Math.max(rgb[0] * 255, 0), 255));
  const g = Math.round(Math.min(Math.max(rgb[1] * 255, 0), 255));
  const b = Math.round(Math.min(Math.max(rgb[2] * 255, 0), 255));
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

/** Clamp RGB to [0, 1]. */
export function clampRgb(rgb: RGB): RGB {
  return [
    Math.min(Math.max(rgb[0], 0), 1),
    Math.min(Math.max(rgb[1], 0), 1),
    Math.min(Math.max(rgb[2], 0), 1),
  ];
}

/** WCAG relative luminance from sRGB [0,1]. */
export function relativeLuminance(rgb: RGB): number {
  return (
    0.2126 * srgbToLinear(rgb[0]) +
    0.7152 * srgbToLinear(rgb[1]) +
    0.0722 * srgbToLinear(rgb[2])
  );
}
