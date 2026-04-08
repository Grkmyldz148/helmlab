/**
 * Inline color space converters for OKLab and CIE Lab comparisons.
 * No external dependencies — used by playground React components.
 */

const { pow, cbrt, max, min, round } = Math;

// ── Hex / sRGB conversions ─────────────────────────────────────

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

export function rgbToHex(r: number, g: number, b: number): string {
  const clamp = (v: number) => max(0, min(255, round(v * 255)));
  return '#' + [clamp(r), clamp(g), clamp(b)]
    .map(v => v.toString(16).padStart(2, '0'))
    .join('');
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

function linearToSrgb(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055;
}

// ── OKLab ──────────────────────────────────────────────────────

export function srgbToOklab(r: number, g: number, b: number): [number, number, number] {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);

  const l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
  const m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
  const s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

  const l_ = cbrt(l);
  const m_ = cbrt(m);
  const s_ = cbrt(s);

  return [
    0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
  ];
}

export function oklabToSrgb(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;

  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;

  const r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

  return [
    max(0, min(1, linearToSrgb(r))),
    max(0, min(1, linearToSrgb(g))),
    max(0, min(1, linearToSrgb(bl))),
  ];
}

// ── CIE Lab ────────────────────────────────────────────────────

export function srgbToCieLab(r: number, g: number, b: number): [number, number, number] {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);

  const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047;
  const y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb;
  const z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883;

  const f = (t: number) => t > 0.008856 ? cbrt(t) : 7.787 * t + 16 / 116;
  const fx = f(x), fy = f(y), fz = f(z);

  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

export function cieLabToSrgb(L: number, a: number, b: number): [number, number, number] {
  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - b / 200;

  const finv = (t: number) => t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787;

  const x = finv(fx) * 0.95047;
  const y = finv(fy);
  const z = finv(fz) * 1.08883;

  const r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
  const g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
  const bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

  return [
    max(0, min(1, linearToSrgb(r))),
    max(0, min(1, linearToSrgb(g))),
    max(0, min(1, linearToSrgb(bl))),
  ];
}

// ── XYZ conversions (for ColorPicker) ──────────────────────────

export function srgbToXyz(r: number, g: number, b: number): [number, number, number] {
  const lr = srgbToLinear(r);
  const lg = srgbToLinear(g);
  const lb = srgbToLinear(b);
  return [
    0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb,
    0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb,
    0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb,
  ];
}

// ── Interpolation helpers ──────────────────────────────────────

export function interpolateOklab(
  hex1: string, hex2: string, steps: number
): string[] {
  const [r1, g1, b1] = hexToRgb(hex1);
  const [r2, g2, b2] = hexToRgb(hex2);
  const lab1 = srgbToOklab(r1, g1, b1);
  const lab2 = srgbToOklab(r2, g2, b2);

  const result: string[] = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0 : i / (steps - 1);
    const L = lab1[0] + t * (lab2[0] - lab1[0]);
    const a = lab1[1] + t * (lab2[1] - lab1[1]);
    const b = lab1[2] + t * (lab2[2] - lab1[2]);
    const [r, g, bl] = oklabToSrgb(L, a, b);
    result.push(rgbToHex(r, g, bl));
  }
  return result;
}

export function interpolateCieLab(
  hex1: string, hex2: string, steps: number
): string[] {
  const [r1, g1, b1] = hexToRgb(hex1);
  const [r2, g2, b2] = hexToRgb(hex2);
  const lab1 = srgbToCieLab(r1, g1, b1);
  const lab2 = srgbToCieLab(r2, g2, b2);

  const result: string[] = [];
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0 : i / (steps - 1);
    const L = lab1[0] + t * (lab2[0] - lab1[0]);
    const a = lab1[1] + t * (lab2[1] - lab1[1]);
    const b = lab1[2] + t * (lab2[2] - lab1[2]);
    const [r, g, bl] = cieLabToSrgb(L, a, b);
    result.push(rgbToHex(r, g, bl));
  }
  return result;
}

/** Generate OKLab lightness palette (like Tailwind scale). */
export function oklabPalette(hex: string, steps: number): string[] {
  const [r, g, b] = hexToRgb(hex);
  const [, oa, ob] = srgbToOklab(r, g, b);
  const Lhi = 0.97;
  const Llo = 0.15;
  const C = Math.sqrt(oa * oa + ob * ob);
  const aDir = C > 0 ? oa / C : 0;
  const bDir = C > 0 ? ob / C : 0;
  const result: string[] = [];
  for (let i = 0; i < steps; i++) {
    const L = Lhi - (i / (steps - 1)) * (Lhi - Llo);
    // Binary search max in-gamut chroma at this L
    let lo = 0, hi = C * 1.5;
    for (let j = 0; j < 20; j++) {
      const mid = (lo + hi) / 2;
      const [tr, tg, tb] = oklabToSrgbRaw(L, aDir * mid, bDir * mid);
      if (tr >= -0.002 && tr <= 1.002 && tg >= -0.002 && tg <= 1.002 && tb >= -0.002 && tb <= 1.002) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const cUse = Math.min(C, lo);
    const [sr, sg, sb] = oklabToSrgb(L, aDir * cUse, bDir * cUse);
    result.push(rgbToHex(sr, sg, sb));
  }
  return result;
}

/** OKLab to linear sRGB (no clamping) for gamut checking. */
function oklabToSrgbRaw(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  return [
    +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
}

/** Relative luminance from sRGB [0,1]. */
export function relativeLuminance(r: number, g: number, b: number): number {
  return 0.2126 * srgbToLinear(r) + 0.7152 * srgbToLinear(g) + 0.0722 * srgbToLinear(b);
}

/** Check if an sRGB triplet is in gamut. */
export function isInSrgb(r: number, g: number, b: number): boolean {
  return r >= -0.001 && r <= 1.001 && g >= -0.001 && g <= 1.001 && b >= -0.001 && b <= 1.001;
}

/**
 * Gradient perceptual uniformity CV (Coefficient of Variation).
 * Lower = more uniform steps. Uses CIEDE2000-simplified step distances.
 * Input: array of hex color strings.
 */
export function gradientCV(colors: string[]): number {
  if (colors.length < 3) return 0;
  // Compute CIE Lab for each color, then step deltaE
  const labs = colors.map((hex) => {
    const [r, g, b] = hexToRgb(hex);
    return srgbToCieLab(r, g, b);
  });
  const steps: number[] = [];
  for (let i = 1; i < labs.length; i++) {
    const [L1, a1, b1] = labs[i - 1];
    const [L2, a2, b2] = labs[i];
    const dL = L2 - L1;
    const C1 = Math.sqrt(a1 * a1 + b1 * b1);
    const C2 = Math.sqrt(a2 * a2 + b2 * b2);
    const dC = C2 - C1;
    const dH = Math.sqrt(max(0, (a2 - a1) ** 2 + (b2 - b1) ** 2 - dC ** 2));
    const Lb = (L1 + L2) / 2;
    const Cb = (C1 + C2) / 2;
    const SL = 1 + 0.015 * (Lb - 50) ** 2 / Math.sqrt(20 + (Lb - 50) ** 2);
    const SC = 1 + 0.045 * Cb;
    const SH = 1 + 0.015 * Cb;
    steps.push(Math.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2));
  }
  const mean = steps.reduce((a, b) => a + b, 0) / steps.length;
  if (mean < 1e-10) return 0;
  const variance = steps.reduce((a, s) => a + (s - mean) ** 2, 0) / steps.length;
  return (Math.sqrt(variance) / mean) * 100; // percentage
}
