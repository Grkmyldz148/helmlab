/** Helmlab — UI design system utility layer for perceptual color.
 *
 * Composes two purpose-built color spaces:
 *   MetricSpace — full enriched pipeline (distance, deltaE)
 *   GenSpace    — generation-optimized pipeline (palette, gradient, gamut map)
 */

import type { Lab, XYZ, RGB, Hex, SemanticScale, WCAGLevel } from './types.js';
import { AnalyticalSpace } from './core/analytical.js';
import { GenSpace } from './spaces/gen.js';
import {
  compileParams, getDefaultParams,
  compileGenParams, getDefaultGenParams,
  type HelmlabParams, type CompiledParams,
  type GenParams, type CompiledGenParams,
} from './core/params.js';
import { hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb, xyzToDisplayP3, xyzToRec2020, clampRgb, relativeLuminance } from './utils/srgb.js';
import { gamutMap, isInGamut, type SpaceLike } from './utils/gamut.js';
import { contrastRatio as wcagCR } from './utils/contrast.js';
import { TokenExporter } from './export.js';
import { assessConfidence, type Confidence } from './core/confidence.js';

const { sqrt, atan2, cos, sin, PI, pow, abs } = Math;

// ── CIE Lab + CIEDE2000 (for arc-length reparameterization) ──────
function srgbToCieLab(rgb: RGB): [number, number, number] {
  const lr = rgb[0] <= 0.04045 ? rgb[0] / 12.92 : pow((rgb[0] + 0.055) / 1.055, 2.4);
  const lg = rgb[1] <= 0.04045 ? rgb[1] / 12.92 : pow((rgb[1] + 0.055) / 1.055, 2.4);
  const lb = rgb[2] <= 0.04045 ? rgb[2] / 12.92 : pow((rgb[2] + 0.055) / 1.055, 2.4);
  const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047;
  const y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb;
  const z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883;
  const f = (t: number) => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  const fx = f(x), fy = f(y), fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function ciede2000(L1: number, a1: number, b1: number, L2: number, a2: number, b2: number): number {
  const C1 = sqrt(a1 * a1 + b1 * b1), C2 = sqrt(a2 * a2 + b2 * b2);
  const Cab = (C1 + C2) / 2, Cab7 = pow(Cab, 7), p257 = pow(25, 7);
  const G = 0.5 * (1 - sqrt(Cab7 / (Cab7 + p257)));
  const ap1 = (1 + G) * a1, ap2 = (1 + G) * a2;
  const Cp1 = sqrt(ap1 * ap1 + b1 * b1), Cp2 = sqrt(ap2 * ap2 + b2 * b2);
  let hp1 = atan2(b1, ap1), hp2 = atan2(b2, ap2);
  if (hp1 < 0) hp1 += 2 * PI; if (hp2 < 0) hp2 += 2 * PI;
  const dLp = L2 - L1, dCp = Cp2 - Cp1;
  let dhp = 0;
  if (Cp1 * Cp2 !== 0) { dhp = hp2 - hp1; if (dhp > PI) dhp -= 2 * PI; if (dhp < -PI) dhp += 2 * PI; }
  const dHp = 2 * sqrt(Cp1 * Cp2) * sin(dhp / 2);
  const Lp = (L1 + L2) / 2, Cp = (Cp1 + Cp2) / 2;
  let hp = 0;
  if (Cp1 * Cp2 === 0) hp = hp1 + hp2;
  else { hp = (hp1 + hp2) / 2; if (abs(hp1 - hp2) > PI) { hp += hp < PI ? PI : -PI; } }
  const T = 1 - 0.17 * cos(hp - PI / 6) + 0.24 * cos(2 * hp) + 0.32 * cos(3 * hp + PI / 30) - 0.20 * cos(4 * hp - 63 * PI / 180);
  const Lp50 = Lp - 50;
  const SL = 1 + 0.015 * Lp50 * Lp50 / sqrt(20 + Lp50 * Lp50);
  const SC = 1 + 0.045 * Cp, SH = 1 + 0.015 * Cp * T;
  const Cp7 = pow(Cp, 7), RC = 2 * sqrt(Cp7 / (Cp7 + p257));
  const hpDeg = hp * 180 / PI;
  const dth = 30 * Math.exp(-pow((hpDeg - 275) / 25, 2));
  const RT = -sin(2 * dth * PI / 180) * RC;
  return sqrt(pow(dLp / SL, 2) + pow(dCp / SC, 2) + pow(dHp / SH, 2) + RT * (dCp / SC) * (dHp / SH));
}

export interface HelmlabOptions {
  params?: HelmlabParams;
  genParams?: GenParams;
  neutralCorrection?: boolean;
  abRotateDeg?: number;
}

export class Helmlab {
  private readonly metric: AnalyticalSpace;
  private readonly gen: GenSpace;
  private readonly cp: CompiledParams;
  private readonly gcp: CompiledGenParams;
  private readonly genWhiteL: number;

  constructor(opts: HelmlabOptions = {}) {
    const p = compileParams(opts.params ?? getDefaultParams());
    this.cp = p;
    this.metric = new AnalyticalSpace(p, {
      neutralCorrection: opts.neutralCorrection ?? true,
      abRotateDeg: opts.abRotateDeg ?? -28.2,
    });

    const gp = compileGenParams(opts.genParams ?? getDefaultGenParams());
    this.gcp = gp;
    this.gen = new GenSpace(gp);

    // Cache white L for palette/scale range
    const D65: XYZ = [0.95047, 1.0, 1.08883];
    this.genWhiteL = this.gen.fromXYZ(D65)[0];
  }

  // ── Full-pipeline conversions (MetricSpace — public API) ────────

  /** Hex '#rrggbb' → Helmlab Lab [L, a, b] (metric pipeline). */
  fromHex(hex: Hex): Lab {
    return this.fromSrgb(hexToSrgb(hex));
  }

  /** Helmlab Lab → hex '#rrggbb' (metric pipeline, gamut mapped). */
  toHex(lab: Lab): Hex {
    return srgbToHex(this.toSrgb(lab));
  }

  /** sRGB [0,1] → Helmlab Lab (metric pipeline). */
  fromSrgb(rgb: RGB): Lab {
    return this.metric.fromXYZ(srgbToXyz(rgb));
  }

  /** Helmlab Lab → sRGB [0,1] (metric pipeline, gamut mapped, clamped). */
  toSrgb(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.metric, 'srgb');
    return clampRgb(xyzToSrgb(this.metric.toXYZ(mapped)));
  }

  /** CIE XYZ → Helmlab Lab (metric pipeline). */
  fromXYZ(xyz: XYZ): Lab {
    return this.metric.fromXYZ(xyz);
  }

  /** Helmlab Lab → CIE XYZ (metric pipeline). */
  toXYZ(lab: Lab): XYZ {
    return this.metric.toXYZ(lab);
  }

  /** Helmlab Lab → Display P3 [0,1] (gamut mapped). */
  toDisplayP3(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.metric, 'display-p3');
    return clampRgb(xyzToDisplayP3(this.metric.toXYZ(mapped)));
  }

  /** Helmlab Lab → CSS color(display-p3 r g b) string. */
  toHexP3(lab: Lab): string {
    const p3 = this.toDisplayP3(lab);
    return `color(display-p3 ${p3[0].toFixed(4)} ${p3[1].toFixed(4)} ${p3[2].toFixed(4)})`;
  }

  /** Helmlab Lab → Rec2020 / BT.2020 [0,1] (gamut mapped). */
  toRec2020(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.metric, 'rec2020');
    return clampRgb(xyzToRec2020(this.metric.toXYZ(mapped)));
  }

  /** Helmlab Lab → CSS color(rec2020 r g b) string. */
  toHexRec2020(lab: Lab): string {
    const r = this.toRec2020(lab);
    return `color(rec2020 ${r[0].toFixed(4)} ${r[1].toFixed(4)} ${r[2].toFixed(4)})`;
  }

  /** Check if Lab is within sRGB gamut. */
  isInSrgb(lab: Lab): boolean {
    return isInGamut(lab, this.metric, 'srgb');
  }

  /** Check if Lab is within Display P3 gamut. */
  isInP3(lab: Lab): boolean {
    return isInGamut(lab, this.metric, 'display-p3');
  }

  /** Check if Lab is within Rec2020 / BT.2020 gamut. */
  isInRec2020(lab: Lab): boolean {
    return isInGamut(lab, this.metric, 'rec2020');
  }

  // ── GenSpace conversions (for generation) ──────────────────────

  /** Hex → Gen Lab [L, a, b] (generation pipeline). */
  genFromHex(hex: Hex): Lab {
    return this.gen.fromXYZ(srgbToXyz(hexToSrgb(hex)));
  }

  /** Gen Lab → hex (generation pipeline, gamut mapped). */
  genToHex(lab: Lab): Hex {
    return srgbToHex(this._genToSrgb(lab));
  }

  /** sRGB [0,1] → Gen Lab [L, a, b] (generation pipeline). */
  genFromSrgb(rgb: RGB): Lab {
    return this.gen.fromXYZ(srgbToXyz(rgb));
  }

  /** Gen Lab → sRGB [0,1] (gamut mapped, clamped). */
  genToSrgb(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.gen, 'srgb');
    return clampRgb(xyzToSrgb(this.gen.toXYZ(mapped)));
  }

  /** @internal Alias kept for internal callers. */
  private _genToSrgb(lab: Lab): RGB {
    return this.genToSrgb(lab);
  }

  // ── Deprecated base* aliases → gen* ─────────────────────────────

  /** @deprecated Use genFromHex(). */
  baseFromHex(hex: Hex): Lab {
    return this.genFromHex(hex);
  }

  /** @deprecated Use genToHex(). */
  baseToHex(lab: Lab): Hex {
    return this.genToHex(lab);
  }

  /** @deprecated Use genFromSrgb(). */
  baseFromSrgb(rgb: RGB): Lab {
    return this.genFromSrgb(rgb);
  }

  /** @deprecated Use genToSrgb(). */
  baseToSrgb(lab: Lab): RGB {
    return this.genToSrgb(lab);
  }

  // ── Contrast ─────────────────────────────────────────────────

  /** WCAG contrast ratio between two hex colors (1.0–21.0). */
  contrastRatio(fg: Hex, bg: Hex): number {
    return wcagCR(hexToSrgb(fg), hexToSrgb(bg));
  }

  /** Adjust fg lightness to meet min contrast ratio against bg. Uses GenSpace. */
  ensureContrast(fg: Hex, bg: Hex, minRatio = 4.5): Hex {
    if (this.contrastRatio(fg, bg) >= minRatio) return fg;

    const fgLab = this.genFromHex(fg);
    const bgRgb = hexToSrgb(bg);
    const origL = fgLab[0];

    let bestHex = fg;
    let bestRatio = 0;
    let bestL = origL;

    for (const dir of ['darken', 'lighten'] as const) {
      let lo = dir === 'darken' ? 0 : origL;
      let hi = dir === 'darken' ? origL : 1.5;
      const candidate: Lab = [...fgLab];

      for (let i = 0; i < 40; i++) {
        const mid = (lo + hi) / 2;
        candidate[0] = mid;
        const srgb = this._genToSrgb(candidate);
        const hexQ = hexToSrgb(srgbToHex(srgb));
        const ratio = wcagCR(hexQ, bgRgb);

        if (dir === 'darken') {
          if (ratio >= minRatio) lo = mid; else hi = mid;
        } else {
          if (ratio >= minRatio) hi = mid; else lo = mid;
        }
      }

      let safeL = dir === 'darken' ? lo : hi;
      if (dir === 'darken') {
        safeL = Math.max(0, safeL - 0.003);
      } else {
        safeL = Math.min(1.5, safeL + 0.003);
      }
      candidate[0] = safeL;
      const candHex = srgbToHex(this._genToSrgb(candidate));
      const ratio = wcagCR(hexToSrgb(candHex), bgRgb);

      if (ratio >= minRatio) {
        const dist = Math.abs(safeL - origL);
        const bestDist = Math.abs(bestL - origL);
        if (bestRatio < minRatio || dist < bestDist) {
          bestHex = candHex;
          bestRatio = ratio;
          bestL = safeL;
        }
      }
    }

    if (bestRatio < minRatio) {
      for (const fallback of ['#000000', '#ffffff'] as const) {
        if (this.contrastRatio(fallback, bg) >= minRatio) return fallback;
      }
      return this.contrastRatio('#000000', bg) > this.contrastRatio('#ffffff', bg)
        ? '#000000' : '#ffffff';
    }

    return bestHex;
  }

  /** Check if fg/bg meets WCAG contrast level. */
  meetsContrast(fg: Hex, bg: Hex, level: WCAGLevel = 'AA'): boolean {
    const threshold = level === 'AAA' ? 7 : 4.5;
    return this.contrastRatio(fg, bg) >= threshold;
  }

  // ── Distance ─────────────────────────────────────────────────

  /**
   * Euclidean Lab distance between two hex colors (uncompressed, ΔE76-style).
   *
   * Returns `sqrt(dL² + da² + db²)` in Helmlab metric Lab. Range ≈ 0–1.6
   * across sRGB primaries (`#000000` ↔ `#ffffff` ≈ 1.12, `#ff0000` ↔ `#00ff00`
   * ≈ 1.62). For the compressed perceptual metric (COMBVD-fit), use
   * {@link perceptualDistance}.
   */
  deltaE(color1: Hex, color2: Hex): number {
    const lab1 = this.fromHex(color1);
    const lab2 = this.fromHex(color2);
    return sqrt((lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2);
  }

  /**
   * Compressed perceptual distance (Minkowski + compression) between two Lab values.
   *
   * Optimized against COMBVD-class human-judgment data (STRESS ≈ 22.5 vs
   * CIEDE2000's ~29.2). Saturates near ~0.15 for very dissimilar pairs —
   * relative order preserved, absolute magnitudes plateau. For uncompressed
   * Euclidean use {@link deltaE}.
   */
  perceptualDistance(lab1: Lab, lab2: Lab): number {
    return this.metric.distance(lab1, lab2);
  }

  /**
   * Alias of {@link perceptualDistance} — for cross-language API parity with
   * Python's `Helmlab.perceptual_distance()` and `MetricSpace.distance_from_lab()`.
   */
  distanceFromLab(lab1: Lab, lab2: Lab): number {
    return this.metric.distanceFromLab(lab1, lab2);
  }

  /**
   * Recommended entry point: perceptual color difference between two hex colors.
   *
   * Returns the trained perceptual distance (the COMBVD-fit metric, STRESS ≈ 22.5)
   * directly from hex — no need to convert to Lab or pick a lower-level method.
   * For the uncompressed Euclidean analogue use {@link deltaE}; for a reliability
   * estimate use {@link differenceWithConfidence}.
   */
  difference(color1: Hex, color2: Hex): number {
    return this.perceptualDistance(this.fromHex(color1), this.fromHex(color2));
  }

  /**
   * EXPERIMENTAL (beta): perceptual difference plus how much observers will
   * disagree about it. Returns `{ de, disagreement, reliability, reliable, ... }`.
   * Small / low-chroma differences come back unreliable — that is the point.
   * See {@link Confidence} and `core/confidence.ts` for provenance and limits.
   */
  differenceWithConfidence(color1: Hex, color2: Hex): Confidence {
    const lab1 = this.fromHex(color1);
    const lab2 = this.fromHex(color2);
    const de = this.perceptualDistance(lab1, lab2);
    const chroma = 0.5 * (sqrt(lab1[1] ** 2 + lab1[2] ** 2) + sqrt(lab2[1] ** 2 + lab2[2] ** 2));
    return assessConfidence(de, chroma);
  }

  // ── Palette Generation (GenSpace) ──────────────────────────────

  /** Generate lightness palette from base color. Uses GenSpace. */
  palette(baseHex: Hex, steps = 10): Hex[] {
    if (steps < 1) return [];
    const lab = this.genFromHex(baseHex);
    if (steps === 1) return [this.genToHex(lab)];
    const Lhi = this.genWhiteL - 0.01;
    const Llo = 0.05;
    const result: Hex[] = [];
    for (let i = 0; i < steps; i++) {
      const L = Lhi - (i / (steps - 1)) * (Lhi - Llo);
      result.push(this.genToHex([L, lab[1], lab[2]]));
    }
    return result;
  }

  /** Generate hue ring at fixed lightness and chroma. Uses GenSpace. */
  paletteHues(lightness = 0.6, chroma = 0.15, steps = 12): Hex[] {
    const result: Hex[] = [];
    for (let i = 0; i < steps; i++) {
      const h = (2 * PI * i) / steps;
      result.push(this.genToHex([lightness, chroma * cos(h), chroma * sin(h)]));
    }
    return result;
  }

  // ── Semantic Scale (GenSpace) ──────────────────────────────────

  /** Generate Tailwind-style semantic scale (50–950). Uses GenSpace. */
  semanticScale(baseHex: Hex, levels?: number[]): SemanticScale {
    if (!levels) levels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

    const lab = this.genFromHex(baseHex);
    const baseL = lab[0];
    const Llight = this.genWhiteL - 0.01;
    const Ldark = 0.05;

    const result: SemanticScale = {};
    for (const level of levels) {
      let L: number;
      if (level <= 500) {
        const t = level / 500;
        L = Llight + t * (baseL - Llight);
      } else {
        const t = (level - 500) / 450;
        L = baseL + t * (Ldark - baseL);
      }
      result[String(level)] = this.genToHex([L, lab[1], lab[2]]);
    }
    return result;
  }

  // ── Gradient ────────────────────────────────────────────────────

  /** Generate a perceptually uniform gradient between two hex colors.
   *  Uses GenSpace Lab path with CIEDE2000 arc-length reparameterization
   *  for equal perceptual step sizes on any color pair. */
  gradient(start: Hex, end: Hex, steps = 16): Hex[] {
    if (steps === 1) return [start];

    const lab1 = this.genFromHex(start);
    const lab2 = this.genFromHex(end);
    const dL = lab2[0] - lab1[0], da = lab2[1] - lab1[1], db = lab2[2] - lab1[2];

    // Fine-sample the GenSpace Lab line and build cumulative CIEDE2000 arc length
    const N = 256;
    const cumDist: number[] = [0];
    let prevCie = srgbToCieLab(clampRgb(xyzToSrgb(this.gen.toXYZ([lab1[0], lab1[1], lab1[2]]))));
    for (let i = 1; i <= N; i++) {
      const t = i / N;
      const srgb = clampRgb(xyzToSrgb(this.gen.toXYZ([
        lab1[0] + dL * t, lab1[1] + da * t, lab1[2] + db * t,
      ])));
      const cie = srgbToCieLab(srgb);
      cumDist.push(cumDist[i - 1] + ciede2000(prevCie[0], prevCie[1], prevCie[2], cie[0], cie[1], cie[2]));
      prevCie = cie;
    }
    const totalDist = cumDist[N];

    // Binary search for t values that produce equal cumulative distances
    const result: Hex[] = [];
    for (let s = 0; s < steps; s++) {
      const target = (s / (steps - 1)) * totalDist;
      let lo = 0, hi = N;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (cumDist[mid] < target) lo = mid; else hi = mid;
      }
      const frac = (cumDist[hi] - cumDist[lo]) > 1e-12
        ? (target - cumDist[lo]) / (cumDist[hi] - cumDist[lo]) : 0;
      const t = (lo + frac) / N;
      result.push(this.genToHex([
        lab1[0] + dL * t, lab1[1] + da * t, lab1[2] + db * t,
      ]));
    }
    return result;
  }

  // ── Dark/Light Mode ─────────────────────────────────────────

  /** Adapt color between light and dark mode via soft L-inversion (GenSpace). */
  adaptToMode(colorHex: Hex, fromMode: 'light' | 'dark' = 'light', toMode: 'light' | 'dark' = 'dark'): Hex {
    if (fromMode === toMode) return colorHex;

    const Lmax = this.genWhiteL - 0.02;
    const LIGHT_LO = 0.05, LIGHT_HI = Lmax;
    const DARK_LO = 0.08, DARK_HI = Lmax - 0.05;

    const lab = this.genFromHex(colorHex);
    const L = lab[0];

    const [srcLo, srcHi] = fromMode === 'light' ? [LIGHT_LO, LIGHT_HI] : [DARK_LO, DARK_HI];
    const [dstLo, dstHi] = fromMode === 'light' ? [DARK_LO, DARK_HI] : [LIGHT_LO, LIGHT_HI];

    const t = Math.min(Math.max((L - srcLo) / (srcHi - srcLo), 0), 1);
    const Lnew = dstHi - t * (dstHi - dstLo);

    return this.genToHex([Lnew, lab[1], lab[2]]);
  }

  /** Adapt fg/bg pair to target mode, ensuring contrast. */
  adaptPair(fgHex: Hex, bgHex: Hex, fromMode: 'light' | 'dark' = 'light', toMode: 'light' | 'dark' = 'dark', minRatio = 4.5): [Hex, Hex] {
    const newFg = this.adaptToMode(fgHex, fromMode, toMode);
    const newBg = this.adaptToMode(bgHex, fromMode, toMode);
    const adjusted = this.ensureContrast(newFg, newBg, minRatio);
    return [adjusted, newBg];
  }

  // ── Info ──────────────────────────────────────────────────────

  /** Return color info for a hex value. */
  info(hex: Hex): { hex: Hex; srgb: RGB; xyz: XYZ; lab: Lab; L: number; C: number; H: number; luminance: number } {
    const srgb = hexToSrgb(hex);
    const xyz = srgbToXyz(srgb);
    const lab = this.metric.fromXYZ(xyz);
    const C = sqrt(lab[1] ** 2 + lab[2] ** 2);
    const H = ((atan2(lab[2], lab[1]) * 180 / PI) % 360 + 360) % 360;
    return { hex, srgb, xyz, lab, L: lab[0], C, H, luminance: relativeLuminance(srgb) };
  }

  /** Return a TokenExporter for this Helmlab instance. */
  export(): TokenExporter {
    return new TokenExporter(this);
  }
}
