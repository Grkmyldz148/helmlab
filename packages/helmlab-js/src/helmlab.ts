/** Helmlab — UI design system utility layer for perceptual color. */

import type { Lab, XYZ, RGB, Hex, SemanticScale, WCAGLevel } from './types.js';
import { AnalyticalSpace } from './core/analytical.js';
import { compileParams, getDefaultParams, type HelmlabParams } from './core/params.js';
import { hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb, xyzToDisplayP3, clampRgb } from './utils/srgb.js';
import { gamutMap, isInGamut } from './utils/gamut.js';
import { contrastRatio as wcagCR } from './utils/contrast.js';

const { sqrt, atan2, cos, sin, PI } = Math;

export interface HelmlabOptions {
  params?: HelmlabParams;
  neutralCorrection?: boolean;
  abRotateDeg?: number;
}

export class Helmlab {
  private readonly space: AnalyticalSpace;

  constructor(opts: HelmlabOptions = {}) {
    const p = compileParams(opts.params ?? getDefaultParams());
    this.space = new AnalyticalSpace(p, {
      neutralCorrection: opts.neutralCorrection ?? true,
      abRotateDeg: opts.abRotateDeg ?? -28.2,
    });
  }

  // ── Core conversions ────────────────────────────────────────

  /** Hex '#rrggbb' → Helmlab Lab [L, a, b]. */
  fromHex(hex: Hex): Lab {
    return this.fromSrgb(hexToSrgb(hex));
  }

  /** Helmlab Lab → hex '#rrggbb' (gamut mapped). */
  toHex(lab: Lab): Hex {
    return srgbToHex(this.toSrgb(lab));
  }

  /** sRGB [0,1] → Helmlab Lab. */
  fromSrgb(rgb: RGB): Lab {
    return this.space.fromXYZ(srgbToXyz(rgb));
  }

  /** Helmlab Lab → sRGB [0,1] (gamut mapped, clamped). */
  toSrgb(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.space, 'srgb');
    return clampRgb(xyzToSrgb(this.space.toXYZ(mapped)));
  }

  /** CIE XYZ → Helmlab Lab. */
  fromXYZ(xyz: XYZ): Lab {
    return this.space.fromXYZ(xyz);
  }

  /** Helmlab Lab → CIE XYZ. */
  toXYZ(lab: Lab): XYZ {
    return this.space.toXYZ(lab);
  }

  /** Helmlab Lab → Display P3 [0,1] (gamut mapped). */
  toDisplayP3(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.space, 'display-p3');
    return clampRgb(xyzToDisplayP3(this.space.toXYZ(mapped)));
  }

  /** Check if Lab is within sRGB gamut. */
  isInSrgb(lab: Lab): boolean {
    return isInGamut(lab, this.space, 'srgb');
  }

  /** Check if Lab is within Display P3 gamut. */
  isInP3(lab: Lab): boolean {
    return isInGamut(lab, this.space, 'display-p3');
  }

  // ── Contrast ─────────────────────────────────────────────────

  /** WCAG contrast ratio between two hex colors (1.0–21.0). */
  contrastRatio(fg: Hex, bg: Hex): number {
    return wcagCR(hexToSrgb(fg), hexToSrgb(bg));
  }

  /** Adjust fg lightness to meet min contrast ratio against bg. */
  ensureContrast(fg: Hex, bg: Hex, minRatio = 4.5): Hex {
    if (this.contrastRatio(fg, bg) >= minRatio) return fg;

    const fgLab = this.fromHex(fg);
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
        const srgb = this.toSrgb(candidate);
        const hexQ = hexToSrgb(srgbToHex(srgb));
        const ratio = wcagCR(hexQ, bgRgb);

        if (dir === 'darken') {
          if (ratio >= minRatio) lo = mid; else hi = mid;
        } else {
          if (ratio >= minRatio) hi = mid; else lo = mid;
        }
      }

      const safeL = dir === 'darken' ? lo : hi;
      candidate[0] = safeL;
      const candHex = srgbToHex(this.toSrgb(candidate));
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

  /** Euclidean distance in Helmlab Lab space between two hex colors. */
  deltaE(color1: Hex, color2: Hex): number {
    const lab1 = this.fromHex(color1);
    const lab2 = this.fromHex(color2);
    return sqrt((lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2);
  }

  /** Full perceptual distance (Minkowski + compression) between two Lab values. */
  perceptualDistance(lab1: Lab, lab2: Lab): number {
    return this.space.distance(lab1, lab2);
  }

  // ── Palette Generation ───────────────────────────────────────

  /** Generate lightness palette from base color. */
  palette(baseHex: Hex, steps = 10): Hex[] {
    const lab = this.fromHex(baseHex);
    const result: Hex[] = [];
    for (let i = 0; i < steps; i++) {
      const L = 0.95 - (i / (steps - 1)) * 0.8; // 0.95 to 0.15
      result.push(this.toHex([L, lab[1], lab[2]]));
    }
    return result;
  }

  /** Generate hue ring at fixed lightness and chroma. */
  paletteHues(lightness = 0.6, chroma = 0.15, steps = 12): Hex[] {
    const result: Hex[] = [];
    for (let i = 0; i < steps; i++) {
      const h = (2 * PI * i) / steps;
      result.push(this.toHex([lightness, chroma * cos(h), chroma * sin(h)]));
    }
    return result;
  }

  // ── Semantic Scale ───────────────────────────────────────────

  /** Generate Tailwind-style semantic scale (50–950). */
  semanticScale(baseHex: Hex, levels?: number[]): SemanticScale {
    if (!levels) levels = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

    const lab = this.fromHex(baseHex);
    const baseL = lab[0];
    const Llight = 0.97, Ldark = 0.10;

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
      result[String(level)] = this.toHex([L, lab[1], lab[2]]);
    }
    return result;
  }

  // ── Info ──────────────────────────────────────────────────────

  /** Return color info for a hex value. */
  info(hex: Hex): { hex: Hex; lab: Lab; L: number; C: number; H: number } {
    const lab = this.fromHex(hex);
    const C = sqrt(lab[1] ** 2 + lab[2] ** 2);
    const H = ((atan2(lab[2], lab[1]) * 180 / PI) % 360 + 360) % 360;
    return { hex, lab, L: lab[0], C, H };
  }
}
