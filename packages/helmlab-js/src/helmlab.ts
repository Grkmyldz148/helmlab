/** Helmlab 1.0 — two-space color API.
 *
 * One `Helmlab` instance exposes three namespaces:
 *
 *   hl.gen     — CREATE colors (GenSpace): gradient, palette, scale,
 *                harmonies, contrast, dark mode. Wide-gamut output via the
 *                `gamut` option on every generation function.
 *   hl.metric  — MEASURE colors (MetricSpace): difference (trained metric),
 *                euclidean, ciede2000, jnd, confidence, nearest, info.
 *   hl.tokens  — EXPORT design tokens (hex/CSS strings in, strings out).
 *
 * The two spaces have branded Lab types (`GenLab` / `MetricLab`); passing
 * one space's Lab into the other's API throws. Color-string parameters
 * accept '#rrggbb', '#rgb', 'color(display-p3 r g b)' and
 * 'color(rec2020 r g b)' everywhere.
 *
 * Mirrors the Python sibling exactly (camelCase ↔ snake_case). See API.md.
 */

import type { Lab, XYZ, RGB, Hex, LCh, SemanticScale, WCAGLevel, GenLab, MetricLab } from './types.js';
import { AnalyticalSpace } from './core/analytical.js';
import { GenSpace } from './spaces/gen.js';
import {
  compileParams, getDefaultParams,
  compileGenParams, getDefaultGenParams,
  type HelmlabParams, type GenParams,
} from './core/params.js';
import {
  hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb,
  xyzToDisplayP3, displayP3ToXyz, xyzToRec2020, rec2020ToXyz,
  clampRgb, relativeLuminance,
} from './utils/srgb.js';
import { gamutMap, isInGamut, maxChroma as rawMaxChroma, findCusp, type SpaceLike, type GamutMapMethod } from './utils/gamut.js';
import { contrastRatio as wcagCR } from './utils/contrast.js';
import { assessConfidence, JND_DE, type Confidence } from './core/confidence.js';
import { Tokens } from './tokens.js';

const { sqrt, atan2, cos, sin, PI, pow, abs } = Math;

export type Gamut = 'srgb' | 'display-p3' | 'rec2020';
const VALID_GAMUTS: readonly Gamut[] = ['srgb', 'display-p3', 'rec2020'];

export type HarmonyKind =
  | 'complementary' | 'analogous' | 'triadic' | 'tetradic' | 'split_complementary';

const HARMONY_OFFSETS: Record<HarmonyKind, readonly number[]> = {
  complementary: [0, 180],
  analogous: [0, -30, 30],
  triadic: [0, 120, 240],
  tetradic: [0, 90, 180, 270],
  split_complementary: [0, 150, 210],
};

const DEFAULT_SCALE_LEVELS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

/** Thrown by `ensureContrast(..., { strict: true })` when the requested
 * ratio is unreachable against the given background. */
export class ContrastError extends Error {
  override name = 'ContrastError';
}

// ═══════════════════════════════════════════════════════════════════
// Branded Lab helpers (runtime tags backing the GenLab/MetricLab types)
// ═══════════════════════════════════════════════════════════════════

type SpaceTag = 'gen' | 'metric';

function brand<T>(lab: readonly number[], space: SpaceTag): T {
  const out: Lab = [lab[0], lab[1], lab[2]];
  Object.defineProperty(out, '__space', { value: space, enumerable: false });
  return out as unknown as T;
}

function acceptLab(lab: readonly number[], thisNs: SpaceTag, thatNs: SpaceTag): Lab {
  const tag = (lab as { __space?: SpaceTag }).__space;
  if (tag !== undefined && tag !== thisNs) {
    const wrong = tag === 'gen' ? 'GenLab' : 'MetricLab';
    throw new TypeError(
      `hl.${thisNs} got a ${wrong} — that Lab belongs to hl.${thatNs}. The two ` +
      `spaces have different coordinates; convert the original color with ` +
      `hl.${thisNs}.fromHex() instead.`,
    );
  }
  return [lab[0], lab[1], lab[2]];
}

// ═══════════════════════════════════════════════════════════════════
// Color-string parsing (shared boundary)
// ═══════════════════════════════════════════════════════════════════

const CSS_COLOR_RE =
  /^\s*color\(\s*(display-p3|rec2020)\s+([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s+([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s+([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*\)\s*$/;

/** Any supported color string → CIE XYZ (D65).
 * Accepts '#rrggbb', '#rgb', 'color(display-p3 r g b)', 'color(rec2020 r g b)'. */
export function parseColor(color: string): XYZ {
  if (typeof color !== 'string') {
    throw new TypeError(
      `expected a color string, got ${typeof color} — Lab arrays go to the ` +
      'lab-typed methods (toHex, distance, ...)',
    );
  }
  const s = color.trim();
  if (s.startsWith('color(')) {
    const m = CSS_COLOR_RE.exec(s);
    if (!m) {
      throw new Error(
        `unparseable CSS color() string: '${color}' — supported: ` +
        `'color(display-p3 r g b)', 'color(rec2020 r g b)'`,
      );
    }
    const comps: RGB = [parseFloat(m[2]), parseFloat(m[3]), parseFloat(m[4])];
    return m[1] === 'display-p3' ? displayP3ToXyz(comps) : rec2020ToXyz(comps);
  }
  return srgbToXyz(hexToSrgb(s));
}

/** Color string → sRGB [0,1] (wide-gamut inputs clipped — used only for
 * sRGB-defined operations like WCAG luminance). */
function colorToSrgb(color: string): RGB {
  const s = color.trim();
  if (s.startsWith('color(')) return clampRgb(xyzToSrgb(parseColor(s)));
  return hexToSrgb(s);
}

function checkGamut(gamut: string): asserts gamut is Gamut {
  if (!VALID_GAMUTS.includes(gamut as Gamut)) {
    throw new Error(`unknown gamut '${gamut}'; use one of ${VALID_GAMUTS.join(', ')}`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// CIELAB / CIEDE2000 (reference formulas)
// ═══════════════════════════════════════════════════════════════════

function srgbToCieLab(rgb: RGB): [number, number, number] {
  const lr = rgb[0] <= 0.04045 ? rgb[0] / 12.92 : pow((rgb[0] + 0.055) / 1.055, 2.4);
  const lg = rgb[1] <= 0.04045 ? rgb[1] / 12.92 : pow((rgb[1] + 0.055) / 1.055, 2.4);
  const lb = rgb[2] <= 0.04045 ? rgb[2] / 12.92 : pow((rgb[2] + 0.055) / 1.055, 2.4);
  const x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047;
  const y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb;
  const z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883;
  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x), fy = f(y), fz = f(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function xyzToCieLab(xyz: XYZ): [number, number, number] {
  const x = xyz[0] / 0.95047;
  const y = xyz[1];
  const z = xyz[2] / 1.08883;
  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
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

// ═══════════════════════════════════════════════════════════════════
// hl.gen — generation namespace
// ═══════════════════════════════════════════════════════════════════

/** Generation namespace — everything that CREATES colors (GenSpace).
 *
 * All generation functions take `{ gamut: 'srgb' | 'display-p3' | 'rec2020' }`
 * and return '#rrggbb' for sRGB or 'color(<gamut> r g b)' CSS strings for
 * the wide gamuts. Color-string inputs accept the same forms everywhere. */
export class Gen {
  /** Raw GenSpace (advanced access). */
  readonly space: GenSpace;
  private readonly whiteL: number;

  constructor(space: GenSpace) {
    this.space = space;
    this.whiteL = space.fromXYZ([0.95047, 1.0, 1.08883])[0];
  }

  // ── Conversions ─────────────────────────────────────────────────

  /** Color string → GenLab. Accepts '#rrggbb', '#rgb', 'color(display-p3 …)', 'color(rec2020 …)'. */
  fromHex(color: string): GenLab {
    return brand<GenLab>(this.space.fromXYZ(parseColor(color)), 'gen');
  }

  /** GenLab → '#rrggbb' (gamut-mapped to sRGB). */
  toHex(lab: GenLab | Lab): Hex {
    return this.emit(this.accept(lab), 'srgb');
  }

  /** GenLab → CSS color string in the requested gamut (gamut-mapped). */
  toCss(lab: GenLab | Lab, gamut: Gamut = 'display-p3'): string {
    return this.emit(this.accept(lab), gamut);
  }

  /** sRGB [0,1] → GenLab. */
  fromSrgb(rgb: RGB): GenLab {
    return brand<GenLab>(this.space.fromXYZ(srgbToXyz(rgb)), 'gen');
  }

  /** GenLab → sRGB [0,1] (gamut-mapped, clamped). */
  toSrgb(lab: GenLab | Lab): RGB {
    return this.toSrgbInternal(this.accept(lab));
  }

  /** Branded GenLab constructor from raw coordinates. */
  lab(L: number, a: number, b: number): GenLab {
    return brand<GenLab>([L, a, b], 'gen');
  }

  /** Cylindrical constructor → GenLab. ⚠ L and C are 0–1-scale, not 0–100. */
  lch(L: number, C: number, hDeg: number): GenLab {
    const rad = (hDeg * PI) / 180;
    return brand<GenLab>([L, C * cos(rad), C * sin(rad)], 'gen');
  }

  /** GenLab → [L, C, h°] (plain array); h in [0, 360).
   * Same coordinates as the `helmgenlch` space registered on Color.js.
   * For hue rotations prefer {@link rotateHue} / {@link harmonies}. */
  toLch(lab: GenLab | Lab): LCh {
    const nd = this.accept(lab);
    const C = sqrt(nd[1] ** 2 + nd[2] ** 2);
    let h = (atan2(nd[2], nd[1]) * 180) / PI;
    if (h < 0) h += 360;
    return [nd[0], C, h];
  }

  /** [L, C, h°] → GenLab. */
  fromLch(lch: LCh): GenLab {
    return this.lch(lch[0], lch[1], lch[2]);
  }

  /** Map a GenLab into the requested gamut.
   * `method='chroma'` (default) reduces chroma at constant L and hue —
   * predictable, hue-exact. `method='adaptive'` is the Ottosson-style
   * projection toward the cusp that trades a little L for more chroma —
   * better for saturated photographic content. */
  gamutMap(lab: GenLab | Lab, gamut: Gamut = 'srgb', method: GamutMapMethod = 'chroma'): GenLab {
    checkGamut(gamut);
    return brand<GenLab>(gamutMap(this.accept(lab), this.space, gamut, method), 'gen');
  }

  /** Maximum in-gamut chroma at fixed lightness and hue.
   * The gamut-boundary query behind {@link vivid} — use it to build
   * saturation ramps or to know how much chroma headroom a wide gamut adds. */
  maxChroma(lightness: number, hueDeg: number, gamut: Gamut = 'srgb'): number {
    checkGamut(gamut);
    return rawMaxChroma(lightness, (hueDeg * PI) / 180, this.space, gamut);
  }

  /** The (L, C) cusp of a hue — the single most colorful point of that hue
   * leaf in the requested gamut. GenSpace resolves a clean cusp at all 360
   * hues in sRGB, P3 and Rec2020 (the ColorBench 360/360/360 result); this
   * exposes that geometry directly. */
  cusp(hueDeg: number, gamut: Gamut = 'srgb'): [number, number] {
    checkGamut(gamut);
    return findCusp((hueDeg * PI) / 180, this.space, gamut);
  }

  /** The most saturated version of a color: same lightness, same hue,
   * chroma pushed to the gamut boundary. With `gamut: 'display-p3'` this is
   * the honest "P3 makes this color pop" upgrade — hue and lightness stay
   * put, only colorfulness grows. */
  vivid(color: string, opts: { gamut?: Gamut } = {}): string {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    const lch = this.toLch(this.fromHex(color));
    const Cmax = this.maxChroma(lch[0], lch[2], gamut);
    return this.emit(acceptLab(this.lch(lch[0], Cmax, lch[2]), 'gen', 'metric'), gamut);
  }

  /** Whether a GenLab or color string is inside the requested gamut. */
  inGamut(labOrColor: GenLab | Lab | string, gamut: Gamut = 'srgb'): boolean {
    checkGamut(gamut);
    const nd = typeof labOrColor === 'string'
      ? this.space.fromXYZ(parseColor(labOrColor))
      : this.accept(labOrColor);
    return isInGamut(nd, this.space, gamut);
  }

  // ── Generation ──────────────────────────────────────────────────

  /** Perceptually uniform gradient between two colors.
   *
   * Equal perceptual step sizes on any color pair (GenSpace Lab path,
   * CIEDE2000 arc-length reparameterization). Step spacing is measured on
   * the sRGB projection of the path; with a wide `gamut` the emitted stops
   * keep the extra chroma the target gamut allows.
   *
   * Edge cases: `steps=1` → `[start]` (end not included); `steps<1` → `[]`. */
  gradient(start: string, end: string, steps = 16, opts: { gamut?: Gamut } = {}): string[] {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    if (steps < 1) return [];
    const lab1 = acceptLab(this.fromHex(start), 'gen', 'metric');
    if (steps === 1) return [this.emit(lab1, gamut)];
    const lab2 = acceptLab(this.fromHex(end), 'gen', 'metric');
    const fractions: number[] = [];
    for (let s = 0; s < steps; s++) fractions.push(s / (steps - 1));
    return this.pathPoints(lab1, lab2, fractions).map((p) => this.emit(p, gamut));
  }

  /** The color at fraction `t` (0 → a, 1 → b) along the SAME perceptual
   * arc-length path {@link gradient} uses — `mix(a, b, 0.5)` is the visual
   * midpoint, not the coordinate average. */
  mix(a: string, b: string, t = 0.5, opts: { gamut?: Gamut } = {}): string {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    const tc = Math.min(Math.max(t, 0), 1);
    const lab1 = acceptLab(this.fromHex(a), 'gen', 'metric');
    const lab2 = acceptLab(this.fromHex(b), 'gen', 'metric');
    return this.emit(this.pathPoints(lab1, lab2, [tc])[0], gamut);
  }

  /** Lightness ramp from a base color, light → dark.
   *
   * Evenly spaced GenSpace L; hue and chroma preserved, gamut mapped. The
   * base color is only approximate inside the result (its L is re-spaced) —
   * for a scale where the input is returned exactly at level 500 use
   * {@link scale}. `steps<1` returns `[]`. */
  palette(base: string, steps = 10, opts: { gamut?: Gamut } = {}): string[] {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    if (steps < 1) return [];
    const lab = acceptLab(this.fromHex(base), 'gen', 'metric');
    if (steps === 1) return [this.emit(lab, gamut)];
    const Lhi = this.whiteL - 0.01;
    const Llo = 0.05;
    const result: string[] = [];
    for (let i = 0; i < steps; i++) {
      const L = Lhi - (i / (steps - 1)) * (Lhi - Llo);
      result.push(this.emit([L, lab[1], lab[2]], gamut));
    }
    return result;
  }

  /** Tailwind-style semantic scale (50–950). `base` maps to level 500
   * EXACTLY; lower levels are lighter, higher levels darker. */
  scale(base: string, levels?: number[], opts: { gamut?: Gamut } = {}): SemanticScale {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    const lvls = levels ?? DEFAULT_SCALE_LEVELS;
    const lab = acceptLab(this.fromHex(base), 'gen', 'metric');
    const baseL = lab[0];
    const Llight = this.whiteL - 0.01;
    const Ldark = 0.05;
    const result: SemanticScale = {};
    for (const level of lvls) {
      let L: number;
      if (level <= 500) {
        const t = level / 500;
        L = Llight + t * (baseL - Llight);
      } else {
        const t = (level - 500) / 450;
        L = baseL + t * (Ldark - baseL);
      }
      result[String(level)] = this.emit([L, lab[1], lab[2]], gamut);
    }
    return result;
  }

  /** Categorical palette: `count` evenly spaced hues at fixed lightness and
   * chroma (equal perceptual weight per category). */
  hueRing(count = 12, opts: { lightness?: number; chroma?: number; gamut?: Gamut } = {}): string[] {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    const lightness = opts.lightness ?? 0.6;
    const chroma = opts.chroma ?? 0.15;
    const result: string[] = [];
    for (let i = 0; i < count; i++) {
      const h = (2 * PI * i) / count;
      result.push(this.emit([lightness, chroma * cos(h), chroma * sin(h)], gamut));
    }
    return result;
  }

  /** Classic hue harmonies with the base color first.
   *
   * `kind`: complementary [0°, 180°] · analogous [0°, ±30°] · triadic
   * [0°, 120°, 240°] · tetradic [0°, 90°, 180°, 270°] · split_complementary
   * [0°, 150°, 210°]. Rotations happen in GenSpace LCh at constant L and C —
   * perceptual lightness and colorfulness stay matched across the set (the
   * property hue harmonies assume but sRGB HSL rotations don't deliver). */
  harmonies(base: string, kind: HarmonyKind = 'complementary', opts: { gamut?: Gamut } = {}): string[] {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    const offsets = HARMONY_OFFSETS[kind];
    if (!offsets) {
      throw new Error(
        `unknown harmony kind '${kind}'; use one of ${Object.keys(HARMONY_OFFSETS).sort().join(', ')}`,
      );
    }
    const lch = this.toLch(this.fromHex(base));
    return offsets.map((off) =>
      this.emit(acceptLab(this.lch(lch[0], lch[1], (((lch[2] + off) % 360) + 360) % 360), 'gen', 'metric'), gamut),
    );
  }

  /** Rotate a color's hue by `degrees` at constant L and C (GenLCh). */
  rotateHue(color: string, degrees: number, opts: { gamut?: Gamut } = {}): string {
    const gamut = opts.gamut ?? 'srgb';
    checkGamut(gamut);
    const lch = this.toLch(this.fromHex(color));
    const h = (((lch[2] + degrees) % 360) + 360) % 360;
    return this.emit(acceptLab(this.lch(lch[0], lch[1], h), 'gen', 'metric'), gamut);
  }

  // ── Contrast ────────────────────────────────────────────────────

  /** WCAG 2.1 contrast ratio between two colors (1.0–21.0). */
  contrastRatio(fg: string, bg: string): number {
    return wcagCR(colorToSrgb(fg), colorToSrgb(bg));
  }

  /** Whether fg/bg meets WCAG level AA (4.5:1) or AAA (7:1). */
  meetsContrast(fg: string, bg: string, level: WCAGLevel = 'AA'): boolean {
    const threshold = level === 'AAA' ? 7 : 4.5;
    return this.contrastRatio(fg, bg) >= threshold;
  }

  /** Return a variant of `fg` meeting `ratio` against `bg`.
   *
   * Binary search on the GenSpace L axis — hue and chroma are preserved
   * while a lightness-only adjustment can reach the ratio. If no lightness
   * of this hue reaches it, falls back to pure #000000/#ffffff (hue lost).
   * If even black/white cannot reach the ratio (possible for mid-gray
   * backgrounds with high ratios, e.g. 7:1 vs #808080):
   * `strict: true` throws {@link ContrastError}; `strict: false` (default)
   * warns via console.warn and returns the best effort BELOW the requested
   * ratio. Check with {@link meetsContrast} for hard requirements. */
  ensureContrast(fg: string, bg: string, ratio = 4.5, opts: { strict?: boolean } = {}): Hex {
    const current = this.contrastRatio(fg, bg);
    if (current >= ratio) return srgbToHex(colorToSrgb(fg));

    const fgLab = acceptLab(this.fromHex(fg), 'gen', 'metric');
    const bgRgb = colorToSrgb(bg);
    const origL = fgLab[0];

    let bestHex = srgbToHex(colorToSrgb(fg));
    let bestRatio = current;
    let bestL = origL;

    for (const dir of ['darken', 'lighten'] as const) {
      let lo = dir === 'darken' ? 0 : origL;
      let hi = dir === 'darken' ? origL : 1.5;
      const candidate: Lab = [...fgLab];

      for (let i = 0; i < 40; i++) {
        const mid = (lo + hi) / 2;
        candidate[0] = mid;
        const hexQ = hexToSrgb(srgbToHex(this.toSrgbInternal(candidate)));
        const r = wcagCR(hexQ, bgRgb);
        if (dir === 'darken') {
          if (r >= ratio) lo = mid; else hi = mid;
        } else {
          if (r >= ratio) hi = mid; else lo = mid;
        }
      }

      let safeL = dir === 'darken' ? lo : hi;
      safeL = dir === 'darken' ? Math.max(0, safeL - 0.003) : Math.min(1.5, safeL + 0.003);
      candidate[0] = safeL;
      const candHex = srgbToHex(this.toSrgbInternal(candidate));
      const r = wcagCR(hexToSrgb(candHex), bgRgb);

      if (r >= ratio) {
        const dist = Math.abs(safeL - origL);
        const bestDist = Math.abs(bestL - origL);
        if (bestRatio < ratio || dist < bestDist) {
          bestHex = candHex;
          bestRatio = r;
          bestL = safeL;
        }
      }
    }

    if (bestRatio < ratio) {
      for (const fallback of ['#000000', '#ffffff'] as const) {
        if (this.contrastRatio(fallback, bg) >= ratio) return fallback;
      }
      const rBlack = this.contrastRatio('#000000', bg);
      const rWhite = this.contrastRatio('#ffffff', bg);
      const best = rBlack > rWhite ? '#000000' : '#ffffff';
      const msg =
        `ensureContrast: no color reaches ${ratio}:1 against ${bg} ` +
        `(best achievable: ${Math.max(rBlack, rWhite).toFixed(2)}:1 with ${best}).`;
      if (opts.strict) throw new ContrastError(msg);
      console.warn(`helmlab ${msg} Returning best effort below the requested ratio.`);
      return best;
    }

    return bestHex;
  }

  // ── Dark/light mode ─────────────────────────────────────────────

  /** Adapt a color between light and dark mode (soft L-inversion, GenSpace). */
  adaptToMode(color: string, fromMode: 'light' | 'dark' = 'light', toMode: 'light' | 'dark' = 'dark'): Hex {
    if (fromMode === toMode) return srgbToHex(colorToSrgb(color));

    const Lmax = this.whiteL - 0.02;
    const LIGHT_LO = 0.05, LIGHT_HI = Lmax;
    const DARK_LO = 0.08, DARK_HI = Lmax - 0.05;

    const lab = acceptLab(this.fromHex(color), 'gen', 'metric');
    const L = lab[0];

    const [srcLo, srcHi] = fromMode === 'light' ? [LIGHT_LO, LIGHT_HI] : [DARK_LO, DARK_HI];
    const [dstLo, dstHi] = fromMode === 'light' ? [DARK_LO, DARK_HI] : [LIGHT_LO, LIGHT_HI];

    const t = Math.min(Math.max((L - srcLo) / (srcHi - srcLo), 0), 1);
    const Lnew = dstHi - t * (dstHi - dstLo);
    return this.emit([Lnew, lab[1], lab[2]], 'srgb');
  }

  /** Adapt an fg/bg pair to the target mode, then re-ensure contrast. */
  adaptPair(fg: string, bg: string, fromMode: 'light' | 'dark' = 'light',
            toMode: 'light' | 'dark' = 'dark', ratio = 4.5): [Hex, Hex] {
    const newFg = this.adaptToMode(fg, fromMode, toMode);
    const newBg = this.adaptToMode(bg, fromMode, toMode);
    return [this.ensureContrast(newFg, newBg, ratio), newBg];
  }

  // ── Internals ───────────────────────────────────────────────────

  private accept(lab: readonly number[]): Lab {
    return acceptLab(lab, 'gen', 'metric');
  }

  private toSrgbInternal(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.space, 'srgb');
    return clampRgb(xyzToSrgb(this.space.toXYZ(mapped)));
  }

  /** Gamut-map in GenSpace and emit the color string for the gamut. */
  private emit(lab: Lab, gamut: Gamut): string {
    if (gamut === 'srgb') return srgbToHex(this.toSrgbInternal(lab));
    const mapped = gamutMap(lab, this.space, gamut);
    const xyz = this.space.toXYZ(mapped);
    if (gamut === 'display-p3') {
      const c = clampRgb(xyzToDisplayP3(xyz));
      return `color(display-p3 ${c[0].toFixed(4)} ${c[1].toFixed(4)} ${c[2].toFixed(4)})`;
    }
    const c = clampRgb(xyzToRec2020(xyz));
    return `color(rec2020 ${c[0].toFixed(4)} ${c[1].toFixed(4)} ${c[2].toFixed(4)})`;
  }

  /** Points along the CIEDE2000 arc-length-parameterized GenLab line at the
   * given arc fractions (0..1). Shared by gradient() and mix(). Samples go
   * through the gamut-MAPPED sRGB projection (like the final stops and the
   * Python sibling) — raw clamping made the arc length disagree with what
   * is displayed whenever the Lab line leaves sRGB. */
  private pathPoints(lab1: Lab, lab2: Lab, fractions: number[]): Lab[] {
    const dL = lab2[0] - lab1[0], da = lab2[1] - lab1[1], db = lab2[2] - lab1[2];
    const N = 256;
    const cum: number[] = [0];
    let prev = srgbToCieLab(this.toSrgbInternal([lab1[0], lab1[1], lab1[2]]));
    for (let i = 1; i <= N; i++) {
      const t = i / N;
      const cie = srgbToCieLab(this.toSrgbInternal([lab1[0] + dL * t, lab1[1] + da * t, lab1[2] + db * t]));
      cum.push(cum[i - 1] + ciede2000(prev[0], prev[1], prev[2], cie[0], cie[1], cie[2]));
      prev = cie;
    }
    const total = cum[N];

    return fractions.map((f) => {
      const target = f * total;
      let lo = 0, hi = N;
      while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (cum[mid] < target) lo = mid; else hi = mid;
      }
      const frac = cum[hi] - cum[lo] > 1e-12 ? (target - cum[lo]) / (cum[hi] - cum[lo]) : 0;
      const t = (lo + frac) / N;
      return [lab1[0] + dL * t, lab1[1] + da * t, lab1[2] + db * t] as Lab;
    });
  }
}

// ═══════════════════════════════════════════════════════════════════
// hl.metric — measurement namespace
// ═══════════════════════════════════════════════════════════════════

/** Measurement namespace — everything that MEASURES colors (MetricSpace).
 *
 * Four clearly-named difference metrics: {@link difference} (trained,
 * saturates ~0.15), {@link euclidean} (unbounded ΔE76-analogue),
 * {@link ciede2000} (industry standard), {@link jnd} (difference in
 * just-noticeable-difference units). */
export class Metric {
  /** Raw AnalyticalSpace (advanced access). */
  readonly space: AnalyticalSpace;

  constructor(space: AnalyticalSpace) {
    this.space = space;
  }

  // ── Conversions ─────────────────────────────────────────────────

  /** Color string → MetricLab. Accepts '#rrggbb', '#rgb', 'color(display-p3 …)', 'color(rec2020 …)'. */
  fromHex(color: string): MetricLab {
    return brand<MetricLab>(this.space.fromXYZ(parseColor(color)), 'metric');
  }

  /** MetricLab → '#rrggbb' (gamut-mapped to sRGB). */
  toHex(lab: MetricLab | Lab): Hex {
    return srgbToHex(this.toSrgbInternal(this.accept(lab)));
  }

  /** MetricLab → CSS color string in the requested gamut (gamut-mapped). */
  toCss(lab: MetricLab | Lab, gamut: Gamut = 'display-p3'): string {
    checkGamut(gamut);
    const nd = this.accept(lab);
    if (gamut === 'srgb') return this.toHex(nd);
    const mapped = gamutMap(nd, this.space, gamut);
    const xyz = this.space.toXYZ(mapped);
    if (gamut === 'display-p3') {
      const c = clampRgb(xyzToDisplayP3(xyz));
      return `color(display-p3 ${c[0].toFixed(4)} ${c[1].toFixed(4)} ${c[2].toFixed(4)})`;
    }
    const c = clampRgb(xyzToRec2020(xyz));
    return `color(rec2020 ${c[0].toFixed(4)} ${c[1].toFixed(4)} ${c[2].toFixed(4)})`;
  }

  /** sRGB [0,1] → MetricLab. */
  fromSrgb(rgb: RGB): MetricLab {
    return brand<MetricLab>(this.space.fromXYZ(srgbToXyz(rgb)), 'metric');
  }

  /** MetricLab → sRGB [0,1] (gamut-mapped, clamped). */
  toSrgb(lab: MetricLab | Lab): RGB {
    return this.toSrgbInternal(this.accept(lab));
  }

  /** CIE XYZ (D65) → MetricLab. */
  fromXyz(xyz: XYZ): MetricLab {
    return brand<MetricLab>(this.space.fromXYZ(xyz), 'metric');
  }

  /** MetricLab → CIE XYZ (D65). */
  toXyz(lab: MetricLab | Lab): XYZ {
    return this.space.toXYZ(this.accept(lab));
  }

  /** Branded MetricLab constructor from raw coordinates. */
  lab(L: number, a: number, b: number): MetricLab {
    return brand<MetricLab>([L, a, b], 'metric');
  }

  /** Cylindrical constructor → MetricLab. ⚠ L and C are 0–1-scale, not 0–100. */
  lch(L: number, C: number, hDeg: number): MetricLab {
    const rad = (hDeg * PI) / 180;
    return brand<MetricLab>([L, C * cos(rad), C * sin(rad)], 'metric');
  }

  /** MetricLab → [L, C, h°] (plain array); h in [0, 360).
   * Cylindrical view of Metric Lab — the same C and H that {@link info}
   * reports. For hue manipulation prefer the GENERATION space
   * (`hl.gen.rotateHue` / `harmonies`): Metric Lab is tuned for difference
   * prediction, not for producing colors. */
  toLch(lab: MetricLab | Lab): LCh {
    const nd = this.accept(lab);
    const C = sqrt(nd[1] ** 2 + nd[2] ** 2);
    let h = (atan2(nd[2], nd[1]) * 180) / PI;
    if (h < 0) h += 360;
    return [nd[0], C, h];
  }

  /** [L, C, h°] → MetricLab. */
  fromLch(lch: LCh): MetricLab {
    return this.lch(lch[0], lch[1], lch[2]);
  }

  /** Whether a MetricLab or color string is inside the requested gamut. */
  inGamut(labOrColor: MetricLab | Lab | string, gamut: Gamut = 'srgb'): boolean {
    checkGamut(gamut);
    const nd = typeof labOrColor === 'string'
      ? this.space.fromXYZ(parseColor(labOrColor))
      : this.accept(labOrColor);
    return isInGamut(nd, this.space, gamut);
  }

  // ── The four difference metrics ─────────────────────────────────

  /** THE trained perceptual difference between two colors (v21 metric,
   * COMBVD STRESS 22.48 — 23% better than CIEDE2000's 29.20).
   *
   * Best for near-threshold judgments. Monotonic compression saturates near
   * ~0.15 for very different colors: rank is preserved but absolute values
   * plateau — for an unbounded number use {@link euclidean}, for threshold
   * units use {@link jnd}. */
  difference(a: string, b: string): number {
    return this.space.distance(this.space.fromXYZ(parseColor(a)), this.space.fromXYZ(parseColor(b)));
  }

  /** Uncompressed Euclidean distance in Metric Lab (ΔE76-analogue).
   *
   * Unbounded companion to {@link difference}: range ≈ 0–1.6 across sRGB
   * (#000000 ↔ #ffffff ≈ 1.12, #ff0000 ↔ #00ff00 ≈ 1.62). */
  euclidean(a: string, b: string): number {
    const lab1 = this.fromHex(a);
    const lab2 = this.fromHex(b);
    return sqrt((lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2);
  }

  /** CIEDE2000 ΔE between two colors (CIELAB scale, ~0–100).
   *
   * The industry-standard formula, self-contained. Recommended for
   * fixed-catalog nearest-color matching: measurably more stable under tiny
   * input perturbations than the trained metric (7 vs 16 selection flips per
   * 100 targets at ±2/255), because {@link difference} is fit for
   * near-threshold judgments, not suprathreshold ranking.
   * Sanity anchor: `ciede2000('#ff0000', '#00ff00') ≈ 86.6`. */
  ciede2000(a: string, b: string): number {
    const lab1 = this.cielab(a);
    const lab2 = this.cielab(b);
    return ciede2000(lab1[0], lab1[1], lab1[2], lab2[0], lab2[1], lab2[2]);
  }

  /** Difference in just-noticeable-difference units.
   *
   * `jnd = difference / 0.0356` where 0.0356 de is the point at which the
   * median observer of the v2 ordinal confidence model rates a pair at
   * least "moderately different" (`TAU2 / MU_SCALE`). Interpretation: < 1
   * likely unnoticed, 1–2 subtle, > 2 clearly visible. Because
   * {@link difference} saturates near ~0.15, values above ≈ 4.2 mean "far
   * above threshold", not a precise multiple. */
  jnd(a: string, b: string): number {
    return this.difference(a, b) / JND_DE;
  }

  // ── Lab-level distance (identical contract in both languages) ───

  /** Trained perceptual distance between two MetricLab values.
   *
   * Identical contract in Python and JS: MetricLab in, distance out (the
   * 0.x Python-XYZ/JS-Lab asymmetry is gone; the XYZ-in variant lives only
   * on Python's raw `MetricSpace` class). Same saturation behavior as
   * {@link difference}. */
  distance(labA: MetricLab | Lab, labB: MetricLab | Lab): number {
    return this.space.distance(this.accept(labA), this.accept(labB));
  }

  // ── Confidence / catalog / info ─────────────────────────────────

  /** Perceptual difference plus how much observers will disagree.
   *
   * EXPERIMENTAL (v2 ordinal model, n=47 pairs — see `core/confidence.ts`
   * for provenance and limits). Returns `de`, `latent`, `disagreement`,
   * `reliability`, `pNoticeable`, `reliable` and `extrapolated` (true when
   * de is beyond the training range — treat the numbers as indicative). */
  confidence(a: string, b: string): Confidence {
    const lab1 = this.space.fromXYZ(parseColor(a));
    const lab2 = this.space.fromXYZ(parseColor(b));
    const de = this.space.distance(lab1, lab2);
    const chroma = 0.5 * (sqrt(lab1[1] ** 2 + lab1[2] ** 2) + sqrt(lab2[1] ** 2 + lab2[2] ** 2));
    return assessConfidence(de, chroma);
  }

  /** Pick the perceptually nearest color to `target` from `palette`.
   *
   * `metric`: 'ciede2000' (default — most perturbation-stable for catalog
   * argmax), 'difference' (trained, near-threshold), 'euclidean' (fast
   * unbounded). Returns the pick plus `runnerUp` and `margin` (relative gap
   * to the runner-up — a small margin means a fragile pick). */
  nearest(
    target: string,
    palette: string[],
    metric: 'ciede2000' | 'difference' | 'euclidean' = 'ciede2000',
  ): { hex: string; index: number; distance: number; runnerUp: string;
       runnerUpDistance: number; margin: number; metric: string } {
    if (!palette.length) throw new Error('palette is empty');
    const fn = metric === 'ciede2000' ? (a: string, b: string) => this.ciede2000(a, b)
      : metric === 'difference' ? (a: string, b: string) => this.difference(a, b)
      : metric === 'euclidean' ? (a: string, b: string) => this.euclidean(a, b)
      : null;
    if (!fn) throw new Error(`unknown metric '${metric}'; use ciede2000 | difference | euclidean`);
    const dists = palette.map((c) => fn(target, c));
    const order = dists.map((_, i) => i).sort((x, y) => dists[x] - dists[y]);
    const best = order[0], second = order.length > 1 ? order[1] : order[0];
    const d0 = dists[best], d1 = dists[second];
    return {
      hex: palette[best], index: best, distance: d0,
      runnerUp: palette[second], runnerUpDistance: d1,
      margin: d0 > 0 ? (d1 - d0) / d0 : Infinity, metric,
    };
  }

  /** Descriptive object for a color: hex, srgb, xyz, lab, L, C, H,
   * luminance, inSrgb, inP3, inRec2020. */
  info(color: string): {
    hex: Hex; srgb: RGB; xyz: XYZ; lab: Lab; L: number; C: number; H: number;
    luminance: number; inSrgb: boolean; inP3: boolean; inRec2020: boolean;
  } {
    const xyz = parseColor(color);
    const srgb = clampRgb(xyzToSrgb(xyz));
    const lab = this.space.fromXYZ(xyz);
    const C = sqrt(lab[1] ** 2 + lab[2] ** 2);
    const H = ((atan2(lab[2], lab[1]) * 180 / PI) % 360 + 360) % 360;
    return {
      hex: srgbToHex(srgb), srgb, xyz, lab, L: lab[0], C, H,
      luminance: relativeLuminance(srgb),
      inSrgb: isInGamut(lab, this.space, 'srgb'),
      inP3: isInGamut(lab, this.space, 'display-p3'),
      inRec2020: isInGamut(lab, this.space, 'rec2020'),
    };
  }

  // ── Internals ───────────────────────────────────────────────────

  private accept(lab: readonly number[]): Lab {
    return acceptLab(lab, 'metric', 'gen');
  }

  private toSrgbInternal(lab: Lab): RGB {
    const mapped = gamutMap(lab, this.space, 'srgb');
    return clampRgb(xyzToSrgb(this.space.toXYZ(mapped)));
  }

  private cielab(color: string): [number, number, number] {
    const s = color.trim();
    if (s.startsWith('color(')) return xyzToCieLab(parseColor(s));
    return srgbToCieLab(hexToSrgb(s));
  }
}

// ═══════════════════════════════════════════════════════════════════
// Helmlab — the entry point
// ═══════════════════════════════════════════════════════════════════

export interface HelmlabOptions {
  metricParams?: HelmlabParams;
  genParams?: GenParams;
  neutralCorrection?: boolean;
  abRotateDeg?: number;
}

/** Two purpose-built color spaces behind three namespaces.
 *
 * `hl.gen` creates colors (GenSpace — ColorBench 62-9-19 vs OKLab),
 * `hl.metric` measures them (MetricSpace — COMBVD STRESS 22.48),
 * `hl.tokens` exports design tokens. See API.md. */
export class Helmlab {
  readonly gen: Gen;
  readonly metric: Metric;
  readonly tokens: Tokens;

  constructor(opts: HelmlabOptions = {}) {
    const p = compileParams(opts.metricParams ?? getDefaultParams());
    const metricSpace = new AnalyticalSpace(p, {
      neutralCorrection: opts.neutralCorrection ?? true,
      abRotateDeg: opts.abRotateDeg ?? -28.2,
    });
    const genSpace = new GenSpace(compileGenParams(opts.genParams ?? getDefaultGenParams()));

    this.gen = new Gen(genSpace);
    this.metric = new Metric(metricSpace);
    this.tokens = new Tokens(this.metric);
  }
}
