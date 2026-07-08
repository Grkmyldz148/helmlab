/** Design-token export — CSS, Android, iOS, Tailwind formats.
 *
 * 1.0 replacement for the 0.x `TokenExporter`: every method takes color
 * STRINGS ('#rrggbb', 'color(display-p3 …)', …), never Lab arrays — the
 * Gen-Lab-into-the-exporter footgun is structurally gone. Access via
 * `hl.tokens`. Mirrors Python's `helmlab.tokens.Tokens`.
 */

import type { XYZ, SemanticScale } from './types.js';

const { sqrt, atan2, abs, round, sign, PI } = Math;
const cbrt = Math.cbrt;

// ── Oklab matrices (Björn Ottosson) — for CSS oklch() output ────────

const M1_OKLAB = [
  0.8189330101, 0.3618667424, -0.1288597137,
  0.0329845436, 0.9293118715,  0.0361456387,
  0.0482003018, 0.2643662691,  0.6338517070,
];

const M2_OKLAB = [
  0.2104542553,  0.7936177850, -0.0040720468,
  1.9779984951, -2.4285922050,  0.4505937099,
  0.0259040371,  0.7827717662, -0.8086757660,
];

function xyzToOklab(xyz: XYZ): [number, number, number] {
  const [x, y, z] = xyz;
  const l = M1_OKLAB[0] * x + M1_OKLAB[1] * y + M1_OKLAB[2] * z;
  const m = M1_OKLAB[3] * x + M1_OKLAB[4] * y + M1_OKLAB[5] * z;
  const s = M1_OKLAB[6] * x + M1_OKLAB[7] * y + M1_OKLAB[8] * z;
  const lc = sign(l) * cbrt(abs(l));
  const mc = sign(m) * cbrt(abs(m));
  const sc = sign(s) * cbrt(abs(s));
  return [
    M2_OKLAB[0] * lc + M2_OKLAB[1] * mc + M2_OKLAB[2] * sc,
    M2_OKLAB[3] * lc + M2_OKLAB[4] * mc + M2_OKLAB[5] * sc,
    M2_OKLAB[6] * lc + M2_OKLAB[7] * mc + M2_OKLAB[8] * sc,
  ];
}

/** Minimal view of the Metric namespace that Tokens needs (avoids a
 * circular import with helmlab.ts). */
interface MetricLike {
  fromHex(color: string): readonly number[];
  toHex(lab: readonly number[]): string;
  toSrgb(lab: readonly number[]): readonly number[];
  toXyz(lab: readonly number[]): XYZ;
  toCss(lab: readonly number[], gamut: 'srgb' | 'display-p3' | 'rec2020'): string;
}

export type TokenCssFormat = 'hex' | 'rgb' | 'hsl' | 'oklch' | 'p3' | 'rec2020';

const CSS_FORMATS: readonly TokenCssFormat[] = ['hex', 'rgb', 'hsl', 'oklch', 'p3', 'rec2020'];

/** Design-token export namespace (`hl.tokens`). Color strings in, platform
 * token strings out. */
export class Tokens {
  private readonly metric: MetricLike;

  constructor(metric: MetricLike) {
    this.metric = metric;
  }

  // ── Single-color formats ─────────────────────────────────────────

  /** One CSS value for a color. `format`: hex, rgb, hsl, oklch,
   * p3 ('color(display-p3 …)'), rec2020. */
  css(color: string, format: TokenCssFormat = 'oklch'): string {
    if (!CSS_FORMATS.includes(format)) {
      throw new Error(`unknown format '${format}'; use one of ${CSS_FORMATS.join(', ')}`);
    }
    const m = this.metric;
    const lab = m.fromHex(color);
    switch (format) {
      case 'hex':
        return m.toHex(lab);
      case 'rgb': {
        const [r, g, b] = m.toSrgb(lab).map((v) => round(v * 255));
        return `rgb(${r}, ${g}, ${b})`;
      }
      case 'hsl':
        return hsl(m.toSrgb(lab));
      case 'oklch': {
        const ok = xyzToOklab(m.toXyz(lab));
        const C = sqrt(ok[1] * ok[1] + ok[2] * ok[2]);
        const H = ((atan2(ok[2], ok[1]) * 180 / PI) % 360 + 360) % 360;
        return `oklch(${(ok[0] * 100).toFixed(1)}% ${C.toFixed(4)} ${H.toFixed(1)})`;
      }
      case 'p3':
        return m.toCss(lab, 'display-p3');
      case 'rec2020':
        return m.toCss(lab, 'rec2020');
    }
  }

  /** Color → '0xFFrrggbb' (Android ARGB int literal). */
  android(color: string): string {
    const [r, g, b] = this.metric.toSrgb(this.metric.fromHex(color)).map((v) => round(v * 255));
    const h = (v: number) => v.toString(16).padStart(2, '0');
    return `0xFF${h(r)}${h(g)}${h(b)}`;
  }

  /** Color → {r, g, b} Display P3 floats (UIColor displayP3). */
  iosP3(color: string): { r: number; g: number; b: number } {
    const css = this.metric.toCss(this.metric.fromHex(color), 'display-p3');
    const vals = css.slice('color(display-p3 '.length, -1).split(' ');
    const r4 = (v: string) => Math.round(parseFloat(v) * 10000) / 10000;
    return { r: r4(vals[0]), g: r4(vals[1]), b: r4(vals[2]) };
  }

  /** Color → Swift `Color(.displayP3, …)` literal. */
  swift(color: string): string {
    const p3 = this.iosP3(color);
    return `Color(.displayP3, red: ${p3.r.toFixed(4)}, green: ${p3.g.toFixed(4)}, blue: ${p3.b.toFixed(4)})`;
  }

  // ── Scale exports ────────────────────────────────────────────────

  /** Scale (level → color string) → CSS custom-properties block.
   * Include the `--` in the prefix yourself. */
  cssVariables(scale: SemanticScale, prefix = '--color'): string {
    return Object.entries(scale)
      .sort(([a], [b]) => parseInt(a, 10) - parseInt(b, 10))
      .map(([level, value]) => `  ${prefix}-${level}: ${value};`)
      .join('\n');
  }

  /** Scale → Tailwind config-compatible object `{name: {level: value}}`. */
  tailwind(scale: SemanticScale, name: string): Record<string, SemanticScale> {
    return { [name]: { ...scale } };
  }

  /** Scale → nested object `{name: {level: {format: value}}}`.
   * Default formats: hex, oklch, p3. */
  multiFormat(
    scale: SemanticScale,
    name: string,
    formats: (TokenCssFormat | 'android')[] = ['hex', 'oklch', 'p3'],
  ): Record<string, Record<string, Record<string, string>>> {
    const result: Record<string, Record<string, string>> = {};
    for (const [level, value] of Object.entries(scale)) {
      const entry: Record<string, string> = {};
      for (const fmt of formats) {
        entry[fmt] = fmt === 'android' ? this.android(value) : this.css(value, fmt);
      }
      result[level] = entry;
    }
    return { [name]: result };
  }

  /** Multiple scales (name → scale) → multi-format JSON string. */
  json(scales: Record<string, SemanticScale>): string {
    const result: Record<string, Record<string, Record<string, string>>> = {};
    for (const [name, scale] of Object.entries(scales)) {
      Object.assign(result, this.multiFormat(scale, name));
    }
    return JSON.stringify(result, null, 2);
  }
}

// ── Internals ────────────────────────────────────────────────────────

function hsl(srgb: readonly number[]): string {
  const [r, g, b] = srgb;
  const cmax = Math.max(r, g, b);
  const cmin = Math.min(r, g, b);
  const delta = cmax - cmin;
  const l = (cmax + cmin) / 2;
  let h = 0;
  let s = 0;
  if (delta >= 1e-10) {
    s = abs(2 * l - 1) < 1 ? delta / (1 - abs(2 * l - 1)) : 1;
    if (cmax === r) h = 60 * (((g - b) / delta) % 6);
    else if (cmax === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
    h = ((h % 360) + 360) % 360;
  }
  return `hsl(${h.toFixed(0)}, ${(s * 100).toFixed(0)}%, ${(l * 100).toFixed(0)}%)`;
}
