/** Design token export — CSS, Android, iOS, Tailwind formats.
 *
 * Converts Helmlab Lab colors to platform-specific token strings.
 * Includes Oklab/oklch conversion for CSS oklch() output.
 */

import type { Lab, Hex, RGB, XYZ } from './types.js';

const { sqrt, atan2, max, min, abs, round, sign, pow, PI } = Math;
const cbrt = Math.cbrt;

// ── Oklab matrices (Björn Ottosson) ─────────────────────────────────

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

function oklabToOklch(lab: [number, number, number]): [number, number, number] {
  const [L, a, b] = lab;
  const C = sqrt(a * a + b * b);
  const H = ((atan2(b, a) * 180 / PI) % 360 + 360) % 360;
  return [L, C, H];
}

// ── Helmlab-like interface ──────────────────────────────────────────

interface HelmlabLike {
  toHex(lab: Lab): Hex;
  toSrgb(lab: Lab): RGB;
  toXYZ(lab: Lab): XYZ;
  toDisplayP3(lab: Lab): RGB;
  toHexP3(lab: Lab): string;
  fromHex(hex: Hex): Lab;
}

// ── TokenExporter ───────────────────────────────────────────────────

export class TokenExporter {
  private readonly hl: HelmlabLike;

  constructor(helmlab: HelmlabLike) {
    this.hl = helmlab;
  }

  /** Accept a hex string anywhere a Metric Lab is expected — hex is
   *  unambiguous, so the Gen-Lab-into-the-exporter footgun can't happen. */
  private coerce(c: Lab | Hex): Lab {
    return typeof c === 'string' ? this.hl.fromHex(c) : c;
  }

  // ── Single color formats ──────────────────────────────────────

  /** Helmlab Lab → '#rrggbb'. */
  toCssHex(color: Lab | Hex): string {
    const lab = this.coerce(color);
    return this.hl.toHex(lab);
  }

  /** Helmlab Lab → 'rgb(r, g, b)'. */
  toCssRgb(color: Lab | Hex): string {
    const lab = this.coerce(color);
    const srgb = this.hl.toSrgb(lab);
    const r = round(min(max(srgb[0] * 255, 0), 255));
    const g = round(min(max(srgb[1] * 255, 0), 255));
    const b = round(min(max(srgb[2] * 255, 0), 255));
    return `rgb(${r}, ${g}, ${b})`;
  }

  /** Helmlab Lab → 'oklch(L% C H)' via XYZ → Oklab → oklch. */
  toCssOklch(color: Lab | Hex): string {
    const lab = this.coerce(color);
    const xyz = this.hl.toXYZ(lab);
    const oklab = xyzToOklab(xyz);
    const [L, C, H] = oklabToOklch(oklab);
    return `oklch(${(L * 100).toFixed(1)}% ${C.toFixed(4)} ${H.toFixed(1)})`;
  }

  /** Helmlab Lab → 'color(display-p3 r g b)'. */
  toCssDisplayP3(color: Lab | Hex): string {
    const lab = this.coerce(color);
    return this.hl.toHexP3(lab);
  }

  /** Helmlab Lab → 'hsl(H, S%, L%)'. */
  toCssHsl(color: Lab | Hex): string {
    const lab = this.coerce(color);
    const srgb = this.hl.toSrgb(lab);
    const r = srgb[0], g = srgb[1], b = srgb[2];
    const cmax = max(r, g, b);
    const cmin = min(r, g, b);
    const delta = cmax - cmin;
    const l = (cmax + cmin) / 2;
    let h = 0, s = 0;
    if (delta >= 1e-10) {
      s = abs(2 * l - 1) < 1 ? delta / (1 - abs(2 * l - 1)) : 1;
      if (cmax === r) {
        h = 60 * (((g - b) / delta) % 6);
      } else if (cmax === g) {
        h = 60 * ((b - r) / delta + 2);
      } else {
        h = 60 * ((r - g) / delta + 4);
      }
      h = ((h % 360) + 360) % 360;
    }
    return `hsl(${round(h)}, ${round(s * 100)}%, ${round(l * 100)}%)`;
  }

  // ── Platform-specific ─────────────────────────────────────────

  /** Helmlab Lab → '0xFFrrggbb' (Android ARGB int). */
  toAndroidArgb(color: Lab | Hex): string {
    const lab = this.coerce(color);
    const srgb = this.hl.toSrgb(lab);
    const r = round(min(max(srgb[0] * 255, 0), 255));
    const g = round(min(max(srgb[1] * 255, 0), 255));
    const b = round(min(max(srgb[2] * 255, 0), 255));
    return `0xFF${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  }

  /** Helmlab Lab → {r, g, b} (UIColor Display P3). */
  toIosP3(color: Lab | Hex): { r: number; g: number; b: number } {
    const lab = this.coerce(color);
    const p3 = this.hl.toDisplayP3(lab);
    return {
      r: parseFloat(p3[0].toFixed(4)),
      g: parseFloat(p3[1].toFixed(4)),
      b: parseFloat(p3[2].toFixed(4)),
    };
  }

  /** Helmlab Lab → Swift Color literal with Display P3. */
  toSwiftLiteral(color: Lab | Hex): string {
    const lab = this.coerce(color);
    const p3 = this.hl.toDisplayP3(lab);
    return `Color(.displayP3, red: ${p3[0].toFixed(4)}, green: ${p3[1].toFixed(4)}, blue: ${p3[2].toFixed(4)})`;
  }

  // ── Scale/palette export ──────────────────────────────────────

  /** Export a semantic scale to multiple formats. */
  exportScale(
    scale: Record<string, Hex>,
    name: string,
    formats: string[] = ['hex', 'oklch', 'p3'],
  ): Record<string, Record<string, Record<string, string>>> {
    const formatFns: Record<string, (lab: Lab) => string> = {
      hex: (l) => this.toCssHex(l),
      rgb: (l) => this.toCssRgb(l),
      oklch: (l) => this.toCssOklch(l),
      p3: (l) => this.toCssDisplayP3(l),
      hsl: (l) => this.toCssHsl(l),
      android: (l) => this.toAndroidArgb(l),
    };

    const result: Record<string, Record<string, string>> = {};
    for (const [level, hex] of Object.entries(scale)) {
      const lab = this.hl.fromHex(hex);
      const levelData: Record<string, string> = {};
      for (const fmt of formats) {
        if (fmt in formatFns) {
          levelData[fmt] = formatFns[fmt](lab);
        }
      }
      result[level] = levelData;
    }

    return { [name]: result };
  }

  /** Export scale as CSS custom properties. */
  exportCssCustomProperties(scale: Record<string, Hex>, prefix = '--color'): string {
    const sorted = Object.entries(scale).sort((a, b) => parseInt(a[0]) - parseInt(b[0]));
    return sorted.map(([level, hex]) => `  ${prefix}-${level}: ${hex};`).join('\n');
  }

  /** Export scale as Tailwind config-compatible dict. */
  exportTailwind(scale: Record<string, Hex>, name: string): Record<string, Record<string, Hex>> {
    return { [name]: { ...scale } };
  }

  /** Export multiple scales as multi-format JSON. */
  exportJson(scales: Record<string, Record<string, Hex>>): string {
    const result: Record<string, Record<string, Record<string, string>>> = {};
    for (const [name, scale] of Object.entries(scales)) {
      const exported = this.exportScale(scale, name);
      Object.assign(result, exported);
    }
    return JSON.stringify(result, null, 2);
  }
}
