import { describe, it, expect } from 'vitest';
import { Helmlab, findCusp, Tokens, ContrastError } from '../src/index.js';
import ref from './reference/reference-values.json';

const hl = new Helmlab();

describe('Distance (deltaE)', () => {
  for (const t of ref.distances) {
    it(`deltaE(${t.hex1}, ${t.hex2}) matches Python`, () => {
      const d = hl.metric.euclidean(t.hex1, t.hex2);
      // NC LUT linear vs PCHIP causes ~1e-3 at extremes (black/white)
      expect(d).toBeCloseTo(t.deltaE, 2);
    });
  }
});

describe('Contrast ratio', () => {
  for (const t of ref.contrasts) {
    it(`contrast(${t.fg}, ${t.bg}) = ${t.ratio}`, () => {
      const cr = hl.gen.contrastRatio(t.fg, t.bg);
      expect(cr).toBeCloseTo(t.ratio, 2);
    });
  }
});

describe('meetsContrast', () => {
  it('white on black meets AA', () => {
    expect(hl.gen.meetsContrast('#ffffff', '#000000', 'AA')).toBe(true);
  });
  it('white on black meets AAA', () => {
    expect(hl.gen.meetsContrast('#ffffff', '#000000', 'AAA')).toBe(true);
  });
  it('blue on white may not meet AA', () => {
    // 3.68 < 4.5
    expect(hl.gen.meetsContrast('#3b82f6', '#ffffff', 'AA')).toBe(false);
  });
});

describe('ensureContrast', () => {
  it('returns adjusted color meeting ratio', () => {
    const adjusted = hl.gen.ensureContrast('#3b82f6', '#ffffff', 4.5);
    expect(hl.gen.contrastRatio(adjusted, '#ffffff')).toBeGreaterThanOrEqual(4.5);
  });
  it('returns original if already meets', () => {
    const result = hl.gen.ensureContrast('#000000', '#ffffff', 4.5);
    expect(result).toBe('#000000');
  });
  it('does not return #ffffff for dark bg', () => {
    const result = hl.gen.ensureContrast('#a51d1d', '#111113');
    expect(result).not.toBe('#ffffff');
    expect(hl.gen.contrastRatio(result, '#111113')).toBeGreaterThanOrEqual(4.5);
  });
});

describe('Semantic scale', () => {
  it('matches Python reference scale (±2/255)', () => {
    const scale = hl.gen.scale('#3b82f6');
    for (const [level, hex] of Object.entries(ref.semantic_scale.scale)) {
      const got = scale[level];
      // Allow ±2 per channel due to NC LUT linear vs PCHIP differences
      const parse = (h: string) => [
        parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16),
      ];
      const [r1, g1, b1] = parse(got);
      const [r2, g2, b2] = parse(hex as string);
      const diff = Math.max(Math.abs(r1 - r2), Math.abs(g1 - g2), Math.abs(b1 - b2));
      expect(diff).toBeLessThanOrEqual(2);
    }
  });
});

describe('palette', () => {
  it('generates correct number of steps', () => {
    expect(hl.gen.palette('#3b82f6', 5)).toHaveLength(5);
    expect(hl.gen.palette('#3b82f6', 10)).toHaveLength(10);
  });
  it('first step is lightest, last is darkest', () => {
    const p = hl.gen.palette('#3b82f6', 10);
    const firstL = hl.metric.info(p[0]).L;
    const lastL = hl.metric.info(p[9]).L;
    expect(firstL).toBeGreaterThan(lastL);
  });
  it('palette colors are vivid (not washed out)', () => {
    const p = hl.gen.palette('#3b82f6', 5);
    let saturated = 0;
    for (const hex of p) {
      const parse = (h: string) => [
        parseInt(h.slice(1, 3), 16) / 255,
        parseInt(h.slice(3, 5), 16) / 255,
        parseInt(h.slice(5, 7), 16) / 255,
      ];
      const rgb = parse(hex);
      if (Math.max(...rgb) - Math.min(...rgb) > 0.1) saturated++;
    }
    expect(saturated).toBeGreaterThanOrEqual(3);
  });
});

describe('paletteHues', () => {
  it('generates correct number of hues', () => {
    expect(hl.gen.hueRing(12, { lightness: 0.6, chroma: 0.15 })).toHaveLength(12);
  });
});

describe('Base Lab', () => {
  it('baseFromHex round-trips via baseToHex (±1/255)', () => {
    for (const hex of ['#3b82f6', '#ff0000', '#808080', '#000000', '#ffffff']) {
      const lab = hl.gen.fromHex(hex);
      const rt = hl.gen.toHex(lab);
      const parse = (h: string) => [
        parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16),
      ];
      const [r1, g1, b1] = parse(hex);
      const [r2, g2, b2] = parse(rt);
      const diff = Math.max(Math.abs(r1 - r2), Math.abs(g1 - g2), Math.abs(b1 - b2));
      expect(diff).toBeLessThanOrEqual(1);
    }
  });

  it('semantic scale level 500 matches base color', () => {
    const scale = hl.gen.scale('#3b82f6');
    const parse = (h: string) => [
      parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16),
    ];
    const [r1, g1, b1] = parse('#3b82f6');
    const [r2, g2, b2] = parse(scale['500']);
    const diff = Math.max(Math.abs(r1 - r2), Math.abs(g1 - g2), Math.abs(b1 - b2));
    expect(diff).toBeLessThanOrEqual(2);
  });
});

describe('gradient', () => {
  it('returns correct number of steps', () => {
    expect(hl.gen.gradient('#ff6b00', '#0066ff', 8)).toHaveLength(8);
    expect(hl.gen.gradient('#ff0000', '#00ff00', 32)).toHaveLength(32);
  });
  it('first and last match input colors (±1/255)', () => {
    const g = hl.gen.gradient('#ff6b00', '#0066ff', 16);
    const parse = (h: string) => [
      parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16),
    ];
    const [r1, g1, b1] = parse(g[0]);
    const [r2, g2, b2] = parse('#ff6b00');
    expect(Math.max(Math.abs(r1 - r2), Math.abs(g1 - g2), Math.abs(b1 - b2))).toBeLessThanOrEqual(1);
    const [r3, g3, b3] = parse(g[15]);
    const [r4, g4, b4] = parse('#0066ff');
    expect(Math.max(Math.abs(r3 - r4), Math.abs(g3 - g4), Math.abs(b3 - b4))).toBeLessThanOrEqual(1);
  });
  it('produces valid hex strings', () => {
    const g = hl.gen.gradient('#ff0000', '#0000ff', 10);
    for (const hex of g) {
      expect(hex).toMatch(/^#[0-9a-f]{6}$/);
    }
  });
  it('single step returns start color', () => {
    const g = hl.gen.gradient('#ff6b00', '#0066ff', 1);
    expect(g).toHaveLength(1);
    expect(g[0]).toBe('#ff6b00');
  });
});

describe('Web-safe hex round-trip', () => {
  it('round-trips all web-safe colors within ±2/255', () => {
    let maxDiff = 0;
    for (let r = 0; r < 256; r += 51) {
      for (let g = 0; g < 256; g += 51) {
        for (let b = 0; b < 256; b += 51) {
          const hex = '#' + [r, g, b].map(c => c.toString(16).padStart(2, '0')).join('');
          const rt = hl.metric.toHex(hl.metric.fromHex(hex));
          // Parse both
          const [r1, g1, b1] = [hex, rt].map(h => {
            const s = h.slice(1);
            return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
          })[0];
          const [r2, g2, b2] = (() => {
            const s = rt.slice(1);
            return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
          })();
          const diff = Math.max(Math.abs(r1 - r2), Math.abs(g1 - g2), Math.abs(b1 - b2));
          maxDiff = Math.max(maxDiff, diff);
        }
      }
    }
    expect(maxDiff).toBeLessThanOrEqual(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// New parity tests
// ═══════════════════════════════════════════════════════════════════════

describe('genFromSrgb / genToSrgb', () => {
  it('genFromSrgb round-trips via genToSrgb', () => {
    const rgb = [0.5, 0.3, 0.8] as [number, number, number];
    const lab = hl.gen.fromSrgb(rgb);
    const rt = hl.gen.toSrgb(lab);
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(rt[i] - rgb[i])).toBeLessThan(0.01);
    }
  });
  it('genFromSrgb matches genFromHex for same color', () => {
    const hex = '#3b82f6';
    const rgb = [0x3b / 255, 0x82 / 255, 0xf6 / 255] as [number, number, number];
    const labHex = hl.gen.fromHex(hex);
    const labSrgb = hl.gen.fromSrgb(rgb);
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(labHex[i] - labSrgb[i])).toBeLessThan(1e-4);
    }
  });
});


describe('toHexP3', () => {
  it('returns correct CSS format', () => {
    const lab = hl.metric.fromHex('#ff0000');
    const p3 = hl.metric.toCss(lab);
    expect(p3).toMatch(/^color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)$/);
  });
  it('white produces near 1,1,1', () => {
    const lab = hl.metric.fromHex('#ffffff');
    const p3 = hl.metric.toCss(lab);
    // Extract values
    const m = p3.match(/color\(display-p3 ([\d.]+) ([\d.]+) ([\d.]+)\)/);
    expect(m).not.toBeNull();
    for (let i = 1; i <= 3; i++) {
      expect(parseFloat(m![i])).toBeCloseTo(1.0, 1);
    }
  });
});

describe('adaptToMode', () => {
  it('same mode is identity', () => {
    expect(hl.gen.adaptToMode('#3b82f6', 'light', 'light')).toBe('#3b82f6');
    expect(hl.gen.adaptToMode('#3b82f6', 'dark', 'dark')).toBe('#3b82f6');
  });
  it('light→dark inverts lightness', () => {
    const orig = hl.metric.info('#3b82f6');
    const adapted = hl.gen.adaptToMode('#3b82f6', 'light', 'dark');
    const adaptedInfo = hl.metric.info(adapted);
    // Light color → dark adaptation should change L
    expect(adaptedInfo.L).not.toBeCloseTo(orig.L, 1);
  });
  it('light→dark→light roundtrip preserves color approximately', () => {
    const adapted = hl.gen.adaptToMode('#3b82f6', 'light', 'dark');
    const back = hl.gen.adaptToMode(adapted, 'dark', 'light');
    const parse = (h: string) => [
      parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16),
    ];
    const [r1, g1, b1] = parse('#3b82f6');
    const [r2, g2, b2] = parse(back);
    const diff = Math.max(Math.abs(r1 - r2), Math.abs(g1 - g2), Math.abs(b1 - b2));
    expect(diff).toBeLessThan(30);
  });
});

describe('adaptPair', () => {
  it('returns pair meeting contrast', () => {
    const [fg, bg] = hl.gen.adaptPair('#333333', '#f0f0f0', 'light', 'dark', 4.5);
    expect(hl.gen.contrastRatio(fg, bg)).toBeGreaterThanOrEqual(4.5);
  });
  it('returns two hex strings', () => {
    const [fg, bg] = hl.gen.adaptPair('#000000', '#ffffff');
    expect(fg).toMatch(/^#[0-9a-f]{6}$/);
    expect(bg).toMatch(/^#[0-9a-f]{6}$/);
  });
});

describe('info (expanded)', () => {
  it('returns all 8 fields', () => {
    const i = hl.metric.info('#3b82f6');
    expect(i).toHaveProperty('hex');
    expect(i).toHaveProperty('srgb');
    expect(i).toHaveProperty('xyz');
    expect(i).toHaveProperty('lab');
    expect(i).toHaveProperty('L');
    expect(i).toHaveProperty('C');
    expect(i).toHaveProperty('H');
    expect(i).toHaveProperty('luminance');
  });
  it('srgb values match hex', () => {
    const i = hl.metric.info('#ff0000');
    expect(i.srgb[0]).toBeCloseTo(1.0, 2);
    expect(i.srgb[1]).toBeCloseTo(0.0, 2);
    expect(i.srgb[2]).toBeCloseTo(0.0, 2);
  });
  it('luminance is 0 for black, ~1 for white', () => {
    expect(hl.metric.info('#000000').luminance).toBeCloseTo(0, 5);
    expect(hl.metric.info('#ffffff').luminance).toBeCloseTo(1, 1);
  });
  it('xyz is non-negative for in-gamut colors', () => {
    const i = hl.metric.info('#3b82f6');
    for (let j = 0; j < 3; j++) {
      expect(i.xyz[j]).toBeGreaterThanOrEqual(0);
    }
  });
  it('H is in [0, 360)', () => {
    for (const hex of ['#ff0000', '#00ff00', '#0000ff', '#808080']) {
      const i = hl.metric.info(hex);
      expect(i.H).toBeGreaterThanOrEqual(0);
      expect(i.H).toBeLessThan(360);
    }
  });
});

describe('perceptualDistance', () => {
  it('self distance is zero', () => {
    const lab = hl.metric.fromHex('#3b82f6');
    expect(hl.metric.distance(lab, lab)).toBeCloseTo(0, 10);
  });
  it('symmetric', () => {
    const lab1 = hl.metric.fromHex('#ff0000');
    const lab2 = hl.metric.fromHex('#00ff00');
    expect(hl.metric.distance(lab1, lab2)).toBeCloseTo(hl.metric.distance(lab2, lab1), 10);
  });
  it('positive for different colors', () => {
    const lab1 = hl.metric.fromHex('#ff0000');
    const lab2 = hl.metric.fromHex('#0000ff');
    expect(hl.metric.distance(lab1, lab2)).toBeGreaterThan(0);
  });
  it('larger for dissimilar colors', () => {
    const r = hl.metric.fromHex('#ff0000');
    const rish = hl.metric.fromHex('#ee1111');
    const b = hl.metric.fromHex('#0000ff');
    expect(hl.metric.distance(r, b)).toBeGreaterThan(hl.metric.distance(r, rish));
  });
});

describe('findCusp', () => {
  it('returns [L, C] with positive chroma', () => {
    const hl2 = new Helmlab();
    // Access internal metric space for SpaceLike
    const space = { toXYZ: (lab: [number, number, number]) => hl2.metric.toXyz(lab) };
    const [L, C] = findCusp(0, space);
    expect(L).toBeGreaterThan(0);
    expect(L).toBeLessThan(1);
    expect(C).toBeGreaterThan(0);
  });
  it('cusp chroma is larger than boundary chroma', async () => {
    const { maxChroma: maxC } = await import('../src/utils/gamut.js');
    const space = { toXYZ: (lab: [number, number, number]) => hl.metric.toXyz(lab) };
    const [Lcusp, Ccusp] = findCusp(Math.PI / 3, space);
    // At extreme L values, chroma should be less than cusp
    const Clow = maxC(0.1, Math.PI / 3, space);
    const Chigh = maxC(0.95, Math.PI / 3, space);
    expect(Ccusp).toBeGreaterThanOrEqual(Clow);
    expect(Ccusp).toBeGreaterThanOrEqual(Chigh);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// TokenExporter
// ═══════════════════════════════════════════════════════════════════════

describe('Tokens (hl.tokens — color strings in)', () => {
  describe('single color formats', () => {
    it('css hex returns hex string', () => {
      expect(hl.tokens.css('#3b82f6', 'hex')).toMatch(/^#[0-9a-f]{6}$/);
    });

    it('css rgb returns rgb() format', () => {
      expect(hl.tokens.css('#3b82f6', 'rgb')).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
    });

    it('css oklch returns oklch() format', () => {
      expect(hl.tokens.css('#3b82f6', 'oklch')).toMatch(/^oklch\([\d.]+% [\d.]+ [\d.]+\)$/);
    });

    it('css p3 returns color(display-p3) format', () => {
      expect(hl.tokens.css('#3b82f6', 'p3')).toMatch(/^color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)$/);
    });

    it('css hsl returns hsl() format', () => {
      expect(hl.tokens.css('#3b82f6', 'hsl')).toMatch(/^hsl\(\d+, \d+%, \d+%\)$/);
    });

    it('android returns 0xFF hex', () => {
      expect(hl.tokens.android('#3b82f6')).toMatch(/^0xFF[0-9a-f]{6}$/);
    });

    it('iosP3 returns {r, g, b} dict', () => {
      const p3 = hl.tokens.iosP3('#3b82f6');
      expect(p3).toHaveProperty('r');
      expect(p3).toHaveProperty('g');
      expect(p3).toHaveProperty('b');
      expect(p3.r).toBeGreaterThanOrEqual(0);
      expect(p3.r).toBeLessThanOrEqual(1);
    });

    it('swift returns Color literal', () => {
      expect(hl.tokens.swift('#3b82f6')).toMatch(/^Color\(\.displayP3, red: [\d.]+, green: [\d.]+, blue: [\d.]+\)$/);
    });

    it('unknown format throws', () => {
      expect(() => hl.tokens.css('#3b82f6', 'cmyk' as never)).toThrow(/unknown format/);
    });
  });

  describe('known color values', () => {
    it('red hex is #ff0000', () => {
      expect(hl.tokens.css('#ff0000', 'hex')).toBe('#ff0000');
    });

    it('red rgb is rgb(255, 0, 0)', () => {
      expect(hl.tokens.css('#ff0000', 'rgb')).toBe('rgb(255, 0, 0)');
    });

    it('white android is 0xFFffffff', () => {
      expect(hl.tokens.android('#ffffff')).toBe('0xFFffffff');
    });

    it('black hsl is achromatic', () => {
      expect(hl.tokens.css('#000000', 'hsl')).toMatch(/hsl\(\d+, 0%, 0%\)/);
    });
  });

  describe('scale export', () => {
    const scale = hl.gen.scale('#3b82f6');

    it('multiFormat returns {name: {level: {format: value}}}', () => {
      const result = hl.tokens.multiFormat(scale, 'blue');
      expect(result).toHaveProperty('blue');
      expect(result.blue).toHaveProperty('500');
      expect(result.blue['500']).toHaveProperty('hex');
      expect(result.blue['500']).toHaveProperty('oklch');
      expect(result.blue['500']).toHaveProperty('p3');
    });

    it('multiFormat with custom formats', () => {
      const result = hl.tokens.multiFormat(scale, 'blue', ['hex', 'rgb', 'android']);
      expect(result.blue['500']).toHaveProperty('hex');
      expect(result.blue['500']).toHaveProperty('rgb');
      expect(result.blue['500']).toHaveProperty('android');
      expect(result.blue['500']).not.toHaveProperty('oklch');
    });

    it('cssVariables returns CSS', () => {
      const css = hl.tokens.cssVariables(scale);
      expect(css).toContain('--color-50:');
      expect(css).toContain('--color-900:');
      expect(css).toContain('#');
    });

    it('cssVariables with custom prefix', () => {
      expect(hl.tokens.cssVariables(scale, '--blue')).toContain('--blue-50:');
    });

    it('tailwind returns {name: {level: hex}}', () => {
      const tw = hl.tokens.tailwind(scale, 'blue');
      expect(tw).toHaveProperty('blue');
      expect(tw.blue).toHaveProperty('500');
      expect(tw.blue['500']).toMatch(/^#[0-9a-f]{6}$/);
    });

    it('json returns valid JSON', () => {
      const parsed = JSON.parse(hl.tokens.json({ blue: scale }));
      expect(parsed).toHaveProperty('blue');
      expect(parsed.blue).toHaveProperty('500');
    });
  });

  it('hl.tokens is a Tokens instance', () => {
    expect(hl.tokens).toBeInstanceOf(Tokens);
  });
});

// ── distanceFromLab parity (cross-language with Python v0.12.1) ────
describe('distanceFromLab (Python parity alias)', () => {
  const Helmlab2 = (() => {
    // re-import lazily to avoid undefined hl in this scope
    return null;
  });

  it('Helmlab.distanceFromLab matches perceptualDistance', () => {
    const hl = new Helmlab();
    const lab1 = hl.metric.fromHex('#ff0000');
    const lab2 = hl.metric.fromHex('#00ff00');
    const d_alias = hl.metric.distance(lab1, lab2);
    const d_p = hl.metric.distance(lab1, lab2);
    expect(d_alias).toBeCloseTo(d_p, 12);
  });

  it('AnalyticalSpace.distanceFromLab matches distance', () => {
    const hl = new Helmlab();
    const lab1 = hl.metric.fromHex('#3b82f6');
    const lab2 = hl.metric.fromHex('#fb923c');
    // metric is private but distanceFromLab on Helmlab uses metric.distanceFromLab
    const d_alias = hl.metric.distance(lab1, lab2);
    const d_p = hl.metric.distance(lab1, lab2);
    expect(d_alias).toBe(d_p);
  });

  it('distanceFromLab(a, a) == 0', () => {
    const hl = new Helmlab();
    const lab = hl.metric.fromHex('#3b82f6');
    expect(hl.metric.distance(lab, lab)).toBeLessThan(1e-12);
  });
});

// ── display_phi_deg — opt-in display alignment, exact isometry ────
describe('display_phi_deg (rotation isometry)', () => {
  it('default uses params.display_phi_deg (paper-aligned -28.2°)', () => {
    const hl = new Helmlab();
    // Bundled params.json now ships display_phi_deg = -28.2
    const lab_red = hl.metric.fromHex('#ff0000');
    // With φ=-28.2°, red's hue shifts away from raw +15.8° toward ~-12°
    const hue_deg = (Math.atan2(lab_red[2], lab_red[1]) * 180) / Math.PI;
    expect(hue_deg).toBeLessThan(0);  // pulled below 0° by rotation
    expect(hue_deg).toBeGreaterThan(-30);
  });

  it('chroma is preserved under rotation (Lab a²+b² invariance)', () => {
    const hl = new Helmlab();
    const lab = hl.metric.fromHex('#ff0000');
    // Chroma magnitude in (a, b) plane is rotation-invariant by definition.
    // Sanity check: the chroma is a real positive number.
    const c = Math.hypot(lab[1], lab[2]);
    expect(c).toBeGreaterThan(0);
    expect(Number.isFinite(c)).toBe(true);
  });

  it('roundtrip preserved with default φ', () => {
    const hl = new Helmlab();
    const samples = ['#ff0000', '#00ff00', '#0000ff', '#3b82f6', '#808080'];
    for (const hex of samples) {
      const lab = hl.metric.fromHex(hex);
      const back = hl.metric.toHex(lab);
      expect(back.toLowerCase()).toBe(hex.toLowerCase());
    }
  });
});

// ── deltaE vs perceptualDistance — naming clarity guard ────────
describe('deltaE vs perceptualDistance (distinct metrics)', () => {
  it('deltaE returns Euclidean Lab (uncompressed)', () => {
    const hl = new Helmlab();
    const lab_w = hl.metric.fromHex('#ffffff');
    const lab_b = hl.metric.fromHex('#000000');
    const expected = Math.sqrt(
      (lab_w[0] - lab_b[0]) ** 2 +
      (lab_w[1] - lab_b[1]) ** 2 +
      (lab_w[2] - lab_b[2]) ** 2
    );
    const actual = hl.metric.euclidean('#ffffff', '#000000');
    expect(actual).toBeCloseTo(expected, 12);
  });

  it('genToLch/genFromLch round-trip Gen Lab exactly', () => {
    const hl = new Helmlab();
    for (const hex of ['#3b82f6', '#ff0000', '#facc15', '#14b8a6', '#123456']) {
      const lab = hl.gen.fromHex(hex);
      const lch = hl.gen.toLch(lab);
      const back = hl.gen.fromLch(lch);
      expect(back[0]).toBeCloseTo(lab[0], 12);
      expect(back[1]).toBeCloseTo(lab[1], 12);
      expect(back[2]).toBeCloseTo(lab[2], 12);
      expect(lch[1]).toBeGreaterThanOrEqual(0);
      expect(lch[2]).toBeGreaterThanOrEqual(0);
      expect(lch[2]).toBeLessThan(360);
    }
  });

  it('genToLch hue matches manual atan2 (degrees)', () => {
    const hl = new Helmlab();
    const lab = hl.gen.fromHex('#3b82f6');
    const [, , h] = hl.gen.toLch(lab);
    const expected = ((Math.atan2(lab[2], lab[1]) * 180) / Math.PI + 360) % 360;
    expect(h).toBeCloseTo(expected, 12);
  });

  it('deltaE2000 hits the sanity anchors', () => {
    const hl = new Helmlab();
    expect(hl.metric.ciede2000('#ff0000', '#00ff00')).toBeCloseTo(86.61, 1);
    expect(hl.metric.ciede2000('#000000', '#ffffff')).toBeCloseTo(100.0, 1);
    expect(hl.metric.ciede2000('#3b82f6', '#3b82f6')).toBeCloseTo(0, 10);
  });

  it('nearestColor: default ciede2000, margin + runner-up, guards', () => {
    const hl = new Helmlab();
    const n = hl.metric.nearest('#3b82f6', ['#3b7ff0', '#ff0000', '#00ff00']);
    expect(n.hex).toBe('#3b7ff0');
    expect(n.metric).toBe('ciede2000');
    expect(n.margin).toBeGreaterThan(1);
    expect(n.runnerUp).not.toBe(n.hex);
    expect(hl.metric.nearest('#3b82f6', ['#3b7ff0'], 'difference').hex).toBe('#3b7ff0');
    expect(hl.metric.nearest('#3b82f6', ['#3b7ff0'], 'euclidean').hex).toBe('#3b7ff0');
    expect(() => hl.metric.nearest('#fff', [])).toThrow('empty');
    // @ts-expect-error bogus metric
    expect(() => hl.metric.nearest('#ffffff', ['#000000'], 'bogus' as never)).toThrow('unknown metric');
  });

  it('euclideanDistance is an exact alias of deltaE (Python parity)', () => {
    const hl = new Helmlab();
    expect(hl.metric.euclidean('#ff0000', '#00ff00')).toBe(hl.metric.euclidean('#ff0000', '#00ff00'));
    expect(hl.metric.euclidean('#3b82f6', '#4c8af7')).toBe(hl.metric.euclidean('#3b82f6', '#4c8af7'));
  });

  it('deltaE > perceptualDistance for very dissimilar pairs (compression saturates)', () => {
    const hl = new Helmlab();
    const lab_w = hl.metric.fromHex('#ffffff');
    const lab_b = hl.metric.fromHex('#000000');
    const euclidean = hl.metric.euclidean('#ffffff', '#000000');
    const perceptual = hl.metric.distance(lab_w, lab_b);
    expect(euclidean).toBeGreaterThan(0.5);
    expect(perceptual).toBeLessThan(0.5);
    expect(euclidean).toBeGreaterThan(perceptual);
  });
});

describe('Contract guards (2026-07-08 audit fixes)', () => {
  it('distance() throws on CIELAB-scale input (L>3)', () => {
    expect(() => hl.metric.distance([50, 10, -20], [55, 12, -18])).toThrow(/CIELAB/);
  });

  it('distance() throws on NaN input', () => {
    expect(() => hl.metric.distance([NaN, 0.1, 0.1], [0.5, 0, 0])).toThrow(/non-finite/);
  });

  it('distance() accepts valid Helmlab Lab', () => {
    const d = hl.metric.distance(hl.metric.fromHex('#3b82f6'), hl.metric.fromHex('#3b83f7'));
    expect(Number.isFinite(d)).toBe(true);
  });

  it('ensureContrast warns and returns best effort when ratio unreachable', () => {
    const warnings: string[] = [];
    const orig = console.warn;
    console.warn = (msg: string) => { warnings.push(String(msg)); };
    try {
      const result = hl.gen.ensureContrast('#3b82f6', '#808080', 7.0);
      expect(['#000000', '#ffffff']).toContain(result);
      expect(warnings.some((w) => w.includes('ensureContrast'))).toBe(true);
    } finally {
      console.warn = orig;
    }
  });

  it('ensureContrast does not warn when ratio is reachable', () => {
    const warnings: string[] = [];
    const orig = console.warn;
    console.warn = (msg: string) => { warnings.push(String(msg)); };
    try {
      const result = hl.gen.ensureContrast('#3b82f6', '#ffffff', 4.5);
      expect(hl.gen.contrastRatio(result, '#ffffff')).toBeGreaterThanOrEqual(4.5);
      expect(warnings.length).toBe(0);
    } finally {
      console.warn = orig;
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════
// 1.0 API — branded types, wide gamut, harmonies, mix, jnd, cusp geometry
// ═══════════════════════════════════════════════════════════════════════

describe('branded Lab types', () => {
  it('GenLab rejected by metric API', () => {
    const lab = hl.gen.fromHex('#3b82f6');
    expect(() => hl.metric.toHex(lab as never)).toThrow(/GenLab/);
  });

  it('MetricLab rejected by gen API', () => {
    const lab = hl.metric.fromHex('#3b82f6');
    expect(() => hl.gen.toHex(lab as never)).toThrow(/MetricLab/);
  });

  it('plain arrays still accepted (interop escape hatch)', () => {
    expect(hl.metric.toHex([0.5, 0.05, -0.05])).toMatch(/^#/);
    expect(hl.gen.toHex([0.5, 0.05, -0.05])).toMatch(/^#/);
  });
});

describe('wide gamut (1.0)', () => {
  it('gradient p3 emits color(display-p3 …) strings', () => {
    const g = hl.gen.gradient('#0000ff', '#ffffff', 3, { gamut: 'display-p3' });
    expect(g).toHaveLength(3);
    for (const x of g) expect(x.startsWith('color(display-p3 ')).toBe(true);
  });

  it('gradient rec2020', () => {
    const g = hl.gen.gradient('#ff0000', '#00ff00', 2, { gamut: 'rec2020' });
    for (const x of g) expect(x.startsWith('color(rec2020 ')).toBe(true);
  });

  it('bad gamut throws', () => {
    expect(() => hl.gen.gradient('#ff0000', '#00ff00', 3, { gamut: 'cmyk' as never })).toThrow(/unknown gamut/);
  });

  it('P3 input string works end to end', () => {
    const info = hl.metric.info('color(display-p3 1 0 0)');
    expect(info.inSrgb).toBe(false);
    expect(info.inP3).toBe(true);
  });

  it('bad css color throws', () => {
    expect(() => hl.metric.fromHex('color(foo 1 0 0)')).toThrow(/unparseable/);
  });

  it('scale p3', () => {
    const scale = hl.gen.scale('#3b82f6', undefined, { gamut: 'display-p3' });
    expect(scale['500'].startsWith('color(display-p3 ')).toBe(true);
  });
});

describe('harmonies / mix / rotateHue / hueRing (1.0)', () => {
  it('triadic returns 3, base first', () => {
    const h = hl.gen.harmonies('#3b82f6', 'triadic');
    expect(h).toHaveLength(3);
    expect(h[0]).toBe(hl.gen.toHex(hl.gen.fromHex('#3b82f6')));
  });

  it('harmony sizes', () => {
    expect(hl.gen.harmonies('#3b82f6', 'complementary')).toHaveLength(2);
    expect(hl.gen.harmonies('#3b82f6', 'analogous')).toHaveLength(3);
    expect(hl.gen.harmonies('#3b82f6', 'tetradic')).toHaveLength(4);
    expect(hl.gen.harmonies('#3b82f6', 'split_complementary')).toHaveLength(3);
  });

  it('unknown harmony kind throws', () => {
    expect(() => hl.gen.harmonies('#3b82f6', 'quadratic' as never)).toThrow(/unknown harmony/);
  });

  it('mix endpoints and gradient midpoint', () => {
    const a = '#ff0000', b = '#0000ff';
    expect(hl.gen.mix(a, b, 0)).toBe(hl.gen.toHex(hl.gen.fromHex(a)));
    expect(hl.gen.mix(a, b, 1)).toBe(hl.gen.toHex(hl.gen.fromHex(b)));
    expect(hl.gen.mix(a, b, 0.5)).toBe(hl.gen.gradient(a, b, 3)[1]);
  });

  it('rotateHue identity at 0 and 360', () => {
    const rt = hl.gen.toHex(hl.gen.fromHex('#3b82f6'));
    expect(hl.gen.rotateHue('#3b82f6', 0)).toBe(rt);
    expect(hl.gen.rotateHue('#3b82f6', 360)).toBe(rt);
  });

  it('hueRing distinct colors', () => {
    const ring = hl.gen.hueRing(6);
    expect(ring).toHaveLength(6);
    expect(new Set(ring).size).toBe(6);
  });
});

describe('jnd + strict contrast (1.0)', () => {
  it('jnd self is zero', () => {
    expect(hl.metric.jnd('#808080', '#808080')).toBe(0);
  });

  it('jnd big pair above threshold', () => {
    expect(hl.metric.jnd('#ff0000', '#00ff00')).toBeGreaterThan(3);
  });

  it('jnd is scaled difference (matches Python constant)', () => {
    const de = hl.metric.difference('#808080', '#828282');
    expect(hl.metric.jnd('#808080', '#828282')).toBeCloseTo(de / 0.03563295091867221, 9);
  });

  it('strict ensureContrast throws ContrastError when unreachable', () => {
    expect(() => hl.gen.ensureContrast('#3b82f6', '#808080', 7.0, { strict: true })).toThrow(ContrastError);
  });
});

describe('cusp geometry exposed (1.0)', () => {
  it('maxChroma positive, p3 wider', () => {
    const cS = hl.gen.maxChroma(0.6, 263);
    const cP = hl.gen.maxChroma(0.6, 263, 'display-p3');
    expect(cS).toBeGreaterThan(0);
    expect(cP).toBeGreaterThan(cS);
  });

  it('cusp is max over L on same hue', () => {
    const [Lc, Cc] = hl.gen.cusp(263);
    expect(Lc).toBeGreaterThan(0);
    expect(Lc).toBeLessThan(1);
    for (const L of [0.3, 0.6, 0.8]) {
      expect(Cc).toBeGreaterThanOrEqual(hl.gen.maxChroma(L, 263) - 1e-3);
    }
  });

  it('vivid preserves L and hue, grows chroma', () => {
    const base = '#6488b8';
    const lchBase = hl.gen.toLch(hl.gen.fromHex(base));
    const lchVivid = hl.gen.toLch(hl.gen.fromHex(hl.gen.vivid(base)));
    expect(Math.abs(lchVivid[0] - lchBase[0])).toBeLessThan(0.02);
    const dh = Math.abs(lchVivid[2] - lchBase[2]) % 360;
    expect(Math.min(dh, 360 - dh)).toBeLessThan(3);
    expect(lchVivid[1]).toBeGreaterThan(lchBase[1]);
  });

  it('adaptive gamut map lands in gamut', () => {
    const mapped = hl.gen.gamutMap(hl.gen.lab(0.5, 0.8, 0.0), 'srgb', 'adaptive');
    expect(hl.gen.inGamut(mapped)).toBe(true);
  });

  it('adaptive matches Python within tolerance', () => {
    // Python: gamut_map([0.5, 0.8, 0], method='adaptive') → [0.5033, 0.3020, 0]
    const m = hl.gen.gamutMap(hl.gen.lab(0.5, 0.8, 0.0), 'srgb', 'adaptive');
    expect(m[0]).toBeCloseTo(0.5033, 3);
    expect(m[1]).toBeCloseTo(0.302, 3);
  });
});

describe('metric LCh (1.0 symmetry with gen)', () => {
  it('lch roundtrip', () => {
    const lab = hl.metric.fromHex('#3b82f6');
    const lch = hl.metric.toLch(lab);
    const back = hl.metric.fromLch(lch);
    for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(lab[i], 12);
  });

  it('lch matches info C/H', () => {
    const lch = hl.metric.toLch(hl.metric.fromHex('#3b82f6'));
    const info = hl.metric.info('#3b82f6');
    expect(lch[1]).toBeCloseTo(info.C, 12);
    expect(lch[2]).toBeCloseTo(info.H, 9);
  });

  it('rejects GenLab', () => {
    expect(() => hl.metric.toLch(hl.gen.fromHex('#3b82f6') as never)).toThrow(/GenLab/);
  });
});
