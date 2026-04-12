import { describe, it, expect } from 'vitest';
import { Helmlab, findCusp, TokenExporter } from '../src/index.js';
import ref from './reference/reference-values.json';

const hl = new Helmlab();

describe('Distance (deltaE)', () => {
  for (const t of ref.distances) {
    it(`deltaE(${t.hex1}, ${t.hex2}) matches Python`, () => {
      const d = hl.deltaE(t.hex1, t.hex2);
      // NC LUT linear vs PCHIP causes ~1e-3 at extremes (black/white)
      expect(d).toBeCloseTo(t.deltaE, 2);
    });
  }
});

describe('Contrast ratio', () => {
  for (const t of ref.contrasts) {
    it(`contrast(${t.fg}, ${t.bg}) = ${t.ratio}`, () => {
      const cr = hl.contrastRatio(t.fg, t.bg);
      expect(cr).toBeCloseTo(t.ratio, 2);
    });
  }
});

describe('meetsContrast', () => {
  it('white on black meets AA', () => {
    expect(hl.meetsContrast('#ffffff', '#000000', 'AA')).toBe(true);
  });
  it('white on black meets AAA', () => {
    expect(hl.meetsContrast('#ffffff', '#000000', 'AAA')).toBe(true);
  });
  it('blue on white may not meet AA', () => {
    // 3.68 < 4.5
    expect(hl.meetsContrast('#3b82f6', '#ffffff', 'AA')).toBe(false);
  });
});

describe('ensureContrast', () => {
  it('returns adjusted color meeting ratio', () => {
    const adjusted = hl.ensureContrast('#3b82f6', '#ffffff', 4.5);
    expect(hl.contrastRatio(adjusted, '#ffffff')).toBeGreaterThanOrEqual(4.5);
  });
  it('returns original if already meets', () => {
    const result = hl.ensureContrast('#000000', '#ffffff', 4.5);
    expect(result).toBe('#000000');
  });
  it('does not return #ffffff for dark bg', () => {
    const result = hl.ensureContrast('#a51d1d', '#111113');
    expect(result).not.toBe('#ffffff');
    expect(hl.contrastRatio(result, '#111113')).toBeGreaterThanOrEqual(4.5);
  });
});

describe('Semantic scale', () => {
  it('matches Python reference scale (±2/255)', () => {
    const scale = hl.semanticScale('#3b82f6');
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
    expect(hl.palette('#3b82f6', 5)).toHaveLength(5);
    expect(hl.palette('#3b82f6', 10)).toHaveLength(10);
  });
  it('first step is lightest, last is darkest', () => {
    const p = hl.palette('#3b82f6', 10);
    const firstL = hl.info(p[0]).L;
    const lastL = hl.info(p[9]).L;
    expect(firstL).toBeGreaterThan(lastL);
  });
  it('palette colors are vivid (not washed out)', () => {
    const p = hl.palette('#3b82f6', 5);
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
    expect(hl.paletteHues(0.6, 0.15, 12)).toHaveLength(12);
  });
});

describe('Base Lab', () => {
  it('baseFromHex round-trips via baseToHex (±1/255)', () => {
    for (const hex of ['#3b82f6', '#ff0000', '#808080', '#000000', '#ffffff']) {
      const lab = hl.baseFromHex(hex);
      const rt = hl.baseToHex(lab);
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
    const scale = hl.semanticScale('#3b82f6');
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
    expect(hl.gradient('#ff6b00', '#0066ff', 8)).toHaveLength(8);
    expect(hl.gradient('#ff0000', '#00ff00', 32)).toHaveLength(32);
  });
  it('first and last match input colors (±1/255)', () => {
    const g = hl.gradient('#ff6b00', '#0066ff', 16);
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
    const g = hl.gradient('#ff0000', '#0000ff', 10);
    for (const hex of g) {
      expect(hex).toMatch(/^#[0-9a-f]{6}$/);
    }
  });
  it('single step returns start color', () => {
    const g = hl.gradient('#ff6b00', '#0066ff', 1);
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
          const rt = hl.toHex(hl.fromHex(hex));
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
    const lab = hl.genFromSrgb(rgb);
    const rt = hl.genToSrgb(lab);
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(rt[i] - rgb[i])).toBeLessThan(0.01);
    }
  });
  it('genFromSrgb matches genFromHex for same color', () => {
    const hex = '#3b82f6';
    const rgb = [0x3b / 255, 0x82 / 255, 0xf6 / 255] as [number, number, number];
    const labHex = hl.genFromHex(hex);
    const labSrgb = hl.genFromSrgb(rgb);
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(labHex[i] - labSrgb[i])).toBeLessThan(1e-4);
    }
  });
});

describe('baseFromSrgb / baseToSrgb (deprecated aliases)', () => {
  it('baseFromSrgb matches genFromSrgb', () => {
    const rgb = [0.4, 0.6, 0.2] as [number, number, number];
    const lab1 = hl.baseFromSrgb(rgb);
    const lab2 = hl.genFromSrgb(rgb);
    expect(lab1).toEqual(lab2);
  });
  it('baseToSrgb matches genToSrgb', () => {
    const lab = hl.genFromHex('#ff6b00');
    const s1 = hl.baseToSrgb(lab);
    const s2 = hl.genToSrgb(lab);
    expect(s1).toEqual(s2);
  });
});

describe('toHexP3', () => {
  it('returns correct CSS format', () => {
    const lab = hl.fromHex('#ff0000');
    const p3 = hl.toHexP3(lab);
    expect(p3).toMatch(/^color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)$/);
  });
  it('white produces near 1,1,1', () => {
    const lab = hl.fromHex('#ffffff');
    const p3 = hl.toHexP3(lab);
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
    expect(hl.adaptToMode('#3b82f6', 'light', 'light')).toBe('#3b82f6');
    expect(hl.adaptToMode('#3b82f6', 'dark', 'dark')).toBe('#3b82f6');
  });
  it('light→dark inverts lightness', () => {
    const orig = hl.info('#3b82f6');
    const adapted = hl.adaptToMode('#3b82f6', 'light', 'dark');
    const adaptedInfo = hl.info(adapted);
    // Light color → dark adaptation should change L
    expect(adaptedInfo.L).not.toBeCloseTo(orig.L, 1);
  });
  it('light→dark→light roundtrip preserves color approximately', () => {
    const adapted = hl.adaptToMode('#3b82f6', 'light', 'dark');
    const back = hl.adaptToMode(adapted, 'dark', 'light');
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
    const [fg, bg] = hl.adaptPair('#333333', '#f0f0f0', 'light', 'dark', 4.5);
    expect(hl.contrastRatio(fg, bg)).toBeGreaterThanOrEqual(4.5);
  });
  it('returns two hex strings', () => {
    const [fg, bg] = hl.adaptPair('#000000', '#ffffff');
    expect(fg).toMatch(/^#[0-9a-f]{6}$/);
    expect(bg).toMatch(/^#[0-9a-f]{6}$/);
  });
});

describe('info (expanded)', () => {
  it('returns all 8 fields', () => {
    const i = hl.info('#3b82f6');
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
    const i = hl.info('#ff0000');
    expect(i.srgb[0]).toBeCloseTo(1.0, 2);
    expect(i.srgb[1]).toBeCloseTo(0.0, 2);
    expect(i.srgb[2]).toBeCloseTo(0.0, 2);
  });
  it('luminance is 0 for black, ~1 for white', () => {
    expect(hl.info('#000000').luminance).toBeCloseTo(0, 5);
    expect(hl.info('#ffffff').luminance).toBeCloseTo(1, 1);
  });
  it('xyz is non-negative for in-gamut colors', () => {
    const i = hl.info('#3b82f6');
    for (let j = 0; j < 3; j++) {
      expect(i.xyz[j]).toBeGreaterThanOrEqual(0);
    }
  });
  it('H is in [0, 360)', () => {
    for (const hex of ['#ff0000', '#00ff00', '#0000ff', '#808080']) {
      const i = hl.info(hex);
      expect(i.H).toBeGreaterThanOrEqual(0);
      expect(i.H).toBeLessThan(360);
    }
  });
});

describe('perceptualDistance', () => {
  it('self distance is zero', () => {
    const lab = hl.fromHex('#3b82f6');
    expect(hl.perceptualDistance(lab, lab)).toBeCloseTo(0, 10);
  });
  it('symmetric', () => {
    const lab1 = hl.fromHex('#ff0000');
    const lab2 = hl.fromHex('#00ff00');
    expect(hl.perceptualDistance(lab1, lab2)).toBeCloseTo(hl.perceptualDistance(lab2, lab1), 10);
  });
  it('positive for different colors', () => {
    const lab1 = hl.fromHex('#ff0000');
    const lab2 = hl.fromHex('#0000ff');
    expect(hl.perceptualDistance(lab1, lab2)).toBeGreaterThan(0);
  });
  it('larger for dissimilar colors', () => {
    const r = hl.fromHex('#ff0000');
    const rish = hl.fromHex('#ee1111');
    const b = hl.fromHex('#0000ff');
    expect(hl.perceptualDistance(r, b)).toBeGreaterThan(hl.perceptualDistance(r, rish));
  });
});

describe('findCusp', () => {
  it('returns [L, C] with positive chroma', () => {
    const hl2 = new Helmlab();
    // Access internal metric space for SpaceLike
    const space = { toXYZ: (lab: [number, number, number]) => hl2.toXYZ(lab) };
    const [L, C] = findCusp(0, space);
    expect(L).toBeGreaterThan(0);
    expect(L).toBeLessThan(1);
    expect(C).toBeGreaterThan(0);
  });
  it('cusp chroma is larger than boundary chroma', async () => {
    const { maxChroma: maxC } = await import('../src/utils/gamut.js');
    const space = { toXYZ: (lab: [number, number, number]) => hl.toXYZ(lab) };
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

describe('TokenExporter', () => {
  const exp = hl.export();

  describe('single color formats', () => {
    const lab = hl.fromHex('#3b82f6');

    it('toCssHex returns hex string', () => {
      const hex = exp.toCssHex(lab);
      expect(hex).toMatch(/^#[0-9a-f]{6}$/);
    });

    it('toCssRgb returns rgb() format', () => {
      const rgb = exp.toCssRgb(lab);
      expect(rgb).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
    });

    it('toCssOklch returns oklch() format', () => {
      const oklch = exp.toCssOklch(lab);
      expect(oklch).toMatch(/^oklch\([\d.]+% [\d.]+ [\d.]+\)$/);
    });

    it('toCssDisplayP3 returns color(display-p3) format', () => {
      const p3 = exp.toCssDisplayP3(lab);
      expect(p3).toMatch(/^color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)$/);
    });

    it('toCssHsl returns hsl() format', () => {
      const hsl = exp.toCssHsl(lab);
      expect(hsl).toMatch(/^hsl\(\d+, \d+%, \d+%\)$/);
    });

    it('toAndroidArgb returns 0xFF hex', () => {
      const argb = exp.toAndroidArgb(lab);
      expect(argb).toMatch(/^0xFF[0-9a-f]{6}$/);
    });

    it('toIosP3 returns {r, g, b} dict', () => {
      const p3 = exp.toIosP3(lab);
      expect(p3).toHaveProperty('r');
      expect(p3).toHaveProperty('g');
      expect(p3).toHaveProperty('b');
      expect(p3.r).toBeGreaterThanOrEqual(0);
      expect(p3.r).toBeLessThanOrEqual(1);
    });

    it('toSwiftLiteral returns Color literal', () => {
      const swift = exp.toSwiftLiteral(lab);
      expect(swift).toMatch(/^Color\(\.displayP3, red: [\d.]+, green: [\d.]+, blue: [\d.]+\)$/);
    });
  });

  describe('known color values', () => {
    it('red hex is #ff0000', () => {
      const lab = hl.fromHex('#ff0000');
      expect(exp.toCssHex(lab)).toBe('#ff0000');
    });

    it('red rgb is rgb(255, 0, 0)', () => {
      const lab = hl.fromHex('#ff0000');
      expect(exp.toCssRgb(lab)).toBe('rgb(255, 0, 0)');
    });

    it('white android is 0xFFffffff', () => {
      const lab = hl.fromHex('#ffffff');
      expect(exp.toAndroidArgb(lab)).toBe('0xFFffffff');
    });

    it('black hsl is achromatic', () => {
      const lab = hl.fromHex('#000000');
      const hsl = exp.toCssHsl(lab);
      expect(hsl).toMatch(/hsl\(\d+, 0%, 0%\)/);
    });
  });

  describe('scale export', () => {
    const scale = hl.semanticScale('#3b82f6');

    it('exportScale returns {name: {level: {format: value}}}', () => {
      const result = exp.exportScale(scale, 'blue');
      expect(result).toHaveProperty('blue');
      expect(result.blue).toHaveProperty('500');
      expect(result.blue['500']).toHaveProperty('hex');
      expect(result.blue['500']).toHaveProperty('oklch');
      expect(result.blue['500']).toHaveProperty('p3');
    });

    it('exportScale with custom formats', () => {
      const result = exp.exportScale(scale, 'blue', ['hex', 'rgb', 'android']);
      expect(result.blue['500']).toHaveProperty('hex');
      expect(result.blue['500']).toHaveProperty('rgb');
      expect(result.blue['500']).toHaveProperty('android');
      expect(result.blue['500']).not.toHaveProperty('oklch');
    });

    it('exportCssCustomProperties returns CSS', () => {
      const css = exp.exportCssCustomProperties(scale);
      expect(css).toContain('--color-50:');
      expect(css).toContain('--color-900:');
      expect(css).toContain('#');
    });

    it('exportCssCustomProperties with custom prefix', () => {
      const css = exp.exportCssCustomProperties(scale, '--blue');
      expect(css).toContain('--blue-50:');
    });

    it('exportTailwind returns {name: {level: hex}}', () => {
      const tw = exp.exportTailwind(scale, 'blue');
      expect(tw).toHaveProperty('blue');
      expect(tw.blue).toHaveProperty('500');
      expect(tw.blue['500']).toMatch(/^#[0-9a-f]{6}$/);
    });

    it('exportJson returns valid JSON', () => {
      const json = exp.exportJson({ blue: scale });
      const parsed = JSON.parse(json);
      expect(parsed).toHaveProperty('blue');
      expect(parsed.blue).toHaveProperty('500');
    });
  });

  it('export() returns TokenExporter instance', () => {
    const exp2 = hl.export();
    expect(exp2).toBeInstanceOf(TokenExporter);
  });
});
