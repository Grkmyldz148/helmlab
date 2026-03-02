import { describe, it, expect } from 'vitest';
import { Helmlab } from '../src/index.js';
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
