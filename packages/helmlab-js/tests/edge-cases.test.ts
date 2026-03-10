/**
 * Comprehensive edge-case tests for MetricSpace (via Helmlab) and GenSpace.
 *
 * Covers: black point, near-black, achromatic axis, white point,
 * NC LUT boundaries, extreme inputs, and round-trip stress tests.
 */
import { describe, it, expect } from 'vitest';
import { Helmlab, GenSpace, compileGenParams, getDefaultGenParams, hexToSrgb, srgbToXyz, xyzToSrgb, displayP3ToXyz } from '../src/index.js';
import type { XYZ, Lab } from '../src/index.js';

const hl = new Helmlab();
const gen = new GenSpace(compileGenParams(getDefaultGenParams()));

const D65: XYZ = [0.95047, 1.0, 1.08883];

function chroma(lab: Lab): number {
  return Math.sqrt(lab[1] * lab[1] + lab[2] * lab[2]);
}

function maxAbsDiff(a: number[], b: number[]): number {
  return Math.max(...a.map((v, i) => Math.abs(v - b[i])));
}

// ── Black point ──────────────────────────────────────────────────────

describe('Black point', () => {
  it('MetricSpace: XYZ=[0,0,0] → Lab=[0,0,0]', () => {
    const lab = hl.fromXYZ([0, 0, 0]);
    expect(lab[0]).toBeCloseTo(0, 10);
    expect(lab[1]).toBeCloseTo(0, 10);
    expect(lab[2]).toBeCloseTo(0, 10);
  });

  it('GenSpace: XYZ=[0,0,0] → Lab=[0,0,0]', () => {
    const lab = gen.fromXYZ([0, 0, 0]);
    expect(lab[0]).toBeCloseTo(0, 10);
    expect(lab[1]).toBeCloseTo(0, 10);
    expect(lab[2]).toBeCloseTo(0, 10);
  });

  it('MetricSpace: black round-trip XYZ → Lab → XYZ', () => {
    const xyz: XYZ = [0, 0, 0];
    const lab = hl.fromXYZ(xyz);
    const rec = hl.toXYZ(lab);
    expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-12);
  });

  it('GenSpace: black round-trip XYZ → Lab → XYZ', () => {
    const xyz: XYZ = [0, 0, 0];
    const lab = gen.fromXYZ(xyz);
    const rec = gen.toXYZ(lab);
    expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-12);
  });

  it('MetricSpace: #000000 has zero chroma', () => {
    const lab = hl.fromHex('#000000');
    expect(chroma(lab)).toBeLessThan(1e-8);
  });

  it('GenSpace: #000000 has zero chroma', () => {
    const srgb = hexToSrgb('#000000');
    const xyz = srgbToXyz(srgb);
    const lab = gen.fromXYZ(xyz);
    expect(chroma(lab)).toBeLessThan(1e-8);
  });
});

// ── Near-black ───────────────────────────────────────────────────────

describe('Near-black', () => {
  const darkYvals = [1e-8, 1e-6, 1e-4, 0.001, 0.005, 0.01];

  for (const Y of darkYvals) {
    it(`MetricSpace: Y=${Y} gray round-trip`, () => {
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = hl.fromXYZ(xyz);
      const rec = hl.toXYZ(lab);
      expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-8);
    });

    it(`GenSpace: Y=${Y} gray round-trip`, () => {
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = gen.fromXYZ(xyz);
      const rec = gen.toXYZ(lab);
      expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-8);
    });

    it(`MetricSpace: Y=${Y} gray has low chroma`, () => {
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = hl.fromXYZ(xyz);
      // MetricSpace enrichment stages produce small residual chroma
      // for extreme darks outside the NC LUT range
      const tol = Y < 0.001 ? 0.05 : 1e-3;
      expect(chroma(lab)).toBeLessThan(tol);
    });

    it(`GenSpace: Y=${Y} gray has near-zero chroma`, () => {
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = gen.fromXYZ(xyz);
      expect(chroma(lab)).toBeLessThan(1e-3);
    });
  }

  const darkHexes = ['#010101', '#020202', '#050505', '#0a0a0a', '#101010'];
  for (const hex of darkHexes) {
    it(`MetricSpace: ${hex} round-trip`, () => {
      const lab = hl.fromHex(hex);
      const rt = hl.toHex(lab);
      const parse = (h: string) => [
        parseInt(h.slice(1, 3), 16),
        parseInt(h.slice(3, 5), 16),
        parseInt(h.slice(5, 7), 16),
      ];
      const diff = Math.max(...parse(hex).map((v, i) => Math.abs(v - parse(rt)[i])));
      expect(diff).toBeLessThanOrEqual(1);
    });
  }
});

// ── White point ──────────────────────────────────────────────────────

describe('White point', () => {
  it('MetricSpace: D65 white has near-zero chroma', () => {
    const lab = hl.fromXYZ(D65);
    expect(chroma(lab)).toBeLessThan(1e-3);
  });

  it('GenSpace: D65 white has near-zero chroma', () => {
    const lab = gen.fromXYZ(D65);
    expect(chroma(lab)).toBeLessThan(1e-3);
  });

  it('MetricSpace: D65 round-trip', () => {
    const lab = hl.fromXYZ(D65);
    const rec = hl.toXYZ(lab);
    expect(maxAbsDiff(rec, D65)).toBeLessThan(1e-8);
  });

  it('GenSpace: D65 round-trip', () => {
    const lab = gen.fromXYZ(D65);
    const rec = gen.toXYZ(lab);
    expect(maxAbsDiff(rec, D65)).toBeLessThan(1e-8);
  });

  for (const hex of ['#f0f0f0', '#f8f8f8', '#fefefe', '#ffffff']) {
    it(`MetricSpace: ${hex} round-trip`, () => {
      const lab = hl.fromHex(hex);
      const rt = hl.toHex(lab);
      const parse = (h: string) => [
        parseInt(h.slice(1, 3), 16),
        parseInt(h.slice(3, 5), 16),
        parseInt(h.slice(5, 7), 16),
      ];
      const diff = Math.max(...parse(hex).map((v, i) => Math.abs(v - parse(rt)[i])));
      expect(diff).toBeLessThanOrEqual(1);
    });
  }
});

// ── Full achromatic axis ─────────────────────────────────────────────

describe('Achromatic axis', () => {
  it('MetricSpace: 256-step gray ramp max chroma < 1e-3', () => {
    let maxC = 0;
    for (let i = 0; i <= 255; i++) {
      const Y = i / 255;
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = hl.fromXYZ(xyz);
      const C = chroma(lab);
      if (C > maxC) maxC = C;
    }
    expect(maxC).toBeLessThan(1e-3);
  });

  it('GenSpace: 256-step gray ramp max chroma < 1e-3', () => {
    let maxC = 0;
    for (let i = 0; i <= 255; i++) {
      const Y = i / 255;
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = gen.fromXYZ(xyz);
      const C = chroma(lab);
      if (C > maxC) maxC = C;
    }
    expect(maxC).toBeLessThan(1e-3);
  });

  it('MetricSpace: lightness monotonically increases from Y=0', () => {
    let prevL = -1;
    // Logarithmic + zero
    const Yvals = [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
    for (const Y of Yvals) {
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = hl.fromXYZ(xyz);
      expect(lab[0]).toBeGreaterThanOrEqual(prevL - 1e-10);
      prevL = lab[0];
    }
  });

  it('GenSpace: lightness monotonically increases from Y=0', () => {
    let prevL = -1;
    const Yvals = [0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
    for (const Y of Yvals) {
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      const lab = gen.fromXYZ(xyz);
      expect(lab[0]).toBeGreaterThanOrEqual(prevL - 1e-10);
      prevL = lab[0];
    }
  });

  it('MetricSpace: hex grays have near-zero chroma', () => {
    let maxC = 0;
    for (let v = 0; v < 256; v += 17) {
      const hex = '#' + [v, v, v].map(c => c.toString(16).padStart(2, '0')).join('');
      const lab = hl.fromHex(hex);
      const C = chroma(lab);
      if (C > maxC) maxC = C;
    }
    expect(maxC).toBeLessThan(1e-3);
  });
});

// ── NC LUT boundary continuity ───────────────────────────────────────

describe('NC LUT continuity', () => {
  it('MetricSpace: no a,b jumps across dark boundary', () => {
    // Fine sampling in the critical dark region
    const labs: Lab[] = [];
    for (let i = 0; i <= 200; i++) {
      const Y = Math.pow(10, -5 + (i / 200) * 4); // 1e-5 to 0.1
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      labs.push(hl.fromXYZ(xyz));
    }
    let maxJumpA = 0, maxJumpB = 0;
    for (let i = 1; i < labs.length; i++) {
      maxJumpA = Math.max(maxJumpA, Math.abs(labs[i][1] - labs[i - 1][1]));
      maxJumpB = Math.max(maxJumpB, Math.abs(labs[i][2] - labs[i - 1][2]));
    }
    expect(maxJumpA).toBeLessThan(0.01);
    expect(maxJumpB).toBeLessThan(0.01);
  });

  it('GenSpace: no a,b jumps across dark boundary', () => {
    const labs: Lab[] = [];
    for (let i = 0; i <= 200; i++) {
      const Y = Math.pow(10, -5 + (i / 200) * 4);
      const xyz: XYZ = [D65[0] * Y, D65[1] * Y, D65[2] * Y];
      labs.push(gen.fromXYZ(xyz));
    }
    let maxJumpA = 0, maxJumpB = 0;
    for (let i = 1; i < labs.length; i++) {
      maxJumpA = Math.max(maxJumpA, Math.abs(labs[i][1] - labs[i - 1][1]));
      maxJumpB = Math.max(maxJumpB, Math.abs(labs[i][2] - labs[i - 1][2]));
    }
    expect(maxJumpA).toBeLessThan(0.01);
    expect(maxJumpB).toBeLessThan(0.01);
  });
});

// ── Extreme inputs ──────────────────────────────────────────────────

describe('Extreme inputs', () => {
  const extremes: { label: string; xyz: XYZ }[] = [
    { label: 'black', xyz: [0, 0, 0] },
    { label: 'D65 white', xyz: [0.95047, 1.0, 1.08883] },
    { label: 'above white', xyz: [2, 2, 2] },
    { label: 'near-zero Z only', xyz: [0, 0, 0.001] },
    { label: 'near-zero X only', xyz: [0.001, 0, 0] },
  ];

  for (const { label, xyz } of extremes) {
    it(`MetricSpace: ${label} round-trip`, () => {
      const lab = hl.fromXYZ(xyz);
      const rec = hl.toXYZ(lab);
      expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-6);
    });

    it(`GenSpace: ${label} round-trip`, () => {
      const lab = gen.fromXYZ(xyz);
      const rec = gen.toXYZ(lab);
      expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-6);
    });
  }

  it('MetricSpace: no NaN on zero input', () => {
    const lab = hl.fromXYZ([0, 0, 0]);
    expect(Number.isNaN(lab[0])).toBe(false);
    expect(Number.isNaN(lab[1])).toBe(false);
    expect(Number.isNaN(lab[2])).toBe(false);
  });

  it('GenSpace: no NaN on zero input', () => {
    const lab = gen.fromXYZ([0, 0, 0]);
    expect(Number.isNaN(lab[0])).toBe(false);
    expect(Number.isNaN(lab[1])).toBe(false);
    expect(Number.isNaN(lab[2])).toBe(false);
  });

  it('MetricSpace: no NaN on tiny input', () => {
    const lab = hl.fromXYZ([1e-20, 1e-20, 1e-20]);
    expect(Number.isNaN(lab[0])).toBe(false);
    expect(Number.isNaN(lab[1])).toBe(false);
    expect(Number.isNaN(lab[2])).toBe(false);
  });

  it('GenSpace: no NaN on tiny input', () => {
    const lab = gen.fromXYZ([1e-20, 1e-20, 1e-20]);
    expect(Number.isNaN(lab[0])).toBe(false);
    expect(Number.isNaN(lab[1])).toBe(false);
    expect(Number.isNaN(lab[2])).toBe(false);
  });
});

// ── sRGB primaries ──────────────────────────────────────────────────

describe('sRGB primaries round-trip', () => {
  const colors = [
    '#ff0000', '#00ff00', '#0000ff',
    '#ffff00', '#ff00ff', '#00ffff',
    '#000000', '#ffffff', '#808080',
  ];

  for (const hex of colors) {
    it(`MetricSpace: ${hex}`, () => {
      const xyz = srgbToXyz(hexToSrgb(hex));
      const lab = hl.fromXYZ(xyz);
      const rec = hl.toXYZ(lab);
      expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-8);
    });

    it(`GenSpace: ${hex}`, () => {
      const xyz = srgbToXyz(hexToSrgb(hex));
      const lab = gen.fromXYZ(xyz);
      const rec = gen.toXYZ(lab);
      expect(maxAbsDiff(rec, xyz)).toBeLessThan(1e-8);
    });
  }
});

// ── Web-safe stress test ────────────────────────────────────────────

describe('Web-safe 216 colors stress test', () => {
  it('MetricSpace: all web-safe round-trip < 1e-6', () => {
    let maxErr = 0;
    for (let r = 0; r < 256; r += 51) {
      for (let g = 0; g < 256; g += 51) {
        for (let b = 0; b < 256; b += 51) {
          const srgb = [r / 255, g / 255, b / 255] as [number, number, number];
          const xyz = srgbToXyz(srgb);
          const lab = hl.fromXYZ(xyz);
          const rec = hl.toXYZ(lab);
          const err = maxAbsDiff(rec, xyz);
          if (err > maxErr) maxErr = err;
        }
      }
    }
    expect(maxErr).toBeLessThan(1e-6);
  });

  it('GenSpace: all web-safe round-trip < 1e-6', () => {
    let maxErr = 0;
    for (let r = 0; r < 256; r += 51) {
      for (let g = 0; g < 256; g += 51) {
        for (let b = 0; b < 256; b += 51) {
          const srgb = [r / 255, g / 255, b / 255] as [number, number, number];
          const xyz = srgbToXyz(srgb);
          const lab = gen.fromXYZ(xyz);
          const rec = gen.toXYZ(lab);
          const err = maxAbsDiff(rec, xyz);
          if (err > maxErr) maxErr = err;
        }
      }
    }
    expect(maxErr).toBeLessThan(1e-6);
  });
});

// ── refRange validation ────────────────────────────────────────────

describe('refRange validation', () => {
  // These match the Color.js refRange values for CSS percentage mapping.
  // L range must cover D65 white, a/b range must cover Display P3 gamut.
  const METRIC_L_MAX = 1.144;
  const METRIC_AB_MAX = 1.0;
  const GEN_L_MAX = 1.169;
  const GEN_AB_MAX = 0.4;

  it('MetricSpace: D65 white L within refRange', () => {
    const lab = hl.fromXYZ(D65);
    expect(lab[0]).toBeGreaterThan(0);
    expect(lab[0]).toBeLessThanOrEqual(METRIC_L_MAX);
  });

  it('GenSpace: D65 white L within refRange', () => {
    const lab = gen.fromXYZ(D65);
    expect(lab[0]).toBeGreaterThan(0);
    expect(lab[0]).toBeLessThanOrEqual(GEN_L_MAX);
  });

  it('MetricSpace: sRGB primaries within ab refRange', () => {
    const primaries = [
      '#ff0000', '#00ff00', '#0000ff',
      '#ffff00', '#ff00ff', '#00ffff',
    ];
    for (const hex of primaries) {
      const xyz = srgbToXyz(hexToSrgb(hex));
      const lab = hl.fromXYZ(xyz);
      expect(Math.abs(lab[1])).toBeLessThanOrEqual(METRIC_AB_MAX + 0.001);
      expect(Math.abs(lab[2])).toBeLessThanOrEqual(METRIC_AB_MAX + 0.001);
    }
  });

  it('GenSpace: sRGB primaries within ab refRange', () => {
    const primaries = [
      '#ff0000', '#00ff00', '#0000ff',
      '#ffff00', '#ff00ff', '#00ffff',
    ];
    for (const hex of primaries) {
      const xyz = srgbToXyz(hexToSrgb(hex));
      const lab = gen.fromXYZ(xyz);
      expect(Math.abs(lab[1])).toBeLessThanOrEqual(GEN_AB_MAX + 0.001);
      expect(Math.abs(lab[2])).toBeLessThanOrEqual(GEN_AB_MAX + 0.001);
    }
  });

  it('MetricSpace: Display P3 primaries within ab refRange', () => {
    // Display P3 primaries at full saturation
    const p3Primaries: [number, number, number][] = [
      [1, 0, 0], [0, 1, 0], [0, 0, 1],
      [1, 1, 0], [1, 0, 1], [0, 1, 1],
    ];
    for (const p3 of p3Primaries) {
      const xyz = displayP3ToXyz(p3);
      const lab = hl.fromXYZ(xyz);
      expect(Math.abs(lab[1])).toBeLessThanOrEqual(METRIC_AB_MAX + 0.001);
      expect(Math.abs(lab[2])).toBeLessThanOrEqual(METRIC_AB_MAX + 0.001);
    }
  });

  it('GenSpace: Display P3 primaries within ab refRange', () => {
    const p3Primaries: [number, number, number][] = [
      [1, 0, 0], [0, 1, 0], [0, 0, 1],
      [1, 1, 0], [1, 0, 1], [0, 1, 1],
    ];
    for (const p3 of p3Primaries) {
      const xyz = displayP3ToXyz(p3);
      const lab = gen.fromXYZ(xyz);
      expect(Math.abs(lab[1])).toBeLessThanOrEqual(GEN_AB_MAX + 0.001);
      expect(Math.abs(lab[2])).toBeLessThanOrEqual(GEN_AB_MAX + 0.001);
    }
  });

  it('MetricSpace: sRGB primaries L within refRange', () => {
    const primaries = [
      '#ff0000', '#00ff00', '#0000ff',
      '#ffff00', '#ff00ff', '#00ffff',
    ];
    for (const hex of primaries) {
      const xyz = srgbToXyz(hexToSrgb(hex));
      const lab = hl.fromXYZ(xyz);
      expect(lab[0]).toBeGreaterThanOrEqual(0);
      expect(lab[0]).toBeLessThanOrEqual(METRIC_L_MAX + 0.001);
    }
  });

  it('GenSpace: sRGB primaries L within refRange', () => {
    const primaries = [
      '#ff0000', '#00ff00', '#0000ff',
      '#ffff00', '#ff00ff', '#00ffff',
    ];
    for (const hex of primaries) {
      const xyz = srgbToXyz(hexToSrgb(hex));
      const lab = gen.fromXYZ(xyz);
      expect(lab[0]).toBeGreaterThanOrEqual(0);
      expect(lab[0]).toBeLessThanOrEqual(GEN_L_MAX + 0.001);
    }
  });
});
