import { describe, it, expect } from 'vitest';
import { Helmlab } from '../src/index.js';
import ref from './reference/reference-values.json';

const hl = new Helmlab();

describe('Analytical forward transform', () => {
  for (const t of ref.forward) {
    it(`fromHex(${t.hex}) matches Python`, () => {
      const lab = hl.fromHex(t.hex);
      // NC LUT uses linear interp (not PCHIP), so near-black tolerance is wider
      const tol = (t.hex === '#000000' || t.hex === '#1a1a1a') ? 1 : 4;
      expect(lab[0]).toBeCloseTo(t.lab[0], tol);
      expect(lab[1]).toBeCloseTo(t.lab[1], tol);
      expect(lab[2]).toBeCloseTo(t.lab[2], tol);
    });
  }
});

describe('Hex round-trip', () => {
  for (const t of ref.forward) {
    it(`toHex(fromHex(${t.hex})) = ${t.rt_hex}`, () => {
      const lab = hl.fromHex(t.hex);
      const rt = hl.toHex(lab);
      expect(rt).toBe(t.rt_hex);
    });
  }
});

describe('Forward → toSrgb matches Python', () => {
  for (const t of ref.forward) {
    it(`toSrgb for ${t.hex}`, () => {
      const lab = hl.fromHex(t.hex);
      const srgb = hl.toSrgb(lab);
      // Allow 2/255 tolerance for gamut mapping differences
      expect(srgb[0]).toBeCloseTo(t.srgb[0], 2);
      expect(srgb[1]).toBeCloseTo(t.srgb[1], 2);
      expect(srgb[2]).toBeCloseTo(t.srgb[2], 2);
    });
  }
});

describe('XYZ round-trip', () => {
  it('fromXYZ → toXYZ preserves values', () => {
    for (const t of ref.xyz) {
      const lab = hl.fromXYZ(t.xyz as [number, number, number]);
      const xyz = hl.toXYZ(lab);
      expect(xyz[0]).toBeCloseTo(t.xyz[0], 5);
      expect(xyz[1]).toBeCloseTo(t.xyz[1], 5);
      expect(xyz[2]).toBeCloseTo(t.xyz[2], 5);
    }
  });
});

describe('info()', () => {
  for (const t of ref.forward) {
    it(`info(${t.hex}).L matches`, () => {
      const info = hl.info(t.hex);
      expect(info.L).toBeCloseTo(t.L, 4);
    });
  }
});
