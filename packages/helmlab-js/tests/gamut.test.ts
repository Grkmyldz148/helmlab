import { describe, it, expect } from 'vitest';
import { Helmlab } from '../src/index.js';

const hl = new Helmlab();

describe('Gamut mapping', () => {
  it('in-gamut colors are not modified', () => {
    const lab = hl.fromHex('#808080');
    expect(hl.isInSrgb(lab)).toBe(true);
  });

  it('pure primaries are in sRGB gamut', () => {
    for (const hex of ['#ff0000', '#00ff00', '#0000ff']) {
      const lab = hl.fromHex(hex);
      expect(hl.isInSrgb(lab)).toBe(true);
    }
  });

  it('out-of-gamut Lab is gamut-mapped to valid sRGB', () => {
    // High chroma, should be OOG
    const srgb = hl.toSrgb([0.5, 0.8, 0.8]);
    expect(srgb[0]).toBeGreaterThanOrEqual(0);
    expect(srgb[0]).toBeLessThanOrEqual(1);
    expect(srgb[1]).toBeGreaterThanOrEqual(0);
    expect(srgb[1]).toBeLessThanOrEqual(1);
    expect(srgb[2]).toBeGreaterThanOrEqual(0);
    expect(srgb[2]).toBeLessThanOrEqual(1);
  });

  it('gamut mapping preserves lightness', () => {
    const L = 0.5;
    const lab = hl.fromHex(hl.toHex([L, 0.8, 0.8]));
    // L should be close (gamut mapping preserves L)
    expect(lab[0]).toBeCloseTo(L, 1);
  });
});
