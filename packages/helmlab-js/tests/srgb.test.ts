import { describe, it, expect } from 'vitest';
import {
  hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb, relativeLuminance,
  xyzToRec2020, rec2020ToXyz, linearToRec2020, rec2020ToLinear,
} from '../src/index.js';
import ref from './reference/reference-values.json';

describe('sRGB utilities', () => {
  it('hexToSrgb parses correctly', () => {
    expect(hexToSrgb('#ff0000')).toEqual([1, 0, 0]);
    expect(hexToSrgb('#00ff00')).toEqual([0, 1, 0]);
    expect(hexToSrgb('#808080')).toEqual([128 / 255, 128 / 255, 128 / 255]);
  });

  it('srgbToHex round-trips', () => {
    expect(srgbToHex([1, 0, 0])).toBe('#ff0000');
    expect(srgbToHex([0, 1, 0])).toBe('#00ff00');
    expect(srgbToHex([0, 0, 1])).toBe('#0000ff');
  });

  it('srgbToXyz matches Python reference', () => {
    for (const t of ref.xyz) {
      const rgb = hexToSrgb(t.hex);
      const xyz = srgbToXyz(rgb);
      expect(xyz[0]).toBeCloseTo(t.xyz[0], 6);
      expect(xyz[1]).toBeCloseTo(t.xyz[1], 6);
      expect(xyz[2]).toBeCloseTo(t.xyz[2], 6);
    }
  });

  it('xyzToSrgb → srgbToXyz round-trip', () => {
    for (const t of ref.xyz) {
      const rgb = hexToSrgb(t.hex);
      const xyz = srgbToXyz(rgb);
      const back = xyzToSrgb(xyz);
      expect(back[0]).toBeCloseTo(rgb[0], 6);
      expect(back[1]).toBeCloseTo(rgb[1], 6);
      expect(back[2]).toBeCloseTo(rgb[2], 6);
    }
  });

  it('relative luminance of white is 1', () => {
    expect(relativeLuminance([1, 1, 1])).toBeCloseTo(1, 6);
  });

  it('relative luminance of black is 0', () => {
    expect(relativeLuminance([0, 0, 0])).toBeCloseTo(0, 6);
  });

  it('rec2020 OETF round-trips on representative values', () => {
    for (const v of [0, 0.005, 0.018, 0.05, 0.2, 0.5, 0.9, 1]) {
      expect(rec2020ToLinear(linearToRec2020(v))).toBeCloseTo(v, 10);
    }
  });

  it('rec2020 OETF endpoints are anchored at 0 and 1', () => {
    expect(linearToRec2020(0)).toBeCloseTo(0, 12);
    expect(linearToRec2020(1)).toBeCloseTo(1, 12);
    expect(rec2020ToLinear(0)).toBeCloseTo(0, 12);
    expect(rec2020ToLinear(1)).toBeCloseTo(1, 12);
  });

  it('rec2020 ↔ XYZ round-trips', () => {
    for (const rgb of [[0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.7, 0.4]]) {
      const xyz = rec2020ToXyz(rgb as [number, number, number]);
      const back = xyzToRec2020(xyz);
      expect(back[0]).toBeCloseTo(rgb[0], 8);
      expect(back[1]).toBeCloseTo(rgb[1], 8);
      expect(back[2]).toBeCloseTo(rgb[2], 8);
    }
  });

  it('Rec2020 white → XYZ matches D65 reference', () => {
    const xyz = rec2020ToXyz([1, 1, 1]);
    expect(xyz[0]).toBeCloseTo(0.95047, 3);
    expect(xyz[1]).toBeCloseTo(1.00000, 3);
    expect(xyz[2]).toBeCloseTo(1.08883, 3);
  });
});
