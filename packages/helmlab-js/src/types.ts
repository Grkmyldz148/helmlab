/** Helmlab Lab coordinates [L, a, b]. L in ~[0,1], a/b unbounded. */
export type Lab = [number, number, number];

/** CIE XYZ tristimulus [X, Y, Z]. Y=1 for reference white. */
export type XYZ = [number, number, number];

/** sRGB or Display P3 [R, G, B] in [0, 1], gamma-encoded. */
export type RGB = [number, number, number];

/** CSS hex string '#rrggbb'. */
export type Hex = string;

/** Semantic scale levels (Tailwind-style). */
export type SemanticScale = Record<string, Hex>;

/** WCAG conformance level. */
export type WCAGLevel = 'AA' | 'AAA';
