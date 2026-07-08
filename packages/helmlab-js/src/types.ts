/** Helmlab Lab coordinates [L, a, b]. L in ~[0,1], a/b unbounded. */
export type Lab = [number, number, number];

declare const GEN_LAB_BRAND: unique symbol;
/** Lab coordinates in the GENERATION space (`hl.gen`). Branded: cannot be
 * passed to `hl.metric` APIs (compile-time), and carries a runtime tag so
 * cross-space misuse throws in plain JS too. Construct via `hl.gen.fromHex`
 * / `hl.gen.lab`. */
export type GenLab = [number, number, number] & { readonly [GEN_LAB_BRAND]: 'gen' };

declare const METRIC_LAB_BRAND: unique symbol;
/** Lab coordinates in the MEASUREMENT space (`hl.metric`). See {@link GenLab};
 * construct via `hl.metric.fromHex` / `hl.metric.lab`. */
export type MetricLab = [number, number, number] & { readonly [METRIC_LAB_BRAND]: 'metric' };

/** CIE XYZ tristimulus [X, Y, Z]. Y=1 for reference white. */
export type XYZ = [number, number, number];

/** sRGB or Display P3 [R, G, B] in [0, 1], gamma-encoded. */
export type RGB = [number, number, number];

/** CSS hex string '#rrggbb'. */
export type Hex = string;

/** Cylindrical Lab [L, C, h]. h in degrees, [0, 360). */
export type LCh = [number, number, number];

/** Semantic scale levels (Tailwind-style). */
export type SemanticScale = Record<string, Hex>;

/** WCAG conformance level. */
export type WCAGLevel = 'AA' | 'AAA';
