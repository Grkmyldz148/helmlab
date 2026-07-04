# helmlab

Perceptual color library for UI design systems — two purpose-built Lab spaces in one zero-dependency package.

[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![bundle size](https://img.shields.io/bundlephobia/minzip/helmlab)](https://bundlephobia.com/package/helmlab)
[![Color.js](https://img.shields.io/badge/Color.js-merged-f97316.svg)](https://colorjs.io)
[![license](https://img.shields.io/npm/l/helmlab.svg)](https://github.com/Grkmyldz148/helmlab/blob/main/packages/helmlab-js/LICENSE)

**[Website](https://helmlab.space)** · **[Docs](https://helmlab.space/docs/)** · **[Playground](https://helmlab.space/playground/)** · **[Benchmark](https://helmlab.space/benchmark/)** · **[Paper](https://arxiv.org/abs/2602.23010)**

- **MetricSpace** — measures how different two colors *look*. STRESS **22.48** on COMBVD with Bradford CAT (CIEDE2000: 29.20; cross-validated estimate ~24.3 — [full overfit analysis](https://helmlab.space/benchmark/)).
- **GenSpace** — creates colors: gradients, palettes, gamut mapping. **62–9** vs OKLab (19 ties) across 90 ColorBench metrics; 360/360 valid gamut cusps in sRGB and Display P3.

~17.8 KB gzipped · zero dependencies · ESM + CJS · full TypeScript types · tree-shakeable (`sideEffects: false`) · works in browsers and Node.js.

```bash
npm install helmlab
```

## Quick start

```ts
import { Helmlab } from 'helmlab';

const hl = new Helmlab();

// Generate — GenSpace under the hood
hl.gradient('#0000ff', '#ffffff', 16);   // stays blue through the midpoint
hl.semanticScale('#3b82f6');             // Tailwind-style { '50': '#e7efff', ..., '950': '#000046' } — 500 is your exact input
hl.palette('#3b82f6', 10);               // lightness ramp, light → dark

// Measure — MetricSpace under the hood
hl.difference('#ff0000', '#00ff00');     // 0.148 — the trained perceptual metric (STRESS 22.48)
hl.euclideanDistance('#ff0000', '#00ff00'); // 1.62 — plain Euclidean Lab, ΔE76-style, fast

// Accessibility
hl.contrastRatio('#ffffff', '#3b82f6');  // 3.68 (WCAG 2.1)
hl.ensureContrast('#3b82f6', '#ffffff'); // darkens until 4.5:1, hue preserved
```

## Two spaces, one rule

Every method routes to the space it was designed for — you never pick manually:

| You call | Space used | Why |
|---|---|---|
| `gradient` `palette` `paletteHues` `semanticScale` `ensureContrast` `adaptToMode` `adaptPair` | **GenSpace** | optimized for smooth, in-gamut color *creation* |
| `difference` `deltaE` `euclideanDistance` `perceptualDistance` `differenceWithConfidence` `info` `toHexP3` `toHexRec2020` | **MetricSpace** | optimized to predict human difference judgments |

The Lab coordinates of the two spaces are **not interchangeable**: `fromHex`/`toHex` speak Metric Lab, `genFromHex`/`genToHex` speak Gen Lab. The `TokenExporter` (`hl.export()`) takes **Metric Lab**.

## Measuring color difference

```ts
// Recommended: the trained metric (Minkowski + compression, fit on COMBVD).
// Saturates near ~0.15 for very dissimilar pairs — order is preserved.
hl.difference('#3b82f6', '#4c8af7');     // 0.0227

// Experimental: difference + how much real observers would disagree about it
hl.differenceWithConfidence('#808080', '#828282');
// { de: 0.0117, disagreement: 29.6, reliability: 0.19, reliable: false, ... }

// Fast Euclidean for quick UI checks (alias: deltaE)
hl.euclideanDistance('#000000', '#ffffff'); // 1.12
```

## Generating colors

```ts
// Perceptually even gradient: CIEDE2000 arc-length reparameterization,
// gamut-mapped sampling — equal visual step sizes on any pair
hl.gradient('#ef4444', '#3b82f6', 16);

// Hue ring at fixed lightness/chroma (categorical palettes)
hl.paletteHues(0.6, 0.15, 12);

// Cylindrical LCh on GenSpace — hue rotations & harmonies
// (same coordinates as the `helmgenlch` space on Color.js)
const lch = hl.genToLch(hl.genFromHex('#3b82f6'));  // [0.5586, 0.2976, 263.1]
const triad = hl.genToHex(hl.genFromLch([lch[0], lch[1], (lch[2] + 120) % 360]));
```

## Wide gamut & tokens

```ts
const lab = hl.fromHex('#ff0000');
hl.toHexP3(lab);        // 'color(display-p3 0.9176 0.2003 0.1386)' — gamut mapped, hue preserved
hl.toHexRec2020(lab);   // 'color(rec2020 0.7920 0.2310 0.0738)'
hl.isInP3(lab);         // gamut tests: isInSrgb / isInP3 / isInRec2020

const ex = hl.export(); // TokenExporter — takes METRIC Lab (hl.fromHex)
ex.toCssOklch(lab);     // 'oklch(62.8% 0.2576 29.2)'
ex.exportTailwind(hl.semanticScale('#3b82f6'), 'primary');
ex.exportCssCustomProperties(hl.semanticScale('#3b82f6'), '--primary');
// also: toCssHex / toCssRgb / toCssHsl / toCssDisplayP3 / toAndroidArgb / toIosP3 / toSwiftLiteral
```

## Dark / light mode

```ts
hl.adaptToMode('#3b82f6', 'light', 'dark');            // '#2a67d9' — soft L-inversion, hue kept
hl.adaptPair('#3366ff', '#ffffff', 'light', 'dark');   // [fg, bg] with contrast ≥ 4.5 guaranteed
hl.meetsContrast('#1e40af', '#ffffff', 'AAA');         // WCAG check without modifying colors
```

## Advanced: raw spaces

```ts
import { GenSpace, MetricSpace, srgbToXyz,
         compileGenParams, getDefaultGenParams,
         compileParams, getDefaultParams } from 'helmlab';

const gen = new GenSpace(compileGenParams(getDefaultGenParams()));
const metric = new MetricSpace(compileParams(getDefaultParams()));
const lab = gen.fromXYZ(srgbToXyz([0.2, 0.5, 0.8]));   // raw Lab, no gamut mapping
```

Custom parameter sets (research / retraining) are accepted by both constructors — see the [docs](https://helmlab.space/docs/).

## Python parity

The [`helmlab` PyPI package](https://pypi.org/project/helmlab/) is the same math with a snake_case API (`difference()` ↔ `difference()`, `genToLch` ↔ `gen_to_lch`). Outputs are cross-checked down to float64 — a battery over the full public API shows **zero differences at 1e-12 tolerance**.

## Using with Color.js

Helmlab is merged into [color-js/color.js](https://github.com/color-js/color.js) master: spaces `helmgen`, `helmgenlch`, `helmlab-metric` plus a `"Helmlab"` deltaE method. Not yet in the published `colorjs.io` release — until then use this package, or `npm install github:color-js/color.js`.

## Honest limits

No color space wins everywhere. OKLab is still the better pick for near-achromatic gradient mastering, CVD-deutan-optimized palettes, native CSS `oklch()`, or a ~2 KB bundle. CIEDE2000 still edges MetricSpace on the small near-threshold tolerance datasets (LEEDS / RIT-DuPont). Full loss list with numbers: [helmlab.space/benchmark](https://helmlab.space/benchmark/).

## License

MIT © [Görkem Yıldız](https://github.com/Grkmyldz148)
