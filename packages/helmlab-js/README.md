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

One `Helmlab` instance, three namespaces: **`hl.gen`** creates colors,
**`hl.metric`** measures them, **`hl.tokens`** exports design tokens.

```ts
import { Helmlab } from 'helmlab';

const hl = new Helmlab();

// Create — GenSpace
hl.gen.gradient('#0000ff', '#ffffff', 16);   // stays blue through the midpoint
hl.gen.scale('#3b82f6');                     // Tailwind-style { '50': ..., '950': ... } — 500 is your exact input
hl.gen.palette('#3b82f6', 10);               // lightness ramp, light → dark

// Measure — MetricSpace
hl.metric.difference('#ff0000', '#00ff00');  // 0.148 — the trained perceptual metric (STRESS 22.48)
hl.metric.jnd('#808080', '#828282');         // 0.33 — in just-noticeable-difference units
hl.metric.euclidean('#ff0000', '#00ff00');   // 1.62 — plain Euclidean Lab, unbounded

// Accessibility
hl.gen.contrastRatio('#ffffff', '#3b82f6');  // 3.68 (WCAG 2.1)
hl.gen.ensureContrast('#3b82f6', '#ffffff'); // darkens until 4.5:1, hue preserved
```

## Two spaces, two Lab types

Each namespace has its own `fromHex`, and their Lab values are **branded
types** (`GenLab` / `MetricLab`). Passing one space's Lab to the other
throws a `TypeError` — the 0.x silent-wrong-color footgun is structurally
gone. Everyday use never touches Lab: color strings in, color strings out.

| Namespace | Space | For |
|---|---|---|
| `hl.gen` | GenSpace | `gradient` `mix` `palette` `scale` `hueRing` `harmonies` `rotateHue` `vivid` `cusp` `maxChroma` `gamutMap` `ensureContrast` `adaptToMode` `adaptPair` |
| `hl.metric` | MetricSpace | `difference` `euclidean` `ciede2000` `jnd` `distance` `confidence` `nearest` `info` `toCss` |
| `hl.tokens` | — | `css` `android` `iosP3` `swift` `cssVariables` `tailwind` `multiFormat` `json` (all take color strings) |

## Measuring color difference

```ts
// Recommended: the trained metric (Minkowski + compression, fit on COMBVD).
// Saturates near ~0.15 for very dissimilar pairs — order is preserved.
hl.metric.difference('#3b82f6', '#4c8af7');   // 0.0227

// In threshold units: <1 likely unnoticed, 1–2 subtle, >2 clearly visible
hl.metric.jnd('#3b82f6', '#4c8af7');

// Experimental: difference + how much real observers would disagree about it
hl.metric.confidence('#808080', '#828282');
// { de: 0.0117, pNoticeable: 0.077, reliability: 0.41, reliable: false, ... }

// Catalog matching (most perturbation-stable for argmax): CIEDE2000
hl.metric.nearest('#3b82f6', ['#3b7ff0', '#ff0000'], 'ciede2000');
```

## Generating colors

```ts
// Perceptually even gradient: CIEDE2000 arc-length reparameterization —
// equal visual step sizes on any pair. Every generation function takes
// { gamut: 'srgb' | 'display-p3' | 'rec2020' }.
hl.gen.gradient('#ef4444', '#3b82f6', 16);
hl.gen.gradient('#0000ff', '#ffffff', 16, { gamut: 'display-p3' });

// The visual midpoint on the same path (not the coordinate average)
hl.gen.mix('#ef4444', '#3b82f6', 0.5);

// Hue ring at fixed lightness/chroma (categorical palettes)
hl.gen.hueRing(12, { lightness: 0.6, chroma: 0.15 });

// Harmonies: constant-L,C hue rotations (matched lightness & colorfulness)
hl.gen.harmonies('#3b82f6', 'triadic');      // also: complementary, analogous, tetradic, split_complementary
hl.gen.rotateHue('#3b82f6', 120);

// Cusp geometry — the 360/360/360 strength, exposed:
hl.gen.cusp(263);                            // [L, C] of the most colorful point of a hue
hl.gen.maxChroma(0.6, 263, 'display-p3');    // chroma headroom of a wide gamut
hl.gen.vivid('#6488b8', { gamut: 'display-p3' }); // same L & hue, chroma → boundary
```

## Wide gamut & tokens

```ts
// Wide-gamut INPUT everywhere a color string is accepted:
hl.metric.info('color(display-p3 1 0 0)');   // { inSrgb: false, inP3: true, ... }

// Wide-gamut OUTPUT:
const lab = hl.metric.fromHex('#ff0000');
hl.metric.toCss(lab, 'display-p3');          // 'color(display-p3 0.9176 0.2003 0.1386)'
hl.metric.toCss(lab, 'rec2020');
hl.metric.inGamut(lab, 'display-p3');

// Tokens: color strings in, platform strings out (no Lab, no footgun)
hl.tokens.css('#3b82f6', 'oklch');           // 'oklch(62.3% 0.1881 259.8)'
hl.tokens.tailwind(hl.gen.scale('#3b82f6'), 'primary');
hl.tokens.cssVariables(hl.gen.scale('#3b82f6'), '--primary');
// also: css(c,'hex'|'rgb'|'hsl'|'p3'|'rec2020') / android / iosP3 / swift / multiFormat / json
```

## Dark / light mode

```ts
hl.gen.adaptToMode('#3b82f6', 'light', 'dark');           // soft L-inversion, hue kept
hl.gen.adaptPair('#3366ff', '#ffffff', 'light', 'dark');  // [fg, bg] re-contrasted
hl.gen.meetsContrast('#1e40af', '#ffffff', 'AAA');        // WCAG check without modifying
hl.gen.ensureContrast('#3b82f6', '#808080', 7, { strict: true }); // throws ContrastError if unreachable
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

The [`helmlab` PyPI package](https://pypi.org/project/helmlab/) is the same math with a snake_case API (`hl.gen.ensureContrast` ↔ `hl.gen.ensure_contrast`). A permanent parity gate (`tests/parity-1.0.test.ts`, reference generated by the Python package) covers the full public surface: **every string output is byte-identical, numeric worst-case difference ~1e-12**, hex round-trips bit-exact on a 1728-color grid in both languages. Conversion precision: XYZ round-trip 2.9e-15 (MetricSpace) / 5.8e-9 (GenSpace).

## Using with Color.js

Helmlab is merged into [color-js/color.js](https://github.com/color-js/color.js) master: spaces `helmgen`, `helmgenlch`, `helmlab-metric` plus a `"Helmlab"` deltaE method. Not yet in the published `colorjs.io` release — until then use this package, or `npm install github:color-js/color.js`.

## Honest limits

No color space wins everywhere. OKLab is still the better pick for near-achromatic gradient mastering, CVD-deutan-optimized palettes, native CSS `oklch()`, or a ~2 KB bundle. CIEDE2000 still edges MetricSpace on the small near-threshold tolerance datasets (LEEDS / RIT-DuPont). Full loss list with numbers: [helmlab.space/benchmark](https://helmlab.space/benchmark/).

## License

MIT © [Görkem Yıldız](https://github.com/Grkmyldz148)
