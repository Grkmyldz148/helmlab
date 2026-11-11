# helmlab

Reference code of Helmlab, in JavaScript/TypeScript. Helmlab defines color spaces for generating colors (gradients, palettes, gamut mapping) and for measuring color differences.

[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![bundle size](https://img.shields.io/bundlephobia/minzip/helmlab)](https://bundlephobia.com/package/helmlab)
[![license](https://img.shields.io/npm/l/helmlab.svg)](https://github.com/Grkmyldz148/helmlab/blob/main/packages/helmlab-js/LICENSE)

[Repository](https://github.com/Grkmyldz148/helmlab) · [API reference](https://github.com/Grkmyldz148/helmlab/blob/main/API.md) · [Changelog](https://github.com/Grkmyldz148/helmlab/blob/main/CHANGELOG.md) · [Paper](https://arxiv.org/abs/2602.23010) · [helmlab.space](https://helmlab.space)

- **GenSpace** (`hl.gen`) creates colors: gradients, palettes, scales, gamut mapping, contrast adjustment.
- **MetricSpace** (`hl.metric`) measures color differences.

No dependencies · ESM + CJS · TypeScript types · `sideEffects: false` · browsers and Node.js.

This README gives no performance figures. They will be published on one evidence page, generated from one file. Each figure there will name its dataset, its size (n), and whether that data was used to fit, tune or design the model.

```bash
npm install helmlab
```

## Helmlab 2026

The current definition, Helmlab 2026, is the output of `helmlab` 1.0.0. Later releases keep these outputs; 1.0.1 has identical outputs, and only its JavaScript speed and documentation changed. The identifiers are `helmgen`, `helmgenlch`, `helmlab-metric` (MetricSpace, base form) and ΔE `helmlab`; see the [repository README](https://github.com/Grkmyldz148/helmlab#readme).

**`hl.metric` returns the neutral-corrected form** (`helmlab-metric-neutral`), not the base form `helmlab-metric` that Color.js and ColorAide implement. The same (L, a, b) triple names different colors in the two forms: (0.7779, 0, 0) is `#808080` in `hl.metric` and `#759dbf` in the base form. `difference()`, `distance()` and `jnd()` remove the correction before measuring, so ΔE `helmlab` is the same for both.

## Quick start

One `Helmlab` instance, three namespaces: `hl.gen` creates colors, `hl.metric` measures them, `hl.tokens` exports design tokens.

```ts
import { Helmlab } from 'helmlab';

const hl = new Helmlab();

// Create (GenSpace)
hl.gen.gradient('#0000ff', '#ffffff', 16);   // 16 steps of about equal CIEDE2000 length
hl.gen.scale('#3b82f6');                     // { '50': ..., '950': ... }; '500' is the input
hl.gen.palette('#3b82f6', 10);               // lightness ramp, light → dark

// Measure (MetricSpace)
hl.metric.difference('#ff0000', '#00ff00');  // ΔE helmlab: 0.148
hl.metric.jnd('#808080', '#828282');         // ΔE helmlab / 0.0356: 0.33
hl.metric.euclidean('#ff0000', '#00ff00');   // uncompressed Euclidean distance: 1.62

// Contrast
hl.gen.contrastRatio('#ffffff', '#3b82f6');  // 3.68 (WCAG 2.1)
hl.gen.ensureContrast('#3b82f6', '#ffffff'); // '#3071e4': darker, same hue, 4.5:1
```

## Two spaces, two Lab types

Each namespace has its own `fromHex`, and their Lab values have distinct types (`GenLab` / `MetricLab`). Passing one space's Lab to the other throws a `TypeError`. Everyday use needs no Lab values: color strings in, color strings out.

| Namespace | Space | Methods |
|---|---|---|
| `hl.gen` | GenSpace | `gradient` `mix` `palette` `scale` `hueRing` `harmonies` `rotateHue` `vivid` `cusp` `maxChroma` `gamutMap` `contrastRatio` `meetsContrast` `ensureContrast` `adaptToMode` `adaptPair` |
| `hl.metric` | MetricSpace | `difference` `euclidean` `ciede2000` `jnd` `distance` `confidence` `nearest` `info` `toCss` |
| `hl.tokens` | — | `css` `android` `iosP3` `swift` `cssVariables` `tailwind` `multiFormat` `json` (all take color strings) |

## Measuring color difference

```ts
// ΔE helmlab (fitted to COMBVD). Its compression levels off near ~0.15 for
// very different pairs; the order is kept.
hl.metric.difference('#3b82f6', '#4c8af7');   // 0.0227

// In units of 0.0356: <1 likely unnoticed, 1–2 subtle, >2 clearly visible
hl.metric.jnd('#3b82f6', '#4c8af7');

// Experimental: ΔE helmlab plus an estimate of how much observers disagree
hl.metric.confidence('#808080', '#828282');
// { de: 0.0117, pNoticeable: 0.077, reliability: 0.41, reliable: false, ... }

// CIEDE2000; also the default of nearest() for catalog matching
hl.metric.ciede2000('#ff0000', '#00ff00');    // 86.6
hl.metric.nearest('#3b82f6', ['#3b7ff0', '#ff0000'], 'ciede2000');
```

For pairs with a color darker than about L* 10, use `ciede2000()` (see Known limitations).

## Generating colors

```ts
// Gradients: a straight path in GenSpace, reparameterized by CIEDE2000 arc
// length. Every generation function takes { gamut: 'srgb' | 'display-p3' | 'rec2020' }.
hl.gen.gradient('#ef4444', '#3b82f6', 16);
hl.gen.gradient('#0000ff', '#ffffff', 16, { gamut: 'display-p3' });

// The midpoint of the same path (not the coordinate average)
hl.gen.mix('#ef4444', '#3b82f6', 0.5);

// Hues at fixed GenSpace lightness and chroma (categorical palettes)
hl.gen.hueRing(12, { lightness: 0.6, chroma: 0.15 });

// Harmonies: hue rotations at constant GenSpace L and C
hl.gen.harmonies('#3b82f6', 'triadic');      // also: complementary, analogous, tetradic, split_complementary
hl.gen.rotateHue('#3b82f6', 120);

// Cusp geometry
hl.gen.cusp(263);                            // [L, C] of the sRGB cusp at hue 263°
hl.gen.maxChroma(0.6, 263, 'display-p3');    // largest chroma at L = 0.6, hue 263°, in Display P3
hl.gen.vivid('#6488b8', { gamut: 'display-p3' }); // same L and hue, chroma raised to the boundary
```

## Wide gamut and tokens

```ts
// Color strings can also be 'color(display-p3 r g b)' or 'color(rec2020 r g b)':
hl.metric.info('color(display-p3 1 0 0)');   // { inSrgb: false, inP3: true, ... }

// Wide-gamut output:
const lab = hl.metric.fromHex('#ff0000');
hl.metric.toCss(lab, 'display-p3');          // 'color(display-p3 0.9176 0.2003 0.1386)'
hl.metric.toCss(lab, 'rec2020');
hl.metric.inGamut(lab, 'display-p3');

// Tokens: color strings in, platform strings out
hl.tokens.css('#3b82f6', 'oklch');           // 'oklch(62.3% 0.1881 259.8)'
hl.tokens.tailwind(hl.gen.scale('#3b82f6'), 'primary');
hl.tokens.cssVariables(hl.gen.scale('#3b82f6'), '--primary');
// also: css(c, 'hex' | 'rgb' | 'hsl' | 'p3' | 'rec2020') / android / iosP3 / swift / multiFormat / json
```

## Dark and light mode

```ts
hl.gen.adaptToMode('#3b82f6', 'light', 'dark');           // lightness inverted, hue kept
hl.gen.adaptPair('#3366ff', '#ffffff', 'light', 'dark');  // [fg, bg] with contrast restored
hl.gen.meetsContrast('#1e40af', '#ffffff', 'AAA');        // WCAG check without changing the color
hl.gen.ensureContrast('#3b82f6', '#808080', 7, { strict: true }); // throws ContrastError if no color reaches it
```

## Raw spaces

```ts
import { GenSpace, MetricSpace, srgbToXyz,
         compileGenParams, getDefaultGenParams,
         compileParams, getDefaultParams } from 'helmlab';

const gen = new GenSpace(compileGenParams(getDefaultGenParams()));
const metric = new MetricSpace(compileParams(getDefaultParams()));   // neutral-corrected (default)
const base = new MetricSpace(compileParams(getDefaultParams()), { neutralCorrection: false }); // base form
const lab = gen.fromXYZ(srgbToXyz([0.2, 0.5, 0.8]));   // raw Lab, no gamut mapping
```

Both constructors also accept custom parameter sets; see the [API reference](https://github.com/Grkmyldz148/helmlab/blob/main/API.md).

## Python

The [`helmlab` PyPI package](https://pypi.org/project/helmlab/) has the same API in snake_case (`hl.gen.ensureContrast` ↔ `hl.gen.ensure_contrast`). A parity test (`tests/parity-1.0.test.ts`) checks this package against outputs of the Python package. On that test set the string outputs are identical, and coordinates and differences agree to about 1e-12. Iterative searches (cusp, maximum chroma, gamut mapping) agree to their internal tolerance, up to 5e-4. The neutral-corrected inverse can differ by up to 4.0e-9, because this package rounds its correction table to 12 decimal places. Hex → Lab → hex round trips are lossless on all 16,777,216 8-bit sRGB colors in both spaces (checked 2026-11-08).

## Other implementations

These libraries are maintained outside Helmlab:

| Library | MetricSpace form |
|---|---|
| [Color.js](https://colorjs.io) 0.7.x (experimental, registered by default): `helmgen`, `helmgenlch`, `helmlab-metric` and a `"Helmlab"` ΔE method | Base form |
| [ColorAide](https://facelessuser.github.io/coloraide/) 8.9 and later (not registered by default) | Base form |
| [color-space](https://github.com/colorjs/color-space) 3.5.0 and 3.6.0, under the name `helmlab-metric` | Neutral-corrected form |

## Known limitations

- **MetricSpace is meant for SDR content** (Y ≤ 1 relative to diffuse white). Above diffuse white L grows quickly (about 1.80 at Y = 3 and 19 at Y = 10), and the neutral-corrected form keeps grays neutral only up to Y = 2 (gray chroma 0.11 at Y = 3, 0.14 from Y ≈ 4). `difference()`, `distance()` and `jnd()` use the base form and do not carry this gray chroma. Above diffuse white, use GenSpace.
- **`difference()` and `jnd()` near black and near white.** MetricSpace was fitted to COMBVD, where only 4 of the 3,813 pairs have a color darker than L* 10 and no color is lighter than L* 92.8. Near black its lightness axis is steep: `#010101` vs `#000000` gives `jnd()` 2.06 and `#ffffff` vs `#f7f7f7` gives 0.82 (CIEDE2000: 0.16 and 1.60). For pairs with a color darker than about L* 10, use `ciede2000()`. `palette()`, `scale()` and `gradient()` run in GenSpace and are not affected ([#5](https://github.com/Grkmyldz148/helmlab/issues/5)).
- **MetricSpace zero-chroma curve.** L is not continuous on a curve of grayish blues (y ≈ 0.276) where the internal chroma is 0. Inside the sRGB gamut (Y from 0.01 to 0.62) the limits of L differ by up to about 0.2, and a one-color call and a batch call can return different L for the same color.
- **GenSpace lightness.** The lightness correction is piecewise linear with 19 knots. Its slope jumps by up to 0.42 (at L = 0.15), and the darkest 5% of L is compressed about five times.
- **GenSpace and negative LMS.** Negative values after M1 are set to 0, so XYZ → Lab → XYZ changes such input: by up to 0.227 for the 566 of 20,000 random XYZ in [−0.05, 1.2]³ that have a negative component.
- **GenSpace inverse.** Iterative (Halley's method). Near gray the round trip changes the color by up to 2.5e-5 × Y; 8-bit hex round trips are lossless.
- **GenSpace in sRGB.** Small separate in-gamut regions near h ≈ 260°, from the roots of the cubic in the inverse. OKLab has the same kind of regions ([color.js#81](https://github.com/color-js/color.js/issues/81)).
- **`color(rec2020 …)` strings.** helmlab encodes and decodes them with the BT.2020 transfer function. CSS Color 4 defines `rec2020` with a 2.4 gamma (BT.1886), and Color.js follows it, so they read helmlab's Rec.2020 values darker: `#808080` comes out near `#6c6c6c`.
- **`jnd()` and `confidence()`.** `jnd()` divides ΔE `helmlab` by 0.0356, a value taken from a fixed confidence model (47 pairs, one session). `confidence()` uses the same model and is experimental.
- **Fitting data.** GenSpace was selected on ColorBench v1 win counts. MetricSpace was fitted to COMBVD, and MacAdam 1974 and He et al. 2022 were used in its fitting or model selection. Results on these datasets are in-sample.
- If you work directly in CSS `oklch()` or need a smaller bundle, OKLab/OKLCH is the simpler choice.

## License

MIT © [Görkem Yıldız](https://github.com/Grkmyldz148)
