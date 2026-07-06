---
name: helmlab
description: Correct usage of the helmlab color library (npm + PyPI) — two-space model (GenSpace for generation, MetricSpace for measurement), full API with verified outputs, and the gotchas that make AI-generated helmlab code fail (exporter input space, distance input contracts, 0–1 L scale). Use whenever code imports or should import helmlab.
---

# helmlab — correct API usage

Perceptual color library, JS (`npm i helmlab`, 17.8KB, zero deps) and Python (`pip install helmlab`) with **identical outputs** (cross-checked to ~1e-15). JS is camelCase, Python snake_case; same math.

## The one mental model you need

One `Helmlab` class wraps TWO spaces — never mix their Lab coordinates:

| Purpose | Methods | Space |
|---|---|---|
| CREATE colors (gradients, palettes, contrast fixes, dark mode) | `gradient` `palette` `paletteHues` `semanticScale` `ensureContrast` `adaptToMode` `adaptPair` + `genFromHex/genToHex/genToLch/genFromLch` | **GenSpace** |
| MEASURE colors (ΔE, info, wide-gamut output, tokens) | `difference` `deltaE` `euclideanDistance` `perceptualDistance` `differenceWithConfidence` `info` `toHexP3` `toHexRec2020` `isInP3` + `fromHex/toHex` | **MetricSpace** |

`fromHex` ≠ `genFromHex`. Feeding Gen Lab into a Metric-Lab API silently produces wrong colors.

## Verified quick reference (JS; Python is the snake_case mirror)

```js
import { Helmlab } from 'helmlab';
const hl = new Helmlab();

// Generation
hl.gradient('#0000ff', '#ffffff', 16);      // equal visual steps, stays blue
hl.semanticScale('#3b82f6');                // {'50':'#e7efff', …'500':'#3b82f6' EXACT, …'950':'#000046'}
hl.palette('#3b82f6', 10);                  // lightness ramp, LIGHT → DARK ('#fafcff' … '#000046')
hl.paletteHues(0.6, 0.15, 12);              // categorical hue ring
hl.ensureContrast('#3b82f6', '#ffffff');    // darkens to 4.5:1, hue kept
hl.adaptToMode('#3b82f6', 'light', 'dark'); // '#2a67d9'

// LCh (cylindrical GenSpace — hue rotations/harmonies)
const lch = hl.genToLch(hl.genFromHex('#3b82f6')); // [0.5586, 0.2976, 263.1]  ⚠ L and C are 0–1-scale, NOT 0–100
hl.genToHex(hl.genFromLch([lch[0], lch[1], (lch[2] + 120) % 360]));

// Measurement
hl.difference('#ff0000', '#00ff00');        // 0.148 — trained metric (COMBVD-fit), saturates ~0.15
hl.euclideanDistance('#000000', '#ffffff'); // 1.12 — fast ΔE76-style (alias of deltaE)
hl.contrastRatio('#ffffff', '#3b82f6');     // 3.68 (WCAG 2.1)
hl.differenceWithConfidence('#808080', '#828282');
// { de:0.0117, latent, disagreement, reliability:0.41, pNoticeable:0.077, reliable:false, extrapolated:false }

// Wide gamut + tokens (METRIC Lab in!)
const lab = hl.fromHex('#ff0000');
hl.toHexP3(lab);                            // 'color(display-p3 0.9176 0.2003 0.1386)'
const ex = hl.export();                     // TokenExporter — takes METRIC Lab
ex.toCssOklch(lab);                         // 'oklch(62.8% 0.2576 29.2)'
ex.exportTailwind(hl.semanticScale('#3b82f6'), 'primary');
ex.exportCssCustomProperties(hl.semanticScale('#3b82f6'), '--primary');
```

## Gotchas (every one observed breaking real AI-generated code)

1. **TokenExporter takes METRIC Lab** (`hl.fromHex`). Passing `genFromHex` output silently yields a different color (#3b82f6 → #4d2268).
2. **`distance()` input contract differs by language**: Python `MetricSpace.distance(XYZ1, XYZ2)` takes CIE XYZ (Lab input silently saturates ≈0.15); JS `distance(lab1, lab2)` takes Lab. Cross-language-safe: `distance_from_lab` / `distanceFromLab`, or the hex-in `difference()`.
3. **GenSpace L and C are 0–1-scale.** `L - 15` gives black; darken with e.g. `L - 0.15`.
4. **`difference()` saturates near ~0.15** for very dissimilar pairs — rank is preserved, absolute values plateau. For an unbounded number use `euclideanDistance` (range 0–1.6+).
5. **`palette()` runs light→dark** and the base color is only approximate inside it; `semanticScale()` is the Tailwind-style API where level 500 is your exact input.
6. **These APIs don't exist** (common hallucinations): `hl.hex2lab`, `hl.rgbToLab`, `hl('#hex').mix()`, `hl.mix`. Conversions are `fromHex/genFromHex` etc.
7. **Raw classes need compiled params in JS**: `new GenSpace(compileGenParams(getDefaultGenParams()))` — bare `new GenSpace()` throws. Python: `GenSpace()` works; custom params via `GenParams.load(path)`, not a raw dict.
8. `export_css_custom_properties(scale, prefix="--primary")` — include the `--` yourself.

## Ecosystem

- **Color.js** (master, pending release): spaces `helmgen`, `helmgenlch`, `helmlab-metric` + deltaE method `"Helmlab"`. Until released: `npm i github:color-js/color.js` or use this package.
- **PostCSS**: `npm i postcss-helmlab` → `helmlab()`, `helmlch()`, `helmgen()`, `helmgenlch()` CSS functions with P3/Rec2020 `@supports` fallbacks.
- Python extras: `pip install 'helmlab[datasets]'` for benchmark data loaders; `py.typed` shipped (mypy/pyright OK).

## When NOT to use helmlab

Near-achromatic gradient mastering and deutan-safe palettes → OKLab is measurably better. CSS-only with no JS budget → native `oklch()`. HDR/PQ → Jzazbz/ICtCp. For the full task-routing table with benchmark numbers, see the companion `color-space-routing` skill. Docs: helmlab.space/docs · benchmark honesty tables: helmlab.space/benchmark
