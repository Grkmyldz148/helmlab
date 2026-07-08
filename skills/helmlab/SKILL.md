---
name: helmlab
description: Correct usage of the helmlab color library (npm + PyPI) — 1.0 namespaced API (hl.gen / hl.metric / hl.tokens), branded Lab types, wide-gamut generation, and the measurement metrics (difference/jnd/ciede2000/confidence). Use whenever code imports or should import helmlab.
---

# helmlab — correct API usage (1.0)

Perceptual color library, JS (`npm i helmlab`, zero deps) and Python
(`pip install helmlab`) with **identical outputs** — a permanent parity gate
verifies the full surface: every string output byte-identical, numeric worst
~1e-12, hex round-trips bit-exact on a 1728-color grid in both languages.
JS is camelCase, Python snake_case; same math, same namespaces.

## The one mental model you need

One `Helmlab` instance, **three namespaces**. Color strings in, color
strings out — Lab is the advanced layer:

| Namespace | Purpose | Key methods |
|---|---|---|
| `hl.gen` | CREATE colors (GenSpace) | `gradient` `mix` `palette` `scale` `hueRing` `harmonies` `rotateHue` `vivid` `cusp` `maxChroma` `gamutMap` `ensureContrast` `adaptToMode` `adaptPair` |
| `hl.metric` | MEASURE colors (MetricSpace) | `difference` `euclidean` `ciede2000` `jnd` `distance` `confidence` `nearest` `info` `toCss` `inGamut` `toLch`/`fromLch` |
| `hl.tokens` | EXPORT design tokens | `css` `android` `iosP3`/`ios_p3` `swift` `cssVariables` `tailwind` `multiFormat` `json` |

Each namespace has its own `fromHex`; their Lab types are **branded**
(`GenLab` / `MetricLab`) — cross-passing raises `TypeError` instead of
silently producing a wrong color (this replaces the entire 0.x
`fromHex` vs `genFromHex` footgun family).

Color-string params accept `'#rrggbb'`, `'#rgb'`,
`'color(display-p3 r g b)'`, `'color(rec2020 r g b)'` **everywhere**.

## Verified quick reference (JS; Python is the snake_case mirror)

```js
import { Helmlab } from 'helmlab';
const hl = new Helmlab();

// Generation — every function takes { gamut: 'srgb'|'display-p3'|'rec2020' }
hl.gen.gradient('#0000ff', '#ffffff', 16);           // equal visual steps, stays blue
hl.gen.gradient('#00f', '#fff', 8, { gamut: 'display-p3' }); // wide-gamut stops
hl.gen.mix('#ff0000', '#0000ff');                    // visual midpoint (same path as gradient)
hl.gen.scale('#3b82f6');                             // {'50':'#e7efff', …'500':'#3b82f6' EXACT, …'950':'#000046'}
hl.gen.palette('#3b82f6', 10);                       // lightness ramp, LIGHT → DARK
hl.gen.hueRing(12, { lightness: 0.6, chroma: 0.15 }); // categorical hue ring
hl.gen.harmonies('#3b82f6', 'triadic');              // ['#3b82f6','#f13046','#00a250'] — constant L,C
hl.gen.vivid('#6488b8', { gamut: 'display-p3' });    // same L & hue, chroma → gamut boundary
hl.gen.cusp(263);                                    // [0.491, 0.379] most colorful point of a hue
hl.gen.ensureContrast('#3b82f6', '#ffffff');         // darkens to 4.5:1, hue kept
hl.gen.ensureContrast('#3b82f6', '#808080', 7, { strict: true }); // throws ContrastError if unreachable
hl.gen.adaptToMode('#3b82f6', 'light', 'dark');      // '#2a67d9'

// LCh (cylindrical GenSpace) — ⚠ L and C are 0–1-scale, NOT 0–100
const lch = hl.gen.toLch(hl.gen.fromHex('#3b82f6')); // [0.5586, 0.2976, 263.1]
hl.gen.rotateHue('#3b82f6', 120);                    // prefer this over manual LCh math

// Measurement — four clearly-named metrics
hl.metric.difference('#ff0000', '#00ff00');   // 0.148 — trained (COMBVD-fit), saturates ~0.15
hl.metric.euclidean('#000000', '#ffffff');    // 1.12 — unbounded ΔE76-analogue
hl.metric.ciede2000('#ff0000', '#00ff00');    // 86.6 — industry standard, best for catalog argmax
hl.metric.jnd('#808080', '#828282');          // 0.33 — threshold units (<1 unnoticed, >2 clearly visible)
hl.metric.confidence('#808080', '#828282');   // { de, pNoticeable: 0.077, reliability: 0.41, reliable: false, extrapolated }
hl.gen.contrastRatio('#ffffff', '#3b82f6');   // 3.68 (WCAG 2.1) — contrast lives on gen

// Wide gamut, both directions
hl.metric.info('color(display-p3 1 0 0)');    // { inSrgb: false, inP3: true, ... }
hl.metric.toCss(hl.metric.fromHex('#ff0000'), 'display-p3'); // 'color(display-p3 0.9176 0.2003 0.1386)'

// Tokens — color STRINGS in (never Lab)
hl.tokens.css('#3b82f6', 'oklch');            // 'oklch(62.3% 0.1881 259.8)'
hl.tokens.tailwind(hl.gen.scale('#3b82f6'), 'primary');
hl.tokens.cssVariables(hl.gen.scale('#3b82f6'), '--primary'); // include the -- yourself
```

## Gotchas

1. **0.x flat API is GONE** (1.0 clean break): `hl.fromHex`, `hl.gradient`,
   `hl.semanticScale`, `hl.deltaE`, `hl.genFromHex`, `hl.export()`,
   `TokenExporter`, `hl.toHexP3` no longer exist. Mapping: root conversions
   → `hl.metric.*`; generation → `hl.gen.*` (`semanticScale`→`scale`,
   `paletteHues`→`hueRing(count, {...})`); `deltaE/euclideanDistance` →
   `metric.euclidean`; `deltaE2000` → `metric.ciede2000`;
   `perceptualDistance/distanceFromLab` → `metric.distance`;
   `differenceWithConfidence` → `metric.confidence`; `nearestColor` →
   `metric.nearest`; exporter → `hl.tokens` (hex-in).
2. **GenSpace L and C are 0–1-scale.** `L - 15` gives black; darken with
   e.g. `L - 0.15`.
3. **`difference()` saturates near ~0.15** for very dissimilar pairs — rank
   preserved, absolute values plateau. Unbounded: `euclidean` (0–1.6+);
   threshold units: `jnd`.
4. **`palette()` runs light→dark and the base is only approximate inside
   it**; `scale()` is the Tailwind-style API where level 500 is exact.
5. **`metric.distance(labA, labB)` takes MetricLab in BOTH languages** —
   the 0.x Python-XYZ/JS-Lab asymmetry is gone. XYZ-in lives only on
   Python's raw `MetricSpace` class.
6. **Cross-space Lab throws.** If you see
   `TypeError: hl.metric got a GenLab`, convert the original color with the
   right namespace's `fromHex` — don't cast.
7. **Raw classes** (advanced): `hl.gen.space` / `hl.metric.space` expose
   the live GenSpace/AnalyticalSpace; constructing raw JS spaces still needs
   compiled params (`new GenSpace(compileGenParams(getDefaultGenParams()))`).
8. `ensureContrast` fallback: if the ratio is unreachable even with pure
   black/white, default warns and returns best effort; `strict: true` (JS
   opts / Python kwarg) raises `ContrastError`.

## Ecosystem

- **PostCSS**: `npm i postcss-helmlab` → `helmlab()`, `helmlch()`,
  `helmgen()`, `helmgenlch()` CSS functions; since 1.0 the P3/Rec2020
  output uses the TRUE gamut-mapped paths of both spaces.
- **Color.js** (master): spaces `helmgen`, `helmgenlch`, `helmlab-metric` +
  deltaE method `"Helmlab"`.
- Python extras: `pip install 'helmlab[datasets]'`; `py.typed` shipped.

## When NOT to use helmlab

Near-achromatic gradient mastering and deutan-safe palettes → OKLab is
measurably better. CSS-only with no JS budget → native `oklch()`. HDR/PQ →
Jzazbz/ICtCp. Full routing table: the companion `color-space-routing`
skill. Docs: helmlab.space/docs · honesty tables: helmlab.space/benchmark
