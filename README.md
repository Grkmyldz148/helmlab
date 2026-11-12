# Helmlab

Helmlab defines color spaces for generating colors (gradients, palettes, gamut mapping) and for measuring color differences, and publishes their reference code. This repository holds that code: the `helmlab` package for JavaScript/TypeScript and Python, and the PostCSS plugin `postcss-helmlab`.

[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![PyPI version](https://img.shields.io/pypi/v/helmlab.svg)](https://pypi.org/project/helmlab/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.23010-b31b1b.svg)](https://arxiv.org/abs/2602.23010)

[helmlab.space](https://helmlab.space) · [API reference](https://github.com/helmlab-space/helmlab/blob/main/API.md) · [Changelog](https://github.com/helmlab-space/helmlab/blob/main/CHANGELOG.md) · [Paper](https://arxiv.org/abs/2602.23010) · [Organization](https://github.com/helmlab-space)

This README gives no performance figures. They will be published on one evidence page, generated from one file. Each figure there will name its dataset, its size (n), and whether that data was used to fit, tune or design the model.

## Current definition: Helmlab 2026

Helmlab 2026 is defined by the outputs of `helmlab` 1.0.0 on npm and PyPI. Later releases keep these outputs; `helmlab` 1.0.1 has identical outputs, and only its JavaScript speed and documentation changed.

| Identifier | What it names |
|---|---|
| `helmgen` | GenSpace in Lab coordinates, for generating colors |
| `helmgenlch` | GenSpace in LCh coordinates (lightness, chroma, hue) |
| `helmlab-metric` | MetricSpace in its base form, for measuring color differences |
| ΔE `helmlab` | Color difference computed from `helmlab-metric` coordinates |
| `helmlab-metric-neutral` | MetricSpace with the neutral correction (reserved) |

A specification and test vectors for Helmlab 2026 are in preparation.

### `hl.metric` returns the neutral-corrected form

By default `hl.metric` returns `helmlab-metric-neutral`, not the base form `helmlab-metric`. The neutral correction subtracts the (a, b) of grays at the same lightness, so grays get a and b close to 0 (up to Y = 2); in the base form they do not. The same (L, a, b) triple therefore names different colors in the two forms:

| (L, a, b) | `hl.metric` (neutral-corrected) | Base form `helmlab-metric` |
|---|---|---|
| (0.7779, 0, 0) | `#808080` | `#759dbf` |
| (0.7779, 0.2358, 0.1674) | `#92765a` | `#808080` |

`hl.metric.difference()`, `distance()` and `jnd()` remove the correction before measuring, so ΔE `helmlab` is the same for both forms. For base-form coordinates, use `MetricSpace()` from `helmlab.spaces.metric` in Python (its default is the base form) or `new MetricSpace(compileParams(getDefaultParams()), { neutralCorrection: false })` in JavaScript.

## Install

```bash
npm install helmlab          # JavaScript / TypeScript
pip install helmlab          # Python
npm install postcss-helmlab  # PostCSS plugin
```

The npm package has no dependencies and ships ESM and CJS builds with TypeScript types.

## Quick start

One `Helmlab` instance has three namespaces: `hl.gen` generates colors (GenSpace), `hl.metric` measures them (MetricSpace) and `hl.tokens` exports design tokens. Lab values are typed per space: passing a `hl.metric` value to `hl.gen` raises a `TypeError`.

```ts
import { Helmlab } from 'helmlab';

const hl = new Helmlab();

// Generate (GenSpace)
hl.gen.gradient('#ff0000', '#0000ff', 8);                          // steps of about equal CIEDE2000 length
hl.gen.gradient('#0000ff', '#ffffff', 8, { gamut: 'display-p3' }); // Display P3 output
hl.gen.scale('#3b82f6');                                           // levels 50–950; level 500 is the input
hl.gen.harmonies('#3b82f6', 'triadic');                            // hue rotations at constant L and C
hl.gen.vivid('#6488b8', { gamut: 'display-p3' });                  // chroma raised to the gamut boundary
hl.gen.ensureContrast('#3b82f6', '#ffffff', 4.5);                  // '#3071e4': 4.5:1 against white

// Measure (MetricSpace)
hl.metric.difference('#ff0000', '#00ff00');                        // ΔE helmlab: 0.148
hl.metric.jnd('#808080', '#828282');                               // ΔE helmlab / 0.0356: 0.329
hl.metric.ciede2000('#ff0000', '#00ff00');                         // CIEDE2000: 86.6

// Export
hl.tokens.css('#3b82f6', 'oklch');                                 // 'oklch(62.3% 0.1881 259.8)'
```

The Python API is the same in snake_case:

```python
from helmlab import Helmlab

hl = Helmlab()

hl.gen.gradient("#ff0000", "#0000ff", 8)
hl.gen.mix("#ff0000", "#0000ff")                         # midpoint on the gradient path: '#a7528b'
hl.gen.cusp(263)                                         # (L, C) of the sRGB cusp at hue 263°
hl.gen.ensure_contrast("#3b82f6", "#ffffff", ratio=4.5)  # '#3071e4'
hl.metric.difference("#ff0000", "#00ff00")               # 0.148
hl.metric.confidence("#808080", "#828282")               # experimental: how much observers may disagree
hl.metric.info("color(display-p3 1 0 0)")                # Display P3 input
hl.tokens.css_variables(hl.gen.scale("#3b82f6"), "--primary")
```

Full surface: [API.md](https://github.com/helmlab-space/helmlab/blob/main/API.md). Upgrading from 0.x: the flat `hl.gradient()` / `hl.from_hex()` API was removed in 1.0; every method lives on `hl.gen`, `hl.metric` or `hl.tokens`.

### PostCSS

```css
/* Input */
.btn { color: helmlab(0.7221 -0.2468 -0.2378); }
.bg  { background: linear-gradient(in helmgen, #e63946, #457b9d); }

/* Output (each rule also gets a Display P3 version inside @supports) */
.btn { color: rgb(59, 130, 246); }
.bg  { background: linear-gradient(#e63946 0.0%, #cd525a 11.1%, ..., #457b9d 100.0%); }
```

`helmlab()` takes `hl.metric` coordinates (the neutral-corrected form). `in helmgen` interpolates in GenSpace.

## How the spaces are built

**GenSpace (`helmgen`)**, used by `hl.gen`:

```
XYZ → M1 (negative values set to 0) → cubic transfer y³ + αy = x (α = 0.021) → neutral blend → M2
    → piecewise-linear lightness correction (19 knots) → lightness-gated hue rotation (blue region)
    → chroma compression (C^0.978) → (L, a, b)
```

- The transfer has a finite slope at 0, unlike a cube root. Its root is computed in closed form and refined by one Halley step.
- The hue rotation is aimed at the purple shift of blue-to-white gradients: `hl.gen.mix('#0000ff', '#ffffff')` returns `#649cff`.
- Grays have |a| and |b| of at most 2.5e-15 for 0 ≤ Y ≤ 10.
- The inverse is iterative (Halley's method).
- `gradient()` reparameterizes the path by CIEDE2000 arc length, so its steps have about equal CIEDE2000 length. `mix()` uses the same path.
- GenSpace was selected with CMA-ES on ColorBench v1 win counts.

**MetricSpace (`helmlab-metric`)**, used by `hl.metric`:

```
XYZ → M1 → per-channel power → M2 → hue-dependent rotation → chroma-dependent lightness
    → lightness corrections → chroma scales and power → hue-dependent lightness scale
    → rotation by −28.2° → (L, a, b)
```

- The parameters were fitted with CMA-ES to COMBVD, MacAdam 1974 and Helmlab's own Human Feedback data, weighted 0.55, 0.15 and 0.30; He et al. 2022 was used in model selection.
- The neutral-corrected form (`hl.metric`) adds one stage before the rotation. It subtracts gray (a, b) values, interpolated from a table by lightness.
- ΔE `helmlab` is a separate formula on base-form coordinates. It compresses large differences: values level off near 0.15, so it ranks large differences but does not scale them. `hl.metric.euclidean()` gives the uncompressed Euclidean distance.

## Other implementations

These libraries are maintained outside Helmlab:

| Library | MetricSpace form |
|---|---|
| [Color.js](https://colorjs.io) 0.7.x (experimental, registered by default) | Base form (`helmlab-metric`) |
| [ColorAide](https://facelessuser.github.io/coloraide/) 8.9 and later (not registered by default) | Base form |
| [color-space](https://github.com/colorjs/color-space) 3.5.0 and 3.6.0, under the name `helmlab-metric` | Neutral-corrected form |

## Known limitations

- **GenSpace lightness.** The lightness correction is piecewise linear with 19 knots. Its slope jumps by up to 0.42 (at L = 0.15), and the darkest 5% of L is compressed about five times.
- **GenSpace and negative LMS.** Negative values after M1 are set to 0, so XYZ → Lab → XYZ changes such input: by up to 0.227 for the 566 of 20,000 random XYZ in [−0.05, 1.2]³ that have a negative component.
- **GenSpace inverse.** Iterative (Halley's method). Near gray the round trip changes the color by up to 2.5e-5 × Y; very dark colors (Y below about 1e-35) are moved towards gray. 8-bit hex round trips are lossless (see below).
- **GenSpace in sRGB.** Small separate in-gamut regions near h ≈ 260°, from the roots of the cubic in the inverse. OKLab has the same kind of regions ([color.js#81](https://github.com/color-js/color.js/issues/81)).
- **MetricSpace is meant for SDR content** (Y ≤ 1 relative to diffuse white). Above diffuse white L grows quickly (about 1.80 at Y = 3 and 19 at Y = 10), and the neutral-corrected form keeps grays neutral only up to Y = 2 (gray chroma 0.11 at Y = 3, about 0.14 for Y from 4 to 10, above 0.9 from Y ≈ 10.92 and 3.8 at Y = 100). `difference()`, `distance()` and `jnd()` use the base form and do not carry this gray chroma. Above diffuse white, use GenSpace.
- **`difference()` and `jnd()` near black and near white.** Of the 3,989 color pairs MetricSpace was fitted to, 7 have a color darker than L* 10 and none has a color lighter than L* 92.8. On the 4 near-black, near-neutral pairs among them (COMBVD), ΔE `helmlab` matches the visual differences at L* 6–10 but predicts 1.7–2.5 times the observed difference below L* 2.5; CIEDE2000 predicts 0.4–0.7 times on all four. Across very different lightness levels the two disagree: `#010101` vs `#000000` gives `jnd()` 2.06 and `#ffffff` vs `#f7f7f7` gives 0.82, while CIEDE2000 gives 0.16 and 1.60, closer to the Munsell value scale and CIE L* ([#5](https://github.com/helmlab-space/helmlab/issues/5)). `palette()`, `scale()` and `gradient()` run in GenSpace and are not affected.
- **MetricSpace zero-chroma curve.** L is not continuous on a curve of grayish blues (y ≈ 0.276) where the internal chroma is 0. Inside the sRGB gamut (Y from 0.01 to 0.62) the limits of L differ by up to about 0.2, and a one-color call and a batch call can return different L for the same color.
- **`color(rec2020 …)` strings.** helmlab encodes and decodes them with the BT.2020 transfer function. CSS Color 4 defines `rec2020` with a 2.4 gamma (BT.1886), and Color.js follows it, so they read helmlab's Rec.2020 values darker: `#808080` comes out near `#6c6c6c`.
- **`jnd()` and `confidence()`.** `jnd()` divides ΔE `helmlab` by 0.0356, a value taken from a fixed confidence model (47 pairs, one session). `confidence()` uses the same model and is experimental.
- **Fitting data.** GenSpace was selected on ColorBench v1 win counts. MetricSpace was fitted to COMBVD, MacAdam 1974 and Helmlab's own Human Feedback data (3,552 judgements of 48 color pairs, made on observers' own uncalibrated screens); He et al. 2022 was used in model selection. Results on these datasets are in-sample.
- **Paper erratum.** The paper describes the neutral correction with 384 gray levels for Y from 0.001 to 2.59. The code uses 255 levels and holds the correction constant above L ≈ 1.341. The code governs ([helmlab#4](https://github.com/helmlab-space/helmlab/issues/4)). The paper also calls MacAdam 1974 held-out; the fitting script used it, together with the Human Feedback data (see Fitting data).

## Python and JavaScript

A parity test (`packages/helmlab-js/tests/parity-1.0.test.ts`) checks the JavaScript package against outputs of the Python package (`packages/helmlab-js/tests/reference/reference-1.0.json`). On that test set the string outputs (hex, CSS, tokens) are identical, and coordinates and differences agree to about 1e-12. Iterative searches (cusp, maximum chroma, gamut mapping) agree to their internal tolerance, up to 5e-4. The neutral-corrected inverse can differ by up to 4.0e-9, because the JavaScript correction table is rounded to 12 decimal places. Hex → Lab → hex round trips are lossless on all 16,777,216 8-bit sRGB colors in both spaces (JavaScript, checked 2026-11-08).

## Agent skill

[`skills/color-space-routing`](https://github.com/helmlab-space/helmlab/tree/main/skills/color-space-routing) is an agent skill (a `SKILL.md` file) for coding assistants: which color space to use for which task, with recipes and a checklist of pitfalls. It is also published in [helmlab-space/color-skills](https://github.com/helmlab-space/color-skills).

## Repository layout

```
src/helmlab/               Python package: helmlab.py (API), spaces/, metrics/, utils/, tokens.py, data/
packages/helmlab-js/       npm package (TypeScript)
packages/postcss-helmlab/  PostCSS plugin
tests/                     Python tests
tools/golden/              Output comparison across package versions
skills/                    Agent skills
paper/                     Paper source
site/                      helmlab.space
```

`src/helmlab/spaces/` also holds baseline spaces used for comparison (CAM16-UCS, IPT, JzCzhz, OKLCh). The evaluation suite, ColorBench, has its own repository: [Grkmyldz148/colorbench](https://github.com/Grkmyldz148/colorbench).

## Development

```bash
python -m pytest tests/ -q
cd packages/helmlab-js && npx vitest run
```

## Research archive

[helmlab-space/helmlab-experimental](https://github.com/helmlab-space/helmlab-experimental) holds the experiments, checkpoints and scripts from the development of GenSpace. It is archived and read-only. Its win counts are pre-v4 ColorBench figures and are withdrawn.

## Citation

```bibtex
@article{yildiz2026helmlab,
  title={Helmlab: A Two-Space Family of Analytical, Data-Driven Color Spaces for UI Design Systems},
  author={Y{\i}ld{\i}z, G{\"o}rkem},
  journal={arXiv preprint arXiv:2602.23010},
  year={2026},
  url={https://arxiv.org/abs/2602.23010}
}
```

## License

Code: MIT ([LICENSE](https://github.com/helmlab-space/helmlab/blob/main/LICENSE)). Paper text and figures: CC BY 4.0 ([paper/LICENSE.md](https://github.com/helmlab-space/helmlab/blob/main/paper/LICENSE.md)).

## Contact

Questions and bug reports: [open an issue](https://github.com/helmlab-space/helmlab/issues). Maintained by [Görkem Yıldız](https://gorkemyildiz.com).

How this project is run, including its use of AI tools: [GOVERNANCE.md](https://github.com/helmlab-space/.github/blob/main/GOVERNANCE.md).
