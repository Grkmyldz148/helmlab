# helmlab

[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![npm downloads](https://img.shields.io/npm/dm/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![bundle size](https://img.shields.io/bundlephobia/minzip/helmlab)](https://bundlephobia.com/package/helmlab)
[![license](https://img.shields.io/npm/l/helmlab.svg)](https://github.com/Grkmyldz148/helmlab/blob/main/LICENSE)

A data-driven analytical color space for UI design systems. Two purpose-built spaces: **MetricSpace** for perceptual distance, **GenSpace** for gradients and palettes.

- **~12KB gzipped**, zero dependencies
- ESM + CJS dual output with full TypeScript types
- Trained on 64,000+ human color-difference judgments
- MetricSpace: STRESS 23.30 vs CIEDE2000's 29.18 (20% better)
- `gradient()` with CIEDE2000 arc-length reparameterization (CV ≈ 0% on any pair)

**[Website](https://helmlab.space)** · **[Documentation](https://grkmyldz148.github.io/helmlab/)** · **[npm](https://www.npmjs.com/package/helmlab)**

## Install

```bash
npm install helmlab
```

Also available via CDN:

```html
<script type="module">
  import { Helmlab } from 'https://esm.sh/helmlab';
</script>
```

## Quick Start

```ts
import { Helmlab } from 'helmlab';

const hl = new Helmlab();

// Convert (MetricSpace — perceptual distance)
const lab = hl.fromHex('#3B82F6');       // → [0.713, -0.306, -0.387]
const hex = hl.toHex([0.5, -0.1, 0.2]); // → '#rrggbb'

// Convert (GenSpace — gradients & palettes)
const glab = hl.genFromHex('#3B82F6');   // → GenSpace Lab
const ghex = hl.genToHex(glab);          // → '#rrggbb' (gamut mapped)

// Contrast (WCAG)
hl.contrastRatio('#ffffff', '#3B82F6');            // → 3.68
hl.ensureContrast('#3B82F6', '#ffffff', 4.5);      // → adjusted hex meeting 4.5:1
hl.meetsContrast('#000000', '#ffffff', 'AA');       // → true

// Distance
hl.deltaE('#ff0000', '#00ff00');  // → 1.09

// Gradient (GenSpace + arc-length reparameterization)
hl.gradient('#ff0000', '#0000ff', 8);  // → 8 perfectly uniform hex steps

// Palette
hl.palette('#3B82F6', 5);        // → ['#c4d5ff', '#7eaafc', '#3b82f6', '#0060d0', '#003d8e']
hl.semanticScale('#3B82F6');      // → { '50': '#b3c7ff', '500': '#3b82f6', '900': '#00184b', ... }
hl.paletteHues(0.6, 0.15, 12);   // → 12 evenly-spaced hues
```

## API

### Core Conversions

| Method | Description |
|--------|-------------|
| `fromHex(hex)` | Hex → MetricSpace Lab |
| `toHex(lab)` | MetricSpace Lab → hex (gamut mapped) |
| `fromSrgb(rgb)` | sRGB [0,1] → MetricSpace Lab |
| `toSrgb(lab)` | MetricSpace Lab → sRGB [0,1] (gamut mapped) |
| `fromXYZ(xyz)` | CIE XYZ → MetricSpace Lab |
| `toXYZ(lab)` | MetricSpace Lab → CIE XYZ |
| `genFromHex(hex)` | Hex → GenSpace Lab |
| `genToHex(lab)` | GenSpace Lab → hex (gamut mapped) |
| `toDisplayP3(lab)` | MetricSpace Lab → Display P3 [0,1] |

### Contrast

| Method | Description |
|--------|-------------|
| `contrastRatio(fg, bg)` | WCAG contrast ratio (1–21) |
| `ensureContrast(fg, bg, ratio)` | Adjust fg to meet minimum ratio |
| `meetsContrast(fg, bg, level)` | Check AA (4.5:1) or AAA (7:1) |

### Gradient & Palette

| Method | Description |
|--------|-------------|
| `gradient(start, end, steps)` | Uniform gradient via CIEDE2000 arc-length reparameterization (CV ≈ 0%) |
| `palette(hex, steps)` | Lightness ramp from base color |
| `semanticScale(hex)` | Tailwind-style 50–950 scale |
| `paletteHues(L, C, steps)` | Hue ring at fixed L and chroma |

### Distance & Info

| Method | Description |
|--------|-------------|
| `deltaE(hex1, hex2)` | Euclidean distance in Helmlab Lab |
| `perceptualDistance(lab1, lab2)` | Full perceptual metric (Minkowski + compression) |
| `info(hex)` | Color info: Lab, L, C, H |
| `isInSrgb(lab)` | Gamut check |

## Advanced Usage

Lower-level exports are available for custom pipelines:

```ts
import {
  MetricSpace, GenSpace, compileParams,
  getDefaultParams, getDefaultGenParams,
  hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb,
  gamutMap, isInGamut, contrastRatio,
} from 'helmlab';
```

## How It Works

Helmlab is a family of two purpose-built color spaces:

**MetricSpace** (72 parameters) — optimized for perceptual distance:
```
XYZ → M₁ → γ → M₂ → Hue → H-K → L → C → HL → NC → φ → Lab
```

**GenSpace** (21 parameters) — optimized for generation (gradients, palettes):
```
XYZ → M₁ → γ=⅓ → M₂ → NC → Lab
+ CIEDE2000 arc-length reparameterization for gradient()
```

MetricSpace is trained on 64,000+ human color-difference observations (COMBVD + 6 datasets). Every stage is exactly invertible. GenSpace uses Phase1H-optimized matrices for 6× better hue accuracy than Oklab (5.2° vs 30.1° RMS).

### Gradient Uniformity

CV (coefficient of variation of CIEDE2000 step sizes). Lower is better.

| Method | Red→Blue | Orange→Cyan | Black→White | Technique |
|--------|----------|-------------|-------------|-----------|
| **Helmlab `gradient()`** | **≈ 0%** | **≈ 0%** | **≈ 0%** | arc-length reparam. |
| Helmlab GenSpace | 3.1% | 33.2% | 41.0% | linear interpolation |
| Oklab | 31.5% | 41.4% | 41.2% | linear interpolation |
| CIE Lab | 44.8% | 52.3% | 61.5% | linear interpolation |

> **Note:** `gradient()` achieves ≈ 0% via CIEDE2000 arc-length reparameterization — an algorithm that redistributes steps to equal perceptual spacing. The same technique could be applied to any space; Helmlab ships it built-in.

<details>
<summary><strong>How was STRESS 23.30 measured?</strong></summary>

STRESS (Standardized Residual Sum of Squares) is the CIE-standard metric for evaluating color difference formulas. **COMBVD** is a combined dataset of 3,813 color pairs from 6 psychophysical experiments (64,000+ human judgments). Observers viewed color pairs under controlled D65 lighting and rated perceived differences. STRESS = 100 × √( Σ(ΔEᵢ − F·ΔVᵢ)² / Σ(ΔEᵢ)² ), where 0 = perfect and 100 = no correlation. Helmlab's 72 parameters were optimized with L-BFGS-B (8 random restarts). 5-fold CV confirms generalization (mean ≈ 23.5). Bootstrap 95% CI: Helmlab [22.50, 23.93] vs CIEDE2000 [27.64, 30.84] — zero overlap, p < 10⁻⁴. Full methodology: [arXiv:2602.23010](https://arxiv.org/abs/2602.23010).

</details>

## PostCSS Plugin

Use Helmlab color spaces directly in your CSS:

```bash
npm install postcss-helmlab
```

```css
.card { color: helmlab(0.78 0.52 -0.20); }
.bg   { background: linear-gradient(in helmgen, #e63946, #457b9d); }
```

Transforms to `rgb()` at build time. Supports all four spaces, alpha, gradients, and `color-mix()`. See [postcss-helmlab](https://www.npmjs.com/package/postcss-helmlab).

## License

MIT
