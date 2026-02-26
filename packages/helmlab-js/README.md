# helmlab

[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![npm downloads](https://img.shields.io/npm/dm/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![bundle size](https://img.shields.io/bundlephobia/minzip/helmlab)](https://bundlephobia.com/package/helmlab)
[![license](https://img.shields.io/npm/l/helmlab.svg)](https://github.com/Grkmyldz148/helmlab/blob/main/LICENSE)

Perceptual color space for UI design. Human-tuned Lab with WCAG contrast, gamut mapping, and palette generation.

- **10KB gzipped**, zero dependencies
- ESM + CJS dual output with full TypeScript types
- Trained on 64,000+ human color-difference judgments
- Beats CIEDE2000 and Oklab on perceptual accuracy (STRESS 23.2 vs 29.2 vs 27.5)

**[Documentation](https://grkmyldz148.github.io/helmlab/)** · **[npm](https://www.npmjs.com/package/helmlab)** · **[Interactive Demo](https://grkmyldz148.github.io/helmlab/demo.html)**

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

// Convert
const lab = hl.fromHex('#3B82F6');       // → [0.713, -0.306, -0.387]
const hex = hl.toHex([0.5, -0.1, 0.2]); // → '#rrggbb'

// Contrast (WCAG)
hl.contrastRatio('#ffffff', '#3B82F6');            // → 3.68
hl.ensureContrast('#3B82F6', '#ffffff', 4.5);      // → adjusted hex meeting 4.5:1
hl.meetsContrast('#000000', '#ffffff', 'AA');       // → true

// Distance
hl.deltaE('#ff0000', '#00ff00');  // → 1.09

// Palette
hl.palette('#3B82F6', 5);        // → ['#c4d5ff', '#7eaafc', '#3b82f6', '#0060d0', '#003d8e']
hl.semanticScale('#3B82F6');      // → { '50': '#b3c7ff', '500': '#3b82f6', '900': '#00184b', ... }
hl.paletteHues(0.6, 0.15, 12);   // → 12 evenly-spaced hues
```

## API

### Core Conversions

| Method | Description |
|--------|-------------|
| `fromHex(hex)` | Hex → Helmlab Lab |
| `toHex(lab)` | Helmlab Lab → hex (gamut mapped) |
| `fromSrgb(rgb)` | sRGB [0,1] → Lab |
| `toSrgb(lab)` | Lab → sRGB [0,1] (gamut mapped) |
| `fromXYZ(xyz)` | CIE XYZ → Lab |
| `toXYZ(lab)` | Lab → CIE XYZ |
| `toDisplayP3(lab)` | Lab → Display P3 [0,1] |

### Contrast

| Method | Description |
|--------|-------------|
| `contrastRatio(fg, bg)` | WCAG contrast ratio (1–21) |
| `ensureContrast(fg, bg, ratio)` | Adjust fg to meet minimum ratio |
| `meetsContrast(fg, bg, level)` | Check AA (4.5:1) or AAA (7:1) |

### Palette

| Method | Description |
|--------|-------------|
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
  AnalyticalSpace, compileParams, getDefaultParams,
  hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb,
  gamutMap, isInGamut, contrastRatio,
} from 'helmlab';
```

## How It Works

Helmlab is a 72-parameter analytical color space trained on the Helmholtz–Kohlrausch effect dataset and 6 other perceptual datasets (64,000+ observations). The forward transform is a 13-stage pipeline:

```
XYZ → M1 → power → M2 → hue correction → H-K → cubic L → dark L
    → chroma scale → chroma power → L-chroma → HLC → hue-lightness
    → neutral correction → rotation → Lab
```

Every stage is exactly invertible (Newton iteration where needed). The neutral correction guarantees grays map to a=b=0. A rigid rotation in the ab-plane aligns hue angles with intuition.

## License

MIT
