# Helmlab

A data-driven analytical color space for UI design systems.

Helmlab is a 72-parameter color space optimized end-to-end against psychophysical data. It achieves STRESS 23.22 on COMBVD (3,813 color pairs) — a 20.4% improvement over CIEDE2000 — while maintaining a structurally guaranteed achromatic axis and reasonable hue alignment.

**[Interactive Demo](https://grkmyldz148.github.io/helmlab/demo.html)** | **[Documentation](https://grkmyldz148.github.io/helmlab/)** | **[Paper](paper/helmlab.tex)**

## Key Features

- **State-of-the-art color difference prediction** — STRESS 23.22 vs CIEDE2000's 29.18
- **Achromatic guarantee** — Grays map to C < 10⁻⁶ via neutral correction (no color artifacts in gradients)
- **Free hue improvement** — Rigid rotation reduces hue error (RMS 16.1°) at zero cost to the distance metric
- **Embedded Helmholtz-Kohlrausch** — Lightness is chroma-dependent, learned from data
- **UI tooling** — Gamut mapping, WCAG contrast enforcement, palette generation, dark/light mode adaptation
- **Token export** — CSS (`oklch()`), Android XML, iOS Swift (Display P3), Tailwind, JSON

## Installation

### npm (TypeScript / JavaScript)

[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![bundle size](https://img.shields.io/bundlephobia/minzip/helmlab)](https://bundlephobia.com/package/helmlab)

```bash
npm install helmlab
```

```ts
import { Helmlab } from 'helmlab';

const hl = new Helmlab();

const lab = hl.fromHex('#3B82F6');                    // Hex → Helmlab Lab
const hex = hl.toHex([0.5, -0.1, 0.2]);              // Lab → hex (gamut mapped)
hl.contrastRatio('#ffffff', '#3B82F6');                // → 3.68
hl.ensureContrast('#3B82F6', '#ffffff', 4.5);         // Adjust to meet 4.5:1
hl.deltaE('#ff0000', '#00ff00');                      // Perceptual distance
hl.semanticScale('#3B82F6');                          // Tailwind-style 50–950 scale
```

10KB gzipped, zero dependencies, ESM + CJS with full TypeScript types. See the [npm package README](packages/helmlab-js/README.md) for the full API.

### Python

```bash
pip install -e .
```

## Quick Start (Python)

```python
from colorspace.helmlab import Helmlab

hl = Helmlab()

# sRGB to Helmlab Lab
lab = hl.from_srgb([0.2, 0.5, 0.8])
print(f"L={lab[0]:.3f}, a={lab[1]:.3f}, b={lab[2]:.3f}")

# Back to sRGB (round-trip error < 10⁻¹⁴)
rgb = hl.to_srgb(lab)

# Color difference between two sRGB colors
dist = hl.delta_e("#ff0000", "#00ff00")

# Ensure WCAG AA contrast (4.5:1)
adjusted = hl.ensure_contrast("#ffffff", "#3B82F6", min_ratio=4.5)

# Generate a palette (Tailwind-style 50-950 scale)
scale = hl.semantic_scale("#3B82F6")
```

## How It Works

Helmlab maps CIE XYZ (D65) to a perceptually-organized Lab space through 13 stages:

```
XYZ → M₁(9) → γᵢ(3) → M₂(9) → Hue corr.(8) → H-K(6) → L corr.(5)
    → Dark L(3) → C scale(8) → C power(4) → L×C(2) → HLC(4) → Hue-L(4)
    → NC → Rot φ → Lab
```

All 72 parameters (65 space + 7 distance metric) are jointly optimized against COMBVD using L-BFGS-B with 8 random restarts. See the [documentation](https://grkmyldz148.github.io/helmlab/) for the full mathematical description of each stage.

## Benchmarks

STRESS on COMBVD (3,813 pairs). Each method uses its standard distance formula. Lower is better.

| Method | COMBVD STRESS | vs CIEDE2000 |
|--------|--------------|-------------|
| **Helmlab v19** | **23.22** | **-20.4%** |
| CIEDE2000 | 29.18 | — |
| CIE94 | 33.59 | +15.1% |
| CAM16-UCS (Euclid.) | 33.90 | +16.2% |
| ΔE CMC | 34.04 | +16.6% |
| IPT (Euclid.) | 41.21 | +41.3% |
| CIE Lab ΔE76 | 42.80 | +46.7% |
| Oklab (Euclid.) | 47.46 | +62.7% |

Bootstrap (10,000 iterations): Helmlab 95% CI [22.50, 23.93], CIEDE2000 95% CI [27.64, 30.84]. Zero overlap, p < 10⁻⁴.

## Project Structure

```
src/colorspace/
├── helmlab.py              # Main API (Helmlab class)
├── config.py               # Configuration and constants
├── export.py               # Token export (CSS, Android, iOS, Tailwind)
├── spaces/
│   ├── analytical.py       # Core 72-param transform
│   ├── base.py             # Abstract base class
│   ├── registry.py         # Color space registry
│   ├── cam16ucs.py         # CAM16-UCS baseline
│   ├── ipt.py              # IPT baseline
│   ├── jzczhz.py           # JzCzhz baseline
│   ├── oklch.py            # Oklch baseline
│   └── srgb.py             # sRGB baseline
├── metrics/
│   ├── delta_e.py          # Color difference formulas
│   ├── stress.py           # STRESS computation
│   └── benchmarks.py       # Cross-method benchmarking
├── utils/
│   ├── srgb_convert.py     # sRGB/Display P3 conversions
│   ├── gamut.py            # Gamut mapping (binary search)
│   ├── conversions.py      # XYZ ↔ xyY, Lab ↔ LCh, etc.
│   ├── io.py               # File I/O helpers
│   └── visualization.py    # Plotting utilities
├── data/
│   ├── analytical_params.json  # Trained parameters (v19-NC)
│   ├── combvd.py           # COMBVD dataset loader
│   ├── he2022.py           # He 2022 dataset loader
│   ├── macadam1974.py      # MacAdam 1974 dataset loader
│   ├── munsell.py          # Munsell dataset loader
│   ├── hung_berns.py       # Hung & Berns hue data
│   ├── dataset.py          # Unified dataset interface
│   └── preprocessing.py    # Data preprocessing
├── nn/
│   ├── inn.py              # Invertible Neural Network (Phase 0)
│   ├── mlp.py              # MLP baseline
│   ├── training.py         # Training loop
│   ├── losses.py           # Loss functions
│   └── evaluate.py         # Evaluation utilities
└── feedback/
    ├── generator.py        # Bidirectional test pair generation
    └── collector.py        # Human feedback collection

docs/                       # Documentation + interactive demo
paper/                      # LaTeX paper + figures
tests/                      # 214 tests
```

## Tests

```bash
python -m pytest tests/ -q
```

## Citation

```bibtex
@article{yildiz2025helmlab,
  title={Helmlab: A Data-Driven Analytical Color Space for UI Design Systems},
  author={Y{\i}ld{\i}z, G{\"o}rkem},
  year={2025}
}
```

## License

MIT
