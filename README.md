# Helmlab

A data-driven analytical color space for UI design systems.

Helmlab is a family of purpose-built color spaces: **MetricSpace** (72-parameter enriched pipeline for perceptual distance) and **GenSpace** (generation-optimized pipeline for gradients and palettes). MetricSpace achieves STRESS 23.30 on COMBVD (3,813 color pairs) — a 20.1% improvement over CIEDE2000. GenSpace + arc-length reparameterization produces perfectly uniform gradients (CV ≈ 0% on any color pair).

[![arXiv](https://img.shields.io/badge/arXiv-2602.23010-b31b1b.svg)](https://arxiv.org/abs/2602.23010)
[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![PyPI version](https://img.shields.io/pypi/v/helmlab.svg)](https://pypi.org/project/helmlab/)

**[Website](https://helmlab.space)** | **[Documentation](https://grkmyldz148.github.io/helmlab/)** | **[Paper](https://arxiv.org/abs/2602.23010)**

## Key Features

- **State-of-the-art color difference prediction** — STRESS 23.30 vs CIEDE2000's 29.18 (MetricSpace)
- **Perfectly uniform gradients** — Built-in CIEDE2000 arc-length reparameterization in `gradient()`, CV ≈ 0% on any pair
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
hl.gradient('#ff0000', '#0000ff', 8);                 // Perfectly uniform gradient
hl.semanticScale('#3B82F6');                          // Tailwind-style 50–950 scale
```

~12KB gzipped, zero dependencies, ESM + CJS with full TypeScript types. See the [npm package README](packages/helmlab-js/README.md) for the full API.

### Python (pip)

[![PyPI version](https://img.shields.io/pypi/v/helmlab.svg)](https://pypi.org/project/helmlab/)

```bash
pip install helmlab
```

## Quick Start (Python)

```python
from helmlab import Helmlab

hl = Helmlab()

# sRGB to Helmlab Lab
lab = hl.from_srgb([0.2, 0.5, 0.8])
print(f"L={lab[0]:.3f}, a={lab[1]:.3f}, b={lab[2]:.3f}")

# Color difference between two sRGB colors
dist = hl.delta_e("#ff0000", "#00ff00")

# Perfectly uniform gradient (arc-length reparameterized)
gradient = hl.gradient("#ff0000", "#0000ff", 8)

# Ensure WCAG AA contrast (4.5:1)
adjusted = hl.ensure_contrast("#ffffff", "#3B82F6", min_ratio=4.5)

# Generate a palette (Tailwind-style 50-950 scale)
scale = hl.semantic_scale("#3B82F6")
```

## Architecture

Helmlab is a family of purpose-built color spaces:

```
Helmlab (UI layer)
├── MetricSpace — 72-param enriched pipeline (distance, deltaE)
│   XYZ → M₁ → γ → M₂ → Hue → H-K → L → C → HL → NC → φ → Lab
│
└── GenSpace — generation-optimized pipeline (gradient, palette)
    XYZ → M₁ → γ=⅓ → M₂ → NC → Lab
    + CIEDE2000 arc-length reparameterization for gradient()
```

**MetricSpace** (72 parameters) is jointly optimized against COMBVD using L-BFGS-B with 8 random restarts. 13-stage enriched pipeline with hue correction, Helmholtz-Kohlrausch, chroma scaling, neutral correction, and rigid rotation.

**GenSpace** (21 parameters) uses Phase1H-optimized M1/M2 matrices with shared γ=⅓. No enrichment stages — pure linear-algebra pipeline, fast and invertible. 6× better hue accuracy than Oklab (5.2° vs 30.1° RMS).

## Benchmarks

### Perceptual Distance (MetricSpace)

STRESS on COMBVD (3,813 pairs). Each method uses its standard distance formula. Lower is better.

| Method | COMBVD STRESS | vs CIEDE2000 |
|--------|--------------|-------------|
| **Helmlab v20b** | **23.30** | **-20.1%** |
| CIEDE2000 | 29.18 | — |
| CIE94 | 33.59 | +15.1% |
| CAM16-UCS (Euclid.) | 33.90 | +16.2% |
| ΔE CMC | 34.04 | +16.6% |
| IPT (Euclid.) | 41.21 | +41.3% |
| CIE Lab ΔE76 | 42.80 | +46.7% |
| Oklab (Euclid.) | 47.46 | +62.7% |

Bootstrap (10,000 iterations): Helmlab 95% CI [22.50, 23.93], CIEDE2000 95% CI [27.64, 30.84]. Zero overlap, p < 10⁻⁴.

<details>
<summary><strong>How was STRESS measured?</strong></summary>

STRESS (Standardized Residual Sum of Squares) is the CIE-standard metric for evaluating color difference formulas. **COMBVD** is a combined visual-difference dataset of 3,813 color pairs from 6 independent psychophysical experiments (Luo & Rigg 1986, RIT-DuPont, Witt, Leeds, BFD, He et al. 2022), containing 64,000+ individual human judgments. In each experiment, observers viewed color pairs under controlled D65 lighting and rated perceived differences.

For each pair *i*, let ΔVᵢ = human visual difference, ΔEᵢ = predicted distance. STRESS finds the optimal scale *F* minimizing residuals:

```
STRESS = 100 × √( Σ(ΔEᵢ − F·ΔVᵢ)² / Σ(ΔEᵢ)² )
```

Scale: 0 = perfect, 100 = no correlation. Helmlab's 72 parameters were optimized with L-BFGS-B (8 random restarts, 80/20 split, seed=42). 5-fold CV confirms generalization (mean ≈ 23.5). Full methodology: [arXiv:2602.23010](https://arxiv.org/abs/2602.23010).

</details>

### Gradient Uniformity

CV (coefficient of variation of CIEDE2000 step sizes). Lower is better.

| Method | Red→Blue | Orange→Cyan | Black→White | Technique |
|--------|----------|-------------|-------------|-----------|
| **Helmlab `gradient()`** | **≈ 0%** | **≈ 0%** | **≈ 0%** | arc-length reparam. |
| Helmlab GenSpace | 3.1% | 33.2% | 41.0% | linear interpolation |
| Oklab | 31.5% | 41.4% | 41.2% | linear interpolation |
| CIE Lab | 44.8% | 52.3% | 61.5% | linear interpolation |

> **Note:** Helmlab's `gradient()` achieves ≈ 0% via CIEDE2000 arc-length reparameterization — an algorithm that redistributes steps to equal perceptual spacing. Oklab and CIE Lab values reflect naive linear interpolation, which is how most libraries use them. The same reparameterization technique could be applied to any space; Helmlab ships it built-in.

## Project Structure

```
src/helmlab/
├── helmlab.py              # Main API (Helmlab class)
├── spaces/
│   ├── metric.py           # MetricSpace — 72-param enriched pipeline
│   ├── gen.py              # GenSpace — generation-optimized pipeline
│   ├── analytical.py       # Compatibility shim → MetricSpace
│   ├── base.py             # Abstract base class
│   ├── registry.py         # Color space registry
│   └── ...                 # Baseline spaces (CAM16, IPT, Oklch, etc.)
├── metrics/
│   ├── delta_e.py          # Color difference formulas
│   ├── stress.py           # STRESS computation
│   └── benchmarks.py       # Cross-method benchmarking
├── utils/
│   ├── srgb_convert.py     # sRGB/Display P3 conversions
│   ├── gamut.py            # Gamut mapping (binary search)
│   └── ...                 # Converters, I/O, visualization
├── data/
│   ├── metric_params.json  # MetricSpace params (v20b, STRESS 23.30)
│   ├── gen_params.json     # GenSpace params (Phase1H optimized)
│   └── ...                 # Dataset loaders (COMBVD, Munsell, etc.)
├── export.py               # Token export (CSS, Android, iOS, Tailwind)
└── feedback/               # Human feedback collection tools

packages/helmlab-js/        # npm package (TypeScript)
docs/                       # Documentation + interactive demo
paper/                      # LaTeX paper + figures
tests/                      # 337 tests (233 Python + 104 JavaScript)
```

## Tests

```bash
python -m pytest tests/ -q        # 233 Python tests
cd packages/helmlab-js && npx vitest run  # 104 JS tests
```

## Citation

```bibtex
@article{yildiz2025helmlab,
  title={Helmlab: A Data-Driven Analytical Color Space for UI Design Systems},
  author={Y{\i}ld{\i}z, G{\"o}rkem},
  journal={arXiv preprint arXiv:2602.23010},
  year={2025},
  url={https://arxiv.org/abs/2602.23010}
}
```

## License

MIT
