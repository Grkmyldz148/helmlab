# Helmlab

A family of purpose-built color spaces for UI design systems.

Helmlab provides two complementary color spaces: **MetricSpace** for perceptual distance measurement (STRESS 23.30 on COMBVD — 20% better than CIEDE2000), and **GenSpace** for gradient/palette generation (27-7 vs OKLab on ColorBench's 48 metrics, 360/360 gamut cusps in all gamuts).

[![arXiv](https://img.shields.io/badge/arXiv-2602.23010-b31b1b.svg)](https://arxiv.org/abs/2602.23010)
[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![PyPI version](https://img.shields.io/pypi/v/helmlab.svg)](https://pypi.org/project/helmlab/)

**[Website](https://helmlab.space)** | **[Documentation](https://helmlab.space/docs.html)** | **[Demo](https://helmlab.space/demo.html)** | **[Paper](https://arxiv.org/abs/2602.23010)**

## Key Features

- **State-of-the-art color difference** — MetricSpace: STRESS 23.30 vs CIEDE2000's 29.18 on COMBVD (3,813 pairs)
- **Superior gradient generation** — GenSpace: 27 wins vs OKLab across 48 benchmarks (ColorBench), 360/360 valid cusps in sRGB/P3/Rec.2020
- **Softened cube root transfer** — `f(x) = (|x|+ε)^(1/3) - ε^(1/3)`: eliminates cusp singularities while preserving gradient quality. Exact analytical inverse, no Newton iteration
- **True blue gradients** — Blue→White midpoint is sky blue (G/R = 1.51), not lavender
- **Perfect achromatic axis** — Grays map to C* < 10⁻¹⁵ (structural guarantee from uniform transfer function)
- **Perfectly uniform gradients** — Built-in CIEDE2000 arc-length reparameterization, CV ≈ 0% on any pair
- **Embedded Helmholtz-Kohlrausch** — MetricSpace: lightness is chroma-dependent, learned from data
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

### PostCSS

[![npm version](https://img.shields.io/npm/v/postcss-helmlab.svg)](https://www.npmjs.com/package/postcss-helmlab)

Use Helmlab color spaces directly in CSS — transformed to `rgb()` at build time:

```bash
npm install postcss-helmlab
```

```css
/* Input */
.card { color: helmlab(0.78 0.52 -0.20); }
.bg   { background: linear-gradient(in helmgen, #e63946, #457b9d); }

/* Output */
.card { color: rgb(255, 76, 119); }
.bg   { background: linear-gradient(#e63946 0.0%, ..., #457b9d 100.0%); }
```

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
    XYZ → M₁ → softcbrt(ε) → M₂ → hue_rot → PW_L_corr → Lab
    + CIEDE2000 arc-length reparameterization for gradient()
```

### MetricSpace (72 parameters)

Jointly optimized against COMBVD using L-BFGS-B with 8 random restarts. 13-stage enriched pipeline with hue correction, Helmholtz-Kohlrausch, chroma scaling, neutral correction, and rigid rotation. STRESS 23.30 — the lowest published figure on COMBVD.

### GenSpace (v0.10.0 — Softened Cube Root)

Pipeline: `XYZ → M₁ → softcbrt(ε=0.001) → M₂(rot 20°) → hue_rot → PW L_corr → Lab`

**Transfer function:** `f(x) = sign(x) · ((|x| + ε)^(1/3) − ε^(1/3))`

This softened cube root has a finite derivative at zero (unlike standard `x^(1/3)`), which eliminates gamut boundary singularities while preserving the perceptual uniformity of the cube root for typical color values. The inverse is exact and analytical: `f⁻¹(y) = (|y| + ε^(1/3))³ − ε`.

**Key properties:**
- 360/360 valid cusps in sRGB, Display P3, and Rec.2020 (OKLab: 294, 309, 360)
- Cusp smoothness 0.079 (OKLab: 0.801 — 10x smoother gamut boundaries)
- Blue→White gradient: sky blue midpoint (G/R = 1.51), no lavender shift
- Achromatic: C* < 10⁻¹⁵ (structural guarantee — uniform transfer × orthogonal M₂)
- Munsell Value uniformity: 2.01% (OKLab: 2.80%)
- Exact analytical inverse — no Newton iteration anywhere in the pipeline
- Piecewise-linear L correction with 19 breakpoints (analytically invertible)

**ColorBench evaluation (48 metrics, 3,038 gradient pairs, 3 gamuts):**

| | GenSpace v0.10.0 | OKLab |
|---|---|---|
| **Head-to-head** | **27 wins** | 7 wins |
| sRGB valid cusps | **360** | 294 |
| Cusp smoothness | **0.079** | 0.801 |
| Blue→White G/R | **1.51** | 1.41 |
| Hue RMS | **13°** | 30° |
| Munsell Value | **2.01%** | 2.80% |
| Gradient CV (mean) | 38.2% | 38.0% |
| Achromatic C* | **< 10⁻¹⁵** | 3.7×10⁻⁸ |
| 1000-trip roundtrip | **3.9×10⁻¹⁴** | 5.0×10⁻¹³ |
| Palette harmony | **10.2°** | 11.7° |

> GenSpace's 7 losses are all non-critical: 3 floating-point precision (10⁻⁸ / 10⁻¹⁵), 1 sRGB matrix precision artifact, 1 CVD structural (shared M₁ basis), 1 paradigm difference (CIE Lab hue agreement), 1 photo gamut mapping (5% difference). None are visible to the human eye or affect practical use.

## Benchmarks

### Perceptual Distance (MetricSpace)

STRESS on COMBVD (3,813 pairs). Lower is better.

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

<details>
<summary><strong>How was STRESS measured?</strong></summary>

STRESS (Standardized Residual Sum of Squares) is the CIE-standard metric for evaluating color difference formulas. **COMBVD** is a combined visual-difference dataset of 3,813 color pairs from 6 independent psychophysical experiments (Luo & Rigg 1986, RIT-DuPont, Witt, Leeds, BFD, He et al. 2022), containing 64,000+ individual human judgments.

For each pair *i*, let ΔVᵢ = human visual difference, ΔEᵢ = predicted distance. STRESS finds the optimal scale *F* minimizing residuals:

```
STRESS = 100 × √( Σ(ΔEᵢ − F·ΔVᵢ)² / Σ(ΔEᵢ)² )
```

Scale: 0 = perfect, 100 = no correlation. Full methodology: [arXiv:2602.23010](https://arxiv.org/abs/2602.23010).

</details>

### Gradient Quality (GenSpace)

Helmlab GenSpace vs OKLab — head-to-head on [ColorBench](https://github.com/Grkmyldz148/colorbench) (48 metrics, 3,038 gradient pairs across sRGB, Display P3, and Rec.2020 gamuts):

| Category | GenSpace wins | OKLab wins | Tie |
|----------|-------------|------------|-----|
| Gamut geometry | 5 | 0 | 4 |
| Gradient uniformity | 6 | 0 | 8 |
| Hue stability | 6 | 1 | 1 |
| Perceptual quality | 5 | 2 | 1 |
| Numerical precision | 3 | 4 | 0 |
| Other | 2 | 0 | 0 |
| **Total** | **27** | **7** | **14** |

### Gradient Uniformity

CV (coefficient of variation of CIEDE2000 step sizes). Lower is better.

| Method | Red→Blue | Orange→Cyan | Black→White | Technique |
|--------|----------|-------------|-------------|-----------|
| **Helmlab `gradient()`** | **≈ 0%** | **≈ 0%** | **≈ 0%** | arc-length reparam. |
| Helmlab GenSpace | 30.3% | 26.5% | 40.7% | linear interpolation |
| Oklab | 31.5% | 41.4% | 41.2% | linear interpolation |
| CIE Lab | 44.8% | 52.3% | 61.5% | linear interpolation |

> **Note:** `gradient()` achieves ≈ 0% via CIEDE2000 arc-length reparameterization. This redistributes steps to equal perceptual spacing — an algorithm that could be applied to any space. Helmlab ships it built-in.

## Project Structure

```
src/helmlab/
├── helmlab.py              # Main API (Helmlab class)
├── spaces/
│   ├── metric.py           # MetricSpace — 72-param enriched pipeline
│   ├── gen.py              # GenSpace — softcbrt + PW L_corr pipeline
│   ├── base.py             # Abstract base class
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
│   ├── gen_params.json     # GenSpace params (v0.10.0, softcbrt)
│   └── ...                 # Dataset loaders (COMBVD, Munsell, etc.)
├── export.py               # Token export (CSS, Android, iOS, Tailwind)
└── feedback/               # Human feedback collection tools

packages/helmlab-js/        # npm package (TypeScript)
packages/postcss-helmlab/   # PostCSS plugin
colorbench/                 # ColorBench evaluation suite (48 metrics)
tests/                      # 602 tests (406 Python + 196 JavaScript)
```

## Tests

```bash
python -m pytest tests/ -q              # 406 Python tests
cd packages/helmlab-js && npx vitest run # 196 JS tests
```

## Research

The optimization experiments, checkpoints, and analysis scripts that led to the current GenSpace v0.10.0 are available in a separate repository:

**[helmlab-experimental](https://github.com/Grkmyldz148/helmlab-experimental)** — 480+ experiments across 4 transfer functions, 3 M₁ variants, and systematic grid searches. Includes all checkpoints, optimization scripts, and the full experiment report.

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

## Author

**[Gorkem Yildiz](https://gorkemyildiz.com)** — [helmlab.space](https://helmlab.space)

## License

MIT
