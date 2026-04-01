# Helmlab

A family of purpose-built color spaces for UI design systems.

Helmlab provides two complementary color spaces: **MetricSpace** for perceptual distance measurement (STRESS 23.30 on COMBVD — 20% better than CIEDE2000), and **GenSpace** for gradient/palette generation (**50-6 vs OKLab** on ColorBench's 61 metrics, 360/360 gamut cusps, zero gamut holes).

[![arXiv](https://img.shields.io/badge/arXiv-2602.23010-b31b1b.svg)](https://arxiv.org/abs/2602.23010)
[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![PyPI version](https://img.shields.io/pypi/v/helmlab.svg)](https://pypi.org/project/helmlab/)

**[Website](https://helmlab.space)** | **[Documentation](https://helmlab.space/docs.html)** | **[Demo](https://helmlab.space/demo.html)** | **[Paper](https://arxiv.org/abs/2602.23010)**

## Key Features

- **State-of-the-art color difference** — MetricSpace: STRESS 23.30 vs CIEDE2000's 29.18 on COMBVD (3,813 pairs)
- **Superior gradient generation** — GenSpace: **50 wins** vs OKLab's 6 across 61 ColorBench metrics (3,038 gradient pairs, 3 gamuts), 360/360/360 valid cusps, zero gamut holes
- **Depressed cubic transfer** — `y³ + αy = x` (α=0.02): eliminates cusp singularities while preserving gradient quality. Exact analytical inverse via hyperbolic functions
- **L-gated hue enrichment** — Targeted hue rotation in the blue region, gated by lightness. Fixes blue→white purple shift without affecting other colors
- **True blue gradients** — Blue→White midpoint is sky blue (G/R = 1.52), not lavender
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
    XYZ → M₁ → depcubic(α) → M₂ → L-gated enrichment → PW_L_corr → Lab
    + CIEDE2000 arc-length reparameterization for gradient()
```

### MetricSpace (72 parameters)

Jointly optimized against COMBVD using L-BFGS-B with 8 random restarts. 13-stage enriched pipeline with hue correction, Helmholtz-Kohlrausch, chroma scaling, neutral correction, and rigid rotation. STRESS 23.30 — the lowest published figure on COMBVD.

### GenSpace (v0.11.0 — Depressed Cubic + L-Gated Enrichment)

Pipeline: `XYZ → M₁ → depcubic(α=0.02) → M₂ → L-gated hue enrichment → PW L_corr → Lab`

**Transfer function:** `y³ + αy = x` (depressed cubic, α=0.02)

Solved analytically via `y = 2s·sinh(arcsinh(x/2s³)/3)` where `s = √(α/3)`, refined with a single Halley iteration for full precision. The inverse is trivial: `x = y³ + αy`. This depressed cubic has a finite derivative at zero (unlike standard `x^(1/3)`), which eliminates gamut boundary singularities.

**L-gated hue enrichment:** A targeted hue rotation `h' = h + A·gate(L)·gauss(h−center)` applied post-M2, where the gate function `sin²(π·(L−L_lo)/(L_hi−L_lo))` activates only in mid-to-high lightness and the Gaussian targets the blue hue region. This fixes the blue→white purple shift with zero impact on achromatic colors or other hue regions. Invertible via Halley iteration (cubic convergence, 8 iterations).

**Key properties:**
- 360/360 valid cusps in sRGB, Display P3, and Rec.2020 (OKLab: 299)
- Zero gamut holes in all gamuts (OKLab: 474 interior holes in sRGB)
- Blue→White gradient: sky blue midpoint (G/R = 1.52), no lavender shift
- Achromatic: C* < 10⁻¹⁵ (structural guarantee — uniform transfer × orthogonal M₂)
- Munsell Value uniformity: 0.03% (OKLab: 2.80% — 93x better)
- Adaptive gamut clipping with cusp-finding (Ottosson-style L0 calculation)
- Piecewise-linear L correction with 19 breakpoints (analytically invertible)

**ColorBench evaluation (61 metrics, 3,038 gradient pairs, 3 gamuts):**

| | GenSpace v0.11.0 | OKLab |
|---|---|---|
| **Head-to-head** | **50 wins** | 6 wins |
| sRGB valid cusps | **360** | 299 |
| Gamut holes | **0** | 474 |
| Blue→White G/R | **1.52** | 1.41 |
| Munsell Value | **0.03%** | 2.80% |
| Cusp smoothness | **0.079** | 0.801 |
| Gray precision | **< 10⁻¹⁵** | 3.7×10⁻⁸ |
| Round-trip (1000×) | **4×10⁻¹⁴** | 5×10⁻¹³ |

> GenSpace's 6 losses are non-critical: floating-point precision differences, CVD structural (shared M₁ basis), and paradigm differences (CIE Lab hue agreement). 5 ties. None affect practical use.

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

Helmlab GenSpace vs OKLab — head-to-head on [ColorBench](https://github.com/Grkmyldz148/colorbench) (61 metrics, 3,038 gradient pairs across sRGB, Display P3, and Rec.2020 gamuts):

| Category | GenSpace wins | OKLab wins | Tie |
|----------|-------------|------------|-----|
| Gradient quality | 14 | 2 | 0 |
| Gamut integrity | 10 | 1 | 0 |
| Perceptual accuracy | 8 | 2 | 0 |
| Hue uniformity | 9 | 1 | 0 |
| Accessibility (CVD) | 5 | 0 | 2 |
| Engineering | 4 | 0 | 3 |
| **Total** | **50** | **6** | **5** |

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
│   ├── gen.py              # GenSpace — depcubic + enrichment pipeline
│   ├── base.py             # Abstract base class
│   └── ...                 # Baseline spaces (CAM16, IPT, Oklch, etc.)
├── metrics/
│   ├── delta_e.py          # Color difference formulas
│   ├── stress.py           # STRESS computation
│   └── benchmarks.py       # Cross-method benchmarking
├── utils/
│   ├── srgb_convert.py     # sRGB/Display P3 conversions
│   ├── gamut.py            # Gamut mapping (binary search + adaptive clipping)
│   └── ...                 # Converters, I/O, visualization
├── data/
│   ├── metric_params.json  # MetricSpace params (v20b, STRESS 23.30)
│   ├── gen_params.json     # GenSpace params (v0.11.0, depcubic + enrichment)
│   └── ...                 # Dataset loaders (COMBVD, Munsell, etc.)
├── export.py               # Token export (CSS, Android, iOS, Tailwind)
└── feedback/               # Human feedback collection tools

packages/helmlab-js/        # npm package (TypeScript)
packages/postcss-helmlab/   # PostCSS plugin
colorbench/                 # ColorBench evaluation suite (48 metrics)
tests/                      # 609+ tests (413 Python + 196 JavaScript)
```

## Tests

```bash
python -m pytest tests/ -q              # 413 Python tests
cd packages/helmlab-js && npx vitest run # 196 JS tests
```

## Research

The optimization experiments, checkpoints, and analysis scripts that led to the current GenSpace v0.11.0 are available in a separate repository:

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
