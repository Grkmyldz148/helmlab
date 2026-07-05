# Helmlab

A family of purpose-built color spaces for UI design systems.

Helmlab provides two complementary color spaces: **MetricSpace** for perceptual distance measurement (STRESS 22.48 on COMBVD with Bradford CAT — 23% better than CIEDE2000's 29.20), and **GenSpace** for gradient/palette generation (**62-9-19 vs OKLab** on ColorBench's 90 metrics including independent datasets, 360/360/360 gamut cusps, zero monotonicity violations in sRGB/P3 — 1 in Rec.2020).

[![arXiv](https://img.shields.io/badge/arXiv-2602.23010-b31b1b.svg)](https://arxiv.org/abs/2602.23010)
[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![PyPI version](https://img.shields.io/pypi/v/helmlab.svg)](https://pypi.org/project/helmlab/)
[![Color.js](https://img.shields.io/badge/Color.js-merged-f97316.svg)](https://colorjs.io/docs/spaces.html#helmgen)

**[Website](https://helmlab.space)** | **[Documentation](https://helmlab.space/docs/)** | **[Playground](https://helmlab.space/playground/)** | **[Paper](https://arxiv.org/abs/2602.23010)**

## Key Features

- **State-of-the-art color difference** — MetricSpace: STRESS 22.48 vs CIEDE2000's 29.20 on COMBVD (3,813 pairs, with Bradford CAT pre-processing)
- **Superior gradient generation** — GenSpace: **62 wins** vs OKLab's 9 (19 ties) across 90 ColorBench metrics (3,038 gradient pairs, 3 gamuts), 360/360/360 valid cusps, zero monotonicity violations in sRGB/P3 (1 in Rec.2020)
- **Depressed cubic transfer** — `y³ + αy = x` (α=0.021): eliminates cusp singularities while preserving gradient quality. Exact analytical inverse via hyperbolic functions
- **Chroma power** — Mild compression (C^0.978) improves gradient step uniformity across 3,038 pairs
- **L-gated hue enrichment** — Targeted hue rotation in the blue region, gated by lightness. Fixes blue→white purple shift without affecting other colors
- **True blue gradients** — Blue→White midpoint is sky blue (G/R = 1.51), not lavender
- **Perfect achromatic axis** — Grays map to C* ≈ 10⁻¹⁵ (structural guarantee from uniform transfer function)
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
hl.difference('#ff0000', '#00ff00');                  // Perceptual color difference (the good metric)
hl.deltaE('#ff0000', '#00ff00');                      // Euclidean Lab distance (uncompressed, ΔE76-style)
hl.gradient('#ff0000', '#0000ff', 8);                 // Perfectly uniform gradient
hl.semanticScale('#3B82F6');                          // Tailwind-style 50–950 scale
```

~17.8KB gzipped, zero dependencies, ESM + CJS with full TypeScript types. Full API reference: [helmlab.space/docs](https://helmlab.space/docs/).

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
    XYZ → M₁ → depcubic(α) → M₂ → PW_L_corr → L-gated enrichment → C^cp → NC → Lab
    + CIEDE2000 arc-length reparameterization for gradient()
```

### MetricSpace (72 parameters)

Jointly optimized with CMA-ES against COMBVD (with Bradford CAT pre-processing). 13-stage enriched pipeline with hue correction, Helmholtz-Kohlrausch, chroma scaling, neutral correction, and rigid rotation. STRESS 22.48 with Bradford CAT (22.73 without) — the lowest published figure on COMBVD; cross-validated estimate ~24.3.

### GenSpace (v0.11.1 — Depressed Cubic + Chroma Power + L-Gated Enrichment)

Pipeline: `XYZ → M₁ → depcubic(α=0.021) → M₂ → PW L_corr → L-gated hue enrichment → chroma_power(0.978) → NC → Lab`

**Transfer function:** `y³ + αy = x` (depressed cubic, α=0.021)

Solved analytically via `y = 2s·sinh(arcsinh(x/2s³)/3)` where `s = √(α/3)`, refined with a single Halley iteration for full precision. The inverse is trivial: `x = y³ + αy`. This depressed cubic has a finite derivative at zero (unlike standard `x^(1/3)`), which eliminates gamut boundary singularities.

**Chroma power:** `C' = C^0.978` — mild chroma compression applied post-M2 that improves gradient step uniformity. Analytically invertible (`C = C'^(1/0.978)`).

**L-gated hue enrichment:** A targeted hue rotation `h' = h + A·gate(L)·gauss(h−center)` applied post-M2, where the gate function `sin²(π·(L−L_lo)/(L_hi−L_lo))` activates only in mid-to-high lightness and the Gaussian targets the blue hue region. This fixes the blue→white purple shift with zero impact on achromatic colors or other hue regions. Invertible via Halley iteration (cubic convergence, 8 iterations).

**Key properties:**
- 360/360/360 valid cusps in sRGB, Display P3, and Rec.2020 (OKLab: 299/308/360)
- Zero invalid cusps across all gamuts; zero monotonicity violations in sRGB/P3 (1 in Rec.2020)
- Blue→White gradient: sky blue midpoint (G/R = 1.51), no lavender shift
- Achromatic: C* ≈ 10⁻¹⁵ (structural guarantee — uniform transfer × orthogonal M₂)
- Munsell Value uniformity: 0.16% (OKLab: 2.80% — 18x better)
- Adaptive gamut clipping with cusp-finding (Ottosson-style L0 calculation)
- Piecewise-linear L correction with 19 breakpoints (analytically invertible)

**ColorBench evaluation (90 metrics, 3,038 gradient pairs, 3 gamuts):**

| Category | GenSpace wins | OKLab wins | Tie |
|----------|---------------|------------|-----|
| Gamut geometry | **24** | 0 | 3 |
| Application | **8** | 1 | 3 |
| Independent (Hung-Berns, Ebner-Fairchild, Pointer) | **6** | 1 | 0 |
| Gradient quality | **5** | 3 | 3 |
| Perceptual accuracy | **5** | 0 | 0 |
| Structural | **4** | 2 | 2 |
| Achromatic | **2** | 0 | 0 |
| Advanced | **2** | 0 | 4 |
| Hue | **2** | 0 | 0 |
| Special | **2** | 1 | 0 |
| Accessibility | 1 | 1 | 0 |
| Banding | **1** | 0 | 1 |
| Numerical stability | 0 | 0 | 3 |
| **Total** | **62** | **9** | **19** |

**Known trade-offs:** Slightly reduced round-trip precision in sRGB (~5.6×10⁻⁸ vs OKLab's ~1.6×10⁻¹⁵, due to enrichment Halley iteration — invisible in 8-bit pipelines), minor primary hue discontinuity at exact primary vertices, and reduced near-achromatic gradient uniformity in very low chroma regions.

**Blue-region gamut fold:** All power-law based M1→f→M2 spaces exhibit tiny non-contiguous gamut regions near h≈260° in sRGB due to cubic polynomial roots in the inverse. OKLab has 46 such holes; GenSpace has 5, each ~0.001 chroma wide (sub-pixel, invisible). See [color.js#81](https://github.com/color-js/color.js/issues/81).

## Benchmarks

### Perceptual Distance (MetricSpace)

STRESS on COMBVD (3,813 pairs), all methods without CAT for a like-for-like protocol. Lower is better. (With Bradford CAT pre-processing Helmlab v21 reaches the headline **22.48**.)

| Method | COMBVD STRESS | vs CIEDE2000 |
|--------|--------------|-------------|
| **Helmlab v21** | **22.73** | **-22.1%** |
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

Helmlab GenSpace vs OKLab — head-to-head on [ColorBench](https://github.com/Grkmyldz148/colorbench) (90 metrics, 3,038 gradient pairs across sRGB, Display P3, and Rec.2020 gamuts):

| Category | GenSpace wins | OKLab wins | Tie |
|----------|---------------|------------|-----|
| Gamut geometry | **24** | 0 | 3 |
| Application | **8** | 1 | 3 |
| Independent (Hung-Berns, Ebner-Fairchild, Pointer) | **6** | 1 | 0 |
| Gradient quality | **5** | 3 | 3 |
| Perceptual accuracy | **5** | 0 | 0 |
| Structural | **4** | 2 | 2 |
| Achromatic | **2** | 0 | 0 |
| Advanced | **2** | 0 | 4 |
| Hue | **2** | 0 | 0 |
| Special | **2** | 1 | 0 |
| Accessibility | 1 | 1 | 0 |
| Banding | **1** | 0 | 1 |
| Numerical stability | 0 | 0 | 3 |
| **Total** | **62** | **9** | **19** |

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
│   ├── metric_params.json  # MetricSpace params (v21, STRESS 22.48 w/ Bradford CAT)
│   ├── gen_params.json     # GenSpace params (v0.11.1, depcubic + enrichment)
│   └── ...                 # Dataset loaders (COMBVD, Munsell, etc.)
├── export.py               # Token export (CSS, Android, iOS, Tailwind)
└── feedback/               # Human feedback collection tools

packages/helmlab-js/        # npm package (TypeScript)
packages/postcss-helmlab/   # PostCSS plugin
tests/                      # 588 tests (336 Python + 252 JavaScript)

# ColorBench (the 90-metric evaluation suite) lives in its own repo:
# https://github.com/Grkmyldz148/colorbench
```

## Tests

```bash
python -m pytest tests/ -q              # 336 Python tests (334 pass + 2 skip)
cd packages/helmlab-js && npx vitest run # 252 JS tests
```

## AI skill: color-space routing

There is no perfect color space — but there is a right one per task. [`skills/color-space-routing`](skills/color-space-routing/) packages our benchmark evidence as a drop-in skill for Claude Code / Cursor / any AI assistant: task → space routing table, measured numbers (including where each space loses), copy-paste recipes, and a pitfall checklist.

## Research

The optimization experiments, checkpoints, and analysis scripts that led to the current GenSpace v0.11.1 are available in a separate repository:

**[helmlab-experimental](https://github.com/Grkmyldz148/helmlab-experimental)** — 480+ experiments across 4 transfer functions, 3 M₁ variants, and systematic grid searches. Includes all checkpoints, optimization scripts, and the full experiment report.

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

## Author

**[Gorkem Yildiz](https://gorkemyildiz.com)** — [helmlab.space](https://helmlab.space)

## License

MIT
