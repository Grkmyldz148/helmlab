# Helmlab

A family of purpose-built color spaces for UI design systems.

Helmlab provides two complementary color spaces: **MetricSpace** for perceptual distance measurement and **GenSpace** for gradient/palette generation. MetricSpace STRESS (lower = closer agreement with observers) is 22.48 on COMBVD (3,813 pairs, Bradford CAT to D65; CIEDE2000: 29.20 — both models were derived from COMBVD, so this is a fit-data comparison) and 29.65 / 33.06 on independent display data (SCDd set 18, 588 pairs each on grey / black background; CIEDE2000: 34.23 / 39.37). GenSpace has a valid gamut cusp at each of 3,600 hues in sRGB, Display P3 and Rec.2020. Details: [Benchmarks](#benchmarks).

[![arXiv](https://img.shields.io/badge/arXiv-2602.23010-b31b1b.svg)](https://arxiv.org/abs/2602.23010)
[![npm version](https://img.shields.io/npm/v/helmlab.svg)](https://www.npmjs.com/package/helmlab)
[![PyPI version](https://img.shields.io/pypi/v/helmlab.svg)](https://pypi.org/project/helmlab/)
[![Color.js](https://img.shields.io/badge/Color.js-merged-f97316.svg)](https://colorjs.io/docs/spaces.html#helmgen)

**[Website](https://helmlab.space)** | **[Documentation](https://helmlab.space/docs/)** | **[Playground](https://helmlab.space/playground/)** | **[Paper](https://arxiv.org/abs/2602.23010)**

## Key Features

- **Color difference** — MetricSpace: STRESS 22.48 vs CIEDE2000's 29.20 on COMBVD (3,813 pairs, Bradford CAT to D65; fit data for both models); 29.65 / 33.06 vs 34.23 / 39.37 on independent SCDd set 18 (588 pairs each, grey / black background)
- **Gradient generation** — GenSpace: a valid gamut cusp at each of 3,600 hues in sRGB, Display P3 and Rec.2020
- **Depressed cubic transfer** — `y³ + αy = x` (α=0.021): finite derivative at zero (unlike `x^(1/3)`). Exact analytical inverse via hyperbolic functions
- **Chroma power** — Mild chroma compression (C^0.978), analytically invertible
- **L-gated hue enrichment** — Targeted hue rotation in the blue region, gated by lightness, aimed at the blue→white purple shift. Achromatic colors are unchanged
- **Blue gradients** — Blue→White midpoint stays blue (`hl.gen.mix('#0000ff', '#ffffff')` → `#649cff`), not lavender
- **Neutral achromatic axis** — GenSpace grays map to C* ≈ 10⁻¹⁵, up to at least Y = 10 (structural: same transfer function on every channel)
- **Equal-step gradients** — `gradient()` reparameterizes the path by CIEDE2000 arc length, so steps are approximately equal in CIEDE2000 on any pair
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

hl.gen.gradient('#ff0000', '#0000ff', 8);             // equal CIEDE2000 steps
hl.gen.gradient('#00f', '#fff', 8, { gamut: 'display-p3' }); // wide-gamut output
hl.gen.scale('#3B82F6');                              // Tailwind-style 50-950 scale
hl.gen.harmonies('#3B82F6', 'triadic');               // constant-L,C hue rotations
hl.gen.vivid('#6488b8', { gamut: 'display-p3' });     // chroma to the gamut boundary
hl.gen.ensureContrast('#3B82F6', '#ffffff', 4.5);     // adjust to meet 4.5:1
hl.metric.difference('#ff0000', '#00ff00');           // trained perceptual metric
hl.metric.jnd('#808080', '#828282');                  // just-noticeable-difference units
hl.metric.ciede2000('#ff0000', '#00ff00');            // industry standard, ~86.6
hl.tokens.css('#3B82F6', 'oklch');                    // design-token export
```

About 21 KB gzipped (ESM build, gzip -9), zero dependencies, ESM + CJS with full TypeScript types. Full API reference: [helmlab.space/docs](https://helmlab.space/docs/).

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

One `Helmlab` instance, three namespaces: **`hl.gen`** creates colors,
**`hl.metric`** measures them, **`hl.tokens`** exports design tokens.
The two spaces have branded Lab types — mixing them raises a `TypeError`
instead of silently producing a wrong color.

```python
from helmlab import Helmlab

hl = Helmlab()

# ── Create (GenSpace) ──────────────────────────────────────────────
hl.gen.gradient("#ff0000", "#0000ff", 8)              # equal perceptual steps
hl.gen.gradient("#0000ff", "#ffffff", 8, gamut="display-p3")  # wide-gamut output
hl.gen.mix("#ff0000", "#0000ff")                      # visual midpoint (same path)
hl.gen.scale("#3b82f6")                               # Tailwind 50-950, 500 == input
hl.gen.harmonies("#3b82f6", "triadic")                # constant-L,C hue rotations
hl.gen.vivid("#6488b8", gamut="display-p3")           # chroma -> gamut boundary
hl.gen.cusp(263)                                      # most colorful point of a hue
hl.gen.ensure_contrast("#ffffff", "#3b82f6", ratio=4.5, strict=True)

# ── Measure (MetricSpace) ──────────────────────────────────────────
hl.metric.difference("#ff0000", "#00ff00")   # trained metric (fit on COMBVD)
hl.metric.jnd("#808080", "#828282")          # in just-noticeable-difference units
hl.metric.ciede2000("#ff0000", "#00ff00")    # industry standard, ~86.6
hl.metric.confidence("#808080", "#828282")   # observer-disagreement estimate
hl.metric.info("color(display-p3 1 0 0)")    # wide-gamut input works everywhere

# ── Export (hex in, tokens out) ────────────────────────────────────
hl.tokens.css("#3b82f6", "oklch")            # 'oklch(62.3% 0.1881 259.8)'
hl.tokens.css_variables(hl.gen.scale("#3b82f6"), "--primary")
```

JS is the camelCase mirror (`hl.gen.ensureContrast(...)`). Parity is
enforced by a permanent cross-language gate: **every string output is
byte-identical, numeric worst-case difference ~1e-12**, and hex round-trips
are bit-exact on a 1728-color grid in both spaces and both languages.
In JS the hex round-trip was also checked on all 16,777,216 8-bit sRGB
colors: 0 losses in both spaces (2026-11-08).
Full surface: [API.md](API.md). Upgrading from 0.x: the flat
`hl.gradient()/hl.from_hex()` API was removed in 1.0 — every method now
lives on `hl.gen` / `hl.metric` / `hl.tokens`.

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

Jointly optimized with CMA-ES against COMBVD (with Bradford CAT pre-processing); MacAdam 1974 and He et al. 2022 were also used during fitting and model selection. 13-stage enriched pipeline with hue correction, Helmholtz-Kohlrausch, chroma scaling, neutral correction, and rigid rotation. STRESS 22.48 on COMBVD with Bradford CAT to D65 (fit data; CIEDE2000, also derived from COMBVD: 29.20). Independent display data: see [Benchmarks](#benchmarks).

### GenSpace (v0.11.1 — Depressed Cubic + Chroma Power + L-Gated Enrichment)

Pipeline: `XYZ → M₁ → depcubic(α=0.021) → M₂ → PW L_corr → L-gated hue enrichment → chroma_power(0.978) → NC → Lab`

**Transfer function:** `y³ + αy = x` (depressed cubic, α=0.021)

Solved analytically via `y = 2s·sinh(arcsinh(x/2s³)/3)` where `s = √(α/3)`, refined with a single Halley iteration for full precision. The inverse is trivial: `x = y³ + αy`. This depressed cubic has a finite derivative at zero (unlike standard `x^(1/3)`), which eliminates gamut boundary singularities.

**Chroma power:** `C' = C^0.978` — mild chroma compression applied post-M2. Analytically invertible (`C = C'^(1/0.978)`).

**L-gated hue enrichment:** A targeted hue rotation `h' = h + A·gate(L)·gauss(h−center)` applied post-M2, where the gate function `sin²(π·(L−L_lo)/(L_hi−L_lo))` activates only in mid-to-high lightness and the Gaussian targets the blue hue region. It is aimed at the blue→white purple shift; achromatic colors are unchanged (a hue rotation leaves C = 0 at 0), and the Gaussian window confines the rotation mainly to blue hues. Invertible via Halley iteration (cubic convergence, up to 8 iterations).

**Key properties:**
- A valid gamut cusp at each of 3,600 hues in sRGB, Display P3 and Rec.2020 (checked 2026-11-08)
- Blue→White gradient: midpoint `#649cff` (blue, no lavender shift)
- Achromatic: C* ≈ 10⁻¹⁵ up to at least Y = 10 (structural: M₁ maps D65 grays to equal cone responses, the same transfer applies to every channel, and the a and b rows of M₂ sum to zero)
- Adaptive gamut clipping with cusp-finding (Ottosson-style L0 calculation)
- Piecewise-linear L correction with 19 breakpoints (analytically invertible)

**Known trade-offs:** The enrichment inverse is iterative (Halley), so float round-trips are not exact to machine precision; 8-bit hex round-trips are lossless (all 16,777,216 sRGB colors, JS, 2026-11-08). Minor primary hue discontinuity at exact primary vertices, and reduced near-achromatic gradient uniformity in very low chroma regions.

**Blue-region gamut fold:** M1→f→M2 spaces such as OKLab and GenSpace can have small non-contiguous in-gamut regions near h≈260° in sRGB, caused by cubic polynomial roots in the inverse. See [color.js#81](https://github.com/color-js/color.js/issues/81).

### Known limitations

- **MetricSpace coordinates are intended for SDR content** (Y ≤ 1 relative to diffuse white). Above diffuse white the neutral axis carries residual chroma (about 0.02 at Y = 2.2, 0.11 at Y = 3, 0.14 from Y ≈ 4) and L grows quickly (≈ 1.80 at Y = 3, ≈ 19 at Y = 10). `difference()`, `distance()` and `jnd()` are unaffected because they remove the neutral correction before measuring. For coordinates above diffuse white use GenSpace (gray axis stays neutral to ~1e-15 up to at least Y = 10).

## Benchmarks

### Perceptual Distance (MetricSpace)

STRESS of MetricSpace (v21) and CIEDE2000. Lower STRESS = closer agreement with observers.

| Dataset | Pairs | Role | MetricSpace | CIEDE2000 |
|---------|-------|------|-------------|-----------|
| COMBVD (BFD-P, Leeds, RIT-DuPont, Witt), Bradford CAT to D65 | 3,813 | fit data: both models were derived from it | 22.48 | 29.20 |
| SCDd set 18, grey background (Liang/Georgoula 2015, ZJU) | 588 | independent: used to fit neither model | 29.65 | 34.23 |
| SCDd set 18, black background (Liang/Georgoula 2015, ZJU) | 588 | independent: used to fit neither model | 33.06 | 39.37 |

COMBVD by subset (fit data for both models, same protocol):

| COMBVD subset | Pairs | MetricSpace | CIEDE2000 |
|---|---|---|---|
| BFD-P | 2,776 | 21.81 | 29.55 |
| Leeds | 307 | 21.84 | 19.25 |
| RIT-DuPont | 312 | 21.90 | 19.47 |
| Witt | 418 | 30.93 | 30.22 |

CIEDE2000 has the lower STRESS on Leeds, RIT-DuPont and Witt; MetricSpace's lower pooled value comes from BFD-P, the largest subset.

MacAdam 1974 and He et al. 2022 (He, Xiao, Pointer, Melgosa & Bressler, *Materials* 15:4055, 3D-printed spheres) were used in MetricSpace's fitting and model selection, so no held-out result is reported for them.

<details>
<summary><strong>How was STRESS measured?</strong></summary>

STRESS (Standardized Residual Sum of Squares) is the CIE-standard metric for evaluating color difference formulas. **COMBVD** is the combined visual-difference dataset of 3,813 color pairs from BFD-P, Leeds, RIT-DuPont and Witt; CIEDE2000 was derived from it.

For each pair *i*, let ΔVᵢ = human visual difference, ΔEᵢ = predicted distance. STRESS finds the optimal scale *F* minimizing residuals:

```
STRESS = 100 × √( Σ(ΔEᵢ − F·ΔVᵢ)² / Σ(ΔEᵢ)² )
```

Scale: 0 = perfect, 100 = no correlation. Full methodology: [arXiv:2602.23010](https://arxiv.org/abs/2602.23010).

</details>

### Gradient Quality (GenSpace)

The GenSpace vs OKLab benchmark is being re-run with the audited [ColorBench](https://github.com/Grkmyldz148/colorbench); results will be published there.

### Gradient Uniformity

`gradient()` reparameterizes the interpolation path by CIEDE2000 arc length, so successive steps are approximately equal in CIEDE2000. The same algorithm could be applied to any space; Helmlab ships it built-in.

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
│   ├── metric_params.json  # MetricSpace params (v21)
│   ├── gen_params.json     # GenSpace params (v0.11.1, depcubic + enrichment)
│   └── ...                 # Dataset loaders (COMBVD, Munsell, etc.)
├── export.py               # Token export (CSS, Android, iOS, Tailwind)
└── feedback/               # Human feedback collection tools

packages/helmlab-js/        # npm package (TypeScript)
packages/postcss-helmlab/   # PostCSS plugin
tests/                      # 670 tests (371 Python + 299 JavaScript)

# ColorBench (the evaluation suite) lives in its own repo:
# https://github.com/Grkmyldz148/colorbench
```

## Tests

```bash
python -m pytest tests/ -q              # 371 Python tests (369 pass + 2 skip)
cd packages/helmlab-js && npx vitest run # 299 JS tests
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
