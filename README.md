# Helmlab

A data-driven analytical color space for UI design systems.

Helmlab is a 72-parameter color space optimized end-to-end against psychophysical data. It achieves STRESS 23.22 on COMBVD (3,813 color pairs) — a 20.4% improvement over CIEDE2000 — while maintaining a structurally guaranteed achromatic axis and reasonable hue alignment.

**[Paper](new-paper/helmlab.tex)** | **[Interactive Demo](https://grkmyldz148.github.io/helmlab/)**

## Key Features

- **State-of-the-art color difference prediction** — STRESS 23.22 vs CIEDE2000's 29.18
- **Achromatic guarantee** — Grays map to C < 10⁻⁶ via neutral correction (no color artifacts in gradients)
- **Free hue improvement** — Rigid rotation reduces hue error (RMS 16.1°) at zero cost to the distance metric
- **Embedded Helmholtz-Kohlrausch** — Lightness is chroma-dependent, learned from data
- **UI tooling** — Gamut mapping, WCAG contrast enforcement, palette generation, dark/light mode adaptation
- **Token export** — CSS (`oklch()`), Android XML, iOS Swift (Display P3), Tailwind, JSON

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from colorspace.helmlab import Helmlab

hl = Helmlab()

# sRGB to Helmlab Lab
lab = hl.from_srgb([0.2, 0.5, 0.8])
print(f"L={lab[0]:.3f}, a={lab[1]:.3f}, b={lab[2]:.3f}")

# Back to sRGB (round-trip error < 10⁻¹⁴)
rgb = hl.to_srgb(lab)

# Color difference between two sRGB colors
dist = hl.distance_srgb([1, 0, 0], [0, 0.5, 0])

# Ensure WCAG AA contrast (4.5:1)
adjusted = hl.ensure_contrast([0.2, 0.5, 0.8], [1, 1, 1], min_ratio=4.5)

# Generate a palette (Tailwind-style 50-950 scale)
from colorspace.export import TokenExporter
exporter = TokenExporter(hl)
tokens = exporter.tailwind_scale("blue", [0.2, 0.5, 0.8])
```

## How It Works

Helmlab maps CIE XYZ (D65) to a perceptually-organized Lab space through 11 stages:

```
XYZ → M₁(9) → γᵢ(3) → M₂(9) → Hue corr.(8) → H-K(6) → L corr.(8)
    → C proc.(18) → Hue-L(4) → NC → Rot φ → Lab
```

All 72 parameters (65 space + 7 distance metric) are jointly optimized against COMBVD using L-BFGS-B with 8 random restarts.

### The Measurement-Generation Tradeoff

Most color spaces optimize for either distance prediction or coordinate usability — not both. Without neutral correction, Helmlab's measurement-optimal configuration maps grays to chroma ~0.34 (unusable for gradients). The post-pipeline neutral correction resolves this at a cost of just +0.04 STRESS points.

## Benchmarks

| Method | STRESS | vs CIEDE2000 |
|--------|--------|-------------|
| **Helmlab** | **23.22** | **-20.4%** |
| CIEDE2000 | 29.18 | — |
| Helmlab (Euclidean only) | ~30.2 | -3.6% |
| CAM16-UCS (Euclidean) | 33.90 | +16.2% |
| CIE76 | 42.80 | +46.7% |
| Oklab (Euclidean) | 47.46 | +62.7% |

Note: Baselines use plain Euclidean distance. Helmlab's full metric includes pair-dependent weighting and compression. The "Euclidean only" row shows the space transform's contribution alone.

## Project Structure

```
src/colorspace/
├── helmlab.py              # Main API (Helmlab class)
├── spaces/
│   ├── analytical.py       # Core 72-param transform
│   └── registry.py         # Color space registry
├── utils/
│   ├── srgb_convert.py     # sRGB/Display P3 conversions
│   └── gamut.py            # Gamut mapping
├── export.py               # Token export (CSS, Android, iOS, Tailwind)
├── data/
│   └── analytical_params.json  # Trained parameters (v19-NC)
└── feedback/
    ├── generator.py        # Bidirectional test pair generation
    └── collector.py        # Human feedback collection

scripts/                    # Optimization scripts (v14→v19)
tests/                      # 158 tests
new-paper/                  # LaTeX paper + figures
data/                       # COMBVD, He 2022, MacAdam 1974 datasets
checkpoints/                # All optimization checkpoints (v1→v19)
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
