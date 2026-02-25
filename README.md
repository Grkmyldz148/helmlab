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

| Method | COMBVD STRESS | vs CIEDE2000 |
|--------|--------------|-------------|
| **Helmlab v19** | **23.22** | **-20.4%** |
| Oklab | 27.50 | -5.8% |
| CIEDE2000 | 29.18 | — |
| CIE Lab ΔE76 | 30.30 | +3.8% |

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

docs/                       # Documentation + interactive demo
paper/                      # LaTeX paper + figures
scripts/                    # Optimization scripts (v14→v19)
tests/                      # 158 tests
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
