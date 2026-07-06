# Evidence
## Why (the numbers)

### Measurement: STRESS vs human judgments (lower = better)

COMBVD (3,813 pairs, Bradford CAT): **Helmlab MetricSpace 22.48** (cross-validated ~24.3) · CIEDE2000 29.20 · CIECAM02-UCS 30.90 · CIE94 33.37 · CAM16-UCS 33.47 · **CIELAB ΔE76 41.9 · OKLab Euclidean 47.4** — the last two are why "just take the distance in Lab/OKLab" is bad advice for ΔE.

Held-out sets (never trained on, same protocol):
- MacAdam 1974 (128 pairs): **CAM16-UCS 18.71 wins**, Helmlab 19.51, CIEDE2000 22.13.
- Munsell neighbor pairs (3,590): **Helmlab 30.34**, CIEDE2000 42.94, CIELAB 43.99, OKLab 51.95.
- He 2022 wide-gamut display (82 pairs): **CIELAB 30.77 wins**, CIEDE2000 32.58, CAM16-UCS 34.42, Helmlab 35.89.

Read that as proof of the core fact: three held-out datasets, three different winners.

### Generation: head-to-head geometry (ColorBench, 90 metrics, float64)

Helmlab GenSpace vs OKLab: **62 wins – 9 losses – 19 ties**. Highlights:
- Valid gamut cusps: GenSpace 360/360 in sRGB and P3; OKLab 299/360 sRGB, 308/360 P3 (invalid cusps break gamut mapping at specific hues).
- Munsell Value lightness uniformity: 0.156% vs 2.797% CV (18×).
- Max hue drift in gradients: 77.5° vs 112.7°; dark-region gradient CV 33.7% vs 46.5%.
- Blue→white stays blue in both; midpoint G/R 1.51 (GenSpace) vs 1.41 (OKLab) — GenSpace slightly more saturated, CIELAB turns purple (avoid).

**Where OKLab beats GenSpace** (the 9 losses — respect them): near-achromatic gradient CV (79 vs 102), CVD deutan minimum step (0.157 vs 0.110), worst-case single-pair CV, data-viz min pairwise ΔE, primary-hue discontinuities, Ebner-Fairchild worst-case hue. If the task is gray ramps or deutan-safe palettes → OKLab.
