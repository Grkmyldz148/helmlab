# Changelog

## 1.0.2 — 2026-11-12

Documentation release. Outputs are bit-identical to 1.0.1 and 1.0.0.

### Documentation

- The READMEs (repository, npm, PyPI) were rewritten. They carry no performance figures; figures will be published on one evidence page. They describe Helmlab 2026 and its identifiers, say that `hl.metric` returns the neutral-corrected form (`helmlab-metric-neutral`) and not the base form `helmlab-metric`, and list the known limitations, including lightness near black and near white (#5) and the transfer function of `color(rec2020 …)` strings.
- Docstrings of `difference()`, `ciede2000()`, `distance()`, `confidence()`, `nearest()`, `gradient()`, `mix()`, `hueRing()` / `hue_ring()`, `harmonies()`, `gamutMap()` / `gamut_map()` and the `Helmlab` class no longer carry performance figures or comparative claims. They name the fitting data of MetricSpace: COMBVD, MacAdam 1974 and the Human Feedback data.
- New package descriptions on npm and PyPI.
- The Python wheel now contains the license file, and the links in the README also work on PyPI.
- postcss-helmlab 0.3.1: README only; its example outputs are now those of the published packages.

### Outputs bit-identical to 1.0.1

Compared against npm and PyPI `helmlab` 1.0.1 with zero tolerance:

- all 219,147 outputs of the golden harness in JavaScript (`tools/golden`);
- the same 219,147 calls in Python;
- all 16,777,216 8-bit sRGB colors (GenSpace and MetricSpace forward, hex round trip), 262,144 Display P3 and 262,144 Rec.2020 grid points, and gamut cusps at 3,600 hues in each of sRGB, Display P3 and Rec.2020 (JavaScript, `exhaustive.mjs`).

### Note on the 1.0.1 provenance commit

npm `helmlab` 1.0.1 was built from commit a2f3a63. Its tree (bf06a2b) is identical to the tree of tag v1.0.1 (0c72024).

## 1.0.1 — 2026-11-08

JavaScript performance release. Outputs are bit-identical to 1.0.0. The Python
package has no functional change (version bump only).

### Performance (JS)

- Hue inverse (Newton): each sin/cos is evaluated once per step and shared
  between the hue correction and its derivative.
- Newton/Halley inverse loops stop as soon as a step leaves the value exactly
  unchanged (`Object.is`); every remaining step would also leave it unchanged.
- Piecewise-linear L-correction breakpoints are computed on demand instead of
  building an array on every inverse call.

Measured with Node 26 on one machine (JS):

| Operation | 1.0.0 | 1.0.1 | Speed-up |
|---|---|---|---|
| GenSpace inverse | 1.82 µs/color | 0.83 µs/color | 2.2× |
| `gen.gradient()`, 16 steps | 3.3 ms | 1.7 ms | 1.9× |
| `gen.scale()` | 107 µs | 49 µs | 2.2× |
| MetricSpace inverse | 4.31 µs/color | 3.24 µs/color | 1.3× |

Forward transforms and `metric.difference()` run at the same speed as in 1.0.0.

### Outputs bit-identical to 1.0.0

Compared against npm `helmlab` 1.0.0 with zero tolerance:

- all 219,147 outputs of a golden harness over the `hl.gen`, `hl.metric` and
  `hl.tokens` API;
- all 16,777,216 8-bit sRGB colors (GenSpace and MetricSpace forward, hex
  round-trip);
- 262,144 Display P3 and 262,144 Rec.2020 grid points;
- gamut cusps at 3,600 hues in each of sRGB, Display P3 and Rec.2020.

### Documentation

- README (also the PyPI page) and the npm README: benchmark statements now name
  the dataset, the number of pairs, and whether the data was used to fit the
  model. Added STRESS on independent display data (SCDd set 18). Corrected the
  COMBVD description. Removed the ColorBench win/loss tables; the comparison is
  being re-run with the audited ColorBench. Bundle size updated to about 21 KB
  gzipped. Added a Known limitations section (MetricSpace above diffuse white).

## 1.0.0 — 2026-07-08

Clean-break namespaced API (`hl.gen` / `hl.metric` / `hl.tokens`) with branded
`GenLab` / `MetricLab` types.
