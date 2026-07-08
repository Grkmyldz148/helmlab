# Helmlab 1.0 — API Specification

> Single source of truth for the 1.0 clean-break API. Python (snake_case) and
> JS (camelCase) mirror each other word-for-word. Written 2026-07-08.

## Design principles

1. **Boundaries speak hex/CSS.** Every everyday task works with color strings
   in and color strings out. Lab coordinates are the advanced layer, not the
   default currency.
2. **Two namespaces, two Lab types.** `hl.gen` (create colors — GenSpace) and
   `hl.metric` (measure colors — MetricSpace). Each namespace has its own
   `from_hex`; their Lab types (`GenLab` / `MetricLab`) are branded. Passing
   one space's Lab into the other's API raises `TypeError` (Python) / fails
   to compile and throws (TS/JS). The 0.x silent-wrong-color footgun is
   structurally impossible.
3. **Wide gamut is an option, not a separate API.** Every generation function
   takes `gamut: 'srgb' | 'display-p3' | 'rec2020'`; color-string inputs
   accept `#rrggbb`, `#rgb`, `color(display-p3 r g b)` and
   `color(rec2020 r g b)` everywhere.
4. **Functions do what their names say — or say so loudly.** Anything
   best-effort has a `strict` option that turns the fallback into an
   exception; nothing silently under-delivers.

## Entry point

```python
from helmlab import Helmlab
hl = Helmlab()          # options: metric_params=, gen_params=, surround=,
                        #          neutral_correction=, ab_rotate_deg=
hl.gen                  # generation namespace (GenSpace)
hl.metric               # measurement namespace (MetricSpace)
hl.tokens               # design-token export (hex in, string out)
hl.set_surround(S)      # viewing context 0..1
```

```js
import { Helmlab } from 'helmlab';
const hl = new Helmlab();
hl.gen / hl.metric / hl.tokens
```

Advanced/raw layer (unchanged from 0.x): `GenSpace`, `MetricSpace` classes
remain exported; `hl.gen.space` / `hl.metric.space` expose the live instances.

## Color-string inputs

Everywhere a parameter is named `color`, these forms are accepted:
`'#rrggbb'`, `'#rgb'`, `'color(display-p3 r g b)'`, `'color(rec2020 r g b)'`.
Invalid strings raise. (Shared parser `parse_color() → XYZ`.)

## Output `gamut` option

- `'srgb'` (default) → `'#rrggbb'`
- `'display-p3'` → `'color(display-p3 r g b)'` (gamut-mapped in the owning space)
- `'rec2020'` → `'color(rec2020 r g b)'`

---

## `hl.gen` — generation (GenSpace)

### Conversions
| Python | Returns | Notes |
|---|---|---|
| `from_hex(color)` | `GenLab` | accepts all color-string forms |
| `to_hex(lab)` | `'#rrggbb'` | gamut-mapped to sRGB |
| `to_css(lab, gamut='display-p3')` | css string | gamut-mapped |
| `from_srgb(rgb)` / `to_srgb(lab)` | `GenLab` / `[r,g,b]` | |
| `lab(L, a, b)` | `GenLab` | branded constructor |
| `lch(L, C, h)` | `GenLab` | cylindrical constructor; **L, C are 0–1-scale** |
| `to_lch(lab)` | `[L, C, h°]` | plain array |
| `gamut_map(lab, gamut='srgb', method='chroma')` | `GenLab` | `method='adaptive'` = Ottosson-style cusp projection (trades a little L for chroma) |
| `in_gamut(lab_or_color, gamut='srgb')` | `bool` | |

### Cusp geometry (the 360/360/360 strength, exposed)
| Python | Returns | Semantics |
|---|---|---|
| `max_chroma(lightness, hue_deg, gamut='srgb')` | `float` | maximum in-gamut chroma at fixed L, h |
| `cusp(hue_deg, gamut='srgb')` | `(L, C)` | the most colorful point of a hue leaf |
| `vivid(color, *, gamut='srgb')` | `str` | same L and hue, chroma pushed to the gamut boundary — the honest "make it pop in P3" operation |

### Generation
| Python | Returns | Semantics |
|---|---|---|
| `gradient(start, end, steps=16, *, gamut='srgb')` | `list[str]` | equal perceptual steps (CIEDE2000 arc-length). `steps=1 → [start]`, `steps<1 → []` |
| `mix(a, b, t=0.5, *, gamut='srgb')` | `str` | point at fraction `t` along the SAME arc-length path as `gradient` |
| `palette(base, steps=10, *, gamut='srgb')` | `list[str]` | lightness ramp light→dark; base only approximate — use `scale` for exact base |
| `scale(base, levels=None, *, gamut='srgb')` | `dict[str,str]` | Tailwind-style; level 500 == input exactly. Default levels 50…950 |
| `hue_ring(count=12, *, lightness=0.6, chroma=0.15, gamut='srgb')` | `list[str]` | categorical ring, evenly spaced hues |
| `harmonies(base, kind, *, gamut='srgb')` | `list[str]` | `kind ∈ {complementary, analogous, triadic, tetradic, split_complementary}`; equal-L,C hue rotations in GenLCh (base included, base first) |
| `rotate_hue(color, degrees, *, gamut='srgb')` | `str` | L and C preserved |

Harmony hue offsets (degrees, base first):
complementary `[0, 180]` · analogous `[0, -30, 30]` · triadic `[0, 120, 240]`
· tetradic `[0, 90, 180, 270]` · split_complementary `[0, 150, 210]`.

### Contrast & modes
| Python | Returns | Semantics |
|---|---|---|
| `contrast_ratio(fg, bg)` | `float` | WCAG 2.1, 1–21 |
| `meets_contrast(fg, bg, level='AA')` | `bool` | AA=4.5, AAA=7 |
| `ensure_contrast(fg, bg, ratio=4.5, *, strict=False)` | `'#rrggbb'` | lightness-only fix, hue+chroma preserved. Unreachable for this hue → falls back to #000/#fff (hue lost). Unreachable even then: `strict=True` → raises `ContrastError`; `strict=False` → warns + best effort |
| `adapt_to_mode(color, from_mode='light', to_mode='dark')` | `'#rrggbb'` | |
| `adapt_pair(fg, bg, from_mode, to_mode, ratio=4.5)` | `(fg, bg)` | adapts both then re-ensures contrast |

---

## `hl.metric` — measurement (MetricSpace)

### Conversions
| Python | Returns | Notes |
|---|---|---|
| `from_hex(color)` | `MetricLab` | accepts all color-string forms (P3/Rec2020 INPUT now first-class) |
| `to_hex(lab)` | `'#rrggbb'` | gamut-mapped |
| `to_css(lab, gamut='display-p3')` | css string | replaces `to_hex_p3`/`to_hex_rec2020` |
| `from_srgb/to_srgb`, `from_xyz/to_xyz` | | |
| `lab(L, a, b)` | `MetricLab` | branded constructor |
| `lch(L, C, h)` / `to_lch(lab)` / `from_lch(lch)` | `MetricLab` / `[L,C,h°]` | cylindrical view (same C/H as `info`); for hue MANIPULATION prefer `hl.gen` |
| `in_gamut(lab_or_color, gamut='srgb')` | `bool` | replaces `is_in_srgb/p3/rec2020` |

### Measurement — four clearly-named metrics
| Python | Range | What it is |
|---|---|---|
| `difference(a, b)` | 0…~0.15 (saturates) | **the trained metric** (v21, COMBVD STRESS 22.48). Best for near-threshold judgments; rank preserved, absolute value plateaus for very different colors |
| `euclidean(a, b)` | 0…~1.6 | uncompressed ΔE76-analogue in Metric Lab; unbounded companion |
| `ciede2000(a, b)` | 0…~100 | industry-standard CIEDE2000 (CIELAB scale); most stable for suprathreshold catalog argmax |
| `jnd(a, b)` | 0…∞ | `difference / 0.035633` — difference in **just-noticeable-difference units**. 1.0 = the point where the median observer of the ordinal model rates the pair at least "moderately different" (`tau[1]/mu_scale` from the v2 confidence fit). ≥ ~4.2 JND = the saturation zone; values there mean "far above threshold", not a precise multiple |

Plus:
| Python | Returns | |
|---|---|---|
| `distance(lab_a, lab_b)` | `float`/batch | trained metric, **MetricLab inputs, identical contract in both languages** (kills the 0.x Python-XYZ/JS-Lab asymmetry; the XYZ-in variant lives only on the raw `MetricSpace` class) |
| `confidence(a, b)` | `dict` | de + latent, disagreement, reliability, p_noticeable, reliable, extrapolated (v2 ordinal model, EXPERIMENTAL, n=47 provenance documented) |
| `nearest(target, palette, metric='ciede2000')` | `dict` | hex, index, distance, runner_up, margin |
| `info(color)` | `dict` | hex, srgb, xyz, lab, L, C, H, luminance, in_srgb, in_p3, in_rec2020 |

---

## `hl.tokens` — design-token export (hex in, string out)

TokenExporter's Lab-input footgun is gone: every method takes color STRINGS.

| Python | Returns |
|---|---|
| `css(color, format='oklch')` | one CSS value; `format ∈ {hex, rgb, hsl, oklch, p3, rec2020}` |
| `android(color)` | `'0xFFrrggbb'` |
| `ios_p3(color)` | `{r, g, b}` (Display P3 floats) |
| `swift(color)` | Swift `Color(.displayP3, …)` literal |
| `css_variables(scale, prefix='--color')` | CSS custom-properties block |
| `tailwind(scale, name)` | Tailwind config dict |
| `multi_format(scale, name, formats=['hex','oklch','p3'])` | nested dict |
| `json(scales)` | multi-format JSON string |

---

## Branded Lab types

**Python** — `GenLab` / `MetricLab` are `np.ndarray` subclasses (full numpy
ergonomics preserved: indexing, `.copy()`, arithmetic). Namespace methods
raise `TypeError` when handed the *other* brand; plain lists/ndarrays pass
(interop escape hatch).

**TS/JS** — `type GenLab = [number, number, number] & { readonly __gen: true }`
(and `__metric` for MetricLab). Compile-time: cross-passing or raw literals
don't typecheck (use `gen.lab()` / `metric.lab()` or the `from*` functions).
Runtime: constructors attach a non-enumerable `__space` tag; consuming
methods throw on a mismatched tag, accept untagged arrays.

## Errors & warnings

- `ContrastError` (Python: exported exception; JS: `Error` with
  `name='ContrastError'`) — `ensure_contrast(strict=True)` when unreachable.
- Cross-space Lab → `TypeError` (Py) / `Error` (JS) with a message naming
  both spaces and the fix.
- Non-finite Lab into `distance` → raise/throw (kept from 0.17 fixes).

## Measured precision (2026-07-08 parity gate)

Cross-language parity is enforced by a permanent test
(`packages/helmlab-js/tests/parity-1.0.test.ts` against a Python-generated
reference covering the full surface):

- **Every string output is byte-identical** between Python and JS (hex,
  `color()` strings, oklch, all generation and token outputs).
- Numeric worst-case Py↔JS difference: **~1e-12** (Lab coords, all four
  metrics, confidence fields). Iterative-search APIs (cusp, maxChroma,
  adaptive gamut map): ~1e-4, bounded by their internal tolerance.
- Round-trips: hex→Lab→hex **bit-exact on a 1728-color grid in both spaces
  and both languages (0 misses)**; XYZ→Lab→XYZ max error 2.9e-15
  (MetricSpace, machine precision) / 5.8e-9 (GenSpace, ~6 orders below the
  8-bit quantum).

## Removed in 1.0 (was 0.x)

`gen_from_hex/gen_to_hex/gen_*` (→ `hl.gen.*`), `from_hex/to_hex/...` on the
root (→ `hl.metric.*`), `base_*` deprecated aliases (deleted),
`palette_hues` (→ `hue_ring`), `semantic_scale` (→ `scale`),
`delta_e`/`euclidean_distance` (→ `metric.euclidean`), `delta_e_2000`
(→ `metric.ciede2000`), `perceptual_distance`/`distance_from_lab` at facade
level (→ `metric.distance`), `difference_with_confidence` (→
`metric.confidence`), `nearest_color` (→ `metric.nearest`),
`to_hex_p3/to_hex_rec2020/to_displayp3/to_rec2020` (→ `metric.to_css`),
`is_in_srgb/p3/rec2020` (→ `in_gamut`), `export()`/`TokenExporter` (→
`hl.tokens`, hex-in). Root-level conversions removed from `Helmlab`.
