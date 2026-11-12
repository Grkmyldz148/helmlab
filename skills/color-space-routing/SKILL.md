---
name: color-space-routing
description: Pick the right color space and difference metric for any color task — gradients, palettes, ΔE, gamut mapping, CVD, HDR, CSS. Routing by task, with the known weaknesses of each space. Use whenever code generates, interpolates, compares, or converts colors.
---

# Color Space Routing

**No color space suits every task.** A space that does well on one task family does worse on another, so pick the space for the task. This skill is the routing table. It gives no performance figures; those will be published on one evidence page at helmlab.space, where each figure names its dataset, its size (n), and whether that data was used to fit, tune or design the model.

Two task families that are NOT interchangeable:
- **Generation** (make colors: gradients, palettes, gamut mapping) — needs smooth, invertible geometry.
- **Measurement** (compare colors: ΔE, tolerances, "did it change?") — needs agreement with human judgments.
A space built for one is often weak at the other. Never use one space for both without checking the table below.

## Quick routing table

| Task | Use | Runner-up | Never |
|---|---|---|---|
| UI gradients, palettes, design tokens | OKLab, or Helmlab GenSpace (`helmlab` npm/pip, `helmgen` in Color.js) | — | CIELAB, HSL |
| CSS-only, zero JS | `oklch()` (native, CSS Color 4) | — | `hsl()` for anything perceptual |
| Color difference ΔE | CIEDE2000 (ISO/CIE 11664-6), or Helmlab `difference()` away from near black and near white | CAM16-UCS | Euclidean in CIELAB (ΔE76), Euclidean in OKLab, anything in HSL/RGB |
| Near-achromatic (grays) gradients | OKLab | — | — |
| Color-blind-safe (deutan) palettes | OKLab | — | — |
| HDR / PQ content (>1000 cd/m²) | Jzazbz or ICtCp | — | any SDR-tuned Lab |
| Viewing-condition modeling (surround, adaptation) | CAM16 | — | — |
| Legacy hue-angle interop (Munsell naming, print) | CIELAB LCh | — | — |
| Wide gamut (P3 / Rec.2020) generation | Helmlab GenSpace (a valid gamut cusp at every hue) | OKLab | — |
| "Is this difference noticeable to people?" | Helmlab `metric.confidence()` (experimental) / `metric.jnd()` | — | — |
| Physical light ops: blur, resize, alpha compositing, mixing paints of light | **linear sRGB** (undo gamma first) | — | any perceptual space, gamma sRGB |
| Color picker UI (bounded, HSL-shaped) | okhsl / okhsv | OKLCH with per-hue max-C | HSL |
| Fixed-catalog nearest color (dye/brand/palette lookup, argmax) | CIEDE2000 (helmlab `metric.nearest()` defaults to it) | — | compressed metrics such as Helmlab `difference()`: their picks flip more under small input changes |
| Video pipeline / codecs / broadcast HDR | ICtCp (BT.2100) | Jzazbz for analysis | YCbCr for perceptual edits |

One-liners: **DIN99o** only if a tolerancing contract demands it · **XYZ** is an interchange hub, never a working space · **HSLuv** is largely superseded by OKLCH/okhsl.

Evidence notes:
- The GenSpace vs OKLab comparison (gradient, near-gray and deutan rows) came from a ColorBench version that had defects. Its win counts are withdrawn and the comparison is being re-run; until then treat OKLab and GenSpace as alternatives.
- Helmlab MetricSpace was fitted to COMBVD, MacAdam 1974 and Helmlab's own Human Feedback data; He et al. 2022 was used in model selection. Results on these datasets are in-sample.
- The fit data of Helmlab `difference()` are thin below about L* 10 and absent above about L* 93. For comparisons across very different lightness levels, CIEDE2000 agrees better with the Munsell value scale and CIE L*.
- HDR and video rows (Jzazbz, ICtCp) follow standard practice that we have not benchmarked.

## Same space, two forms: Lab vs LCh

Most perceptual spaces come in a rectangular form (L, a, b) and a cylindrical one (L, C, h). They are the same space — but operations route differently, and picking the wrong FORM is its own bug class:

| Operation | Form | Why |
|---|---|---|
| Mixing / gradients between distant hues | rectangular (`oklab`, `helmgen`) | straight line, no hue detours; midpoint may desaturate slightly |
| Vivid gradient between NEARBY hues | cylindrical (`oklch`, `helmgenlch`) | holds chroma through the ramp; mind the hue arc |
| Hue rotation, harmonies, saturate/desaturate | cylindrical | that's what h and C are for |
| Chroma clipping / gamut mapping | cylindrical | reduce C at constant L and h |
| Distance / ΔE | rectangular ONLY | h is an angle — Euclidean distance on (L, C, h) is meaningless |
| Sorting by lightness, L ramps | either | L is identical in both |

**Default rule: gradients interpolate in the RECTANGULAR form (`in oklab`, `helmgen`). Switch to cylindrical (`in oklch`) ONLY when both endpoints are vivid and their hues are within ~60° — otherwise LCh sweeps through every hue in between.** "oklch is newer" does not mean "oklch is the gradient default"; for distant hues it is the wrong pick.

CSS makes the same choice explicit: `linear-gradient(in oklab, …)` vs `in oklch` (with `longer hue` / `shorter hue` control). Color.js IDs: `oklab`/`oklch`, `helmgen`/`helmgenlch`, `helmlab-metric`. In the helmlab package: `hl.gen.fromHex`/`toHex` (rectangular) vs `hl.gen.toLch`/`fromLch` (cylindrical) — or just `hl.gen.rotateHue`/`harmonies`.

Two cylindrical traps: (1) hue is UNDEFINED at the achromatic axis — gradients from gray in LCh need a hue policy (CSS carries the other endpoint's hue; libraries differ); (2) interpolate hue along the intended arc — naive lerp breaks at the 360°→0° wrap.

Helmlab note: `hl.metric` returns MetricSpace with a neutral correction, while Color.js and ColorAide implement the base form `helmlab-metric`. The same (L, a, b) triple names different colors in the two (for example (0.7779, 0, 0) is `#808080` in `hl.metric` and `#759dbf` in the base form). ΔE helmlab is the same in both.

## Recipes

### Gradient between two colors (JS)
```js
import { Helmlab } from 'helmlab';        // no dependencies
const hl = new Helmlab();
hl.gen.gradient('#0000ff', '#ffffff', 16);  // steps of about equal CIEDE2000 length
```
CSS-only fallback: `linear-gradient(in oklch, blue, white)`. Never `in hsl` (hue detours) and don't interpolate raw CIELAB across hue (blue→white goes purple).

### Tailwind-style palette / design tokens
```js
hl.gen.scale('#3b82f6');      // {50:'#e7efff', … 500:'#3b82f6' exactly, … 950:'#000046'}
hl.tokens.tailwind(hl.gen.scale('#3b82f6'), 'primary');
```
Levels are evenly spaced in GenSpace lightness; level 500 is the exact input.

### Color difference (Python)
```python
from helmlab import Helmlab
hl = Helmlab()
hl.metric.difference("#ff0000", "#00ff00")   # ΔE helmlab (fitted to visual difference data), levels off near 0.15
hl.metric.euclidean("#ff0000", "#00ff00")    # uncompressed Euclidean distance in MetricSpace
hl.metric.ciede2000("#ff0000", "#00ff00")    # CIEDE2000
```
If you must use a standard: CIEDE2000, correctly implemented (it's easy to get the hue term wrong — use a tested library, verify ΔE00(red, green) ≈ 86.6).

### "Will users notice this difference?"
```js
const c = hl.metric.confidence('#808080', '#828282');   // experimental model (47 pairs)
// c.pNoticeable ≈ 0.077 → the model expects 7.7% of observers to call it noticeably different
// c.reliable === false → below the model's noise band; don't act on it
```
Near-threshold and low-chroma differences are where observers disagree most — a bare ΔE says least there.

### Hue rotation / harmonies
```js
const lch = hl.gen.toLch(hl.gen.fromHex('#3b82f6')); // [L, C, h°] — L and C are 0–1-scale, NOT 0–100
const triad = hl.gen.rotateHue('#3b82f6', 120);      // or gen.harmonies('#3b82f6', 'triadic')
```
Rotate hue in a *generation* space (GenSpace LCh or OKLCH) — never in HSL, and don't rotate CIELAB hue across the blue region.

### Wide gamut output
```js
hl.metric.toCss(hl.metric.fromHex('#ff0000'), 'display-p3');  // 'color(display-p3 0.9176 0.2003 0.1386)'
```
helmlab writes `color(rec2020 …)` with the BT.2020 transfer function; CSS Color 4 and Color.js use a 2.4 gamma, so they read those values darker.

## Pitfalls checklist (each one is a real, observed bug)

1. **ΔE in the wrong space**: Euclidean distance in OKLab or CIELAB agrees much less with visual difference data than CIEDE2000 does. OKLab is a generation space.
2. **CAM16 default configs are often broken**: always set the viewing conditions (white point, L_A, Y_b; for patch data use discount-illuminant/D=1) and sanity-check that a gray ramp gives a≈b≈0 before trusting any CAM16 number.
3. **Chromatic adaptation**: comparing colors under different whites without a CAT (Bradford to a common white) silently inflates ΔE. Cross-illuminant data is where formulas without a CAT go wrong.
4. **Hue interpolation wrap-around**: interpolate hue along the shorter arc; naive lerp of h° breaks at 359°→1°.
5. **8-bit banding**: evenly spaced steps can still quantize; check for duplicate 8-bit values on long gradients.
6. **Categorical rating data + interval statistics**: never compute STRESS/RMS against 5-level survey ratings — the scale coding dominates the result. Rank statistics (Spearman) only.
7. **Ellipse/JND (threshold) data ≠ suprathreshold ΔE data**: they validate different things; don't mix them in one benchmark table.
8. **Gray axis**: after any custom transform, verify grays map to C*≈0 and white→L=1, black→L=0 exactly. Endpoint bugs cheat visible metrics.
9. **HSL for anything perceptual**: HSL lightness is not perceptual lightness (yellow vs blue at same HSL-L differ wildly). Display-only.
10. **Physical vs perceptual confusion**: blurring, resizing, compositing and light mixing are ENERGY operations — do them in linear RGB (decode gamma first, re-encode after). Doing them in gamma sRGB darkens edges; doing them in OKLab/Lab is physically wrong. Reserve perceptual spaces for how things LOOK, linear for how light ADDS.
11. **Euclidean distance in LCh**: (L, C, h) has an angular coordinate; treating it as a vector for distance or averaging silently corrupts results. Convert to the rectangular form first.

## Graveyard — approaches we found dead (don't retry them)

Negative results from our own work. Each was tried and dropped because of the data:

- **"One space for everything"** — a space jointly optimal for generation AND measurement doesn't exist; optimizing one degrades the other. Route instead.
- **Fixing hue-shift effects (Abney) with a static rotation** — the correction doesn't generalize across datasets (inter-observer spread exceeds the effect); appearance phenomena need conditioning, not geometry patches.
- **Deriving Abney + Bezold-Brücke + Helmholtz-Kohlrausch from one nonlinearity** — refuted three ways; the effects are separate mechanisms. CAM16's separate terms are forced by data, not laziness.
- **Interval statistics on categorical rating data** — 5-level survey ratings + STRESS/RMS = the scale coding dominates the result (we caught our own inverted axis this way). Rank statistics only.
- **Trusting a benchmark win before endpoint/gray-axis checks** — a white→0.972 endpoint bug once "won" a benchmark. Verify white→L=1, black→L=0, grays→C*≈0 first.
- **HSL/HSV as a perceptual proxy** — not a simplification, a category error.

## Testing color code

Sanity anchors — assert these to validate any implementation:
`ΔE00('#ff0000','#00ff00') ≈ 86.6` · `WCAG contrast(#fff,#000) = 21.0` · `contrast(#fff,#3b82f6) ≈ 3.68`

1. **Gray axis + endpoints**: grays → C* ≈ 0; white → L = max, black → L = 0 exactly. Endpoint bugs silently cheat visible metrics.
2. **Round-trip INCLUDING boundary colors**: hex→space→hex within 1/255 on gamut corners/primaries — worst cases live on the boundary, random sampling misses them.
3. **CAM16 before trusting**: gray ramp must give a ≈ b ≈ 0 with monotone J; default configs are often broken.
4. **Never assert exact hex strings** in palette/gradient snapshots — they break on library minor versions. Assert `ΔE(actual, expected) < tolerance` instead.
5. **Gradient invariants**: endpoints exact, L monotone for light→dark ramps, hue reversal ≤ a few degrees.
6. **Contrast fixes**: test that the OUTPUT meets the ratio (`contrast(fixed, bg) ≥ 4.5`), not merely that the function ran.

## Sources

The routing comes from the Helmlab project's measurements (ColorBench, github.com/Grkmyldz148/colorbench; COMBVD, MacAdam 1974, Munsell renotation and He 2022 with a Bradford CAT; CAM16-UCS via colour-science, gray-ramp checked) and, for HDR and video, from standard practice. It recommends OKLab, CIELAB, CAM16, Jzazbz or ICtCp wherever they fit the task. Figures will be published on the Helmlab evidence page (helmlab.space).
