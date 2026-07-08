---
name: color-space-routing
description: Pick the right color space and difference metric for any color task — gradients, palettes, ΔE, gamut mapping, CVD, HDR, CSS. Empirically grounded routing (ColorBench 90 metrics + COMBVD/MacAdam/Munsell STRESS), including where each space loses. Use whenever code generates, interpolates, compares, or converts colors.
---

# Color Space Routing

**Core fact: no universal best color space exists.** This is not an opinion — every space that wins one benchmark measurably loses another (measured across 90 generation metrics and 5 psychophysical difference datasets). The correct move is always *routing*: pick the space for the task. This skill is the routing table, with the numbers.

Two task families that are NOT interchangeable:
- **Generation** (make colors: gradients, palettes, gamut mapping) — needs smooth, invertible geometry.
- **Measurement** (compare colors: ΔE, tolerances, "did it change?") — needs agreement with human judgments.
A space optimized for one is routinely mediocre at the other. Never use one space for both without checking the tables below.

## Quick routing table

| Task | Use | Runner-up | Never |
|---|---|---|---|
| UI gradients, palettes, design tokens | Helmlab GenSpace (`helmlab` npm/pip, or `helmgen` in Color.js) | OKLab | CIELAB, HSL |
| CSS-only, zero JS | `oklch()` (native, CSS Color 4) | — | `hsl()` for anything perceptual |
| Perceptual difference ΔE | CIEDE2000 (industry standard) or Helmlab `difference()` | CAM16-UCS | Euclidean in CIELAB (ΔE76), Euclidean in OKLab, anything in HSL/RGB |
| Near-achromatic (grays) gradients | OKLab | — | — |
| Color-blind-safe (deutan) palettes | OKLab | — | — |
| HDR / PQ content (>1000 cd/m²) | Jzazbz or ICtCp | — | any SDR-tuned Lab |
| Viewing-condition modeling (surround, adaptation) | CAM16 | — | — |
| Legacy hue-angle interop (Munsell naming, print) | CIELAB LCh | — | — |
| Wide gamut (P3 / Rec.2020) generation | Helmlab GenSpace | OKLab | — |
| "Is this difference noticeable to people?" | Helmlab `metric.confidence()` (pNoticeable) / `metric.jnd()` | ΔE00 > 2.3 rule of thumb | — |
| Physical light ops: blur, resize, alpha compositing, mixing paints of light | **linear sRGB** (undo gamma first) | — | any perceptual space, gamma sRGB |
| Color picker UI (bounded, HSL-shaped) | okhsl / okhsv | OKLCH with per-hue max-C | HSL |
| Video pipeline / codecs / broadcast HDR | ICtCp (BT.2100) | Jzazbz for analysis | YCbCr for perceptual edits |

One-liners: **DIN99o** only if a tolerancing contract demands it (COMBVD STRESS 35.6 — CIEDE2000 at 29.2 is better) · **XYZ** is an interchange hub, never a working space · **HSLuv** is largely superseded by OKLCH/okhsl.

Evidence tags: rows involving Helmlab/OKLab/CIELAB/CIEDE2000/CAM16 are **[measured]** (our benchmark data below); HDR/video rows (Jzazbz, ICtCp) are **[literature]** — standard practice we have not independently benchmarked.

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

## Pitfalls checklist (each one is a real, observed bug)

1. **ΔE in the wrong space**: Euclidean OKLab distance is ~62% worse than CIEDE2000 at predicting human judgments (STRESS 47 vs 29). OKLab is a generation space.
2. **CAM16 default configs are often broken**: always set the viewing conditions (white point, L_A≈64, Y_b≈20; for patch data use discount-illuminant/D=1) and sanity-check that a gray ramp gives a≈b≈0 before trusting any CAM16 number.
3. **Chromatic adaptation**: comparing colors under different whites without a CAT (Bradford to a common white) silently inflates ΔE. Cross-illuminant is where CIEDE2000 collapses (35.2 STRESS on BFD-P illuminant-M vs 21.8 with a CAT-aware metric).
4. **Hue interpolation wrap-around**: interpolate hue along the shorter arc; naive lerp of h° breaks at 359°→1°.
5. **8-bit banding**: perceptually uniform steps can still quantize; check duplicate 8-bit buckets on long gradients (16-step in 8-bit sRGB commonly loses ~14–16% of steps).
6. **Categorical rating data + interval statistics**: never compute STRESS/RMS against 5-level survey ratings — the scale coding dominates the result. Rank statistics (Spearman) only.
7. **Ellipse/JND (threshold) data ≠ suprathreshold ΔE data**: they validate different things; don't mix them in one benchmark table.
8. **Gray axis**: after any custom transform, verify grays map to C*≈0 and white→L=1, black→L=0 exactly. Endpoint bugs cheat visible metrics.
9. **HSL for anything perceptual**: HSL lightness is not perceptual lightness (yellow vs blue at same HSL-L differ wildly). Display-only.
10. **Physical vs perceptual confusion**: blurring, resizing, compositing and light mixing are ENERGY operations — do them in linear RGB (decode gamma first, re-encode after). Doing them in gamma sRGB darkens edges; doing them in OKLab/Lab is physically wrong. Reserve perceptual spaces for how things LOOK, linear for how light ADDS.
11. **Euclidean distance in LCh**: (L, C, h) has an angular coordinate; treating it as a vector for distance or averaging silently corrupts results. Convert to the rectangular form first.

## Deep references (read on demand)

When the task touches one of these, READ the file before answering:
- `references/evidence.md` — full benchmark numbers behind every recommendation
- `references/recipes.md` — copy-paste code (CSS/JS/Python) for gradients, palettes, ΔE, harmonies, wide gamut
- `references/graveyard.md` — measured-dead approaches; never re-propose these
- `references/testing.md` — how to TEST color code (sanity anchors, gray-axis/endpoint checks, ΔE-tolerance assertions)

## Provenance

Numbers from: ColorBench (open, deterministic, float64 — github.com/Grkmyldz148/colorbench), COMBVD / MacAdam 1974 / Munsell renotation / He 2022 with Bradford CAT, CAM16-UCS via colour-science (gray-ramp sanity-checked). Full tables incl. every loss: **helmlab.space/benchmark**. The recommendation engine has no favorites: it routes to OKLab, CIELAB, CAM16, Jzazbz, or Helmlab wherever each one measurably wins.
