# Recipes
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

