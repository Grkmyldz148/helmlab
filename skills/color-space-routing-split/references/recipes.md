# Recipes
## Recipes

### Gradient between two colors (JS)
```js
import { Helmlab } from 'helmlab';        // 17.8KB gzip, zero deps
const hl = new Helmlab();
hl.gradient('#0000ff', '#ffffff', 16);    // CIEDE2000 arc-length: equal visual steps
```
CSS-only fallback: `linear-gradient(in oklch, blue, white)`. Never `in hsl` (hue detours) and don't interpolate raw CIELAB across hue (blue→white goes purple).

### Tailwind-style palette / design tokens
```js
hl.semanticScale('#3b82f6');  // {50:'#e7efff', … 500:'#3b82f6' exactly, … 950:'#000046'}
hl.export().exportTailwind(hl.semanticScale('#3b82f6'), 'primary');
```
Lightness is Munsell-uniform by construction; level 500 is the exact input.

### Perceptual difference (Python)
```python
from helmlab import Helmlab
hl = Helmlab()
hl.difference("#ff0000", "#00ff00")     # trained metric (COMBVD-fit), saturates ~0.15
hl.euclidean_distance("#ff0000", "#00ff00")  # fast ΔE76-style, quick UI checks only
```
If you must use a standard: CIEDE2000, correctly implemented (it's easy to get the hue term wrong — use a tested library, verify ΔE00(red, green) ≈ 86.6).

### "Will users notice this difference?"
```js
const c = hl.differenceWithConfidence('#808080', '#828282');
// c.pNoticeable ≈ 0.077 → 7.7% of observers would call it noticeably different
// c.reliable === false → below the human noise band; don't act on it
```
Near-threshold and low-chroma differences are where humans disagree most — a bare ΔE is least trustworthy exactly there.

### Hue rotation / harmonies
```js
const lch = hl.genToLch(hl.genFromHex('#3b82f6'));   // [L, C, h°] — L and C are 0–1-scale, NOT 0–100
const triad = hl.genToHex(hl.genFromLch([lch[0], lch[1], (lch[2] + 120) % 360]));
```
Rotate hue in a *generation* space (GenSpace LCh or OKLCH) — never in HSL, and don't rotate CIELAB hue across the blue region.

### Wide gamut output
```js
hl.toHexP3(hl.fromHex('#ff0000'));  // 'color(display-p3 0.9176 0.2003 0.1386)' — hue-preserving gamut map
```
