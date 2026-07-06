# Testing color code

Sanity anchors (assert these to validate any implementation):
- `ΔE00('#ff0000','#00ff00') ≈ 86.6` · `WCAG contrast(#fff,#000) = 21.0` · `contrast(#fff,#3b82f6) ≈ 3.68`

Structural tests for any custom space/transform:
1. **Gray axis + endpoints**: grays → C* ≈ 0; white → L = max, black → L = 0 exactly. Endpoint bugs silently cheat visible metrics.
2. **Round-trip INCLUDING boundary colors**: hex→space→hex within 1/255 on gamut corners and primaries, not just random samples — worst cases live on the boundary.
3. **CAM16 before trusting**: a gray ramp must give a ≈ b ≈ 0 and monotone J; default configs are often broken.

Testing color OUTPUT (palettes, gradients, themes):
4. **Never assert exact hex strings** in snapshots — they break on library minor versions. Assert `ΔE(actual, expected) < tolerance` instead.
5. **Gradient invariants**: endpoints exact; L monotone for light→dark ramps; hue reversal ≤ a few degrees; step-size CV bounded.
6. **Contrast fixes**: test the OUTPUT meets the ratio (`contrast(fixed, bg) ≥ 4.5`), not that the function ran.
