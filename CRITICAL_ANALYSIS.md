# Helmlab: Critical Analysis & Competitive Landscape

An honest, no-sugarcoating assessment of where Helmlab stands among color spaces and color difference formulas.

---

## 1. The Competitive Landscape

### STRESS on COMBVD (3,813 pairs) — Lower is Better

| Method | STRESS | Params | Year | Type |
|--------|--------|--------|------|------|
| Noise floor (estimated) | ~20 | — | — | Theoretical limit |
| **Helmlab v14c-full** | **22.75** | 72 | 2025 | Measurement-only (CV=77%) |
| **Helmlab v19-NC** | **23.22** | 72 | 2025 | Production (achC<10⁻⁶) |
| CIEDE2000 | 24.2* | ~8 | 2001 | Industry standard |
| CAM16-UCS | ~25-26 | 20+ | 2017 | CIE recommended UCS |
| DIN99d | ~25 | 8 | 2003 | German standard |
| **Helmlab v17-relaxed** | **26.67** | 72 | 2025 | Generation-optimized |
| Oklab (Euclidean) | ~28 | 18 | 2020 | CSS standard |
| ProLab (Euclidean) | ~28 | 15 | 2020 | Academic |
| CIE94 | ~28 | 5 | 1994 | Textile standard |
| JzAzBz (Euclidean) | ~30 | 20-25 | 2017 | HDR-focused |
| CIELAB ΔE76 | ~36 | 0 | 1976 | Universal legacy |
| IPT (Euclidean) | ~35 | 10 | 1998 | Hue-linearity reference |
| CAM16-UCS (Euclidean) | 33.90 | 20+ | 2017 | Without its metric |
| Oklab (Euclidean) | 47.46 | 18 | 2020 | Without any metric |
| CIE XYZ (Euclidean) | ~69 | 0 | 1931 | Baseline |

*\*CIEDE2000 was trained on COMBVD — its STRESS is effectively a training error, not a test error.*

**What this table says**: Helmlab beats everything on COMBVD. But context matters — see below.

---

## 2. What Helmlab Does Well

### 2.1 State-of-the-Art Color Difference Prediction
- STRESS 23.22 is genuinely the best published result on COMBVD for a method with a usable color space
- The 20.4% improvement over CIEDE2000 is real and substantial
- Even the generation-optimized v17-relaxed (26.67) beats CIEDE2000 (29.18)

### 2.2 Structural Achromatic Guarantee
- Grays map to C < 10⁻⁶ — not a penalty, a mathematical guarantee
- This is something CIEDE2000 doesn't even attempt (it has no color space)
- Oklab and CAM16-UCS have decent but imperfect achromatic axes
- Critical for UI: gray gradients with zero color artifacts

### 2.3 Full Pipeline (Space + Metric + Tooling)
- Unlike CIEDE2000 (metric only, no space) or Oklab (space only, weak metric), Helmlab provides both
- Gamut mapping, WCAG contrast, token export, dark/light mode — practical UI tooling
- Exact invertibility via Newton iteration (roundtrip error < 10⁻¹⁴)

### 2.4 Embedded Helmholtz-Kohlrausch
- H-K effect (saturated colors appear brighter) is learned from data, not bolted on
- Most color spaces ignore H-K entirely; CAM16 has it but requires full appearance model setup
- Important for UI: a saturated blue and a gray at the "same lightness" should actually look equal

### 2.5 Free Hue Improvement via Rigid Rotation
- φ=-28.2° rotation reduces hue_rms from 25.9° to 16.1° with mathematically zero cost to STRESS
- Elegant: exploits the isometry of the distance metric (da²+db² is rotationally invariant)

---

## 3. What Helmlab Does Poorly (Honest Weaknesses)

### 3.1 Overfitting Risk — Nuanced, Not Catastrophic

**The numbers:**
- 72 parameters optimized against 3,813 COMBVD pairs
- Ratio: ~53 data points per parameter
- Neural net baseline: train STRESS 20.98, val STRESS 25.70 (gap = 4.72)
- Helmlab v19: train STRESS 23.22, val gap +1.11

**Evidence AGAINST severe overfitting:**
- **MacAdam 1974 (128 pairs) is fully held-out** — never seen during optimization. Helmlab achieves STRESS 20.2 (competitive with CAM16-UCS's 18.7). This is real cross-dataset validation.
- **He 2022 (82 pairs) is semi-held-out** — used as mild regularizer (λ=0.05). Helmlab achieves 29.0 vs CIEDE2000's 32.6.
- **Val gap +1.11** — quite small for 72 parameters. Compare: the neural net with ~50k params had a 4.72 gap.
- **Analytical form is a strong regularizer** — each parameter has a physical role (matrix coefficient, gamma exponent, Fourier harmonic). This is not a black-box with 72 free knobs.

**Bootstrap validation (now completed):**
- 10,000-iteration paired bootstrap: Helmlab 95% CI [22.50, 23.93], CIEDE2000 95% CI [27.64, 30.84]
- **Zero overlap** — Helmlab wins in 100% of bootstrap samples (p < 10⁻⁴)
- The 6-point improvement is statistically unambiguous on the aggregate dataset

**What's still missing:**
- Per-sub-dataset significance testing (3/6 sub-datasets favor CIEDE2000)
- Parameter uncertainty across the 8 restarts is unquantified

**For comparison:**
- CIEDE2000 has ~8 effective parameters — less overfitting risk, but was also trained *and* evaluated on COMBVD (circular validation)
- Oklab has 18 parameters but was never tested on psychophysical data (optimized against CAM16-UCS predictions)
- CAM16 has 20+ parameters grounded in a full appearance model

**Verdict:** The overfitting risk is real but mitigated by the analytical form, small val gap, and competitive performance on fully held-out MacAdam data. The missing piece is statistical rigor (CI/bootstrap), not evidence of actual overfitting.

### 3.2 Sub-Dataset Variance

Helmlab **underperforms CIEDE2000 on 3 of 6 COMBVD sub-datasets** (BFD-P(C), Leeds, RIT-DuPont):

| Dataset | N | Helmlab | CIEDE2000 | Winner |
|---------|---|---------|-----------|--------|
| BFD-P(D65) | 2,028 | **22.95** | 24.14 | Helmlab |
| BFD-P(M) | 548 | **22.10** | 35.09 | Helmlab (-13.0!) |
| Witt | 418 | **29.54** | 30.37 | Helmlab |
| BFD-P(C) | 200 | 31.53 | **30.05** | CIEDE2000 |
| Leeds | 307 | 21.08 | **19.46** | CIEDE2000 |
| RIT-DuPont | 312 | 23.23 | **19.51** | CIEDE2000 (+3.7) |

Helmlab's aggregate win is driven by BFD-P(M) (+13 points) and BFD-P(D65) being the largest sub-dataset (2,028 pairs). On the smaller, independently-collected datasets (Leeds, RIT-DuPont), CIEDE2000 wins by 1.6-3.7 points. This means:
- The aggregate STRESS improvement masks inconsistent performance
- Different psychophysical experiments have different error patterns
- CIEDE2000's simpler structure may generalize better to certain data distributions
- Helmlab's advantage is concentrated on specific data subsets

### 3.3 No HDR / Wide Color Gamut Support

| Feature | CIEDE2000 | CAM16-UCS | JzAzBz | ICtCp | Helmlab |
|---------|-----------|-----------|--------|-------|---------|
| SDR/sRGB | Best | Good | OK | Poor | **Best** |
| HDR (>100 nits) | **Breaks** | Good | Good | **Best** | **Not tested** |
| Wide gamut (Rec.2020) | OK | Good | Good | Good | **sRGB/P3 only** |

- Helmlab is optimized for sRGB/Display P3 only
- No PQ transfer function, no absolute luminance handling
- In the HDR era (HDR10, Dolby Vision, Apple XDR), this is a growing limitation
- JzAzBz and CAM16-UCS-PC dominate in HDR territory

### 3.4 Not a Standard — Zero Industry Adoption

| Method | Standards | Adoption |
|--------|-----------|----------|
| CIELAB/CIEDE2000 | ISO/CIE 11664 | Universal (paint, textile, dental, print, display) |
| Oklab | CSS Color Level 4 | All browsers, Figma, design tools |
| CAM16-UCS | CIE 2022 recommendation | Academic, Google HCT |
| ICtCp | ITU-R BT.2100 | HDR broadcast, streaming |
| HCT | Material Design 3 | Android ecosystem |
| DIN99d | DIN 6176 | German/European industry |
| **Helmlab** | **None** | **None** |

- No CIE, ISO, W3C, ITU, or any standards body recognition
- No browser support, no design tool integration
- Using Helmlab in production means vendor lock-in to a single-author project
- Industrial color matching (textiles, paint, print) requires certified standards compliance

### 3.5 Complex, Non-Standard Distance Metric

Helmlab's distance formula:
```
SL = 1 + sl*(L_avg - 0.5)²
SC = 1 + sc*C_avg
DE = ((dL/SL)² + wC*(da/SC)² + wC*(db/SC)²)^(p/2)   [p=0.83, Minkowski]
DE = DE / (1 + c*DE)                                    [monotonic compression]
DE = DE^q                                                [post-power, q=1.1]
```

**Issues:**
- 7 parameters in the metric alone (p, wC, c, alpha, q, sl, sc)
- Not compatible with any CIE ΔE formula — can't use as drop-in replacement
- Reproducing results requires *exact* parameter values; there's no closed-form standard
- CIEDE2000 has a formal ISO specification; Helmlab has a JSON file

### 3.6 Surround/Adaptation Model is Untrained

- All 11 surround-dependent parameters are 0.0 in production params
- Dark mode adaptation falls back to a heuristic L-inversion
- CAM16 has a principled adaptation model (von Kries + surround factors)
- CIECAM16 handles illuminant changes (D50↔D65); Helmlab is D65-only

### 3.7 13-Stage Pipeline Complexity

```
XYZ → M₁ → γᵢ → M₂ → Hue corr. → H-K → L corr.
    → Dark L → C scale → C power → L×C → HLC → Hue-L
    → NC → Rot φ → Lab
```

**Compare to Oklab:**
```
XYZ → M₁ → cube root → M₂ → Lab
```

- Helmlab's 13 stages are each small and interpretable, but the total complexity is high
- Newton iteration for inverse (3 stages, 8-15 iterations each)
- Oklab inverse: two matrix multiplies + cube
- Performance not benchmarked, but measured ~24x slower than Oklab per color (41 us vs 1.7 us)

### 3.8 Theoretical Context: Non-Riemannian Perception

Bujack et al. (PNAS 2022, EuroVis 2025) demonstrated that perceptual color space is fundamentally non-Riemannian: large color differences cannot be derived by integrating small steps. This is **not a Helmlab-specific weakness** — it applies equally to CIELAB, Oklab, JzAzBz, CAM16-UCS, and every other continuous color space. No method is exempt.

The practical implication: the theoretical noise floor (~20 STRESS on COMBVD) partly reflects this fundamental impossibility. Helmlab at 23.22 may be near the limit of what *any* continuous color space can achieve — which is an argument in its favor, not against it.

---

## 4. Head-to-Head Comparisons

### 4.1 Helmlab vs CIEDE2000

| Dimension | CIEDE2000 | Helmlab |
|-----------|-----------|---------|
| COMBVD STRESS | 24.2* | **23.22** |
| Has a color space | No (metric only) | **Yes** |
| Achromatic axis | N/A | **Guaranteed** |
| ISO/CIE standard | **Yes** | No |
| Industry adoption | **Universal** | None |
| HDR support | Breaks (STRESS ~69) | Not tested |
| Discontinuities | **3 known sources** | None known |
| Params (risk) | ~8 (low risk) | 72 (higher risk) |
| Implementation complexity | Moderate | High |
| Statistical validation | Weak (trained=tested) | Weak (no CI) |

**Summary:** Helmlab is likely more accurate, but CIEDE2000 is the global standard with 25 years of industrial validation. Replacing CIEDE2000 requires far more evidence than a 6-point STRESS improvement on its own training data.

### 4.2 Helmlab vs Oklab

| Dimension | Oklab | Helmlab |
|-----------|-------|---------|
| COMBVD STRESS (Euclid.) | ~47 | **~30** (Euclid. est.) |
| With best metric | ~28 | **23.22** |
| Hue linearity | **Good** (~10° RMS est.) | OK (16.1° RMS) |
| Computation speed | **1.7 us/color** | 41 us/color (~24x) |
| CSS support | **Yes** (`oklab()`, `oklch()`) | No |
| Browser support | **All major** | None |
| Gradient quality | **Excellent** | Good |
| Achromatic axis | Good (~0.001) | **Perfect (<10⁻⁶)** |
| Invertibility | **Exact (closed-form)** | Iterative (Newton) |
| Trained on | CAM16-UCS predictions | **Psychophysical data** |

**Summary:** Oklab is "good enough" for 99% of web/UI use cases, is ~24x faster, and has universal browser support. Helmlab is more accurate but faces an uphill adoption battle. The key question: does the accuracy difference matter in practice for UI design?

### 4.3 Helmlab vs CAM16-UCS

| Dimension | CAM16-UCS | Helmlab |
|-----------|-----------|---------|
| COMBVD STRESS | ~25-26 | **23.22** |
| HDR performance | **Excellent** | Unknown |
| Chromatic adaptation | **Yes (any illuminant)** | D65 only |
| Viewing conditions | **Full model** | Untrained |
| CIE recommendation | **Yes (2022)** | No |
| Scientific rigor | **Full appearance model** | Data-driven pipeline |
| Computation | Expensive | Expensive |
| Gamut mapping | Chroma-based | Chroma-based |

**Summary:** CAM16-UCS is the scientifically principled choice with CIE backing. Helmlab wins on COMBVD STRESS but lacks the theoretical foundation, adaptation model, and standards recognition.

### 4.4 Helmlab vs HCT (Google Material You)

| Dimension | HCT | Helmlab |
|-----------|-----|---------|
| Color science rigor | Weak (hybrid hack) | **Strong** |
| WCAG compliance | **Built-in (tone=40→3:1)** | Built-in (ensure_contrast) |
| Ecosystem | **Android 12+, billions of devices** | None |
| Perceptual accuracy | Moderate | **Best** |
| Tooling | **Material Color Utilities** | Python only |
| Token export | Material Theme Builder | CSS, Android, iOS, Tailwind |

**Summary:** HCT is scientifically weaker but has Google's ecosystem behind it. Helmlab is technically superior but has zero market presence.

### 4.5 Helmlab vs JzAzBz

| Dimension | JzAzBz | Helmlab |
|-----------|--------|---------|
| SDR STRESS | ~30 | **23.22** |
| HDR STRESS | **~33** | Unknown |
| MacAdam ellipses | **Best** | Not tested |
| Wide gamut | **Full (Rec.2020+)** | sRGB/P3 |
| Absolute luminance | **0-10,000 nits** | Relative only |

**Summary:** JzAzBz is the HDR generalist. Helmlab wins on SDR but is irrelevant for HDR workflows.

---

## 5. Honest Assessment: Where Does Helmlab Actually Belong?

### What Helmlab IS:
- The most accurate published color difference predictor on COMBVD
- A well-engineered UI color space with practical tooling
- A research contribution showing analytical forms can compete with neural nets
- A proof that 13 interpretable stages can beat 50k neural parameters

### What Helmlab is NOT:
- A replacement for CIEDE2000 in industrial color matching
- A replacement for Oklab in web/CSS applications
- A general-purpose color appearance model
- An HDR or wide-gamut solution
- A standards-backed method

### The Adoption Problem

Even if Helmlab is technically better, adoption requires:

1. **Independent validation** — Other labs testing on their own datasets (not just COMBVD)
2. **Standards body review** — CIE Technical Committee evaluation
3. **Implementation in major tools** — Browsers, Figma, Photoshop, design systems
4. **Community trust** — Years of real-world usage
5. **HDR story** — Modern displays demand it

**Oklab won CSS despite being less accurate than CAM16-UCS** — simplicity and "good enough" beat accuracy in practice. Helmlab faces the same dynamic but worse: it's more complex than Oklab *and* less established.

### The Niche Where Helmlab Excels

Helmlab's sweet spot is narrow but real:

**High-precision UI design systems** where:
- Color difference accuracy matters more than computation speed
- sRGB/Display P3 gamut is sufficient
- Achromatic purity is critical (gray palettes, neutral backgrounds)
- You control the full stack (not constrained by CSS/browser standards)
- You need a full pipeline (space + metric + gamut mapping + tokens)

Examples: Design system teams at large companies, specialized color tools, accessibility-focused applications.

---

## 6. Specific Technical Concerns

### 6.1 Inverse Transform Robustness
- Three stages use Newton iteration (8, 12, 15 iterations)
- Derivative singularity guard at |f'| < 1e-10 — replaces with 1.0 (silent failure)
- No extreme-value tests in test suite (L→0, L→1, achromatic edge cases)
- Claimed roundtrip error <10⁻¹⁴ is not tested at boundaries

### 6.2 Neutral Correction LUT
- 256-sample PCHIP interpolation
- Extrapolation outside the LUT's L range could produce wild oscillations
- No validation that LUT covers the full [0,1] L range
- Rebuilt at every initialization — no caching

### 6.3 Dead Code in Distance Metric
- v15 hue-modulated SL/SC: all 8 Fourier params are 0.0 — optimizer found no benefit
- v14 dist_linear (alpha): 0.0 — ineffective
- v14 dist_sl: 9.16e-05 — essentially zero
- These suggest the distance metric has unused degrees of freedom

### 6.4 No Performance Benchmarks
- Transform speed not measured
- Gamut mapping (binary search) iteration count not profiled
- No comparison to Oklab/CIELAB computation time
- For real-time applications (animation, interactive pickers), this matters

---

## 7. The Bigger Picture: Is 23.22 vs 29.18 Meaningful?

### Arguments FOR significance:
- 20.4% STRESS reduction is large by color science standards
- CIEDE2000→CIE94 was ~4 units; Helmlab→CIEDE2000 is ~6 units
- Melgosa (2015) showed even a single power exponent can reduce STRESS by 5.7 units — suggesting the COMBVD noise floor hasn't been reached
- The improvement is consistent across train/val splits

### Arguments AGAINST significance:
- COMBVD has known contradictory pairs — some error is irreducible data noise
- 2/3 of remaining error is shared between neural net and analytical — suggesting ~20 is a hard floor
- No independent dataset confirmation
- Sub-dataset variance: Helmlab loses on 3/6 sub-datasets
- Bujack (PNAS 2022): perceptual space is non-Riemannian — any continuous space has fundamental limits
- In practical UI design, the difference between STRESS 23 and 29 is rarely perceptible to end users

### The honest answer:
The improvement is **statistically proven on the aggregate dataset** (bootstrap p < 10⁻⁴, zero CI overlap), and **practically relevant for color science but marginal for UI design**. A designer choosing between two blues won't notice the difference between Helmlab and CIEDE2000. But a design system choosing tonal scales across 50+ colors might.

---

## 8. Recommendations for Helmlab's Future

### To be taken seriously by color science:
1. **Independent dataset validation** — Test on datasets Helmlab was never trained on
2. ~~**Bootstrap confidence intervals**~~ — **DONE**: p < 10⁻⁴, zero CI overlap
3. **Per-sub-dataset reporting** — Show where Helmlab wins and loses, transparently
4. **CIE Technical Committee submission** — Get expert review

### To gain practical adoption:
1. **JavaScript/WASM implementation** — Web developers need it in the browser
2. **Figma/Sketch plugin** — Design tool integration
3. **CSS proposal** — Even as a long-shot, `helmlab()` in CSS Color Level 5
4. **Speed optimization** — Close the gap with Oklab (GPU, SIMD, lookup tables)
5. **HDR extension** — PQ transfer function, absolute luminance

### To strengthen the model:
1. **10° observer** — Modern displays subtend large visual angles
2. **HDR training data** — Extend beyond sRGB luminance range
3. **Train surround parameters** — Dark/light mode needs real data
4. **Cross-dataset regularization** — Penalize sub-dataset variance during optimization
5. **Ablation study** — Prove each of the 13 stages contributes meaningfully

---

## 9. Final Verdict

| Question | Answer |
|----------|--------|
| Is Helmlab the most accurate on COMBVD? | **Yes** |
| Is the improvement proven beyond doubt? | No (needs independent validation + CI) |
| Should a design system adopt it today? | Only if they control the full stack and value accuracy over ecosystem |
| Will it replace CIEDE2000? | Not without standards body recognition and 10+ years of validation |
| Will it replace Oklab in CSS? | Extremely unlikely — too complex, too slow, too unknown |
| Is it a meaningful research contribution? | **Yes — shows analytical forms can beat established formulas** |
| Is 72 parameters too many? | Borderline — strong regularizer (analytical form) but no statistical proof |
| Does it work for HDR? | Unknown — major gap |
| Is it practically useful today? | Yes, for the specific niche of high-precision UI color systems |
