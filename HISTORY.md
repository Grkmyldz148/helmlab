# Helmlab Optimization History

How we got from a neural network prototype to a 72-parameter analytical color space with STRESS 23.22.

## Version Progression

| Version | Params | COMBVD STRESS | Key Addition |
|---------|--------|--------------|-------------|
| Neural INN | ~50k | 20.98 (overfit) | Baseline teacher model |
| v1 | 24 | 36.04 | Core pipeline (M1 + gamma + M2 + H-K) |
| v4 | 44 | 26.88 | Enrichment stages |
| v6 | 57 | ~25.5 | Multi-objective (He regularization) |
| v7 | 61 | 24.11 | Minkowski distance metric |
| v10 | 66 | 23.18 | H-K relocation + multi-start restarts |
| v12 | 66 | 23.16 | Monotonic distance compression |
| v14 | 70 | 23.18 | Post-power q + Munsell CV tracking |
| v14c | 72 | **22.75** | SL/SC pair-dependent weights (record) |
| v16-gen | 72 | 26.95 | Soft achromatic constraint |
| v17-relaxed | 72 | 26.67 | Reverse hue optimization |
| v18 | 70 | ~24.0 | Shared gamma (structural ach.) |
| **v19-NC** | **72** | **23.22** | **Neutral correction (production)** |

---

## Phase 0: Neural Network Foundation

Before the analytical model, we trained a 3D Invertible Neural Network (INN) to learn the XYZ-to-perceptual mapping from Munsell, Hung & Berns, and COMBVD data.

- **Train STRESS**: 20.98 (impressive, but overfit)
- **Val STRESS**: 25.70 (significant gap)
- **Takeaway**: The neural model showed the theoretical floor is around 20-21, but ~2/3 of remaining errors are shared between neural and analytical approaches, suggesting a noise floor around 20 in the COMBVD data itself.

The neural model served as a teacher for distilling the initial analytical form.

---

## Phase 1: Core Pipeline (v1 - v6)

### v1 — Minimal Pipeline (24 params, STRESS 36.04)

The simplest possible analytical color space:
- M1: 3x3 matrix XYZ → cone-like (9 params)
- Gamma: per-channel power compression (3 params)
- M2: 3x3 matrix → opponent Lab (9 params)
- H-K: basic Helmholtz-Kohlrausch embedding (3 params)

Already beats sRGB Euclidean (67.82) and Oklab Euclidean (47.46) by a large margin.

### v4 — Enrichment Stages (44 params, STRESS ~26.88)

Added the "enrichment pipeline" that makes the space competitive:
- Cubic lightness correction (3 params)
- Hue-dependent chroma scaling, 3-harmonic Fourier (6 params)
- L-dependent chroma scaling (2 params)
- Hue correction rotation, 3-harmonic (6 params)
- Enhanced H-K with hue modulation (3 params)

### v6 — Multi-Objective (57 params, STRESS ~25.5)

- Added chroma power with Fourier hue dependence (4 params)
- Dark lightness compression (1 param)
- Hue-lightness-chroma interaction terms (8 params)
- **Innovation**: Multi-objective loss with He 2022 regularizer (lambda=0.1)

---

## Phase 2: Distance Metric Exploration (v7 - v9)

### v7 — Minkowski Distance (61 params, STRESS 24.11)

The biggest single-version jump. Instead of Euclidean distance in Lab, we learned:
- Minkowski exponent p (not necessarily 2)
- Chroma weight wC (a,b channels vs L channel)
- 4th harmonic hue correction

**Insight**: The distance metric is half the battle. A good space with Euclidean distance scores ~30; the metric brings it down to ~24.

### v8-v9 — Refinements (63-65 params, STRESS ~23.5)

- v8: Hue-dependent lightness correction (captures "blue appears darker" effect)
- v9: Nonlinear distance compression term

---

## Phase 3: H-K and Multi-Start (v10 - v13)

### v10 — H-K Relocation (66 params, STRESS 23.18)

Two changes that each contributed:

1. **H-K moved to raw chroma**: Previously H-K operated on post-scaled chroma (step 7). Moving it to raw chroma (step 3.7) improved the signal — the H-K correction now sees the "true" chroma before hue-dependent scaling distorts it.

2. **Multi-start restarts**: 5 perturbed random seeds instead of 1. The loss landscape has many local minima; restarts find deeper basins.

Cross-validation: He=29.31, MacAdam=20.65.

### v12 — Monotonic Distance (66 params, STRESS 23.16)

- **Honest evaluation**: Train on 80% COMBVD, validate on 20% (seed=42). No more overfitting blindness.
- **Monotonic compression**: Replaced ad-hoc nonlinear distance with `DE_final = DE / (1 + c * DE)`. This guarantees monotonicity (larger raw distance = larger final distance) and prevents triangle inequality violations.

### v13 — Hue-Dependent Dark Compression (68 params)

Added hue modulation to dark-region lightness compression: dark blues and dark yellows need different treatment. Small but consistent gain.

---

## Phase 4: The SL/SC Breakthrough (v14 - v14c)

### v14 — Post-Power and Grid Search (70 params, STRESS 23.18)

Two new distance parameters explored via grid search:
- `dist_post_power` (q): `DE_compressed^q` — modest bias correction at q=1.1
- `dist_linear` (alpha): asymptotic linearity — ineffective, basin too deep

Munsell CV tracking added: v14 achieves 19% CV (excellent uniformity).

### v14c — Pair-Dependent Weights (72 params, STRESS 22.75)

**The single biggest improvement in the entire project.**

Inspired by CIEDE2000, we added pair-dependent weighting:
```
SL = 1 + sl * (L_avg - 0.5)^2    # dark/light pairs weighted differently
SC = 1 + sc * C_avg               # chromatic pairs weighted differently
DE = ((dL/SL)^2 + wC*(da/SC)^2 + wC*(db/SC)^2)^(p/2)
```

The optimizer independently discovered weights similar to CIEDE2000's — a validation of both approaches.

**Critical finding — binary basin behavior**:
- Without SL/SC: optimizer sits in the "v12 basin" (STRESS ~23.18, CV ~19%)
- With SL/SC: optimizer escapes to the "v14c basin" (STRESS 22.75, CV ~77%)
- No smooth Pareto frontier between these basins
- CV penalty (even lambda=2.0) cannot pull v14c back to the v12 basin
- These are fundamentally different optimization landscapes

**Trade-off**: v14c achieves the best STRESS ever (22.75) but has high CV (77%), meaning Munsell constant-hue lines are non-uniform. For measurement (predicting human data), v14c wins. For generation (creating palettes), v14 (23.18, CV 19%) is better.

---

## Phase 5: Generation Optimization (v16 - v17)

The measurement-optimal v14 maps neutral grays to chroma ~0.34 — the achromatic axis is deformed. Gray gradients show visible color artifacts. We need grays at C=0 for practical use.

### v16-gen — Soft Achromatic (STRESS 26.95, achC=0.023)

Added a soft penalty: `lambda * mean(C^2_grays)`.

Pareto sweep over lambda=[0, 50, 100, 200, 500, 1000]:
- lambda=0 to 200 and 1000: stuck in v14 basin, achC stays at 0.34
- **lambda=500: the only value that escaped** to a new basin with achC=0.023

Binary basin behavior again — there's no gradual transition, only escape or not.

Result: STRESS 26.95 (still beats CIEDE2000's 29.18), achC near zero. But hue angles are distorted.

### v17-strict — Tight Hue (STRESS 31.35, hue_rms=5.8deg)

Added hue alignment penalty on top of achromatic constraint. At lambda_hue=100, all primaries land within 8deg of expected positions — but STRESS is worse than CIEDE2000.

### v17-relaxed — Reverse Optimization (STRESS 26.67, hue_rms=21deg)

**Key technique: reverse optimization.** Start from the tight-hue basin (v17-strict) and gradually relax the hue penalty. The optimizer stays in the "good hue neighborhood" while recovering STRESS.

- lambda_hue=5 sweet spot: STRESS 26.67, achC=0.022, hue_rms=21deg
- Beats both CIEDE2000 (29.18) and Oklab (~27.5) while having reasonable hues

Best generation-only version, but significant STRESS gap vs measurement-optimal v14.

---

## Phase 6: Unifying Measurement and Generation

### v18 — Shared Gamma, Structural Achromatic (70 params, STRESS ~24)

**Idea**: Force gamma_0 = gamma_1 = gamma_2 (shared gamma), then project M2 perpendicular to the neutral direction. This mathematically guarantees a=b=0 for all grays.

**Problem**: Loses 2 degrees of freedom (72 -> 70 params). The optimizer is slightly more constrained.

**Result**: Decent but not quite as good as v14. Superseded by v19.

### v19-NC — Neutral Correction (72 params, STRESS 23.22)

**The production model.** Resolves the measurement-generation tradeoff with zero compromise.

**Mathematical insight** (discovered via Codex analysis): With 3 independent gammas, the functions Y^gamma_0, Y^gamma_1, Y^gamma_2 are linearly independent — no linear M2 can map all grays to a=b=0. The v18 shared-gamma approach works but loses DOF. The neutral correction approach preserves all DOF.

**Method**: Run 256 gray levels through the full pipeline (stages 1-9), record the achromatic error a_err(L) and b_err(L), fit PCHIP interpolants. At runtime, subtract:
```
a <- a - a_err(L)
b <- b - b_err(L)
```

**Critical architectural decision**: The correction must be at the END of the pipeline, not after M2. Post-M2 correction shifts hue early, which cascades through H-K, dark compression, and chroma scaling — causing extreme distortions (L diverging from 0.94 to 7.27 for some colors). End-of-pipeline correction lets the enrichment stages see the original hue, correcting only the final output.

**Results**:
- COMBVD STRESS: 23.22 (only +0.04 vs v14's 23.18 — essentially free)
- Achromatic chroma: 0.000001 (structural guarantee, not a penalty)
- Gammas: [0.395, 0.421, 0.434] — all 3 independent, full DOF preserved
- Hue RMS: 25.9deg (before rotation)
- Val gap: +1.11 (good generalization)
- He: 29.03, MacAdam: 20.23, CV: 17.6%

### Rigid Rotation — Free Hue Improvement

**Discovery**: A rigid rotation of the (a,b) plane is an isometry — it preserves da^2 + db^2 for every pair, and since SL/SC depend only on L and C (also rotationally invariant), the entire distance metric is exactly preserved.

Empirical verification: COMBVD STRESS difference = 0.0000000000.

Minimax optimization finds phi = -28.2deg minimizes the worst-case hue error:
- hue_rms: 25.9deg -> 16.1deg
- hue_max: 48.4deg -> 20.2deg

This improvement is literally free — zero cost to color-difference prediction.

---

## Failed Experiments

### v15 — Hue-Dependent SL/SC (80 params)

Added 8 Fourier parameters to make SL/SC hue-dependent, hoping to keep v14c's STRESS gains while improving CV in specific hue regions.

**Result**: Complete failure. All 8 Fourier parameters converged to ~0 across all restarts. The optimizer had no signal to work with — the hue modulation adds DOF without enough data to justify them.

### Achromatic Lambda Sweep (v16)

Most lambda values (0, 50, 100, 200, 1000) failed to escape the v14 measurement basin. Only lambda=500 worked, and only on 1 of 8 restarts. The optimization landscape is treacherous.

### v17 Hue Lambda=200

5 out of 5 restarts produced singular matrices. The hue constraint at this strength pushes the optimizer to the edge of the feasible region where matrices become degenerate.

---

## Key Insights

1. **Distance metric matters as much as the space.** Helmlab Euclidean scores ~30.2; the full 7-param metric brings it to 23.22. Roughly half the improvement comes from each.

2. **Binary basins, not smooth Pareto frontiers.** The loss landscape has discrete basins separated by barriers. You either escape to a new basin or you don't — there's no gradual transition.

3. **Reverse optimization works.** When you need properties A and B that live in different basins: optimize hard for A (find basin A), then gradually relax A while optimizing B. The optimizer explores basin A's neighborhood for B-friendly regions.

4. **Structural > penalty-based constraints.** Penalties create basin conflicts (achromatic penalty vs STRESS). Structural guarantees (neutral correction) resolve the conflict architecturally with near-zero STRESS cost.

5. **End-of-pipeline corrections are safe.** Post-M2 corrections cascade through nonlinear stages unpredictably. End-of-pipeline corrections affect only the final coordinates, leaving the internal pipeline undisturbed.

6. **Restarts are essential.** 8 random restarts consistently find deeper basins than single runs. The v14c discovery and v16 lambda=500 escape both came from specific restarts.

7. **CIEDE2000 was right about SL/SC.** Our optimizer independently converged to pair-dependent lightness/chroma weights similar to CIEDE2000's design — a strong validation of that 2001 design decision.

8. **The noise floor is ~20.** Both the neural network (overfit) and the analytical model converge toward STRESS ~20-21. Roughly 2/3 of remaining error is shared, suggesting inherent noise in the psychophysical data.
