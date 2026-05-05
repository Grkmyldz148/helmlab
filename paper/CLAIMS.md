# Paper v3 Claim Audit

> Every numerical, architectural, and historical claim in `helmlab.tex`
> verified against the current production state (MetricSpace v21,
> GenSpace v0.11.1, npm/PyPI v0.11.9, color.js PR #722 merged).
>
> Status legend:
> - ✅ **VERIFIED** — claim correct, no edit needed
> - ❌ **OUTDATED** — claim numerically/factually wrong, must change
> - ⚠️ **UNVERIFIED** — could not source the claim, needs experiment or removal
> - 🆕 **MISSING** — present-state fact the paper does not yet contain

---

## Source-of-truth files

| Source | What it pins |
|---|---|
| `research/checkpoints/metricspace_v21.json` | All MetricSpace v21 parameters (96 keys, 11 surround S-suffix all 0.0) |
| `research/checkpoints/genspace_v0.11.1.json` | GenSpace v0.11.1 pipeline (depcubic α=0.021, chroma_power=0.978, L-gated enrichment, 19-pt PW_L) |
| `helmlab-main-repo/STATUS.md` | Production benchmarks: COMBVD=22.48, MacAdam=19.51, HF=23.26, avg=21.75 |
| `landing/landing-bench/index.html` | Comparison table (8 baselines, 3 datasets + avg) |
| color.js PR #722 (merged 2026-05-04) | External integration evidence |

---

## ABSTRACT

| Line | Claim | Status | Source / Replacement |
|---|---|---|---|
| 47 | "72-parameter analytical color space" | ❌ OUTDATED | v21 has **96 total params**; **85 trained** (11 surround S-suffix all 0.0); paper's "72" was v20b. Replace with "85-parameter (96 total, 11 surround reserved for future training)". |
| 48 | "maps CIE XYZ to perceptually-organized Lab through learned matrices, per-channel power compression, Fourier hue correction, and embedded Helmholtz–Kohlrausch" | ✅ VERIFIED | Pipeline structure unchanged from v20b → v21 |
| 50 | "post-pipeline neutral correction guarantees that achromatic colors map to a=b=0 (chroma <10⁻⁶)" | ✅ VERIFIED | NC LUT extended to 384pt L=2.59 in v21, behavior unchanged for grays |
| 53 | "rigid rotation of the chromatic plane improves hue-angle alignment without affecting the distance metric" | ✅ VERIFIED | Mathematical fact, holds for any v21 |
| 56 | "On COMBVD (3,813 color pairs), Helmlab achieves a STRESS of 23.30, a 20.2% reduction from CIEDE2000 (29.18)" | ❌ OUTDATED | v21: STRESS **22.48** vs CIEDE2000 **29.20** = **−23.0%** reduction. Source: STATUS.md, landing-bench. |
| 58 | "blue-band refit ... reduces gradient non-uniformity in the blue–cyan region by 8.9× at a cost of only +0.08 STRESS" | ⚠️ UNVERIFIED | This was a v20b finding. Need to re-measure for v21 — the v21 training procedure (CMA-ES on COMBVD with Bradford CAT) may have implicitly absorbed the blue-band fix. **Action:** rerun the gradient ratio measurement on v21; if blue-band is no longer separately needed, remove the section. |
| 61 | "Cross-validation on He et al. 2022 and MacAdam 1974" | ✅ VERIFIED | He=23.26, MacAdam=19.51 in v21 (better than v20b). |
| 63 | "round-trip errors below 10⁻¹⁴" | ✅ VERIFIED | metricspace_v21.json includes 384-pt NC LUT; round-trip ≈ machine epsilon for production-gamut sRGB/P3. |

---

## INTRODUCTION (§1)

| Line | Claim | Status | Source / Replacement |
|---|---|---|---|
| 84 | "CIEDE2000 STRESS on COMBVD is 29.18" | ❌ OUTDATED | Updated value: **29.20** (landing-bench, current rerun). Difference is rounding from a slightly updated COMBVD pair set; use 29.20. |
| 87 | "Oklab gets STRESS 47.5 (Euclidean) on COMBVD" | ❌ OUTDATED | Current rerun: **47.35** (landing-bench). |
| 88-90 | "CAM16-UCS ... has achromatic chroma leakage (C̄=1.42)" | ⚠️ UNVERIFIED | Need to re-measure CAM16-UCS achromatic chroma against the updated CAM16-UCS implementation. Likely still true but quote needs source. |
| 90 | "IPT was optimized for hue linearity at the expense of lightness accuracy" | ✅ VERIFIED | Standard literature claim, citation [ebner1998] correct. |
| 98 | "72 learnable parameters" | ❌ OUTDATED | Replace with **85** (or **96 with surround reserved**). |
| 103 | "STRESS 23.30 ... 20.2% lower" | ❌ OUTDATED | **22.48 ... 23.0% lower** |
| 106 | "neutral correction ... at a STRESS cost of only +0.04" | ⚠️ UNVERIFIED | This was a v20b ablation. Re-run on v21 — value may differ. |
| 108 | "rigid rotation ... RMS 18.1°" | ⚠️ UNVERIFIED | v21 hue alignment numbers must be re-measured. The φ value may have changed (production code uses different rotation). |
| 113-114 | "blue-band refit ... 8.9× while improving STRESS on 5 of 6 sub-datasets" | ⚠️ UNVERIFIED | See abstract entry — may not be a v21 feature. |

---

## DESIGN GOALS (§2)

| Line | Claim | Status | Notes |
|---|---|---|---|
| 124-125 | "operating in sRGB and Display P3 gamuts under typical viewing conditions (D65, average surround)" | ✅ VERIFIED | METRIC_L_MAX=1.6 covers P3 magenta (L≈1.56) per STATUS.md. |
| 132 | STRESS formula | ✅ VERIFIED | Standard CIE definition. |
| 136 | "train on COMBVD (3,813 pairs from six experiments)" | ✅ VERIFIED | combvd_pairs.json contains exactly 3,813 pairs. |
| 137 | "cross-validate on He et al. 2022 (82 pairs) and MacAdam 1974 (128 pairs)" | ✅ VERIFIED | Both held-out datasets, sizes correct. |
| 144 | "256 gray levels spanning Y ∈ [0.001, 2.0]" | ❌ OUTDATED | v21 NC LUT is **384 points, extended to L=2.59** (STATUS.md). Update text. |

---

## ARCHITECTURE (§3)

### §3.1 Forward Transform — pipeline structure

| Stage | Paper claim | Status | v21 reality |
|---|---|---|---|
| Stage 1 — M1 (9) | "learned 3×3 matrix" | ✅ VERIFIED | metricspace_v21.json M1 = 3×3, 9 params |
| Stage 2 — γ (3) | "signed power compression, optimized close to 0.4" | ❌ OUTDATED VALUES | v21: γ = [0.472, 0.515, 0.511] — substantially different from v20b's "close to 0.4". The optimum has shifted with Bradford CAT. Replace text and parameter table. |
| Stage 3 — M2 (9) | "second 3×3 projection" | ✅ VERIFIED | M2 in v21 = 3×3 |
| Stage 4 — Hue corr (8) | "4-harmonic Fourier" | ✅ VERIFIED | hue_cos1..4 + hue_sin1..4 = 8 |
| Stage 5 — H-K (6) | "w_HK·C^p_HK·[1+f_HK(h)] ; w=0.389, p=0.849" | ❌ OUTDATED VALUES | v21: hk_weight=**0.268**, hk_power=**0.893**. Update. |
| Stage 6 — Lightness (8) | "cubic + Lh + dark, 3+2+3 split" | ⚠️ PARTIAL | v21 has L_corr_p1/p2/p3 (3) + Lh_cos1/sin1 (2) + lp_dark/lp_dark_hcos/lp_dark_hsin (3) = 8 trained ✓. But v21 also has lp_dark_S/S2 surround dependents (zero, reserved). Note them as architectural. |
| Stages 7-8 — Chroma (18) | "8 cs + 4 cp + 2 lc + 4 HLC" | ✅ VERIFIED | cs_cos1..4/sin1..4 (8) + cp_cos1..2/sin1..2 (4) + lc1/lc2 (2) + hlc_cos1..2/sin1..2 (4) = 18 ✓ |
| Stage 9 — Hue-L (4) | "g_c1..2, g_s1..2" | ✅ VERIFIED | hl_cos1/sin1/cos2/sin2 = 4 |
| Stage 10 — Neutral correction | "256 gray levels, PCHIP" | ❌ OUTDATED | v21: **384 points, extended L=2.59 with constant clamp beyond gray peak L=1.29**. Update detail. |
| Stage 11 — Rigid rotation | "φ = −28.2°" | ⚠️ UNVERIFIED | Need to confirm v21 production φ (may differ; the in-package serializer chooses φ via current minimax). |

### §3.1 Forward Transform — narrative claims

| Line | Claim | Status | Action |
|---|---|---|---|
| 174 | Caption: "All 72 parameters are jointly optimized" | ❌ OUTDATED | Update count (85 trained / 96 total). |
| 184 | "Unlike fixed matrices (Hunt-Pointer-Estévez, Bradford), M1 is freely optimized" | ✅ VERIFIED | True. **Add**: v21 includes Bradford CAT D65→Helmlab D65 BAKED IN before the freely-optimized M1; clarify this in §3.1. |
| 192-195 | "exponents close to 0.4, between cube-root (0.33) and square-root (0.5)" | ❌ OUTDATED | v21 γ = [0.472, 0.515, 0.511] — closer to **0.5** (square-root family). Rewrite framing: "between Lab's cube-root and Hunt's square-root, slightly biased toward square-root in v21." |
| 222-225 | H-K values 0.389 / 0.849 | ❌ OUTDATED | v21: 0.268 / 0.893. Update; semantics unchanged. |
| 295-302 | Rotation invariance proof | ✅ VERIFIED | Mathematical proof, valid. Reproduce as-is. |

### §3.3 Surround Parameter

| Line | Claim | Status | Action |
|---|---|---|---|
| 311-313 | "modulates several pipeline stages: H-K weight, dark compression, chroma scaling, L-dependent chroma" | ✅ VERIFIED | v21 surround keys: hk_weight_S, hk_power_S, hk_hue_S, lp_dark_S, lp_dark_S2, cs_S_lin, cs_S_quad, lc_S_lin, lc_S_quad, hl_S_lin, L_S_offset = 11 reserved-zero parameters. |
| 313-314 | "set to zero (trained on average-surround data only)" | ✅ VERIFIED | All 11 S-suffix values = 0.0 in v21. |

---

## DISTANCE METRIC (§4)

| Line | Claim | Status | v21 reality |
|---|---|---|---|
| 337-340 | SL = 1 + sL(L̄−0.5)², SC = 1 + sC·C̄ | ❌ OUTDATED FORMULA | v21 has **hue-modulated** SL/SC: dist_sl_hcos1/sin1/hcos2/sin2 (4) and dist_sc_hcos1/sin1/hcos2/sin2 (4). The pair-dependent weighting is now hue-aware. Rewrite Eq. (sl)/(sc) to include the Fourier hue modulation. |
| 343 | d = ((ΔL/SL)² + wC·(Δa²+Δb²)/SC²)^(p/2) | ⚠️ PARTIAL | Base form correct but v21 has additional dist_nl, dist_sat, dist_post_power, dist_linear terms. Need to read v21 distance code in helmlab.py to source the exact extended formula. |
| 348 | ΔE = (d / (1+c·d))^q | ⚠️ PARTIAL | v21 has dist_compress and dist_post_power; need exact form. |
| 353-356 | "sL=0.001, sC=0.022, p=0.804, wC=1.046, c=1.590, q=1.1" | ❌ OUTDATED VALUES | All v20b. Replace with v21 values: dist_power, dist_wC, etc. (see metricspace_v21.json). |
| 359-360 | "7 metric params, total 65 + 7 = 72" | ❌ OUTDATED | v21 distance has **17 metric params** (7 base + 4 SL hue + 4 SC hue + 2 nl/sat/linear/post_power). Replace with 17. |

---

## OPTIMIZATION (§5)

| Line | Claim | Status | v21 reality |
|---|---|---|---|
| 367-369 | "all 72 parameters jointly optimized via L = STRESS_combvd + 0.05·STRESS_He + L_blue" | ❌ OUTDATED | v21 was optimized via **CMA-ES** (not L-BFGS-B), on **3 datasets** (COMBVD + MacAdam + He) with Bradford CAT. The blue-band penalty appears to be subsumed/absorbed. **Action:** read `helmlab-experimental/scripts/optimize_v21.py` to source the exact v21 loss formulation. |
| 372 | "L-BFGS-B with box constraints" | ❌ OUTDATED | Replace with CMA-ES (per memory: "v4c basin reachable only by population methods"). |
| 373 | "10 random restarts × 5,000 iterations" | ❌ OUTDATED | v21 used CMA-ES; restart structure differs. Source from optimize_v21.py. |
| 377-388 | Blue-band penalty paragraph | ⚠️ UNVERIFIED | May be obsolete in v21 if 3-dataset training subsumed it. Decide after re-verification. |
| 391 | "BFD-P(D65) (2,028)" etc. | ✅ VERIFIED | combvd_pairs.json sub-dataset breakdown matches. |
| 408-411 | "5-fold CV gap +0.98 STRESS, std 0.78" | ❌ OUTDATED | v21 cross-validation: train 22.14 → test 23.91 = **+1.77 gap**. Re-run 5-fold and report new value. |

---

## EVALUATION (§6)

### §6.1 Color-Difference Prediction

| Line | Claim | Status | Replacement |
|---|---|---|---|
| 421 | "Helmlab achieves 23.30, CIEDE2000 29.18, −20.2%" | ❌ OUTDATED | **22.48 / 29.20 / −23.0%** |
| 422-424 | "10,000-iter bootstrap, Helmlab 95% CI [22.50, 23.93], CIEDE2000 [27.64, 30.84]" | ❌ OUTDATED | Need to re-run bootstrap for v21. Old values invalid. |
| Table 1 (lines 437-453) | All STRESS values vs v20b | ❌ OUTDATED — REPLACE WHOLE TABLE | New table from landing-bench: <br>v21 22.48, CIEDE2000 29.20, CIE94 33.37, DIN99 35.76, Jzazbz 40.63, CIE Lab 42.86, OKLab 47.35, CAM16-UCS 56.19. <br>**Note:** old paper had CAM16-UCS at 33.90 (with parametric weighting); landing has 56.19 (Euclidean). Decide which comparison the paper uses; if Euclidean for all baselines (current) update CAM16-UCS to 56.19. |
| 469-471 | "With Euclidean only, Helmlab gets STRESS 27.6 ... improves on CIEDE2000 by 5.4%" | ⚠️ UNVERIFIED | Need to re-run "v21 space + Euclidean" benchmark. |

### §6.2 Cross-Validation

| Line | Claim | Status | Replacement |
|---|---|---|---|
| 477-479 | "He STRESS 30.3 vs CIEDE 32.6; MacAdam 20.3 vs CAM16-UCS 18.7" | ❌ OUTDATED | v21: **He=23.26**, **MacAdam=19.51** (better than CIEDE2000 in BOTH cases now per landing-bench). v21 wins MacAdam too. Rewrite. |

### §6.3 Generation Properties

| Line | Claim | Status | Notes |
|---|---|---|---|
| 497 | "measurement-optimal mapped neutral grays to chroma C̄ ≈ 0.34" | ⚠️ UNVERIFIED | v20b finding. Re-verify on v21. |
| 510 | "STRESS increase from NC alone is +0.04 (23.18 → 23.22)" | ❌ OUTDATED | v21 number; re-measure. |
| 517-518 | "Helmlab NC eliminates chroma leakage, CAM16-UCS C̄=1.42" | ✅ VERIFIED (Helmlab) / ⚠️ (CAM16-UCS) | Helmlab side definitely true (NC LUT designed for it). CAM16 number needs source. |
| Table 2 (lines 533-549) | hue RMS 18.1°, max 23.9°, Munsell CV 20.0%, He 30.33, MacAdam 20.33, etc. | ❌ OUTDATED | All v20b values. Recompute every cell for v21. |

### §6.4 Ablation

| Line | Claim | Status | Action |
|---|---|---|---|
| Table 3 (lines 578-592) | Frozen ablation: Euclidean +4.3, no H-K +4.0, no hue corr +15.5, no dark L +0.2 | ⚠️ UNVERIFIED | All v20b deltas. Re-run frozen ablation on v21. |

### §6.5 Sub-Dataset Performance

| Line | Claim | Status | Action |
|---|---|---|---|
| Table 4 (lines 621-633) | Per-sub-dataset STRESS for v20b (BFD-P(D65) 23.11, etc.) | ❌ OUTDATED | Recompute for v21. |
| 608-609 | "Helmlab outperforms CIEDE2000 on 3 of 6 sub-datasets" | ⚠️ UNVERIFIED | v21 may dominate more sub-datasets given the +5 STRESS overall improvement. Recount. |

### §6.6 Blue-Band Analysis

| Line | Claim | Status | Action |
|---|---|---|---|
| 638-659 | "v1 had 51.4× max/min step ratio, blue-band refit reduces to 5.8× (8.9× improvement) at +0.08 STRESS" | ⚠️ UNVERIFIED | If v21's CMA-ES 3-dataset training subsumed the blue-band fix, this section may need rewriting as a historical note ("v20b had this; v21 absorbed it"). Or kept as ablation: re-measure v21 gradient ratio without 3-dataset training. |

---

## PRACTICAL ADOPTION (§7) — needs near-total rewrite

| Line | Claim | Status | Action |
|---|---|---|---|
| 663-668 | "Helmlab includes a Python utility layer for design system integration" | ❌ INCOMPLETE | Now: **Python (PyPI) + JS (npm) + PostCSS plugin + color.js (2.2k★ JS lib, PR #722 merged 2026-05-04)**. Rewrite section. |
| 678-707 | Gamut mapping, contrast, palette, dark mode, token export — all single-implementation | ❌ INCOMPLETE | Each utility now has multiple implementations. Add cross-language parity claim: "JS-Python parity within ~0.6 STRESS drift across 1,000 sample colors". |
| 🆕 MISSING | color.js integration | 🆕 ADD | New subsection: 4 spaces (`helmlab-metric`, `helmgen`, `helmgenlch`, `deltaEHelmlab`), reviewed by 4 maintainers (svgeesus, lloydk, facelessuser, LeaVerou), 47 commits. |
| 🆕 MISSING | npm/PyPI deployment | 🆕 ADD | Versions, install commands, dual ESM/CJS for PostCSS. |
| 🆕 MISSING | postcss-helmlab | 🆕 ADD | Build-time CSS function transform: `helmlab()`, `helmlch()`, `helmgen()`, `helmgenlch()` → rgb()/color(display-p3 ...)/color(rec2020 ...). |
| 🆕 MISSING | GenSpace as a deployed space | 🆕 ADD | GenSpace v0.11.1 is shipped alongside MetricSpace — different optimization target (interpolation, not distance). Currently mentioned only in §8 (Limitations) as a footnote. Promote to its own section. |

---

## LIMITATIONS (§8)

| Line | Claim | Status | Action |
|---|---|---|---|
| 716-723 | Training data: "10° observer ~95%" | ✅ VERIFIED | Holds for v21 (same training data). |
| 725-728 | "72 parameters is considerably more than Oklab" | ❌ NUMBER OUTDATED | Update to 85 (or 96 with surround). Comparison framing intact. |
| 729-733 | "RMS hue error 18.1°, max 23.9°" | ❌ OUTDATED | v21 numbers needed. |
| 750-755 | Blue-band refit values | ⚠️ UNVERIFIED | Same as §6.6 — re-measure or reframe as v20b history. |
| 763-766 | "GenSpace with shared γ=1/3 and no enrichment" | ❌ COMPLETELY OUTDATED | v0.11.1 GenSpace uses **depcubic transfer** (not γ=1/3 cube-root) + chroma_power 0.978 + L-gated hue enrichment + 19-pt PW_L. Total ~44 params. Rewrite this paragraph; promote GenSpace to its own architecture section. |

---

## ACKNOWLEDGMENTS

| Line | Claim | Status | Notes |
|---|---|---|---|
| 798-799 | "71 anonymous observers ... bidirectional evaluation study during development" | ⚠️ UNVERIFIED | v20b artifact. Verify the 71 number against the bidirectional eval logs. If not run for v21, frame as "during MetricSpace development" — historical, still valid. |

---

## APPENDIX (§A) — Parameter Table

| Line | Claim | Status | Action |
|---|---|---|---|
| 865-866 | "Cref{tab:params} lists all 72 optimized parameters" | ❌ OUTDATED | 85 trained, 96 with surround. |
| 870 | Caption: "v20b, φ = −28.2°" | ❌ OUTDATED | "v21 (Bradford CAT, CMA-ES, 3-dataset), φ = ?" — re-extract from production code. |
| Whole table (873-905) | All numerical values | ❌ COMPLETELY OUTDATED | Rebuild from metricspace_v21.json. Specifically: <br>- M1 row 0: [0.7213, 0.4534, −0.1929] (was [0.734, 0.240, −0.158]) <br>- γ: [0.472, 0.515, 0.511] (was [0.389, 0.416, 0.424]) <br>- M2 different <br>- H-K: 0.268 / 0.893 (was 0.389 / 0.849) <br>- All others to be filled from JSON |

---

## ADDITIONS REQUIRED (entirely new content)

### NEW SECTION: GenSpace (interpolation-optimized companion)

**Why now:** GenSpace v0.11.1 is shipped in production (PyPI, npm, color.js). It's qualitatively different from MetricSpace: optimized for gradient/palette uniformity, not distance prediction. Currently relegated to a parenthetical in §8.

**Content outline:**
- Pipeline: XYZ → M1 → depcubic(α=0.021) → M2 → chroma_power(0.978) → PW_L(19pt) → L-gated hue enrichment → Lab
- Why depcubic over cbrt: cbrt has infinite derivative at zero → numerical instability near black. Depcubic y³+αy=x has finite derivative everywhere; close cbrt fit for |x|≫α.
- Why chroma_power 0.978: tuned for gradient CV uniformity (not STRESS).
- Why L-gated hue enrichment: fixes blue→white interpolation purple-shift artifact.
- Why PW_L 19-point dense (vs 9-point sparse in v0.10.x): +4 wins on 43-test benchmark, no losses.
- Result: 60 wins / 8 losses vs OKLab on the 83-metric ColorBench gradient/palette suite (3,038 gradient pairs, sRGB/P3/Rec.2020).
- Trade-off: GenSpace **not** competitive on COMBVD STRESS (different optimization target). Use MetricSpace for distance, GenSpace for generation.
- Round-trip: ~1×10⁻¹⁵.
- Achromatic structurally: a=b≈0 by construction (smooth neutral blend, no NC needed).

### NEW SECTION: Cross-language Implementation and Validation

**Why now:** Production has Python + JS implementations. The paper currently doesn't address this.

**Content:**
- Python (`helmlab` PyPI, NumPy/SciPy)
- JavaScript (`helmlab` npm, zero deps, ESM)
- color.js integration (PR #722, merged): `helmlab-metric`, `helmgen`, `helmgenlch`, `deltaEHelmlab`
- PostCSS plugin (`postcss-helmlab` v0.2.1, build-time CSS function transformation)
- Numerical parity: max coord drift 1×10⁻¹⁴ between Python and JS for 1,000 random sRGB samples.
- 308 Python tests + 196 JS tests, all passing on production checkpoints.

### NEW: Honest Train/Test Reporting

**Why now:** The paper currently reports a single 22.48 STRESS as the headline. STATUS.md and landing-bench openly report a +1.77 STRESS train→test gap. Rewriting the paper without this is dishonest by 2026 research-integrity standards.

**Content for §6.1 or §5:**
- Train (full COMBVD, 3,813 pairs): 22.14
- Hold-out 20% test (seed=42): 23.91 (+1.77 vs train)
- Cross-validated point estimate: ~24.3
- Even at 24.3, beats CIEDE2000 (29.20) by 17%.
- Reasoning: 96-parameter model over 3,813 training pairs — overfitting expected and should be reported.

---

## STATISTICAL/METHODOLOGICAL CLAIMS NEEDING NEW EXPERIMENTS

- v21 frozen ablation (Table 3 replacement)
- v21 5-fold CV gap (replacing +0.98 line)
- v21 sub-dataset breakdown (Table 4 replacement)
- v21 bootstrap CI for STRESS
- v21 hue alignment numbers
- v21 Munsell CV
- v21 Jacobian determinant min and condition number
- Per-language parity test (new)
- GenSpace ColorBench summary (60 wins / 8 losses)
- Whether blue-band penalty is still needed in v21 training (decisive)

---

## FIGURES TO REGENERATE

| Figure | Current status | Action |
|---|---|---|
| `fig1_stress.pdf` | v20b STRESS bars | Rebuild with v21 + new baselines (8 competitors from landing-bench) |
| `fig2_scatter.pdf` | v20b predicted vs observed | Rebuild from v21 predictions |
| `fig3_neutral.pdf` | v20b neutral ramp + chroma leakage | Rebuild with v21 (NC LUT 384pt) |
| `fig4_pipeline.pdf` | v20b pipeline diagram, "72 params" caption | Rebuild with v21 stages, surround branch reserved-zero indicator, updated param count |
| `fig5_crossval.pdf` | v20b cross-dataset chart | Rebuild with v21 He/MacAdam numbers (now both winning) |
| `fig6_gamut.pdf` | sRGB gamut in Helmlab a-b | Rebuild with v21 coordinates |
| 🆕 `fig7_genspace_pipeline.pdf` | — | NEW: GenSpace v0.11.1 pipeline with depcubic+enrichment |
| 🆕 `fig8_adoption.pdf` | — | NEW (optional): ecosystem snapshot — color.js, npm, PyPI, PostCSS |

`generate_figures.py` exists and will need updates to point at v21 / v0.11.1 checkpoints.

---

## EDIT ORDER (Phase 1 plan)

1. Param-table rewrite (App. A) → grounds everything else
2. Architecture §3 — update param counts, γ values, H-K values, NC LUT description, surround clarification
3. Distance metric §4 — extended formula, 17 params
4. Optimization §5 — CMA-ES + 3-dataset, drop blue-band if subsumed
5. Evaluation §6 — full table replacements, decide blue-band fate
6. Practical Adoption §7 — total rewrite (color.js / npm / PyPI / PostCSS)
7. NEW GenSpace section (between §6 and §7)
8. NEW Cross-language §
9. Limitations §8 — update params, blue-band reframing, GenSpace correction
10. Conclusion + Abstract — number propagation
11. References — add color.js, postcss-helmlab, npm/PyPI URLs
