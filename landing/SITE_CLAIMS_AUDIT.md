# Site claims audit — helmlab.space

Every factual claim on the site, its source, and verification status.
Status: ⬜ not verified · ✅ verified · ❌ false/inconsistent · ⚠ needs context

Verification system: ColorBench (canonical eval) + STRESS measurement on COMBVD/MacAdam/HumanFB.

---

## A. MEASUREMENT (MetricSpace) claims

| # | claim | source | status | note |
|---|---|---|---|---|
| M1 | STRESS **22.48** on COMBVD (with Bradford CAT) | stress-scores.ts, README, docs, Hero | ⬜ | my no-CAT measure = 22.73; check with CAT |
| M2 | **23%** better than CIEDE2000 (29.20) on COMBVD | README, docs | ⬜ | 22.48/29.20 = 23.0% ✓ arithmetic; verify both numbers |
| M3 | MacAdam **19.51** | stress-scores.ts | ⬜ | |
| M4 | HumanFeedback **23.26** | stress-scores.ts | ⬜ | |
| M5 | Average **21.75** (3 sources) | stress-scores.ts | ⬜ | (22.48+19.51+23.26)/3 = 21.75 ✓ arithmetic |
| M6 | Baselines: CIE94 combvd 33.37 · CIEDE2000 29.20 · DIN99 35.57 · CIELAB 41.92 | stress-scores.ts | ⬜ | |
| M7 | COMBVD **3,813 pairs**, **64,000+** human judgments | README, docs | ⬜ | pair count checkable |
| M8 | **72-parameter** pipeline | docs, README | ⬜ | count params in v21 json |
| M9 | overfit: v21 train 22.14 / test 23.91 / gap 1.77 | overfit.ts | ⬜ | needs train/test split |

## B. GENERATION (GenSpace) claims

| # | claim | source | status | note |
|---|---|---|---|---|
| G1 | ColorBench record **64-9-17** vs OKLab (90 metrics) | README, docs, Hero | ❌ | **INCONSISTENT**: data file headline = 65-9-16 |
| G2 | **6-1** on independent datasets | README | ⚠ | Hung-Berns data = 6-5-1; independentMetrics=7 → maybe 6-1 there |
| G3 | **360/360/360** valid cusps (sRGB/P3/Rec2020) | docs, README | ⬜ | ColorBench cusp test |
| G4 | **zero** monotonicity violations | docs, README | ⬜ | ColorBench |
| G5 | Blue→White midpoint sky blue, **G/R = 1.51** (not lavender) | docs, README | ⬜ | compute midpoint |
| G6 | Achromatic precision **C* < 10⁻¹⁵** | docs | ⬜ | gray-axis chroma; genspace-results says 1.88e-15 |
| G7 | depressed cubic **α = 0.021** | README, docs | ⬜ | check genspace params |
| G8 | chroma power **C^0.978** | README | ⬜ | check params |
| G9 | Round-trip: sRGB **5.64e-8**, P3 **2.0e-15**, Rec2020 **1.78e-15** | genspace-results.ts | ⬜ | round-trip test |
| G10 | Hung-Berns angular dev: genspace **4.72** · oklab 4.96 · cielab 5.94 | independent.ts | ⬜ | |
| G11 | gradient CV / OSA 37.45 vs OKLab 38.2 | genspace-results.ts | ⬜ | ColorBench |

## C. OTHER claims

| # | claim | source | status | note |
|---|---|---|---|---|
| O1 | bundle **11.6 KB** / ~12KB gzipped, **zero deps** | FaqSection, README | ⬜ | check built dist size |
| O2 | merged into Color.js (helmgen/helmgenlch/helmlab-metric); colorjs.io 0.6.1 doesn't include yet | FaqSection | ⬜ | external claim |
| O3 | contrast_ratio('#fff','#3B82F6') = **3.68** | docs, README | ⬜ | compute |

---

## FOUND INCONSISTENCIES (before measurement)
1. **G1: 64-9-17 (README/docs/Hero) ≠ 65-9-16 (data file headline)** — same 90 total, different split. One is stale.
2. **G2: "6-1 independent" (README) vs 6-5-1 (Hung-Berns data)** — ambiguous; clarify which independent set.
3. M1: STRESS 22.48 is *with Bradford CAT*; without CAT it is 22.73 (my measure). Docs should keep the CAT caveat (they do).

---

## VERIFICATION RESULTS (measured 2026-06-03)

### ✅ Verified correct (measured, match)
| # | claim | measured | verdict |
|---|---|---|---|
| M2 | CIEDE2000 COMBVD 29.20 | 29.18 | ✅ (rounding) |
| M3 | MacAdam 19.51 | 19.51 | ✅ exact |
| M7 | COMBVD 3,813 pairs | 3813 | ✅ exact |
| G6 | achromatic C* ~10⁻¹⁵ | 1.70e-15 (exact D65 method) | ✅ (was 5.7e-5 with wrong white = artifact) |
| G7 | depcubic α = 0.021 | 0.021 | ✅ exact |
| G8 | chroma power C^0.978 | 0.978 | ✅ exact |
| G9 | sRGB round-trip 5.64e-8 | **5.64e-08** (64k structured grid) | ✅ exact (random sample missed worst-case) |
| O3 | contrast(#fff,#3B82F6) 3.68 | 3.68 | ✅ exact |

### ⚠ Approximately correct / needs caveat
| # | claim | measured | note |
|---|---|---|---|
| M1 | STRESS 22.48 (with Bradford CAT) | no-CAT 22.73 / **my-CAT 22.56** | ~correct; 22.48 likely a slightly different CAT (CAT02?) — within 0.1 |
| G5 | Blue→White midpoint G/R = 1.51 | #649cff, **G/R = 1.56** | sky blue ✓ (qualitative claim holds); ratio off by 0.05 |
| G6 | "C* < 10⁻¹⁵" | 1.70e-15 | technically ~1.7e-15 (marginally above 1e-15); say "~10⁻¹⁵" |

### ❌ INCONSISTENT — needs a fix
| # | claim | reality |
|---|---|---|
| G1 | README/docs/Hero: **64-9-17** | data file headline: **65-9-16** (both =90). One is stale — pick the canonical ColorBench run and align all. |
| G2 | README: "**6-1** independent" | Hung-Berns data = **6-5-1**; independentMetrics=7. The "6-1" is unexplained — likely the 7 independent ColorBench metrics (6 win, 1 loss?), needs the ColorBench run to confirm. |
| M8 | docs/README: "**72** parameters" | naive json-key count = 94 (includes derived/metadata keys). "72" is the documented trainable-param count — not independently re-counted; confirm definition. |

### ⬜ STILL TO VERIFY (need ColorBench run — colorbench/run.py)
- G1 exact record (64-9-17 vs 65-9-16) — run canonical ColorBench GenSpace vs OKLab on 90 metrics
- G3 360/360/360 valid cusps (sRGB/P3/Rec2020)
- G4 zero monotonicity violations (colorbench/run_near_mono.py)
- G10 Hung-Berns angular dev (genspace 4.72 / oklab 4.96 / cielab 5.94)
- G11 gradient CV / OSA (genspace 37.45 vs oklab 38.2)
- M4 HumanFeedback 23.26 · M5 average 21.75 · M6 other baselines · M9 overfit train/test
- O1 bundle 11.6 KB · O2 Color.js merge (external)

---

## COLORBENCH VERIFICATION (canonical results, colorbench v1.0, 2026-06-03)

Source: `colorbench/results/HelmCT(genspace_v0.11.1_colorjs_pr.json).json` (the exact params the
site cites) + `results/OKLab.json`, head-to-head via `core.comparison.compare_spaces`.

### ✅ Verified
- **G3 360/360/360 valid cusps** — sRGB 360/0, P3 360/0, Rec2020 360/0. ✅ exact.
- **G10 Hung-Berns** — GenSpace mean 4.716 (claim 4.72) ✅, OKLab 4.959 (claim 4.96) ✅, max 25.16/25.52 ✅.
- **G11 gradient CV p95** — GenSpace 1.39 (=139%, claim 138.78), OKLab 1.38 (=138%, claim 136.69). ✅ approx (CV×100; GenSpace slightly higher, as claimed).

### ❌ FALSE / WRONG — must fix on site
- **G1 win-loss-tie record.** Site: README **64-9-17**, data file **65-9-16**. Actual current ColorBench
  head-to-head: **GenSpace 61 – OKLab 13 – 16 tie** (total 90). NEITHER site number matches. GenSpace wins
  are OVERSTATED (claim 64-65 vs real 61) and OKLab wins UNDERSTATED (claim 9 vs real 13). Qualitative
  ("GenSpace beats OKLab ~4.7:1") holds, exact numbers do not. (Caveat: OKLab.json run 2026-05-30 vs
  GenSpace 2026-05-06, same v1.0/54-metric version — re-run both together for the definitive number.)
- **G4 "zero monotonicity violations" is FALSE.** Canonical result:
  `gamut.Rec2020.monotonicity_violations = 1`. sRGB 0, P3 0, but **Rec2020 = 1**. (channel_mono sub-metrics
  are all 0 — so the claim is true for channel-monotonicity but FALSE for Rec2020 gamut-cusp monotonicity.)
  The blanket "zero monotonicity violations" overstates; should say "0 in sRGB/P3, 1 in Rec2020".

### ⚠ Context
- **"90 metrics"** is defensible: the head-to-head spans 90 scored comparisons (61+13+16). But the
  methodology field says `total_metrics: 54` base categories (expand to 90 with gamut variants). Wording
  "90 ColorBench metrics" is OK; the RECORD attached to it (64-9-17) is what's wrong.
- **G2 "6-1 independent"** — not reproduced; the independent set needs its own head-to-head extraction.

### NET (ColorBench): G3 ✅, G10 ✅, G11 ✅ approx | G1 ❌ (61-13-16 not 64-9-17), G4 ❌ (Rec2020 has 1)

---

## REMAINING (non-ColorBench, measured 2026-06-03)
| # | claim | measured | verdict |
|---|---|---|---|
| M4 | HumanFB STRESS 23.26, n=3552 | **22.96**, n=**3477** | ⚠ STRESS ~ok (0.3 off, CAT); **pair count wrong (3477 not 3552)** |
| M5 | average 21.75 | (22.48+19.51+23.26)/3 = 21.75 | ✅ arithmetic |
| M6 | CIE94 33.37 / CIELab 41.92 / CIEDE2000 29.20 | 33.59 / 42.80 / 29.18 | ⚠ ~ok (within ~1, CAT/method diff) |
| O1 | bundle 11.6 KB (~12KB gzipped) | **16.9 KB gzipped** | ❌ off by ~5KB (my confidence adds ~1-2KB; gap predates them — re-measure & update) |

---

## FINAL TALLY (all claims)
- **✅ Verified exact/near:** M2, M3, M5, M7, G3, G6, G7, G8, G9, G10, O3 (11)
- **⚠ Approx / caveat:** M1 (CAT), M4 (STRESS), M6 (baselines), G5 (1.56 vs 1.51), G11 (CV), G6 ("<10⁻¹⁵"→~1.7e-15) (6)
- **❌ WRONG — fix on site:**
  1. **G1 record 64-9-17 / 65-9-16 → real 61-13-16** (GenSpace wins overstated, OKLab understated)
  2. **G4 "zero monotonicity violations" → Rec2020 has 1** (true only for sRGB/P3 + channel-mono)
  3. **O1 bundle 11.6 KB → ~16.9 KB gzipped**
  4. **M4 HumanFB n=3552 → 3477**
- **⚠ Needs definition:** G2 ("6-1" independent), M8 ("72 params" vs 94 json keys)

Qualitative claims all hold (GenSpace beats OKLab decisively, MetricSpace beats CIEDE2000). The exact
marketing numbers have 4 errors + 2 ambiguities to correct.

---

## ⚠ CORRECTION (2026-06-03) — my first G1 verdict was UNRELIABLE

Görkem asked "is ColorBench itself fair/correct right now?" — good catch. My first comparison (61-13-16)
used `OKLab.json` (run 2026-05-30) vs the GenSpace result (run 2026-05-06). **git log shows a MAJOR
ColorBench refactor between those dates** (Phase 5-10: legacy-file deletion, GenSpace port, 39-metric
re-port, deterministic-RNG fix). So those two results were from DIFFERENT code → the comparison was invalid.
ColorBench's own AUDIT.md also notes 20/37 tests were bias-flagged then fixed — the benchmark evolved.

**Re-ran BOTH spaces fresh on the SAME current code (2026-06-03, 54 metrics):**
- Head-to-head: **GenSpace 62 – OKLab 9 – 19 tie** (total 90).
- **OKLab = 9 now MATCHES the site exactly.** My earlier "OKLab understated (9→13)" was a cross-version
  artifact — WITHDRAWN.
- GenSpace 62 vs site 64-65 (off ~2-3); ties 19 vs site 16-17 (off ~2-3).

**Revised G1 verdict:** the site record is APPROXIMATELY right, not grossly wrong. The 2-3 win gap is
plausibly (a) params — I used `genspace_v0.11.1.json`, the site cited `v0.11.1_colorjs_pr` (the exact
_colorjs_pr input wasn't found), or (b) benchmark-version evolution. The ROBUST, version-independent error
is the **internal inconsistency: README 64-9-17 ≠ data file 65-9-16** (they disagree with each other).
G4 (Rec2020 monotonicity = 1) is re-confirmed on the fresh code — still a real error.

**Honest status of ColorBench fairness:** current version has self-audited + fixed 20 biased tests, so it
is *more* fair than the version that produced the site numbers — but that also means site numbers and
current numbers come from different benchmark generations and shouldn't be expected to match exactly.
Remaining uncertainty: I did not obtain the exact `_colorjs_pr` params, so the 2-3 win gap is not fully
attributed. NOT 100% pinned down.

