# Paper v3 — Status & Final-Check Notes

> Working file for the user's final review. Three commits worth of
> incremental work; please read before pushing to arXiv.

## What's done

### Phase 0 — Audit (✓)
- `paper/CLAIMS.md` — every paper claim categorised
  VERIFIED / OUTDATED / UNVERIFIED / MISSING with source pointers.

### Phase 1 — Architecture & Numerical Rewrite (✓)
- Reframed paper as a two-space family (MetricSpace + GenSpace).
- All v20b numerical values replaced with measured v21 values:
  - STRESS 23.30 → **22.48** (with Bradford CAT pre-processing)
  - CIEDE2000 reference 29.18 → **29.20**
  - γ values [0.389, 0.416, 0.424] → **[0.4723, 0.5149, 0.5113]**
  - H-K weight 0.389/0.849 → **0.2676/0.8935**
  - Distance metric values rewritten with v21's substantially
    different optimum (sL=−0.92, sC=2.93, p=1.97, c=52.5, q=0.48).
  - Hue RMS 18.1° → **26.4°** (honest regression vs v20b)
  - Munsell CV 20% → **32%** (honest regression vs v20b)
- New §7 GenSpace section with depcubic / chroma_power / L-gated
  enrichment rationale, 66/9 vs OKLab on ColorBench.
- §6.1 honest train/test reporting (+1.77 STRESS gap, ~24.3 CV
  point estimate).
- §6.6 Blue-Band reframed as v20b artefact absorbed by v21's
  sub-dataset-balanced loss.

### Phase 2 — Figures (✓)
- `generate_figures.py` updated to load `metricspace_v21.json` from
  `research/checkpoints/`.
- Bradford CAT applied to COMBVD pairs to reproduce headline 22.48.
- All 6 figures regenerated with v21 / v0.11.1 params.
- Fig 1 (STRESS): MetricSpace v21 = 22.5 vs CIEDE2000 = 29.2 ✓
- Fig 2 (scatter): predicted vs observed at v21 STRESS = 22.7 ✓
- Fig 3 (neutral): NC enabled — Helmlab achromatic chroma exact ✓
- Fig 4 (pipeline): structure unchanged, 72 params caption ✓
- Fig 5 (crossval): He 35.9 (lose), MacAdam 19.5 (win) ✓
- Fig 6 (gamut): v21 sRGB gamut at L=0.25/0.5/0.75 ✓

### Phase 3 — Practical Adoption (✓)
- §7 fully rewritten with PyPI/npm/color.js/PostCSS plugin
- Cross-language parity claim (Python ↔ JS within 1e-14)
- color.js PR #722 merged (May 2026) with all four spaces
- postcss-helmlab dual ESM/CJS for Next.js compatibility

### Phase 4 — Rationale Layer (mostly ✓)
- Each architectural choice now has a "why this and not that":
  - Why signed power compression (M1 maps blue → negative LMS)
  - Why depcubic vs cube-root (finite derivative at zero, no cusp)
  - Why chroma_power 0.978 (gradient CV reduction)
  - Why L-gated hue enrichment (blue→white purple shift fix)
  - Why 19-pt dense PW_L (43-test +4W gain over sparse)
  - Why sub-dataset-balanced loss (BFD-P(D65) was dominating)
  - Why low-chroma penalty (CIEDE2000 was beating us at C*=5-25)
  - Why NC as deferred toggle (lets distance basin train freely)

### Phase 5 — Compile (⚠ blocked locally)
- LaTeX not installed locally; needs pdflatex.
- Recommendation: open in Overleaf, or `brew install --cask basictex`
  in a terminal that can prompt for sudo password.

---

## Known issues to review before submission

1. **Hue RMS 26.4° is honest but ugly.** v21 traded hue alignment
   for STRESS. Consider:
   - Add a non-isometric hue-alignment post-step in MetricSpace
     for design tools (similar to v20b's φ but tuned for v21)?
   - Or keep current text framing it as "use GenSpace for geometry"?
   The paper currently uses option B.

2. **He 2022 regression (35.9 vs 32.6) is a real loss.** The text
   attributes it to (i) small N=82 and (ii) high-ΔE tail mismatch,
   but neither is rigorously demonstrated. Worth running an ablation
   that re-trains v21 with different `λ_He` weights to show the
   model can be moved on this set if needed.

3. **CAM16-UCS comparison.** The paper mentions in §1 that we
   re-evaluated baselines. The colorbench numbers include
   ICTCP-style chromatic adaptation handling for these spaces;
   mention this consistently or remove the implicit re-evaluation
   note.

4. **GenSpace section** discusses 66/9 vs OKLab but does not yet
   show a comparable figure. Consider adding `fig7_genspace.pdf`
   summarising the category breakdown.

5. **Bibliography** — the new entries (color.js, helmlab packages,
   postcss-helmlab, Ottosson cusp) use URLs and are not formatted
   to a venue's exact style. ICC submission may need DOI lookups
   or arXiv numbers.

6. **Acknowledgments still says "71 anonymous observers"** — the
   actual `human_feedback.json` has 71+ named observers (not
   anonymous, mostly the author's network). Either rephrase to
   "~71 observers (named in the dataset release)" or anonymise
   the data release.

---

## Post-figure verification

Direct measurement against `research/checkpoints/metricspace_v21.json`:

| Property | Measured | Paper says | Status |
|---|---|---|---|
| STRESS COMBVD (CAT, NC off) | 22.48 | 22.48 | ✓ |
| STRESS COMBVD (NC on) | 28.64 | (not headline) | ✓ |
| STRESS MacAdam | 19.51 | 19.51 | ✓ |
| STRESS He 2022 | 35.9 | 35.9 | ✓ |
| STRESS HumanFB | 23.26 | 23.26 | ✓ |
| Hue RMS (HSL ref, φ=−28.2°) | 26.4° | 26.4° | ✓ |
| Hue max | 53.4° | 53.4° | ✓ |
| Munsell CV | 32.0% | 32.0% | ✓ |
| Round-trip sRGB | 7e-15 | ~7e-15 | ✓ |
| Achromatic max C (NC on) | <1e-6 | <1e-6 | ✓ |
| Achromatic max C (NC off) | ~0.34 | (mentioned) | ✓ |

All headline numbers in the paper now match measurement.

## Files changed (since v2)

```
paper/CLAIMS.md       (new, 307 lines, claim audit)
paper/V3_STATUS.md    (this file)
paper/helmlab.tex     (+1027/−304, near-rewrite)
paper/generate_figures.py  (+80/−10, v21 + Bradford CAT)
paper/figures/fig1..6.{pdf,png}  (regenerated with v21)
```

## Suggested arXiv push

```bash
cd paper/
# Compile via Overleaf or:
pdflatex helmlab.tex && bibtex helmlab && pdflatex helmlab.tex && pdflatex helmlab.tex

# Then upload to arXiv as v3 of 2602.23010
# Don't forget to bump the version date in metadata
```
