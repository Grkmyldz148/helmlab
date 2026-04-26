# Helmlab — Current State

> Bu dosyayı her yeni Claude oturumunda ilk oku.

## Deployed Versions

| Platform | Version | Params | Benchmark |
|----------|---------|--------|-----------|
| PyPI | `helmlab==0.11.8` | `src/helmlab/data/metric_params.json` (MetricSpace v21) | COMBVD=22.48★ |
| npm | `helmlab@0.11.8` | `packages/helmlab-js/src/data/params.json` (MetricSpace v21) | COMBVD=22.48★ |
| color.js PR #722 | — | MetricSpace v21 (NC LUT 384pt, extended to L=2.59) | **59-8** vs OKLab |

## MetricSpace v21 — PRODUCTION READY

| Test Suite | Result |
|---|---|
| Python tests (292+) | ✅ 308 passed, 2 skipped (1 skip: pandas missing — optional dep) |
| JS tests (196) | ✅ 196/196 passed |
| COMBVD STRESS | **22.48** (beats all 9 competitors) |
| MacAdam1974 STRESS | **19.51** ★ |
| HumanFeedback STRESS | **23.26** ★ |

**Key fixes for v21 production:**
- Sign-preserving power: `sign(x)*|x|^γ` in Python + JS (v21 M1 maps sRGB blue → negative LMS)
- NC LUT extended to L=2.59 with constant clamping beyond gray-axis peak (L=1.29)
- Python NC also uses clamped PCHIP (no PCHIP extrapolation beyond gray peak)
- METRIC_L_MAX updated: 1.144 → 1.6 (P3 magenta L≈1.56 with fixed dark_L_compress)

## Active Checkpoints

| Space | File | Score | Notes |
|-------|------|-------|-------|
| GenSpace (PyPI/npm) | `research/checkpoints/genspace_v0.11.1.json` | **60-8** | M2: `0.21186668...` |
| GenSpace (color.js PR) | `research/checkpoints/genspace_v0.11.1_colorjs_pr.json` | **59-8** | M2 renormed: `0.21193779...`, L(white)≈1.0 |
| MetricSpace (production) | `research/checkpoints/metricspace_v21.json` | COMBVD=22.48, Mac=19.51, HF=23.26 | v21 — WITH Bradford CAT |
| MetricSpace (archived) | `helmlab-experimental/checkpoints/metricspace_v20b.json` | COMBVD=27.69 (w/ CAT) | v20b — no CAT optimization |

## GenSpace Pipeline (v0.11.1)

```
XYZ → M1 → depcubic(α=0.021) → M2 → chroma_power(0.978) → PW_L → L-gated enrichment → Lab
```

- M2 L-row: `[0.21186668013760682, 0.7989440040850104, -0.004099375589489282]`
- color.js PR M2 L-row: `[0.21193779684470104, 0.7992121834263127, -0.00410075161564345]` (renormed)

## color.js PR #722

- Repo: `https://github.com/color-js/color.js/pull/722`
- Fork: `/tmp/colorjs-fork/` (local)
- Status: 2 approvals, 1 change requested
- Tests: 219/219 passing (v21 params, updated expected values)

## Run Benchmark

```bash
# GenSpace (production vs OKLab)
python3 colorbench/run.py oklab genspace --json research/checkpoints/genspace_v0.11.1.json

# MetricSpace (STRESS evaluation)
python3 colorbench/run.py metric --json research/checkpoints/metricspace_v21.json
```

## Repo Structure

```
helmlab/
├── STATUS.md                        ← bu dosya
├── src/helmlab/data/                ← Python production params
├── packages/helmlab-js/src/data/    ← JS production params
├── research/
│   ├── council.py                   ← GenSpace optimizer
│   └── checkpoints/
│       └── genspace_v0.11.1.json    ← GenSpace production checkpoint
├── colorjs-pr/                      ← color.js PR dosyaları
├── colorbench/                      ← benchmark engine (ayrı repo)
└── helmlab-experimental/            ← deney arşivi (ayrı repo)
```

## Key Decisions

- **GenSpace M2**: v2_51wins M2 (`0.21186668...`) → 60-8. color.js PR'da renormed versiyon kullanılıyor (L(white)≈1.0).
- **MetricSpace CAT**: helmlab.js Bradford CAT hâlâ mevcut (Color.js D65 ↔ Helmlab D65 farkı için).
- **refRanges**: helmgen a/b=[-0.6,0.6], helmgenlch c=[0,0.65] — Display P3 gamutunu kapsar.
