# Helmlab — Current State

> Bu dosyayı her yeni Claude oturumunda ilk oku.

## Deployed Versions

| Platform | Version | Params | Benchmark |
|----------|---------|--------|-----------|
| PyPI | `helmlab==0.17.0` | `src/helmlab/data/metric_params.json` (MetricSpace v21) | COMBVD=22.48★ |
| npm | `helmlab@0.17.0` | `packages/helmlab-js/src/data/params.json` (MetricSpace v21) | COMBVD=22.48★ |
| color.js PR #722 | — | MetricSpace v21 (NC LUT 384pt, extended to L=2.59) | **59-8** vs OKLab |

> Sürümler 2026-07-08'de registry'den doğrulandı (npm + PyPI ikisi de 0.17.0).
> **1.0.0 working tree'de hazır, HENÜZ YAYINLANMADI**: temiz kırılım — hl.gen / hl.metric / hl.tokens
> namespaced API, branded GenLab/MetricLab tipleri, wide-gamut üretim, cusp/vivid/jnd. Spec: API.md.
> Py 366 + JS 282+14 (parite kapısı) + postcss 27 test yeşil. Deploy = kullanıcı kararı.
> **Ölçülmüş hassasiyet (2026-07-08):** Py↔JS tüm string çıktılar birebir, sayısal worst ~1e-12;
> hex round-trip 1728-renk gridde 0 kayıp (iki uzay, iki dil); XYZ RT 2.9e-15 (Metric) / 5.8e-9 (Gen).
> Kalıcı parite kapısı: packages/helmlab-js/tests/parity-1.0.test.ts (+ generate-reference.py).

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
| GenSpace (PyPI/npm, v0.13.0+) | `helmlab-main-repo/checkpoints/genspace_v0.11.1.json` | **62-9-19 / 90** | M2 L-row renormed (`0.21193779...`) so white(D65) maps to L=1 exactly. Kanonik: 2026-06-03 aynı-kod taze re-run (bkz. landing/SITE_CLAIMS_AUDIT.md correction bölümü). Eski değerler: 64-9-17 (2026-05-06, full CIEDE2000) ve 65-9-16 (simplified ΔE — project_simplified_de2000_bug.md) farklı benchmark nesillerinden. |
| MetricSpace (production) | `helmlab-main-repo/checkpoints/metricspace_v21.json` | COMBVD=22.48, Mac=19.51, HF=23.26 | v21 — WITH Bradford CAT, display_phi_deg=−28.2° |
| MetricSpace (archived) | `helmlab-experimental/checkpoints/metricspace_v20b.json` | COMBVD=27.69 (w/ CAT) | v20b — no CAT optimization |

## GenSpace Pipeline (v0.11.1)

```
XYZ → M1 → depcubic(α=0.021) → M2 → chroma_power(0.978) → PW_L → L-gated enrichment → Lab
```

- M2 L-row: `[0.21193779684470104, 0.7992121834263127, -0.00410075161564345]` (renormed so white(D65) → L=1.0 exactly)

## color.js PR #722

- Repo: `https://github.com/color-js/color.js/pull/722`
- Fork: gerekirse `git clone git@github.com:color-js/color.js` (eski lokal kopya silindi)
- Status: 2 approvals, 1 change requested
- Tests: 219/219 passing (v21 params, updated expected values)

## Run Benchmark

```bash
# GenSpace (production vs OKLab)
python3 colorbench/run.py oklab genspace --json helmlab-main-repo/checkpoints/genspace_v0.11.1.json

# MetricSpace (STRESS evaluation)
python3 colorbench/run.py metric --json helmlab-main-repo/checkpoints/metricspace_v21.json
```

## Repo Structure

```
helmlab-main-repo/
├── STATUS.md                        ← bu dosya
├── src/helmlab/data/                ← Python production params
├── packages/helmlab-js/src/data/    ← JS production params
├── checkpoints/                     ← bundled production params
│   ├── genspace_v0.11.1.json
│   └── metricspace_v21.json
├── landing/marketing/color.js-badge/  ← Color.js entegrasyon rozetleri
└── paper/                           ← LaTeX kaynağı + arXiv tar.gz

../colorbench/                       ← benchmark engine (ayrı repo)
../research/                         ← optimize_metricspace.py + perflab/
../helmlab-experimental/             ← helmlab sürüm geçmişi (ayrı repo)
```

## Key Decisions

- **GenSpace M2**: bundled M2 (`0.21193779...`) is renormed so white(D65) maps to L=1.0 exactly. ColorBench 62-9-19 / 90 (2026-06-03 kanonik run), Python ↔ JS Lab parity 5e-16.
- **MetricSpace CAT**: helmlab.js Bradford CAT hâlâ mevcut (Color.js D65 ↔ Helmlab D65 farkı için).
- **refRanges**: helmgen a/b=[-0.6,0.6], helmgenlch c=[0,0.65] — Display P3 gamutunu kapsar.
