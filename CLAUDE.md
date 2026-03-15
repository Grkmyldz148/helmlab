# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Python tests (408 tests, ~2 min)
python -m pytest tests/ -x -q

# Single test file
python -m pytest tests/test_gen_space.py -v

# JS tests (196 tests, <2s)
cd packages/helmlab-js && npx vitest run

# JS build (ESM + CJS + IIFE + types)
cd packages/helmlab-js && npm run build

# Python build
python -m build

# Full benchmark suite (generates HTML report)
python scripts/benchmark/run.py
```

## Architecture

Helmlab is a **family of two purpose-built color spaces** sharing the same XYZ→LMS→Lab structure but with different optimization objectives:

**MetricSpace** (72 params) — perceptual distance measurement:
```
XYZ → M1 → γ → M2 → hue_correction → H-K → cubic_L → dark_L → chroma_scale
    → chroma_power → chroma_scale(L) → HLC → hue_lightness → NC → rotation → Lab
```
Distance: `DE = ((dL/SL)² + wC*((da/SC)²+(db/SC)²))^(p/2) / (1+c*DE)` then `^q`

**GenSpace** (18 params) — gradient/palette generation:
```
XYZ → M1 → cbrt → M2 → NC → Lab
```
Key constraint: shared γ=1/3 guarantees structural achromatic axis (grays → a=b=0).

Both pipelines are **exactly invertible** (Newton iteration for nonlinear stages). The `Helmlab` class in `helmlab.py` / `helmlab.ts` is the unified UI layer that delegates to the appropriate space.

### Python ↔ JS Parity

Python (`src/helmlab/`) and TypeScript (`packages/helmlab-js/src/`) are independent implementations that must produce identical results. Parameters live in mirrored JSON files:
- `src/helmlab/data/metric_params.json` ↔ `packages/helmlab-js/src/data/metric-params.json`
- `src/helmlab/data/gen_params.json` ↔ `packages/helmlab-js/src/data/gen-params.json`

JS reference values are generated from Python: `python packages/helmlab-js/scripts/generate-reference.py`

### Key Files

- `src/helmlab/spaces/metric.py` / `gen.py` — core space implementations
- `src/helmlab/helmlab.py` — main API (gradient, palette, deltaE, contrast)
- `src/helmlab/utils/gamut.py` — binary-search chroma reduction for gamut mapping
- `src/helmlab/metrics/delta_e.py` — CIEDE2000 and other formulas
- `packages/helmlab-js/src/helmlab.ts` — JS API mirror
- `scripts/benchmark/run.py` — 43-test benchmark comparing spaces

### Optimization

GenSpace matrices are optimized with CMA-ES (not gradient-based). Optimization scripts are in `scripts/optimize_gen*.py`. Checkpoints saved to `checkpoints/`. Current best: v14 (28/43 benchmark wins vs OKLab 6/43).

## CI/CD

GitHub Actions (`.github/workflows/deploy.yml`) runs on every push to main:
1. Tests (Python + JS) must pass
2. Landing page auto-deploys to helmlab.space via SSH/rsync
3. npm publishes automatically when `packages/helmlab-js/package.json` version changes
4. PyPI publishes automatically when `pyproject.toml` version changes

## Conventions

- **Never add `Co-Authored-By: Claude`** to git commits
- **Never commit `gorkemyildizcom.md`** (contains API keys)
- GitHub username: **Grkmyldz148**
- When updating GenSpace matrices: update BOTH Python and JS param JSON files, plus hardcoded copies in `colorjs-pr/src/spaces/helmgen.js`, `docs/demo.html`, `landing/js/main.js`
- After changing gen_params.json: regenerate JS reference values with `python packages/helmlab-js/scripts/generate-reference.py`
- Version numbers must stay in sync: `pyproject.toml`, `src/helmlab/__init__.py`, `packages/helmlab-js/package.json`
