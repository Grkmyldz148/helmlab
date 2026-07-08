#!/usr/bin/env python3
"""Generate 1.0 cross-language reference values + precision report.

Covers the FULL public 1.0 surface (hl.gen / hl.metric / hl.tokens) so the
JS suite can verify value-level parity, plus round-trip precision counts.
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root / "src"))

import numpy as np
from helmlab import Helmlab

hl = Helmlab()

COLORS = [
    "#000000", "#ffffff", "#808080", "#ff0000", "#00ff00", "#0000ff",
    "#ffff00", "#00ffff", "#ff00ff", "#3b82f6", "#ef4444", "#22c55e",
    "#f59e0b", "#8b5cf6", "#ec4899", "#1a1a1a", "#e5e5e5", "#6488b8",
    "#012345", "#fedcba",
]
CSS_INPUTS = ["color(display-p3 1 0 0)", "color(display-p3 0.2 0.7 0.4)", "color(rec2020 0.9 0.1 0.5)"]
PAIRS = list(zip(COLORS[:-1], COLORS[1:])) + [("#ff0000", "#00ff00"), ("#808080", "#828282"), ("#3b82f6", "#4c8af7")]

r10 = lambda x: round(float(x), 12)
labr = lambda lab: [r10(v) for v in lab]

out = {}

# ── conversions ─────────────────────────────────────────────────────
out["conversions"] = []
for c in COLORS + CSS_INPUTS:
    glab = hl.gen.from_hex(c)
    mlab = hl.metric.from_hex(c)
    out["conversions"].append({
        "color": c,
        "gen_lab": labr(glab), "gen_lch": labr(hl.gen.to_lch(glab)),
        "gen_hex": hl.gen.to_hex(glab), "gen_p3": hl.gen.to_css(glab, "display-p3"),
        "metric_lab": labr(mlab), "metric_lch": labr(hl.metric.to_lch(mlab)),
        "metric_hex": hl.metric.to_hex(mlab),
        "metric_p3": hl.metric.to_css(mlab, "display-p3"),
        "metric_rec2020": hl.metric.to_css(mlab, "rec2020"),
    })

# ── generation ──────────────────────────────────────────────────────
out["gradients"] = []
for a, b in [("#ff0000", "#0000ff"), ("#0000ff", "#ffffff"), ("#22c55e", "#ec4899")]:
    out["gradients"].append({
        "a": a, "b": b,
        "srgb": hl.gen.gradient(a, b, 8),
        "p3": hl.gen.gradient(a, b, 8, gamut="display-p3"),
        "mix25": hl.gen.mix(a, b, 0.25), "mix50": hl.gen.mix(a, b, 0.5), "mix75": hl.gen.mix(a, b, 0.75),
    })
out["palette"] = hl.gen.palette("#3b82f6", 10)
out["scale"] = hl.gen.scale("#3b82f6")
out["scale_p3"] = hl.gen.scale("#3b82f6", gamut="display-p3")
out["hue_ring"] = hl.gen.hue_ring(12)
out["harmonies"] = {k: hl.gen.harmonies("#3b82f6", k) for k in
                    ["complementary", "analogous", "triadic", "tetradic", "split_complementary"]}
out["rotate"] = {str(d): hl.gen.rotate_hue("#3b82f6", d) for d in [30, 120, 240, -90]}
out["vivid"] = [{"c": c, "srgb": hl.gen.vivid(c), "p3": hl.gen.vivid(c, gamut="display-p3")}
                for c in COLORS if c not in ("#000000", "#ffffff")]
out["cusp"] = [{"h": h, "srgb": labr(hl.gen.cusp(h))[:2], "p3": labr(hl.gen.cusp(h, "display-p3"))[:2],
                "maxC06": r10(hl.gen.max_chroma(0.6, h)), "maxC06_p3": r10(hl.gen.max_chroma(0.6, h, "display-p3"))}
               for h in range(0, 360, 30)]
out["gamut_map_adaptive"] = [labr(hl.gen.gamut_map(np.array(l), method="adaptive"))
                             for l in [[0.5, 0.8, 0.0], [0.9, 0.1, 0.3], [0.2, -0.4, -0.4], [0.95, 0.0, 0.5]]]
out["contrast"] = [{"fg": fg, "bg": bg, "ratio": r10(hl.gen.contrast_ratio(fg, bg)),
                    "fixed": hl.gen.ensure_contrast(fg, bg, 4.5)}
                   for fg, bg in [("#3b82f6", "#ffffff"), ("#777777", "#ffffff"), ("#a51d1d", "#111113")]]
out["adapt"] = [{"c": c, "dark": hl.gen.adapt_to_mode(c, "light", "dark")} for c in ["#cccccc", "#3b82f6", "#333333"]]
out["adapt_pair"] = list(hl.gen.adapt_pair("#333333", "#ffffff", "light", "dark", 4.5))

# ── measurement ─────────────────────────────────────────────────────
out["metrics"] = []
for a, b in PAIRS:
    out["metrics"].append({
        "a": a, "b": b,
        "difference": r10(hl.metric.difference(a, b)),
        "euclidean": r10(hl.metric.euclidean(a, b)),
        "ciede2000": r10(hl.metric.ciede2000(a, b)),
        "jnd": r10(hl.metric.jnd(a, b)),
        "distance": r10(hl.metric.distance(hl.metric.from_hex(a), hl.metric.from_hex(b))),
    })
out["confidence"] = []
for a, b in [("#808080", "#828282"), ("#ff0000", "#00ff00"), ("#3b82f6", "#4c8af7")]:
    c = hl.metric.confidence(a, b)
    out["confidence"].append({"a": a, "b": b, "de": r10(c["de"]), "latent": r10(c["latent"]),
                              "disagreement": r10(c["disagreement"]), "reliability": r10(c["reliability"]),
                              "p_noticeable": r10(c["p_noticeable"]), "reliable": bool(c["reliable"]),
                              "extrapolated": bool(c["extrapolated"])})
out["nearest"] = hl.metric.nearest("#3b82f6", ["#3b7ff0", "#ff0000", "#00ff00", "#4c8af7"])
out["nearest"]["distance"] = r10(out["nearest"]["distance"])
out["nearest"]["runner_up_distance"] = r10(out["nearest"]["runner_up_distance"])
out["nearest"]["margin"] = r10(out["nearest"]["margin"])
out["info"] = []
for c in COLORS[:8] + CSS_INPUTS:
    i = hl.metric.info(c)
    out["info"].append({"color": c, "hex": i["hex"], "L": r10(i["L"]), "C": r10(i["C"]), "H": r10(i["H"]),
                        "luminance": r10(i["luminance"]), "in_srgb": i["in_srgb"], "in_p3": i["in_p3"],
                        "in_rec2020": i["in_rec2020"]})

# ── tokens ──────────────────────────────────────────────────────────
out["tokens"] = []
for c in ["#3b82f6", "#ff0000", "#808080", "#6488b8", "#012345"]:
    out["tokens"].append({
        "color": c,
        **{f: hl.tokens.css(c, f) for f in ["hex", "rgb", "hsl", "oklch", "p3", "rec2020"]},
        "android": hl.tokens.android(c),
        "ios_p3": hl.tokens.ios_p3(c),
        "swift": hl.tokens.swift(c),
    })
out["css_variables"] = hl.tokens.css_variables(hl.gen.scale("#3b82f6"), "--primary")

# ── round-trip precision (Python side) ──────────────────────────────
N = 12
grid = [f"#{r:02x}{g:02x}{b:02x}"
        for r in range(0, 256, 255 // (N - 1) or 1)[:N]
        for g in range(0, 256, 255 // (N - 1) or 1)[:N]
        for b in range(0, 256, 255 // (N - 1) or 1)[:N]]
gen_miss = sum(1 for h in grid if hl.gen.to_hex(hl.gen.from_hex(h)) != h)
met_miss = sum(1 for h in grid if hl.metric.to_hex(hl.metric.from_hex(h)) != h)
xyz_grid = np.array(np.meshgrid(np.linspace(0.02, 0.9, 6), np.linspace(0.02, 0.95, 6),
                                np.linspace(0.02, 1.0, 6))).reshape(3, -1).T
gen_xyz_err = max(float(np.max(np.abs(hl.gen.space.to_XYZ(hl.gen.space.from_XYZ(x)) - x))) for x in xyz_grid)
met_xyz_err = max(float(np.max(np.abs(hl.metric.space.to_XYZ(hl.metric.space.from_XYZ(x)) - x))) for x in xyz_grid)
out["py_roundtrip"] = {"grid_size": len(grid), "gen_hex_misses": gen_miss, "metric_hex_misses": met_miss,
                       "gen_xyz_max_err": gen_xyz_err, "metric_xyz_max_err": met_xyz_err}
out["grid_n"] = N

path = Path(__file__).parent.parent / "tests" / "reference" / "reference-1.0.json"
path.write_text(json.dumps(out, indent=1))
print(f"wrote {path}")
print("PY ROUNDTRIP:", out["py_roundtrip"])
