#!/usr/bin/env python3
"""Generate reference values for TypeScript tests.

Tests the full pipeline (with NC + rotation) against known hex colors.
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root / "src"))

import numpy as np
from colorspace.helmlab import Helmlab

hl = Helmlab()

# Test colors spanning gamut
test_hexes = [
    "#000000", "#ffffff", "#808080",  # achromatic
    "#ff0000", "#00ff00", "#0000ff",  # primaries
    "#ffff00", "#00ffff", "#ff00ff",  # secondaries
    "#3b82f6", "#ef4444", "#22c55e",  # Tailwind blue/red/green
    "#f59e0b", "#8b5cf6", "#ec4899",  # Tailwind amber/violet/pink
    "#1a1a1a", "#e5e5e5",             # near-black/white
]

results = []
for hex_str in test_hexes:
    lab = hl.from_hex(hex_str)
    rt_hex = hl.to_hex(lab)
    srgb = hl.to_srgb(lab)
    info = hl.info(hex_str)

    results.append({
        "hex": hex_str,
        "lab": [round(float(x), 10) for x in lab],
        "rt_hex": rt_hex,
        "srgb": [round(float(x), 10) for x in srgb],
        "L": round(float(info["L"]), 10),
        "C": round(float(info["C"]), 10),
        "H": round(float(info["H"]), 10),
    })

# Pair distances
pairs = [
    ("#ff0000", "#00ff00"),
    ("#ffffff", "#000000"),
    ("#3b82f6", "#ef4444"),
    ("#808080", "#808080"),
]

distances = []
for h1, h2 in pairs:
    d = hl.delta_e(h1, h2)
    distances.append({
        "hex1": h1, "hex2": h2,
        "deltaE": round(float(d), 10),
    })

# Contrast ratios
contrasts = [
    ("#ffffff", "#000000"),
    ("#3b82f6", "#ffffff"),
    ("#000000", "#ffffff"),
]
contrast_results = []
for fg, bg in contrasts:
    cr = hl.contrast_ratio(fg, bg)
    contrast_results.append({
        "fg": fg, "bg": bg,
        "ratio": round(float(cr), 6),
    })

# Semantic scale for blue-500
scale = hl.semantic_scale("#3b82f6")

# XYZ round-trip
from colorspace.utils.srgb_convert import hex_to_srgb, sRGB_to_XYZ
xyz_tests = []
for hex_str in ["#3b82f6", "#ff0000", "#808080"]:
    srgb = hex_to_srgb(hex_str)
    xyz = sRGB_to_XYZ(srgb)
    xyz_tests.append({
        "hex": hex_str,
        "xyz": [round(float(x), 10) for x in xyz],
    })

output = {
    "forward": results,
    "distances": distances,
    "contrasts": contrast_results,
    "semantic_scale": {"base": "#3b82f6", "scale": scale},
    "xyz": xyz_tests,
}

out_path = Path(__file__).resolve().parent.parent / "tests" / "reference" / "reference-values.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Wrote {len(results)} forward tests, {len(distances)} distance tests")
print(f"to {out_path}")
