#!/usr/bin/env python3
"""Generate neutral-correction LUT for TypeScript package.

Runs 256 gray values through the full analytical pipeline (without NC)
and saves the resulting (L, a_err, b_err) for linear interpolation in TS.
"""
import json
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root / "src"))

import numpy as np
from helmlab.spaces.analytical import AnalyticalSpace

N = 256

space = AnalyticalSpace(neutral_correction=False, ab_rotate_deg=0.0)

# D65 white point
D65 = np.array([0.95047, 1.0, 1.08883])

# Sample Y values (log-spaced for better dark coverage, matching Python)
Y_vals = np.concatenate([
    np.linspace(0.001, 0.01, 10),
    np.linspace(0.01, 0.1, 20),
    np.linspace(0.1, 2.0, N - 30),
])

gray_XYZ = np.outer(Y_vals, D65)
Lab_gray = space.from_XYZ(gray_XYZ)

L_gray = Lab_gray[:, 0]
a_gray = Lab_gray[:, 1]
b_gray = Lab_gray[:, 2]

# Sort by L for monotone interpolation
order = np.argsort(L_gray)
L_sorted = L_gray[order]
a_sorted = a_gray[order]
b_sorted = b_gray[order]

# Remove duplicates
mask = np.diff(L_sorted, prepend=-np.inf) > 1e-12
L_sorted = L_sorted[mask]
a_sorted = a_sorted[mask]
b_sorted = b_sorted[mask]

lut = {
    "L": [round(float(x), 10) for x in L_sorted],
    "a_err": [round(float(x), 10) for x in a_sorted],
    "b_err": [round(float(x), 10) for x in b_sorted],
}

out_path = Path(__file__).resolve().parent.parent / "src" / "data" / "neutral-lut.json"
with open(out_path, "w") as f:
    json.dump(lut, f, separators=(",", ":"))

print(f"Wrote {len(lut['L'])} samples to {out_path}")
print(f"L range: [{lut['L'][0]:.6f}, {lut['L'][-1]:.6f}]")
print(f"a_err range: [{min(lut['a_err']):.6f}, {max(lut['a_err']):.6f}]")
print(f"b_err range: [{min(lut['b_err']):.6f}, {max(lut['b_err']):.6f}]")
