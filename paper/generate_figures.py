#!/usr/bin/env python
"""Generate figures for the Helmlab paper.

Usage:
    python new-paper/generate_figures.py
    python new-paper/generate_figures.py --only 1 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from helmlab.spaces.analytical import AnalyticalSpace, AnalyticalParams
from helmlab.spaces.registry import get_space
from helmlab.metrics.stress import stress
from helmlab.data.combvd import load_combvd
from helmlab.data.he2022 import load_he2022
from helmlab.data.macadam1974 import load_macadam1974
from helmlab.data.munsell import load_munsell
from helmlab.utils.srgb_convert import sRGB_to_XYZ, XYZ_to_sRGB
from helmlab.utils.conversions import XYZ_to_Lab
from helmlab.metrics.delta_e import delta_e_2000, delta_e_76, delta_e_94, delta_e_cmc

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

HELMLAB_COLOR = "#2563EB"
CIEDE2000_COLOR = "#DC2626"
OKLAB_COLOR = "#059669"
CIELAB_COLOR = "#7C3AED"
CAM16_COLOR = "#D97706"
GRAY_COLOR = "#6B7280"


def get_helmlab():
    """Load Helmlab space with v19-NC + rotation."""
    return AnalyticalSpace(neutral_correction=True, ab_rotate_deg=-28.2)


# ═══════════════════════════════════════════════════════════════════
# Fig 1: STRESS Comparison
# ═══════════════════════════════════════════════════════════════════
def fig1_stress():
    print("  Fig 1: STRESS comparison...")
    combvd = load_combvd()
    X1, X2, DV = combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"]

    space = get_helmlab()
    methods = [
        ("Helmlab (ours)", stress(DV, space.distance(X1, X2)), HELMLAB_COLOR),
        ("CIEDE2000", stress(DV, delta_e_2000(X1, X2)), CIEDE2000_COLOR),
        ("CIE94", stress(DV, delta_e_94(X1, X2)), GRAY_COLOR),
        ("CMC", stress(DV, delta_e_cmc(X1, X2)), GRAY_COLOR),
        ("CAM16-UCS", stress(DV, get_space("cam16ucs").distance(X1, X2)), CAM16_COLOR),
        ("IPT", stress(DV, get_space("ipt").distance(X1, X2)), GRAY_COLOR),
        ("CIE Lab", stress(DV, delta_e_76(X1, X2)), CIELAB_COLOR),
        ("Oklab", stress(DV, get_space("oklab").distance(X1, X2)), OKLAB_COLOR),
    ]
    methods.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(range(len(methods)), [m[1] for m in methods],
                   color=[m[2] for m in methods], height=0.7, edgecolor="white", linewidth=0.5)
    for bar, (_, val, _) in zip(bars, methods):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m[0] for m in methods])
    ax.set_xlabel("STRESS (lower is better)")
    ax.set_title("Color Difference Prediction — COMBVD (3813 pairs)")
    ax.invert_yaxis()
    ax.set_xlim(0, max(m[1] for m in methods) + 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig1_stress")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: DV vs DE Scatter
# ═══════════════════════════════════════════════════════════════════
def fig2_scatter():
    print("  Fig 2: DV vs DE scatter...")
    combvd = load_combvd()
    X1, X2, DV = combvd["XYZ_1"], combvd["XYZ_2"], combvd["DV"]

    space = get_helmlab()
    DE_h = space.distance(X1, X2)
    DE_c = delta_e_2000(X1, X2)

    def scale(dv, de): return (np.sum(dv * de) / np.sum(de**2)) * de

    DE_hs, DE_cs = scale(DV, DE_h), scale(DV, DE_c)
    lim = max(DV.max(), DE_hs.max(), DE_cs.max()) * 1.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, DE_s, color, name, de_raw in [
        (ax1, DE_hs, HELMLAB_COLOR, "Helmlab", DE_h),
        (ax2, DE_cs, CIEDE2000_COLOR, "CIEDE2000", DE_c),
    ]:
        ax.scatter(DV, DE_s, s=3, alpha=0.15, c=color, rasterized=True)
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Visual Difference (DV)")
        ax.set_ylabel("Predicted (scaled ΔE)")
        ax.set_title(f"{name} — STRESS = {stress(DV, de_raw):.1f}")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal")
    _save(fig, "fig2_scatter")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Neutral Ramp + Achromatic Chroma
# ═══════════════════════════════════════════════════════════════════
def fig3_neutral():
    print("  Fig 3: Neutral ramp...")
    N = 21
    srgb_vals = np.linspace(0.0, 1.0, N)
    space_h = get_helmlab()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Ramp uniformity
    for label, get_dists, color in [
        ("CIE Lab", lambda: _cielab_dists(srgb_vals), CIELAB_COLOR),
        ("Oklab", lambda: _space_dists(get_space("oklab"), srgb_vals), OKLAB_COLOR),
        ("Helmlab", lambda: _space_dists(space_h, srgb_vals), HELMLAB_COLOR),
    ]:
        dists = get_dists()
        if np.mean(dists) > 1e-12:
            dists_n = dists / np.mean(dists)
            cv = np.std(dists) / np.mean(dists) * 100
        else:
            dists_n, cv = dists, 0
        mid = (srgb_vals[:-1] + srgb_vals[1:]) / 2
        ax1.plot(mid, dists_n, "o-", color=color, label=f"{label} (CV={cv:.0f}%)",
                 markersize=3, linewidth=1.5)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("sRGB gray value"); ax1.set_ylabel("Normalized step size")
    ax1.set_title("Neutral Ramp Uniformity"); ax1.legend(fontsize=8)
    ax1.set_ylim(0, 3.5)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Achromatic chroma
    grays = np.linspace(0.05, 0.95, 19)
    gray_xyz = np.array([sRGB_to_XYZ(np.array([v, v, v])) for v in grays])
    for label, sp, color in [
        ("CIE Lab", None, CIELAB_COLOR),
        ("Oklab", get_space("oklab"), OKLAB_COLOR),
        ("CAM16-UCS", get_space("cam16ucs"), CAM16_COLOR),
        ("Helmlab", space_h, HELMLAB_COLOR),
    ]:
        if sp is None:
            labs = XYZ_to_Lab(gray_xyz)
        else:
            labs = sp.from_XYZ(gray_xyz)
        C = np.sqrt(labs[:, 1]**2 + labs[:, 2]**2)
        ax2.plot(grays, C, "o-", color=color, label=label, markersize=3, linewidth=1.5)
    ax2.set_xlabel("sRGB gray value"); ax2.set_ylabel("Chroma (C)")
    ax2.set_title("Achromatic Chroma Leakage"); ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    _save(fig, "fig3_neutral")


def _cielab_dists(srgb_vals):
    labs = []
    for v in srgb_vals:
        if v < 1e-10: labs.append(np.array([0.0, 0.0, 0.0]))
        else: labs.append(XYZ_to_Lab(sRGB_to_XYZ(np.array([v, v, v])).reshape(1, 3))[0])
    return np.sqrt(np.sum(np.diff(np.array(labs), axis=0)**2, axis=1))


def _space_dists(space, srgb_vals):
    dists = []
    for i in range(1, len(srgb_vals)):
        x0 = sRGB_to_XYZ(np.array([srgb_vals[i-1]]*3)).reshape(1, 3) if srgb_vals[i-1] > 1e-10 else np.array([[0.,0.,0.]])
        x1 = sRGB_to_XYZ(np.array([srgb_vals[i]]*3)).reshape(1, 3) if srgb_vals[i] > 1e-10 else np.array([[0.,0.,0.]])
        dists.append(space.distance(x0, x1)[0])
    return np.array(dists)


# ═══════════════════════════════════════════════════════════════════
# Fig 4: Pipeline Diagram
# ═══════════════════════════════════════════════════════════════════
def fig4_pipeline():
    print("  Fig 4: Pipeline diagram...")
    row1 = [
        ("XYZ", "#F3F4F6", "Input"),
        ("M₁", "#DBEAFE", "9 params"),
        ("γᵢ", "#DBEAFE", "3 params"),
        ("M₂", "#DBEAFE", "9 params"),
        ("Hue corr.", "#FEF3C7", "8 params"),
        ("H-K", "#FCE7F3", "6 params"),
        ("L corr.", "#FEF3C7", "8 params"),
    ]
    row2 = [
        ("C proc.", "#FEF3C7", "18 params"),
        ("Hue-L", "#FEF3C7", "4 params"),
        ("NC", "#D1FAE5", "Neutral corr."),
        ("Rot φ", "#D1FAE5", "−28.2°"),
        ("Lab", "#DCFCE7", "Output"),
    ]
    spacing = 2.0
    bw, bh = 1.6, 1.0
    y1, y2 = 1.8, -0.8  # row centers

    fig, ax = plt.subplots(figsize=(14, 4.5))
    max_cols = max(len(row1), len(row2))
    ax.set_xlim(-1, max_cols * spacing + 0.5)
    ax.set_ylim(-2.5, 3.5)
    ax.axis("off")

    def draw_row(stages, y, start_x=0):
        for i, (name, color, desc) in enumerate(stages):
            x = start_x + i * spacing
            box = FancyBboxPatch((x - bw/2, y - bh/2), bw, bh,
                                 boxstyle="round,pad=0.1", facecolor=color,
                                 edgecolor="#374151", linewidth=1.2)
            ax.add_patch(box)
            ax.text(x, y + 0.05, name, ha="center", va="center", fontsize=9, fontweight="bold")
            ax.text(x, y - bh/2 - 0.15, desc, ha="center", va="top", fontsize=7, color="#6B7280")
            if i < len(stages) - 1:
                ax.annotate("", xy=((start_x + (i+1)*spacing) - bw/2, y),
                            xytext=(x + bw/2, y),
                            arrowprops=dict(arrowstyle="->", color="#374151", linewidth=1.5))

    draw_row(row1, y1)
    draw_row(row2, y2)

    # Connect row1 end → row2 start: straight down from H-K, then left to L corr.
    r1_end_x = (len(row1) - 1) * spacing  # H-K x position
    r2_start_x = 0  # L corr. x position
    mid_y = (y1 - bh/2 + y2 + bh/2) / 2
    # Draw L-shaped path: down from H-K, then left to L corr.
    ax.plot([r1_end_x, r1_end_x], [y1 - bh/2, mid_y], color="#374151", linewidth=1.5)
    ax.plot([r1_end_x, r2_start_x], [mid_y, mid_y], color="#374151", linewidth=1.5)
    ax.annotate("", xy=(r2_start_x, y2 + bh/2), xytext=(r2_start_x, mid_y),
                arrowprops=dict(arrowstyle="->", color="#374151", linewidth=1.5))

    ax.set_title("Helmlab Forward Transform (72 parameters + NC + rotation)", fontsize=12, pad=15)
    _save(fig, "fig4_pipeline")


# ═══════════════════════════════════════════════════════════════════
# Fig 5: Cross-Validation
# ═══════════════════════════════════════════════════════════════════
def fig5_crossval():
    print("  Fig 5: Cross-validation...")
    he, mac = load_he2022(), load_macadam1974()
    space_h = get_helmlab()

    spaces = [
        ("Helmlab", space_h, HELMLAB_COLOR),
        ("CAM16-UCS", get_space("cam16ucs"), CAM16_COLOR),
        ("Oklab", get_space("oklab"), OKLAB_COLOR),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, data, title in [(ax1, he, f"He et al. 2022 ({len(he['DV'])} pairs)"),
                             (ax2, mac, f"MacAdam 1974 ({len(mac['DV'])} pairs)")]:
        labels, scores, colors = [], [], []
        for name, sp, col in spaces:
            s = stress(data["DV"], sp.distance(data["XYZ_1"], data["XYZ_2"]))
            labels.append(name); scores.append(s); colors.append(col)
        s_c = stress(data["DV"], delta_e_2000(data["XYZ_1"], data["XYZ_2"]))
        labels.append("CIEDE2000"); scores.append(s_c); colors.append(CIEDE2000_COLOR)

        order = np.argsort(scores)
        ax.barh([labels[i] for i in order], [scores[i] for i in order],
                color=[colors[i] for i in order], height=0.6)
        for i in order:
            ax.text(scores[i] + 0.3, labels[i], f"{scores[i]:.1f}", va="center", fontsize=8)
        ax.set_xlabel("STRESS"); ax.set_title(title)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _save(fig, "fig5_crossval")


# ═══════════════════════════════════════════════════════════════════
# Fig 6: Gamut Boundary
# ═══════════════════════════════════════════════════════════════════
def fig6_gamut():
    print("  Fig 6: Gamut boundary...")
    space = get_helmlab()
    N = 64
    vals = np.linspace(0, 1, N)
    R, G, B = np.meshgrid(vals, vals, vals, indexing="ij")
    rgb_all = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=-1)
    xyz_all = np.array([sRGB_to_XYZ(rgb) for rgb in rgb_all])
    lab_all = space.from_XYZ(xyz_all)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, L_target in zip(axes, [0.25, 0.50, 0.75]):
        mask = np.abs(lab_all[:, 0] - L_target) < 0.03
        if mask.any():
            ax.scatter(lab_all[mask, 1], lab_all[mask, 2], c=rgb_all[mask],
                       s=1.5, alpha=0.7, rasterized=True)
        ax.set_xlabel("a"); ax.set_ylabel("b")
        ax.set_title(f"L = {L_target:.2f}"); ax.set_aspect("equal")
        ax.axhline(0, color="gray", linewidth=0.3, alpha=0.3)
        ax.axvline(0, color="gray", linewidth=0.3, alpha=0.3)
    fig.suptitle("sRGB Gamut in Helmlab Space", fontsize=11)
    _save(fig, "fig6_gamut")


# ═══════════════════════════════════════════════════════════════════
def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"    → {name}.pdf / .png")


ALL_FIGS = {
    1: ("STRESS Comparison", fig1_stress),
    2: ("DV vs DE Scatter", fig2_scatter),
    3: ("Neutral Ramp + Achromatic", fig3_neutral),
    4: ("Pipeline Diagram", fig4_pipeline),
    5: ("Cross-Validation", fig5_crossval),
    6: ("Gamut Boundary", fig6_gamut),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, nargs="+")
    args = parser.parse_args()
    figs = args.only or list(ALL_FIGS.keys())
    print(f"Generating {len(figs)} figures → {OUT}/\n")
    for n in figs:
        if n in ALL_FIGS:
            print(f"Figure {n}: {ALL_FIGS[n][0]}")
            try: ALL_FIGS[n][1]()
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback; traceback.print_exc()
            print()
    print("Done!")
