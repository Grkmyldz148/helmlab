"""Generate fig7_genspace.pdf — GenSpace vs OKLab category breakdown.

Visualization of Table 6 in §7. Stacked horizontal bars per category showing
GenSpace wins / ties / OKLab wins, ordered by total category size.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from ColorBench v0.11.1 colorjs_pr (CJS canonical) — measured 2026-05-06
# (category, gen_wins, oklab_wins, ties)
# Total: 65W / 9L / 16T across 90 metrics.
DATA = [
    ("Gamut geometry",       24, 0, 3),
    ("Application",           9, 0, 3),
    ("Gradient quality",      7, 3, 1),
    ("Independent",           6, 1, 0),
    ("Perceptual",            5, 0, 0),
    ("Structural",            4, 2, 2),
    ("Hue",                   2, 0, 0),
    ("Achromatic",            2, 0, 0),
    ("Advanced",              2, 0, 4),
    ("Special",               2, 1, 0),
    ("Banding",               1, 0, 1),
    ("Accessibility",         1, 1, 0),
    ("Numerical stability",   0, 1, 2),
]

# Sort by total metrics in category (descending) — biggest categories first
DATA = sorted(DATA, key=lambda r: r[1] + r[2] + r[3], reverse=True)

cats = [d[0] for d in DATA]
gen = np.array([d[1] for d in DATA])
ok = np.array([d[2] for d in DATA])
tie = np.array([d[3] for d in DATA])
totals = gen + ok + tie

fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=200)
y = np.arange(len(cats))

# Colors: GenSpace wins green, ties grey, OKLab wins red
GREEN = "#2e8b57"   # sea green — GenSpace
GREY = "#9e9e9e"
RED = "#c0392b"     # OKLab

ax.barh(y, gen, color=GREEN, label="GenSpace wins", edgecolor="white", linewidth=0.5)
ax.barh(y, tie, left=gen, color=GREY, label="Tie", edgecolor="white", linewidth=0.5)
ax.barh(y, ok, left=gen + tie, color=RED, label="OKLab wins", edgecolor="white", linewidth=0.5)

# Numeric labels inside bars when wide enough
for i, (g, t, o) in enumerate(zip(gen, tie, ok)):
    x = 0
    if g > 0:
        ax.text(x + g / 2, i, str(int(g)), va="center", ha="center",
                color="white", fontsize=9, fontweight="bold")
    x += g
    if t > 0 and t >= 1.5:
        ax.text(x + t / 2, i, str(int(t)), va="center", ha="center",
                color="white", fontsize=9)
    x += t
    if o > 0:
        ax.text(x + o / 2, i, str(int(o)), va="center", ha="center",
                color="white", fontsize=9, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(cats, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Number of metrics", fontsize=10)
ax.set_xlim(0, max(totals) + 2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.25, linestyle=":")
ax.set_axisbelow(True)

# Legend
ax.legend(loc="lower right", fontsize=9, frameon=False, ncol=3,
          bbox_to_anchor=(1.0, -0.18))

# Title-like header summarizing the totals.
# Per-category breakdown sums to 60W/8L/15T across 83 internal metrics;
# including 7 independent gradient/palette metrics the overall ColorBench
# tally is 66W/9L/15T across 90 metrics (Table tab:genspace-vs-oklab).
fig.suptitle(
    "GenSpace v0.11.1 vs OKLab on ColorBench  ·  "
    "65 wins / 9 losses / 16 ties across 90 metrics",
    fontsize=10, fontweight="bold", y=0.99,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_pdf = "/Volumes/harici_ssd/color-space/helmlab-main-repo/paper/figures/fig7_genspace.pdf"
out_png = "/Volumes/harici_ssd/color-space/helmlab-main-repo/paper/figures/fig7_genspace.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")
print(f"Totals: GenSpace {int(gen.sum())}W / OKLab {int(ok.sum())}L / {int(tie.sum())} ties (90 metrics)")
