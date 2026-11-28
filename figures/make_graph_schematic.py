"""Draw the relational graph schematic (figures/graph_schematic.png). PTDF values are illustrative."""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#1f2933", "#6b7785"
CORE_FILL, CORE_EDGE = "#dbe8f6", "#2b6cb0"
OTHER_FILL, OTHER_EDGE = "#eef0f2", "#8a96a3"
CON_FILL, CON_EDGE = "#fde7d2", "#c05621"
POS, NEG = "#2b6cb0", "#c53030"
LINK = "#4a5568"

zones = {  # name: (x, y, in Core)
    "ES": (0.35, 3.4, False), "CH": (0.35, 4.45, False), "IT_NORD": (0.45, 5.45, False),
    "FR": (1.9, 4.0, True), "BE": (2.9, 5.1, True), "NL": (4.0, 5.5, True),
    "DE_LU": (4.75, 4.35, True), "PL": (6.35, 4.7, True), "CZ": (5.75, 3.45, True),
}
links = [("ES", "FR"), ("FR", "BE"), ("BE", "NL"), ("NL", "DE_LU"), ("BE", "DE_LU"), ("FR", "DE_LU"),
         ("DE_LU", "PL"), ("DE_LU", "CZ"), ("CZ", "PL"), ("FR", "CH"), ("CH", "IT_NORD")]
cons = {"line A": (2.75, 1.15, 0.13), "line B": (4.45, 1.15, 0.05), "line C": (6.1, 1.15, 0.60)}
ptdf = {  # constraint: {zone: PTDF}
    "line A": {"FR": 0.15, "BE": 0.12, "NL": 0.04, "DE_LU": -0.06},
    "line B": {"DE_LU": 0.32, "NL": 0.08, "PL": -0.21, "CZ": 0.10},
    "line C": {"PL": 0.18, "CZ": -0.14, "DE_LU": 0.11},
}
labelled = {("line A", "FR"), ("line B", "DE_LU"), ("line B", "PL"), ("line C", "CZ")}

fig = plt.figure(figsize=(13.5, 6.6), dpi=200)
ax = fig.add_axes([0.0, 0.0, 0.70, 1.0]); ax.set_xlim(-0.3, 7.0); ax.set_ylim(0.0, 6.6); ax.axis("off")

# layer labels
ax.text(-0.2, 6.25, "Zone nodes (45 bidding zones)", fontsize=12, weight="bold", color=INK)
ax.text(-0.2, 5.95, "price, load forecast, wind/solar, weather histories", fontsize=9.5, color=MUTED)
ax.text(-0.2, 1.95, "Constraint nodes", fontsize=12, weight="bold", color=INK)
ax.text(-0.2, 1.62, "presolved Core elements,\n~120 per hour, rebuilt daily", fontsize=9.5, color=MUTED, va="top",
        linespacing=1.3)
ax.text(-0.2, 1.0, "features: RAM/Fmax,\npresolved flag", fontsize=9.5, color=MUTED, va="top", linespacing=1.3)
ax.axhline(2.45, xmin=0.02, xmax=0.98, color="#d5dbe1", lw=1, ls=(0, (4, 4)))

# zone-constraint edges (signed PTDF), drawn first so nodes sit on top
for c, zs in ptdf.items():
    cx, cy, _ = cons[c]
    for z, v in zs.items():
        zx, zy, _ = zones[z]
        ax.plot([cx, zx], [cy + 0.28, zy - 0.27], color=POS if v > 0 else NEG, lw=0.8 + 9 * abs(v), alpha=0.55,
                solid_capstyle="round", zorder=1)
        if (c, z) in labelled:
            t = 0.42
            ax.text(cx + t * (zx - cx), cy + 0.28 + t * (zy - 0.27 - cy - 0.28), f"{v:+.2f}", fontsize=9,
                    color=POS if v > 0 else NEG, weight="bold", ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# zone-zone interconnectors
for a, b in links:
    (xa, ya, _), (xb, yb, _) = zones[a], zones[b]
    ax.plot([xa, xb], [ya, yb], color=LINK, lw=2.6, zorder=2, solid_capstyle="round")

for z, (x, y, core) in zones.items():
    w = 0.42 + 0.075 * len(z)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.25), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc=CORE_FILL if core else OTHER_FILL, ec=CORE_EDGE if core else OTHER_EDGE,
                                lw=1.6, zorder=3))
    ax.text(x, y, z, ha="center", va="center", fontsize=10, weight="bold", color=INK, zorder=4)

for c, (x, y, ram) in cons.items():
    ax.add_patch(FancyBboxPatch((x - 0.62, y - 0.28), 1.24, 0.56, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=CON_FILL, ec=CON_EDGE, lw=1.6, zorder=3))
    ax.text(x, y + 0.08, c, ha="center", va="center", fontsize=10, weight="bold", color=INK, zorder=4)
    ax.text(x, y - 0.13, f"RAM/Fmax {ram:.2f}", ha="center", va="center", fontsize=8.5, color=MUTED, zorder=4)
ax.text(6.95, 1.15, "…", fontsize=18, color=MUTED, ha="center", va="center")
ax.text(0.4, 2.85, "non-Core zones:\ninterconnectors only", fontsize=8.5, color=MUTED, ha="center", va="center",
        style="italic", linespacing=1.25)

handles = [
    Line2D([], [], color=LINK, lw=2.6, label="interconnects  (zone–zone): lagged flows, MaxBex"),
    Line2D([], [], color=POS, lw=3, alpha=0.6, label="affects / affects_rev  (zone–constraint): PTDF > 0"),
    Line2D([], [], color=NEG, lw=3, alpha=0.6, label="PTDF < 0 (zone export relieves the line); width ∝ |PTDF|"),
    FancyBboxPatch((0, 0), 1, 1, fc=CORE_FILL, ec=CORE_EDGE, label="Core zone"),
    FancyBboxPatch((0, 0), 1, 1, fc=OTHER_FILL, ec=OTHER_EDGE, label="non-Core zone"),
]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.005), fontsize=8.6, frameon=False, ncol=2,
          handlelength=2.2, columnspacing=1.4)

# pipeline panel
px = fig.add_axes([0.71, 0.0, 0.29, 1.0]); px.set_xlim(0, 1); px.set_ylim(0, 1); px.axis("off")
px.text(0.5, 0.94, "One graph per delivery day", ha="center", fontsize=12, weight="bold", color=INK)
steps = [
    ("Node histories", "zone: price, load, weather lags\nconstraint: RAM/Fmax, presolved"),
    ("GRU encoders", "one shared GRU per node type"),
    ("Relational GATv2 × L", "one attention module per relation\nedge features in the attention scores"),
    ("MLP head", "24 hourly prices per zone\n(asinh residual vs. last week)"),
]
ys = [0.79, 0.60, 0.41, 0.22]
for (title, sub), y in zip(steps, ys):
    px.add_patch(FancyBboxPatch((0.08, y - 0.07), 0.84, 0.14, boxstyle="round,pad=0.01,rounding_size=0.025",
                                fc="#f5f7fa", ec="#9aa5b1", lw=1.3))
    px.text(0.5, y + 0.03, title, ha="center", va="center", fontsize=11, weight="bold", color=INK)
    px.text(0.5, y - 0.03, sub, ha="center", va="center", fontsize=8.6, color=MUTED, linespacing=1.25)
for y0, y1 in zip(ys[:-1], ys[1:]):
    px.add_patch(FancyArrowPatch((0.5, y0 - 0.075), (0.5, y1 + 0.075), arrowstyle="-|>", mutation_scale=14,
                                 color="#52606d", lw=1.4))
px.text(0.5, 0.075, "Messages run zone → constraint → zone,\nso zones sharing a tight line interact\n"
        "even without a common border.", ha="center", va="center", fontsize=9, color=INK, style="italic",
        linespacing=1.3)
px.text(0.5, 0.012, "PTDF and RAM values are illustrative", ha="center", fontsize=7.5, color=MUTED)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_schematic.png")
fig.savefig(out, dpi=200, facecolor="white")
print(out)
