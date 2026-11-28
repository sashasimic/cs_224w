"""Schematics for OVERVIEW.md: the data pipeline (m1), the GATv2 attention-scope model (m2) and the H2
constraint arm (m3). Shapes and sizes are the real ones; PTDF values in m3 are illustrative.

    python figures/make_model_schematics.py            # all three, into figures/data/
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "data")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_150, BLUE_550 = "#dbe8f9", "#1c5cab"
ORANGE_150, AQUA_150 = "#fde3d6", "#d4f1e5"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK})


def box(ax, x, y, w, h, title, body="", fill=NEUTRAL, edge=AXIS, tsize=10.5, bsize=8.5, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec=edge, lw=1.2))
    tx = x + w / 2 if align == "center" else x + 0.15
    ha = "center" if align == "center" else "left"
    if body:
        ax.text(tx, y + h - 0.22, title, ha=ha, va="top", fontsize=tsize, weight="bold", color=INK)
        ax.text(tx, y + h - 0.55, body, ha=ha, va="top", fontsize=bsize, color=INK2, linespacing=1.35)
    else:
        ax.text(tx, y + h / 2, title, ha=ha, va="center", fontsize=tsize, weight="bold", color=INK)


def arrow(ax, x0, y0, x1, y1, color=INK2, text="", lw=1.4, rad=0.0, tsize=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((x0 + x1) / 2 + 0.08, (y0 + y1) / 2, text, fontsize=tsize, color=MUTED, va="center")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}", flush=True)


# ---- m1: data pipeline ---------------------------------------------------------------------------------

def m1_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.text(0.1, 8.4, "Data pipeline: raw sources to scored forecasts", fontsize=14, weight="bold", va="top")
    ax.text(0.1, 8.0, "Cutoff for every input of delivery day D: D-1 10:00 UTC (before the SDAC gate closure)",
            fontsize=9.5, color=MUTED, va="top")

    # stage 0: sources
    ax.text(0.1, 7.45, "STAGE 0  raw sources (data/)", fontsize=9, weight="bold", color=MUTED)
    src = [("ENTSO-E v1.2", "prices (62 areas; 15-min\naveraged from Oct 2025),\nload forecast, physical flows"),
           ("ERA5 / PECD weather", "per bidding zone, 2010-2026:\ntemp, anomaly, wind CF,\nsolar CF (observed)"),
           ("JAO Core flow-based", "9 Jun 2022 - 29 Oct 2026:\npresolved CNECs, RAM, Fmax,\nAMR, 14 hub PTDFs, MaxBex")]
    for i, (t, b) in enumerate(src):
        box(ax, 0.1, 5.35 - i * 2.0 + 0.0, 3.0, 1.75, t, b, fill=ORANGE_150, edge=ORANGE)

    # stage 1
    ax.text(3.6, 7.45, "STAGE 1  build_model_dataset.py", fontsize=9, weight="bold", color=MUTED)
    box(ax, 3.6, 4.5, 3.0, 2.6, "zone_nodes_hourly",
        "4.24 M rows = 45 zones x\n94,223 UTC hours (2016-01-01\nto 2026-09-30)\nprice + mask, price lags,\nload_fc, weather, net_import,\nCore JAO summaries, split")
    box(ax, 3.6, 1.65, 3.0, 2.5, "zone_edges_hourly",
        "17.0 M rows = 180 directed\npairs x 94,223 hours\nflow_phys (+ lags), flow_sched,\nMaxBex (Core borders)")
    arrow(ax, 3.1, 6.4, 3.6, 6.4)                     # ENTSO-E -> nodes
    arrow(ax, 3.1, 5.6, 3.6, 3.6)                     # ENTSO-E flows -> edges
    arrow(ax, 3.1, 4.2, 3.6, 5.2)                     # weather -> nodes
    arrow(ax, 3.1, 2.0, 3.6, 2.5)                     # JAO MaxBex -> edges

    # stage 2
    ax.text(7.1, 7.45, "STAGE 2  python -m epgnn.pipeline.build", fontsize=9, weight="bold", color=MUTED)
    box(ax, 7.1, 5.0, 3.2, 2.1, "features/v1/zone_features",
        "4.24 M x 57: cyclic time, D-1\nprice stats, load shape, weather\n_asof (cutoff-safe), heat stress,\nper-zone JAO exposure (_lag24)")
    box(ax, 7.1, 2.75, 3.2, 1.95, "constraints_hourly",
        "4.27 M rows, ~110 per hour,\n6,319 constraints; synthetic\nrows dropped; zone-to-rest\nPTDF; policy-RAM features")
    box(ax, 7.1, 0.55, 3.2, 1.9, "graph.json",
        "45 zones, 174 directed edges\n(87 links active since Oct\n2018), 87 spread pairs,\n8-dim Laplacian PE")
    arrow(ax, 6.6, 5.8, 7.1, 6.0)
    arrow(ax, 3.1, 1.6, 7.1, 3.2, rad=0.25)            # JAO CNECs -> constraints
    arrow(ax, 6.6, 2.9, 7.1, 1.5)

    # stage 3-4
    ax.text(10.8, 7.45, "STAGES 3-4  tensors and samples", fontsize=9, weight="bold", color=MUTED)
    box(ax, 10.8, 4.75, 3.1, 2.35, "per delivery day, per zone",
        "history (168 h x 8) to cutoff\nday inputs (25 slots x 86):\n  14 own + 72 D-1/D-2/D-7 curve\ntarget asinh(p/50) - asinh(base/50)\n  base = price D-7")
    box(ax, 10.8, 2.55, 3.1, 1.85, "model (sections 5-6)",
        "fit on pretrain + train\nval for early stopping\noptional monthly fine-tune")
    box(ax, 10.8, 0.55, 3.1, 1.65, "scoring",
        "MAE / RMSE / spread MAE,\nslices; weekly block\nbootstrap of MAE differences", fill=BLUE_150, edge=BLUE)
    arrow(ax, 10.3, 6.05, 10.8, 5.9)
    arrow(ax, 10.3, 3.7, 10.8, 3.5, text="")
    arrow(ax, 10.3, 1.5, 10.8, 3.0)
    arrow(ax, 12.35, 4.75, 12.35, 4.4)
    arrow(ax, 12.35, 2.55, 12.35, 2.2)

    # splits strip
    ax.text(0.1, 0.95, "Splits (by delivery day)", fontsize=9, weight="bold", color=MUTED)
    spans = [("pretrain", "2016-01-01 to 2022-06-08", 2.6), ("train", "to 2024-03-31", 1.4),
             ("val", "to 2025-03-31", 1.3), ("test", "2025-04-01 to 2026-09-30", 1.75)]
    x = 0.1
    cols = [NEUTRAL, BLUE_150, AQUA_150, ORANGE_150]
    for (n, d, w), c in zip(spans, cols):
        ax.add_patch(FancyBboxPatch((x, 0.1), w * 1.02 - 0.05, 0.6, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=c, ec=AXIS, lw=0.8))
        ax.text(x + 0.08, 0.52, n, fontsize=8.5, weight="bold", va="center")
        ax.text(x + 0.08, 0.26, d, fontsize=7, color=INK2, va="center")
        x += w * 1.02
    save(fig, "m1_data_pipeline")


# ---- m2: attention-scope model -------------------------------------------------------------------------

def m2_model():
    fig = plt.figure(figsize=(14, 8.2))
    ax = fig.add_axes([0, 0, 0.66, 1]); ax.set_xlim(0, 9.3); ax.set_ylim(0, 11.5); ax.axis("off")
    ax.text(0.1, 11.3, "GATv2 attention-scope model (gat_scope.py, 100,033 parameters)", fontsize=14,
            weight="bold", va="top")
    ax.text(0.1, 10.85, "One forward pass = B delivery days x 45 zones x 25 hourly slots", fontsize=9.5,
            color=MUTED, va="top")

    # inputs row
    box(ax, 0.1, 8.55, 2.8, 1.9, "History", "168 h to the cutoff, 8 cols:\nprice, load_fc, net_import,\nweather (realised, past)",
        fill=ORANGE_150, edge=ORANGE)
    box(ax, 3.2, 8.55, 2.9, 1.9, "Day inputs (per slot)", "14 own: price D-1/D-7 same\nhour, load_fc, _asof weather,\ncalendar  + 72 curve: D-1, D-2,\nD-7 24-h price curves",
        fill=ORANGE_150, edge=ORANGE)
    box(ax, 6.4, 8.55, 2.8, 1.9, "Positional enc.", "8-dim Laplacian eigvecs\nof the interconnector graph\n(random sign flip in training)",
        fill=ORANGE_150, edge=ORANGE)

    box(ax, 0.1, 6.95, 2.8, 1.1, "GRU 8 -> 64", "shared over zones; last state", fill=NEUTRAL)
    arrow(ax, 1.5, 8.55, 1.5, 8.05)
    ax.text(1.62, 6.62, "copied to 25 slots", fontsize=8, color=MUTED)

    box(ax, 0.1, 5.25, 9.1, 1.05, "concat [64 | 86 | 8]  ->  Linear 158 -> 64", "", fill=NEUTRAL)
    arrow(ax, 1.5, 6.95, 1.5, 6.3)
    arrow(ax, 4.65, 8.55, 4.65, 6.3)
    arrow(ax, 7.8, 8.55, 7.8, 6.3)

    box(ax, 0.1, 4.0, 9.1, 0.85, "node_update hook  (identity; the H2 arm adds constraint messages here, see m3)",
        "", fill=AQUA_150, edge=AQUA, tsize=9.5)
    arrow(ax, 4.65, 5.25, 4.65, 4.85)

    box(ax, 0.1, 1.95, 9.1, 1.65, "2 x pre-LN block on each (day, slot) graph of 45 zones",
        "u += Dropout(GATv2(LN(u)))    4 heads x 16, neighbourhood set by the SCOPE (right)\n"
        "u += Dropout(FFN(LN(u)))       64 -> 128 -> 64, ELU         slots never exchange messages",
        fill=BLUE_150, edge=BLUE, bsize=9)
    arrow(ax, 4.65, 4.0, 4.65, 3.6)

    box(ax, 0.1, 0.1, 5.2, 1.45, "Head  LN -> 128 -> 128 -> 1",
        "r = predicted asinh change from D-7;\nprice = 50 sinh(r + asinh(base/50))", fill=NEUTRAL)
    arrow(ax, 2.7, 1.95, 2.7, 1.55)
    box(ax, 5.6, 0.1, 3.6, 1.45, "Loss",
        "masked L1 on r\n+ 0.5 x L1 on cross-border spreads\n(87 interconnected pairs)", fill=NEUTRAL)

    # scope panel
    sx = fig.add_axes([0.67, 0.05, 0.32, 0.9]); sx.set_xlim(0, 4.6); sx.set_ylim(0, 11.5); sx.axis("off")
    sx.text(0.1, 11.3, "The three scopes (same parameters)", fontsize=12, weight="bold", va="top")
    pos = {"FR": (0.7, 1.2), "BE": (1.5, 2.1), "NL": (2.4, 2.4), "DE": (2.6, 1.3), "CH": (1.4, 0.25),
           "PL": (3.6, 1.6), "ES": (0.0, 0.3)}
    links = [("FR", "BE"), ("BE", "NL"), ("NL", "DE"), ("BE", "DE"), ("FR", "DE"), ("FR", "CH"), ("DE", "CH"),
             ("DE", "PL"), ("ES", "FR")]
    panels = [("self", "a zone attends only to itself\n(GRU+MLP-like; no messages)"),
              ("interconnects", "physical neighbours + self\n(174 directed edges)"),
              ("full", "all 45 zones\n(a Transformer over zones)")]
    for k, (name, desc) in enumerate(panels):
        oy = 7.3 - k * 3.45
        sx.text(0.1, oy + 3.05, name, fontsize=11, weight="bold", color=BLUE_550)
        sx.text(0.1, oy + 2.72, desc.replace("\n", " "), fontsize=8, color=MUTED)
        tgt = "DE"
        if name == "interconnects":
            for a, b in links:
                (x0, y0), (x1, y1) = pos[a], pos[b]
                sx.plot([x0 + .5, x1 + .5], [y0 + oy, y1 + oy], color=AXIS, lw=1.2, zorder=1)
        nb = {"self": [], "interconnects": [b if a == tgt else a for a, b in links if tgt in (a, b)],
              "full": [z for z in pos if z != tgt]}[name]
        for z in nb:
            (x0, y0), (x1, y1) = pos[z], pos[tgt]
            sx.add_patch(FancyArrowPatch((x0 + .5, y0 + oy), (x1 + .5, y1 + oy), arrowstyle="-|>", mutation_scale=9,
                                         color=BLUE, lw=1.4, shrinkA=9, shrinkB=11, zorder=2))
        if name == "self":
            sx.add_patch(plt.Circle((pos[tgt][0] + .5, pos[tgt][1] + oy + 0.42), 0.2, fc="none", ec=BLUE, lw=1.4,
                                    zorder=2))
        for z, (x, y) in pos.items():
            hot = z == tgt
            sx.add_patch(plt.Circle((x + .5, y + oy), 0.27, fc=BLUE if hot else SURFACE, ec=BLUE_550 if hot else INK2,
                                    lw=1.2, zorder=3))
            sx.text(x + .5, y + oy, z, ha="center", va="center", fontsize=7, color=SURFACE if hot else INK, zorder=4,
                    weight="bold")
    sx.text(0.1, 0.0, "arrows: messages into DE in one GATv2 layer", fontsize=8, color=MUTED)
    save(fig, "m2_gat_scope_model")


# ---- m3: constraint arm --------------------------------------------------------------------------------

def m3_constraints():
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.text(0.1, 7.25, "H2 constraint arm (gat_constraints.py, +11,168 parameters)", fontsize=14, weight="bold",
            va="top")
    ax.text(0.1, 6.85, "For each slot hour t of day D: the Core constraints presolved at t - 24 h (D-1 domain, "
            "published on D-2; about 110, at most 182)", fontsize=9.5, color=MUTED, va="top")

    cons = [("line A", "RAM/Fmax 0.13", 1.0), ("line B", "RAM/Fmax 0.05", 2.9), ("line C", "RAM/Fmax 0.60", 4.8)]
    zones = {"FR": (1.2, 4.6), "DE_LU": (3.3, 5.3), "PL": (5.4, 4.6)}
    edges = [("line A", "FR", 0.15), ("line A", "DE_LU", -0.06), ("line B", "DE_LU", 0.32), ("line B", "PL", -0.21),
             ("line C", "PL", 0.18), ("line C", "DE_LU", 0.11)]
    cx = {n: x for n, _, x in cons}
    for c, z, p in edges:
        x0, y0 = cx[c] + 0.65, 1.95
        x1, y1 = zones[z][0] + 0.45, zones[z][1] - 0.05
        ax.plot([x0, x1], [y0, y1], color=BLUE if p > 0 else ORANGE, lw=1 + 8 * abs(p), alpha=0.85, zorder=1)
        ax.text(x0 + (x1 - x0) * 0.35 + 0.05, y0 + (y1 - y0) * 0.35, f"{p:+.2f}", fontsize=8,
                color=BLUE_550 if p > 0 else ORANGE, weight="bold")
    ax.plot([zones["FR"][0] + .9, zones["DE_LU"][0]], [4.85, 5.5], color=INK2, lw=2.2, zorder=1)
    ax.plot([zones["DE_LU"][0] + .9, zones["PL"][0]], [5.5, 4.85], color=INK2, lw=2.2, zorder=1)
    for z, (x, y) in zones.items():
        box(ax, x, y - 0.05, 0.9 if z != "DE_LU" else 0.9, 0.55, z, "", fill=BLUE_150, edge=BLUE, tsize=9)
    for n, sub, x in cons:
        box(ax, x, 1.15, 1.3, 0.8, n, sub, fill=ORANGE_150, edge=ORANGE, tsize=9, bsize=7.5)
    ax.text(0.1, 0.65, "edge: |zone-to-rest PTDF| >= 0.05 (slack-invariant);  blue = PTDF > 0, orange = PTDF < 0;  "
            "grey: interconnectors", fontsize=8, color=MUTED)
    ax.text(0.1, 0.3, "Only the 12 Core zones receive messages; non-Core zones, pretrain days and missing domains "
            "get none (gate g = 0).", fontsize=8, color=MUTED)
    ax.text(0.1, 3.4, "Core zones\n(state u_z after\nthe input layer)", fontsize=8.5, color=INK2, va="center")

    # equations
    bx = 7.0
    box(ax, bx, 4.6, 6.9, 1.75, "Constraint encoder and attention (GATv2-style, 4 heads x 16)",
        "x_c  = 8 scale-free features: RAM/Fmax, (RAM-AMR)/Fmax, AMR/Fmax,\n"
        "         min-RAM factor, FRM, F0core, F_LTN over Fmax, log Fmax\n"
        "k_c  = MLP(x_c)\n"
        "s_zc = a . LeakyReLU(Wq u_z + Wk k_c + w_p p_zc + w_a |p_zc|)",
        fill=AQUA_150, edge=AQUA, align="left", bsize=9)
    box(ax, bx, 2.65, 6.9, 1.45, "Message and update (before the zone GAT blocks)",
        "alpha_zc = softmax_c(s_zc)   over the constraints z is exposed to\n"
        "m_z = sum_c alpha_zc (Wv k_c + v_p p_zc)      the signed PTDF enters\n"
        "u_z <- u_z + Dropout(Linear(m_z)) * g_z",
        fill=AQUA_150, edge=AQUA, align="left", bsize=9)
    box(ax, bx, 0.95, 6.9, 1.05, "Then the 2 interconnector GATv2 blocks spread it to neighbours",
        "so non-Core zones see the domain indirectly", fill=BLUE_150, edge=BLUE, align="left", bsize=9)
    save(fig, "m3_constraint_arm")


FIGS = {"m1": m1_pipeline, "m2": m2_model, "m3": m3_constraints}

if __name__ == "__main__":
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
