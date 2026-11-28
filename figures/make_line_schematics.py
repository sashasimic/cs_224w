"""Black-and-white line schematics of every model built or tested, one figure each, for slides and the report.

    python figures/make_line_schematics.py        # -> figures/line/L*.{png,pdf,svg} and figures/line/all_models.pdf

L1 LEAR                     one LASSO per zone and target hour (the EPF benchmark)
L2 XGBoost c1 / c2          per-zone gradient boosting; c2 adds neighbour means (dashed)
L3 GRU + MLP (c2.5)         sequence encoder, no graph
L4 GATv2 attention scope    GRU + 2 GATv2 blocks; self / interconnects / full neighbourhoods
L5 H2 constraint arm        + constraint nodes with signed-PTDF attention before the zone blocks
L6 H2 JAO-summary arm       + six per-zone summaries of the D-1 flow-based domain as day inputs
L7 Training protocol        cutoff rule, splits, fit once vs. monthly rolling fine-tuning

Sizes are the real ones from the code (epgnn/models/*.py); PTDF values in L5 are illustrative.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "line")
INK, GREY = "#111111", "#6b6b6b"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
                     "figure.facecolor": "white", "savefig.facecolor": "white", "text.color": INK,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


# ---- drawing helpers (data units = inches) -------------------------------------------------------------

def canvas(w: float, h: float, title: str, sub: str = ""):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(0.35, h - 0.35, title, ha="left", va="top", fontsize=15, weight="bold")
    if sub:
        ax.text(0.35, h - 0.72, sub, ha="left", va="top", fontsize=9.5, color=GREY)
    return fig, ax


def box(ax, x, y, w, h, title, body="", ls="-", lw=1.1, tsize=10, bsize=8.3, mono_body=False, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc="white", ec=INK,
                                lw=lw, ls=ls))
    if body:
        ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=tsize,
                weight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h - 0.40, body, ha="center", va="top", fontsize=bsize, color=INK, linespacing=1.3,
                family="monospace" if mono_body else None)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize,
                weight="bold" if bold else "normal")
    return (x, y, w, h)


def arrow(ax, p0, p1, label="", ls="-", lw=1.1, rad=0.0, lside="right", lsize=7.8, color=INK):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        if abs(p1[1] - p0[1]) < 0.15:                                    # horizontal: label above
            ax.text(mx, my + 0.07, label, ha="center", va="bottom", fontsize=lsize, color=GREY, family="monospace")
            return
        dx = 0.08 if lside == "right" else -0.08
        ax.text(mx + dx, my, label, ha="left" if lside == "right" else "right", va="center", fontsize=lsize,
                color=GREY, family="monospace")


def line(ax, p0, p1, ls="-", lw=1.0, color=INK):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], ls=ls, lw=lw, color=color, solid_capstyle="round")


def node(ax, x, y, label, r=0.27, lw=1.1, ls="-", fs=7.5, double=False):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK, lw=lw, ls=ls, zorder=3))
    if double:
        ax.add_patch(Circle((x, y), r - 0.05, fc="white", ec=INK, lw=0.8, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)


def note(ax, x, y, text, size=8, ha="left", color=GREY, va="top", **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, linespacing=1.35, **kw)


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def save(fig, name: str, pages: list):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200 if ext == "png" else None)
    pages.append(fig)
    print("->", os.path.join("figures", "line", name + ".png"))


# shared zone layout for the little graphs (a western-European excerpt)
POS = {"ES": (0.2, 0.2), "FR": (0.9, 0.75), "CH": (1.55, 0.25), "BE": (1.25, 1.45), "NL": (1.95, 1.9),
       "DE": (2.25, 1.0), "PL": (3.05, 1.25)}
LINKS = [("ES", "FR"), ("FR", "BE"), ("FR", "CH"), ("FR", "DE"), ("BE", "NL"), ("BE", "DE"), ("NL", "DE"),
         ("CH", "DE"), ("DE", "PL")]


def mini_graph(ax, ox, oy, scope, target="DE", s=0.9):
    """One GATv2 layer's messages into `target` under a scope, drawn at origin (ox, oy)."""
    p = {k: (ox + v[0] * s, oy + v[1] * s) for k, v in POS.items()}
    for a, b in LINKS:
        line(ax, p[a], p[b], ls=":", lw=0.8, color=GREY)
    if scope == "interconnects":
        srcs = [a if b == target else b for a, b in LINKS if target in (a, b)]
    elif scope == "full":
        srcs = [k for k in POS if k != target]
    else:
        srcs = []
    for k in srcs:
        arrow(ax, p[k], p[target], lw=1.0)
    for k, (x, y) in p.items():
        node(ax, x, y, k, r=0.2, fs=6.5, double=(k == target))
    tx, ty = p[target]
    ax.add_patch(FancyArrowPatch((tx - 0.12, ty + 0.17), (tx + 0.12, ty + 0.17), arrowstyle="-|>", mutation_scale=7,
                                 connectionstyle="arc3,rad=-1.6", lw=0.9, color=INK))


# ---- L1 LEAR -------------------------------------------------------------------------------------------

def l1_lear(pages):
    fig, ax = canvas(11, 6.2, "L1  LEAR  (Lago et al. 2021)",
                     "Linear, no graph. One LASSO per zone z and target hour h: 45 zones x 24 hours = 1,080 models")
    ins = [("price curve D-1", "24 h"), ("price curve D-2", "24 h"), ("price curve D-3", "24 h"),
           ("price curve D-7", "24 h"), ("load forecast D", "24 h"), ("load forecast D-7", "24 h"),
           ("weekday dummies", "7")]
    y0 = 4.65
    for i, (t, n) in enumerate(ins):
        y = y0 - i * 0.56
        b = box(ax, 0.4, y, 2.4, 0.42, t, bold=False, tsize=9)
        note(ax, 2.9, y + 0.21, n, va="center", family="monospace")
        arrow(ax, (3.35, y + 0.21), (3.75, 2.85 + (3 - i) * 0.12))
    note(ax, 0.4, 0.78, "the zone's own series only:\nno weather, no neighbours, no constraints", size=8)
    s = box(ax, 3.8, 2.1, 2.0, 1.7, "scale", "median / MAD\nper input,\nthen asinh", bsize=8.5)
    arrow(ax, (5.8, 2.95), (6.3, 2.95), "151")
    lasso = box(ax, 6.35, 1.55, 2.35, 2.8, "LASSO  (z, h)", "y_h = b + sum_j w_j x_j\n\npenalty lambda\nchosen by AIC\n(LassoLarsIC)\n\nx 24 hours\nx 45 zones", bsize=8.3)
    for k in range(3):                                                   # stacked copies
        line(ax, (6.35 + 0.08 * (k + 1), 4.35 + 0.08 * (k + 1)), (8.7 + 0.08 * (k + 1), 4.35 + 0.08 * (k + 1)), lw=0.7)
        line(ax, (8.7 + 0.08 * (k + 1), 1.55 + 0.08 * (k + 1)), (8.7 + 0.08 * (k + 1), 4.35 + 0.08 * (k + 1)), lw=0.7)
    arrow(ax, (9.0, 2.95), (9.45, 2.95))
    box(ax, 9.5, 2.45, 1.25, 1.0, "price", "inverse\nasinh, MAD\n-> EUR/MWh", bsize=7.8)
    note(ax, 3.8, 1.25, "Calibration:  (a) fitted once on train 2022-06 - 2024-03   val 19.18 / test 20.67\n"
                        "                     (b) refitted every month on the trailing 730 days   val 18.25 / test 18.43\n"
                        "1.6 % of hours (gaps in the inputs) fall back to the similar-day naive forecast.", size=8.2)
    save(fig, "L1_lear", pages)


# ---- L2 XGBoost ----------------------------------------------------------------------------------------

def l2_xgb(pages):
    fig, ax = canvas(11, 6.0, "L2  XGBoost per zone  (c1)  and  + neighbours  (c2, dashed)",
                     "No graph, no sequence: every (zone, day D, hour h) is an independent row")
    own = box(ax, 0.4, 3.0, 3.1, 1.9, "own features of zone z, hour h",
              "price_lag24, price_lag168\nload_fc\ntemp, temp_anom, wind, solar\n  (_lag24; _asof in reruns)\nnet_import_lag48\nhour, weekday, month", bsize=8)
    # neighbour averaging mini-graph
    cx, cy = 1.95, 1.35
    node(ax, cx, cy, "z", r=0.22, double=True)
    for k, (dx, dy) in enumerate([(-1.1, 0.45), (-0.9, -0.55), (1.0, 0.5), (1.15, -0.4)]):
        node(ax, cx + dx, cy + dy, f"n{k + 1}", r=0.2, ls="--", fs=6.5)
        line(ax, (cx + dx, cy + dy), (cx, cy), ls="--", lw=0.8)
    note(ax, 0.4, 0.55, "c2: nb_<col> = mean of the 8 numeric features over z's interconnected neighbours", size=8)
    nb = box(ax, 4.0, 1.0, 2.2, 1.1, "neighbour means", "8 x nb_<col>", ls="--", bsize=8.5)
    arrow(ax, (3.2, 1.45), (4.0, 1.5), ls="--")
    cat = box(ax, 4.0, 3.35, 2.2, 1.2, "row", "11 features (c1)\n19 features (c2)", bsize=8.5)
    arrow(ax, (3.5, 3.95), (4.0, 3.95), "11")
    arrow(ax, top(nb), bottom(cat), "8", ls="--")
    # trees
    tx, ty = 6.8, 2.55
    box(ax, tx, ty, 2.45, 2.75, "XGBRegressor (zone z)", " ", bsize=8)
    for k, ox in enumerate([0.35, 1.0, 1.65]):
        bx, by = tx + ox + 0.1, ty + 2.05
        node(ax, bx, by, "", r=0.07)
        for dx in (-0.22, 0.22):
            line(ax, (bx, by), (bx + dx, by - 0.4), lw=0.8)
            node(ax, bx + dx, by - 0.4, "", r=0.06)
            for ddx in (-0.1, 0.1):
                line(ax, (bx + dx, by - 0.4), (bx + dx + ddx, by - 0.75), lw=0.7)
    note(ax, tx + 2.15, ty + 1.75, "...", size=12, color=INK)
    note(ax, tx + 0.15, ty + 0.95, "sum of <= 600 trees, depth 6\nlr 0.05, subsample 0.8\nL1 objective, early stop on val",
         size=7.8, color=INK)
    arrow(ax, (6.2, 3.95), (6.8, 3.95))
    arrow(ax, (9.25, 3.95), (9.65, 3.95))
    box(ax, 9.7, 3.35, 1.05, 1.2, "price", "asinh(p/50)\n-> EUR/MWh", bsize=7.8)
    note(ax, 6.8, 2.25, "one model per zone (45 models)\nfit on pretrain + train (2016-01 - 2024-03)", size=8)
    note(ax, 6.8, 1.35, "test MAE:  c1 21.24   c2 21.03\n(old _lag24 weather; _asof reruns queued)", size=8)
    save(fig, "L2_xgboost", pages)


# ---- L3 GRU + MLP --------------------------------------------------------------------------------------

def l3_gru_mlp(pages):
    fig, ax = canvas(11, 6.2, "L3  GRU + MLP  (c2.5): the neural model without a graph",
                     "Per (zone z, delivery day D); the same weights for all 45 zones. 40,577 parameters")
    hb = box(ax, 0.4, 3.6, 2.8, 1.4, "history  168 x 8", "hours up to D-1 10:00 UTC:\nprice_lag24, price_lag168,\nload_fc, net_import, weather", bsize=8)
    # unrolled GRU
    gx, gy = 3.8, 3.85
    for k in range(4):
        x = gx + k * 0.75
        box(ax, x, gy, 0.55, 0.55, "GRU" if k < 3 else "GRU", tsize=7.5, bold=False)
        if k < 3:
            arrow(ax, (x + 0.55, gy + 0.275), (x + 0.75, gy + 0.275))
        line(ax, (x + 0.275, gy - 0.35), (x + 0.275, gy), lw=0.8)
    for k, lab in enumerate(["t-167", "t-166", "...", "cutoff"]):
        note(ax, gx + k * 0.75 + 0.275, gy - 0.4, lab, size=7, ha="center")
    arrow(ax, (3.2, 4.3), (3.8, 4.12))
    note(ax, gx, gy + 1.0, "GRU 8 -> 64, one layer, oldest first", size=8, color=INK)
    hbox = box(ax, 7.0, 3.85, 1.3, 0.55, "h  (64)", tsize=9)
    arrow(ax, (gx + 2.8, gy + 0.275), (7.0, gy + 0.275))
    xb = box(ax, 0.4, 1.4, 2.8, 1.4, "day-D inputs, slot t  (11)", "price_lag24, price_lag168, load_fc,\nweather lag, net_import_lag48,\nhour, weekday, month", bsize=8)
    cat = box(ax, 4.2, 1.6, 1.6, 1.0, "concat", "[h | x_t] = 75", bsize=8.3)
    arrow(ax, (3.2, 2.1), (4.2, 2.1), "11")
    arrow(ax, bottom(hbox), (5.0, 2.6), "copied to 25 slots", rad=-0.25, lside="left")
    mlp = box(ax, 6.4, 1.35, 2.0, 1.5, "MLP", "75 -> 128, ReLU, drop\n128 -> 128, ReLU\n128 -> 1", bsize=8)
    arrow(ax, (5.8, 2.1), (6.4, 2.1))
    out = box(ax, 9.0, 1.45, 1.75, 1.3, "r_t  ->  price", "p = 50 sinh(r_t +\n asinh(base/50))", bsize=7.8)
    arrow(ax, (8.4, 2.1), (9.0, 2.1))
    note(ax, 0.4, 0.95, "r_t = asinh(p_t / 50) - asinh(base_t / 50), base = price one week earlier.   "
                        "Loss: masked L1 on r + 0.5 x L1 on the price gaps of interconnected pairs.\n"
                        "Zones never exchange information (the spread term only couples them in the loss).   "
                        "test MAE 20.40 (old weather inputs).", size=8)
    save(fig, "L3_gru_mlp", pages)


# ---- L4 GATv2 attention scope --------------------------------------------------------------------------

def gat_trunk(ax, x0=0.4, day_body=None, day_title="day-D inputs, slot t", extra_box=None, n_day="86"):
    hb = box(ax, x0, 6.3, 2.6, 1.25, "history  168 x 8", "to D-1 10:00 UTC\nprices, load, net import,\nrealised weather", bsize=7.8)
    db = box(ax, x0 + 2.85, 6.3, 3.0, 1.25, day_title, day_body or
             "14: price D-1 / D-7 same hour,\nload_fc, _asof weather, sin/cos time\n+ 72: D-1, D-2, D-7 price curves", bsize=7.8)
    pb = box(ax, x0 + 6.1, 6.3, 1.9, 1.25, "position", "8 Laplacian\neigenvectors\n(random sign flip)", bsize=7.8)
    g = box(ax, x0, 5.15, 2.6, 0.6, "GRU 8 -> 64", tsize=9)
    arrow(ax, bottom(hb), top(g))
    lin = box(ax, x0, 4.2, 8.0, 0.55, "concat [64 | " + n_day + " | 8]   ->   Linear -> 64   = u0", tsize=9, bold=False)
    arrow(ax, bottom(g), (x0 + 1.3, 4.75), "x25 slots")
    arrow(ax, bottom(db), (x0 + 4.35, 4.75), n_day)
    arrow(ax, bottom(pb), (x0 + 7.05, 4.75), "8")
    y = 3.35
    arrow(ax, (x0 + 4.0, 4.2), (x0 + 4.0, 3.65)) if not extra_box else None
    if extra_box:
        eb = box(ax, x0, y, 8.0, 0.55, extra_box, tsize=9, bold=False, ls="--")
        arrow(ax, (x0 + 4.0, 4.2), (x0 + 4.0, y + 0.55))
        y -= 0.85
    # one block, drawn as a residual pre-LN unit
    by = y - 1.15                                                        # row of the block's parts
    ax.add_patch(FancyBboxPatch((x0, by - 0.3), 8.0, 1.75, boxstyle="round,pad=0,rounding_size=0.08", fc="white",
                                ec=INK, lw=1.4))
    note(ax, x0 + 0.12, by + 1.37, "block  x2   (one graph per day x slot, 45 zones)", size=8.5, color=INK, va="top")
    arrow(ax, (x0 + 4.0, y + 0.85 if not extra_box else y + 0.85), (x0 + 4.0, by + 1.45)) if not extra_box else None
    parts = [("LN", 0.7), ("GATv2\n4 heads x 16", 1.45), ("drop", 0.75), ("+", 0.45), ("LN", 0.7), ("FFN\n64-128-64 ELU", 1.6),
             ("drop", 0.75), ("+", 0.45)]
    x = x0 + 0.45
    spans = []                                                           # (left, right) of each part
    for t, w in parts:
        if t == "+":
            node(ax, x + 0.2, by + 0.3, "+", r=0.17, fs=10)
            spans.append((x + 0.03, x + 0.37))
        else:
            box(ax, x, by, w - 0.15, 0.6, t, tsize=7.3, bold=t.startswith("GATv2"))
            spans.append((x, x + w - 0.15))
        x += w
    for (_, ra), (lb, _) in zip(spans[:-1], spans[1:]):
        arrow(ax, (ra, by + 0.3), (lb, by + 0.3))
    arrow(ax, (x0 + 0.1, by + 0.3), (spans[0][0], by + 0.3))             # u in
    for start, plus in ((0, 3), (4, 7)):                                 # skip connections
        sx = spans[start][0] - 0.17
        px = (spans[plus][0] + spans[plus][1]) / 2
        line(ax, (sx, by + 0.3), (sx, by + 0.9), lw=0.9)
        line(ax, (sx, by + 0.9), (px, by + 0.9), lw=0.9)
        arrow(ax, (px, by + 0.9), (px, by + 0.47), lw=0.9)
    note(ax, x0 + 0.12, by - 0.08, "u", size=8, color=INK)
    head = box(ax, x0, by - 1.65, 3.6, 0.95, "head", "LN -> 128 ReLU drop -> 128 ReLU -> 1\n= r_t (asinh change from D-7)", bsize=7.6)
    arrow(ax, (x0 + 1.8, by - 0.3), (x0 + 1.8, by - 0.7))
    arrow(ax, (x0 + 3.6, by - 1.18), (x0 + 4.0, by - 1.18))
    loss = box(ax, x0 + 4.0, by - 1.65, 4.0, 0.95, "loss", "masked L1 on r  +  0.5 x L1 on spreads\nof the 87 interconnected pairs", bsize=7.6)
    return by


def l4_gat_scope(pages):
    fig, ax = canvas(13, 9.2, "L4  GRU + GATv2 attention scope  (gat_scope.py)",
                     "Same weights in every arm; only the attention neighbourhood changes. 95,425 parameters (14 day inputs), "
                     "100,033 with the 72 price-curve inputs")
    gat_trunk(ax)
    ox = 9.05
    note(ax, ox, 7.75, "the scope (messages into DE, one layer)", size=9.5, color=INK, weight="bold")
    for k, (scope, desc) in enumerate([("self", "self: only the self-loop\n45 edges; no cross-zone messages"),
                                       ("interconnects", "interconnects: physical neighbours\n174 directed edges + 45 self-loops"),
                                       ("full", "full: every zone (a Transformer over zones)\n1,980 edges + 45 self-loops")]):
        y = 5.35 - k * 2.45
        mini_graph(ax, ox + 0.2, y, scope)
        note(ax, ox, y - 0.12, desc, size=8)
    note(ax, 0.4, 0.42, "Variants run: inputs old (_lag24 weather, 180 edges) / safe (_asof, 174 edges); +/- price curves; fitted once / "
                        "fine-tuned monthly (L7); full without position (full_nope).", size=8)
    save(fig, "L4_gat_scope", pages)


# ---- L5 H2 constraint arm ------------------------------------------------------------------------------

def l5_constraints(pages):
    fig, ax = canvas(13, 7.6, "L5  H2 constraint arm  (gat_constraints.py, 111,201 parameters = L4 + 11,168)",
                     "Constraint nodes from the D-1 flow-based domain (same UTC hour, published on D-2): about 110 per hour, "
                     "at most 182")
    zp = {"FR": (1.2, 5.3), "DE_LU": (3.2, 5.9), "PL": (5.2, 5.3)}
    cp = {"line A\nRAM/Fmax .13": (1.0, 2.6), "line B\nRAM/Fmax .05": (3.2, 2.6), "line C\nRAM/Fmax .60": (5.4, 2.6)}
    line(ax, zp["FR"], zp["DE_LU"], lw=2.2)
    line(ax, zp["DE_LU"], zp["PL"], lw=2.2)
    edges = [("FR", 0, +0.15), ("DE_LU", 0, -0.06), ("DE_LU", 1, +0.32), ("PL", 1, -0.21), ("DE_LU", 2, +0.11), ("PL", 2, +0.18)]
    cl = list(cp.values())
    for z, c, v in edges:
        (x0, y0), (x1, y1) = zp[z], cl[c]
        line(ax, (x0, y0 - 0.3), (x1, y1 + 0.35), ls="-" if v > 0 else "--", lw=0.6 + 4 * abs(v))
        ax.text(x0 + 0.45 * (x1 - x0) + 0.05, y0 - 0.3 + 0.45 * (y1 + 0.35 - y0 + 0.3), f"{v:+.2f}", fontsize=7.5,
                color=INK, bbox={"fc": "white", "ec": "none", "pad": 0.5})
    for z, (x, y) in zp.items():
        node(ax, x, y, z, r=0.36, fs=8)
    for t, (x, y) in cp.items():
        box(ax, x - 0.75, y - 0.35, 1.5, 0.7, t.split("\n")[0], t.split("\n")[1], tsize=8.5, bsize=7.3)
    note(ax, 0.4, 1.7, "thick line: interconnector (zone GAT)     thin line: exposure edge, |zone-to-rest PTDF| >= 0.05\n"
                       "solid: PTDF > 0 (exporting from the zone loads the line)     dashed: PTDF < 0 (relieves it)\n"
                       "line width ~ |PTDF|; values illustrative", size=8)
    x0 = 6.9
    box(ax, x0, 5.25, 5.7, 1.2, "constraint encoder", "x_c = 8 scale-free features: RAM/Fmax, (RAM-AMR)/Fmax,\nAMR/Fmax, "
        "min-RAM factor, FRM, F0core, F_LTN (/Fmax), log Fmax\nk_c = MLP(x_c)  (16)", bsize=7.8)
    box(ax, x0, 3.35, 5.7, 1.3, "zone <- constraint attention  (4 heads)", "s_zc = a . LeakyReLU(Wq u_z + Wk k_c + w_p p_zc + w_a |p_zc|)\n"
        "alpha_zc = softmax over the constraints z is exposed to\nm_z = sum_c alpha_zc (Wv k_c + v_p p_zc)", bsize=7.8)
    box(ax, x0, 1.85, 5.7, 0.95, "update, then the L4 zone blocks", "u_z <- u_z + Drop(Linear(m_z)) * g_z      (g_z = 0: no message)\n"
        "inserted at the node_update hook, before the 2 GATv2 blocks", bsize=7.8)
    arrow(ax, (x0 + 2.85, 5.25), (x0 + 2.85, 4.65))
    arrow(ax, (x0 + 2.85, 3.35), (x0 + 2.85, 2.8))
    note(ax, 0.4, 0.75, "Only the 12 Core zones receive messages; non-Core zones, pretrain days (before 2022-06-09) and days without a "
                        "published domain get none.\nThe interconnector blocks then pass the information on to non-Core neighbours. "
                        "Control: L4 interconnects + curves, fine-tuned monthly.", size=8)
    save(fig, "L5_h2_constraints", pages)


# ---- L6 H2 JAO-summary arm -----------------------------------------------------------------------------

def l6_jao(pages):
    fig, ax = canvas(13, 9.2, "L6  H2 JAO-summary arm  (c4s_gat_interconnects_jao_rolling, 100,417 parameters)",
                     "L4 interconnects + curves, fine-tuned monthly, plus six per-zone summaries of the D-1 domain "
                     "(dashed): no constraint nodes")
    gat_trunk(ax, day_title="day-D inputs, slot t  (86 + 6)", n_day="92",
              day_body="L4 inputs (14 + 72 curves)\n+ 6 jao_*_lag24 (dashed box, right)")
    jb = box(ax, 9.0, 5.3, 3.6, 2.25, "jao_*_lag24  (zone z, hour t-24h)",
             "n_exposed       |PTDF| >= 0.05\nmin_ram_exposed  min RAM/Fmax\ntightness   PTDF-weighted\n"
             "amr_share_exposed\nmin_ram_grid_exposed\ntightness_grid", ls="--", bsize=7.6, mono_body=True)
    line(ax, (10.8, 7.55), (10.8, 7.85), ls="--")
    line(ax, (10.8, 7.85), (4.75, 7.85), ls="--")
    arrow(ax, (4.75, 7.85), (4.75, 7.57), ls="--")
    note(ax, 9.0, 4.95, "12 Core zones only; 0 elsewhere\n(after z-scoring)", size=8)
    save(fig, "L6_h2_jao_summaries", pages)


# ---- L7 training protocol ------------------------------------------------------------------------------

def l7_protocol(pages):
    fig, ax = canvas(13, 6.6, "L7  Training and evaluation protocol",
                     "Same samples and scoring for every model (epgnn/metrics.py); MAE in EUR/MWh over valid zone-hours")
    # cutoff rule
    y = 4.4
    line(ax, (0.6, y), (6.2, y), lw=1.2)
    for x, lab in [(0.8, "D-2"), (2.6, "D-1 00:00"), (3.95, "D-1 10:00 UTC\ncutoff"), (4.9, "D 00:00"), (6.0, "D 24:00")]:
        line(ax, (x, y - 0.1), (x, y + 0.1), lw=1.2)
        note(ax, x, y - 0.18, lab, size=7.5, ha="center", color=INK)
    ax.add_patch(FancyBboxPatch((0.6, y + 0.2), 3.35, 0.32, boxstyle="square,pad=0", fc="white", ec=INK, hatch="////", lw=0.8))
    note(ax, 0.65, y + 0.85, "known: history, D-1 prices (published D-2), load forecast D", size=7.8, color=INK)
    ax.add_patch(FancyBboxPatch((4.9, y + 0.2), 1.1, 0.32, boxstyle="square,pad=0", fc="white", ec=INK, lw=1.2))
    note(ax, 5.45, y + 0.85, "predict 24 h", size=7.8, ha="center", color=INK)
    # splits
    x0, w = 6.9, 5.8
    spans = [("pretrain", 2016.0, 2022.44), ("train", 2022.44, 2024.25), ("val", 2024.25, 2025.25), ("test", 2025.25, 2026.75)]
    sx = lambda t: x0 + (t - 2016.0) / (2026.75 - 2016.0) * w
    for name, a, b in spans:
        ax.add_patch(FancyBboxPatch((sx(a), y + 0.15), sx(b) - sx(a), 0.45, boxstyle="square,pad=0", fc="white",
                                    ec=INK, lw=1.0, hatch="" if name != "test" else "...."))
        ax.text((sx(a) + sx(b)) / 2, y + 0.375, name, fontsize=7.8, ha="center", va="center", color=INK,
                bbox={"fc": "white", "ec": "none", "pad": 1.0})
    for t in (2016, 2018, 2020, 2022, 2024, 2026):
        note(ax, sx(t), y + 0.05, str(t), size=7, ha="center")
    note(ax, x0, y + 1.05, "splits (delivery days)", size=8.5, color=INK, weight="bold")
    # fit once
    y2 = 3.0
    note(ax, 0.6, y2 + 0.45, "fit once  (all neural models)", size=9.5, color=INK, weight="bold")
    box(ax, 0.6, y2 - 0.55, 3.6, 0.75, "Adam lr 1e-3, 16-day batches", "early stop on val loss, patience 8, <= 50 epochs", bsize=7.6)
    arrow(ax, (4.2, y2 - 0.17), (4.8, y2 - 0.17))
    box(ax, 4.85, y2 - 0.55, 1.6, 0.75, "predict", "val and test", bsize=7.6)
    # rolling
    note(ax, 6.9, y2 + 0.45, "fine-tuned monthly  (rolling: true)", size=9.5, color=INK, weight="bold")
    months = ["Apr", "May", "Jun", "...", "Sep 26"]
    for k, m in enumerate(months):
        bx = 6.9 + k * 1.2
        if m == "...":
            note(ax, bx + 0.45, y2 - 0.15, "...", size=12, ha="center", va="center", color=INK)
            continue
        box(ax, bx, y2 - 0.5, 0.9, 0.65, m, tsize=8, bold=False)
        if k > 0:
            ax.add_patch(FancyArrowPatch((bx - 0.25, y2 + 0.15), (bx + 0.2, y2 + 0.15), arrowstyle="-|>", mutation_scale=8,
                                         connectionstyle="arc3,rad=-0.6", lw=0.9, color=INK))
    note(ax, 6.9, y2 - 0.7, "before each month: 2 epochs at lr 1e-4 on the trailing 730 days (ending the day before);\n"
                            "then predict the month with fixed weights. 29 months over val + test; no early stopping.\n"
                            "The fit-once predictions are kept too (pred_<split>_static.npy).", size=7.8)
    note(ax, 0.6, 1.15, "Comparisons: arms differ in one switch only; paired weekly block bootstrap of the MAE difference "
                        "(epgnn/bootstrap.py), 3 seeds for H1.\nBars: LEAR fitted once (20.67 test) for fit-once models, "
                        "LEAR refitted monthly (18.43 test) for rolling models.", size=8)
    save(fig, "L7_protocol", pages)


def main() -> None:
    pages: list = []
    for f in (l1_lear, l2_xgb, l3_gru_mlp, l4_gat_scope, l5_constraints, l6_jao, l7_protocol):
        f(pages)
    with PdfPages(os.path.join(OUT, "all_models.pdf")) as pdf:
        for fig in pages:
            pdf.savefig(fig)
    print("->", os.path.join("figures", "line", "all_models.pdf"))


if __name__ == "__main__":
    main()
