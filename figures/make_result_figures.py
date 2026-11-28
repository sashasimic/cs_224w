#!/usr/bin/env python3
"""Result figures: where and when the graph and the monthly fine-tuning help.

Same style, output folder and conventions as make_data_figures.py (PNG + PDF + CSV in figures/data/).

  r1_monthly_mae      test MAE per month for the three attention scopes (fine-tuned monthly), and what
                      fine-tuning gains per month over the fit-once model; the 15-minute switch marked
  r2_zone_gain_map    per-zone change in test MAE, interconnects - self (fine-tuned, mean over the seeds
                      both arms have): where the graph helps, on the map of Europe
  r3_bootstrap        paired weekly block-bootstrap differences with 95 % CIs, every comparison in
                      results/bootstrap_*.json, by slice
  r4_hour_profile     test MAE by local delivery hour: graph vs no graph, fit once vs fine-tuned
  r5_seeds            test MAE of every seed per arm, fit once and fine-tuned: the seed noise next to
                      the effects
  r6_attention_full   the full-attention arm: share of each zone's cross-zone attention that goes to its
                      physical neighbours, against the share uniform attention would give
  r7_zone_mae         per-zone test MAE of the best model next to the no-graph control, sorted, with the
                      zone's price variability for scale
  r8_constraint_attention  H2: where the constraint-node arm looks. Recomputed from the saved weights on
                      every 7th test day: attention lift (alpha x number of exposed constraints; 1 = uniform)
                      by the constraint's RAM/Fmax and by |zone-to-rest PTDF|, and per Core zone the attention
                      mass on its tightest tenth of constraints against the uniform 10 %

Usage (from the project folder):
    python figures/make_result_figures.py            # all that have their inputs
    python figures/make_result_figures.py r2 r3      # some (prefix match)

Inputs: cs_224w-c1_xgb_zone/runs/<run>/pred_<split>[_static].npy, results/*.json, the node table
(targets, through epgnn.data.samples). A figure whose runs are missing is skipped with a note.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_data_figures as D                                       # noqa: E402  style, helpers, paths
from make_data_figures import (AXIS, BLUE, BLUE_150, BLUE_550, GRID, INK, INK2, MUTED, NEUTRAL,  # noqa: E402
                               ORANGE, SURFACE, YELLOW, AQUA, plt, save, ygrid)

sys.path.insert(0, D.FW)
os.chdir(D.FW)                                                       # epgnn resolves data from the repo root
from epgnn.data import samples as S                                  # noqa: E402

RUNS, RESULTS = os.path.join(D.FW, "runs"), os.path.join(D.FW, "results")
ARMS = {"self": "no graph (self)", "interconnects": "interconnectors", "full": "full attention"}
ARM_COLOR = {"self": INK2, "interconnects": BLUE, "full": ORANGE}
SWITCH = pd.Timestamp("2025-10-01")                                  # 15-minute day-ahead products
ORANGE_150, ORANGE_550 = "#f8c7b3", "#b8461b"                        # diverging pole (orange ramp)


# ---- data ----------------------------------------------------------------------------------------------

_samples: dict = {}


def samples(split: str) -> S.Samples:
    if split not in _samples:
        _samples[split] = S.build(split)
    return _samples[split]


def run_name(arm: str, seed: int = 0, kind: str = "curves_rolling") -> str:
    return f"c3s_gat_{arm}_{kind}" + (f"_s{seed}" if seed else "")


def pred(run: str, split: str = "test", static: bool = False) -> np.ndarray | None:
    f = os.path.join(RUNS, run, f"pred_{split}{'_static' if static else ''}.npy")
    return np.load(f) if os.path.exists(f) else None


def seeds_of(arm: str) -> list[int]:
    return [s for s in range(10) if pred(run_name(arm, s)) is not None]


def abs_err(p: np.ndarray, s: S.Samples) -> np.ndarray:
    ok = s.mask & np.isfinite(s.y) & np.isfinite(p)
    return np.where(ok, np.abs(p - s.y), np.nan)


def seed_mean_err(arm: str, split: str = "test", static: bool = False, seeds=None) -> np.ndarray | None:
    s = samples(split)
    es = [abs_err(p, s) for k in (seeds if seeds is not None else seeds_of(arm))
          if (p := pred(run_name(arm, k), split, static)) is not None]
    return np.mean(es, axis=0) if es else None


def skip(name: str, why: str) -> None:
    print(f"  {name}: skipped ({why})", flush=True)


def month_mae(err: np.ndarray, s: S.Samples) -> pd.Series:
    m = pd.to_datetime(s.day).to_period("M")
    df = pd.DataFrame({"m": np.repeat(m, err.shape[1]), "e": err.ravel()}).dropna()
    return df.groupby("m").e.mean()


# ---- r1: by month ----------------------------------------------------------------------------------------

def r1_monthly_mae():
    rows = []
    for split in ("val", "test"):
        s = samples(split)
        for arm in ARMS:
            for static in (False, True):
                e = seed_mean_err(arm, split, static, seeds=[0])
                if e is None:
                    continue
                for m, v in month_mae(e, s).items():
                    rows.append({"month": m.to_timestamp(), "split": split, "arm": arm,
                                 "fit": "fit once" if static else "fine-tuned", "mae": v})
    if not rows:
        return skip("r1_monthly_mae", "no curves_rolling runs")
    t = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
    ft = t[t.fit == "fine-tuned"]
    for arm in ARMS:
        d = ft[ft.arm == arm].sort_values("month")
        a1.plot(d.month, d.mae, color=ARM_COLOR[arm], lw=1.6 if arm != "self" else 1.3, marker="o", ms=2.8,
                label=ARMS[arm], zorder=3 if arm == "interconnects" else 2)
    ygrid(a1)
    a1.set_ylabel("MAE, EUR/MWh")
    a1.set_title("Test and val MAE by month, fine-tuned monthly (seed 0)", pad=16)
    a1.legend(loc="upper left", fontsize=7, labelcolor=INK2, ncol=3)
    piv = t[t.arm == "interconnects"].pivot_table(index="month", columns="fit", values="mae")
    gain = (piv["fit once"] - piv["fine-tuned"]).rename("gain")
    a2.bar(gain.index, gain.values, width=22, color=np.where(gain.values >= 0, BLUE, ORANGE), zorder=3)
    a2.axhline(0, color=AXIS, lw=0.8)
    ygrid(a2)
    a2.set_ylabel("fit once − fine-tuned")
    a2.set_title("What monthly fine-tuning gains (interconnectors; > 0 = better)", fontsize=8.5, pad=4)
    for ax in (a1, a2):
        ax.axvline(SWITCH, color=MUTED, lw=0.7, ls=(0, (3, 3)))
        ax.axvline(pd.Timestamp(D.SPLITS["test"][0]), color=AXIS, lw=0.7)
    a1.text(SWITCH, 1.0, " 15-min products", transform=a1.get_xaxis_transform(), fontsize=7, color=INK2, va="bottom")
    a1.text(pd.Timestamp(D.SPLITS["val"][0]), 1.0, "val", transform=a1.get_xaxis_transform(), fontsize=7, color=INK2, va="bottom")
    a1.text(pd.Timestamp(D.SPLITS["test"][0]), 1.0, " test", transform=a1.get_xaxis_transform(), fontsize=7, color=INK2, va="bottom")
    a2.text(0, -0.28, "The first val month is never fine-tuned (it follows train directly), so its gain is 0.",
            transform=a2.transAxes, fontsize=6.5, color=MUTED)
    out = t.copy()
    out["month"] = out.month.dt.strftime("%Y-%m")
    save(fig, out, "r1_monthly_mae")


# ---- r2: map of the graph's gain ---------------------------------------------------------------------------

def zone_mae(err: np.ndarray, s: S.Samples) -> pd.Series:
    tot = pd.Series(np.nansum(err, axis=1), index=s.zone).groupby(level=0).sum()
    cnt = pd.Series(np.isfinite(err).sum(axis=1), index=s.zone).groupby(level=0).sum()
    return tot / cnt.where(cnt > 0)                                  # hour-weighted, like the headline MAE


def zone_table(a: str = "interconnects", b: str = "self") -> tuple[pd.DataFrame, list[int]] | None:
    common = sorted(set(seeds_of(a)) & set(seeds_of(b)))
    if not common:
        return None
    s = samples("test")
    ea, eb = seed_mean_err(a, seeds=common), seed_mean_err(b, seeds=common)
    t = pd.DataFrame({"mae_a": zone_mae(ea, s), "mae_b": zone_mae(eb, s)})
    t["delta"] = t.mae_a - t.mae_b
    t["rel"] = t.delta / t.mae_b
    yy = np.where(s.mask, s.y, np.nan)
    with np.errstate(invalid="ignore"):
        y = pd.Series(np.nanstd(yy, axis=1), index=s.zone).groupby(level=0).mean()
    t["price_std"] = y
    return t.reset_index(names="zone"), common


def r2_zone_gain_map():
    zt = zone_table()
    if zt is None:
        return skip("r2_zone_gain_map", "no common seeds")
    t, seeds = zt
    zs, ne = D.zone_shapes()
    g = zs.merge(t, on="zone", how="left")
    lim = float(np.nanpercentile(np.abs(t.delta), 95))
    bins = [-np.inf, -lim * 0.6, -lim * 0.25, -0.05, 0.05, lim * 0.25, lim * 0.6, np.inf]
    colors = [BLUE_550, "#4f8fde", BLUE_150, GRID, ORANGE_150, "#f0936b", ORANGE_550]
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ne.plot(ax=ax, color=NEUTRAL, edgecolor=SURFACE, linewidth=0.4)
    for k in range(len(colors)):
        sel = g[(g.delta > bins[k]) & (g.delta <= bins[k + 1])]
        if len(sel):
            sel.plot(ax=ax, color=colors[k], edgecolor=SURFACE, linewidth=0.6)
    g[g.delta.isna()].plot(ax=ax, color=NEUTRAL, edgecolor=AXIS, linewidth=0.6, hatch="////")
    pts = g.set_index("zone").geometry.representative_point()
    for z, p in pts.items():
        d = t.set_index("zone").delta.get(z, np.nan)
        if np.isfinite(d):
            dx, dy = {**D.ANCHOR_SHIFT, "DK_2": (70, -10), "DK_1": (-20, 10)}.get(z, (0, 0))
            ax.annotate(f"{z.replace('_', '')}\n{d:+.1f}".replace("-", "−"), (p.x + dx * 1e3, p.y + dy * 1e3), ha="center", va="center",
                        fontsize=5.6, color=INK if abs(d) < lim * 0.6 else SURFACE, zorder=5, linespacing=0.95)
    ax.set_xlim(2.55e6, 6.45e6)
    ax.set_ylim(1.35e6, 5.45e6)
    ax.set_axis_off()
    allz = t.delta.mean()
    ax.set_title(f"Where the interconnector graph helps: test MAE change vs no graph (zone mean {allz:+.2f})".replace("-", "−"),
                 loc="left", pad=8)
    f1 = lambda v: f"{v:.1f}".replace("-", "−")                     # noqa: E731
    lab = [f"< {f1(bins[1])}", f"{f1(bins[1])} to {f1(bins[2])}", f"{f1(bins[2])} to −0.05", "±0.05",
           f"0.05 to {f1(bins[5])}", f"{f1(bins[5])} to {f1(bins[6])}", f"> {f1(bins[6])}"]
    leg = [D.Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(colors, lab)]
    leg.append(D.Patch(facecolor=NEUTRAL, edgecolor=AXIS, hatch="////", label="no price in test (GB, BA)"))
    ax.legend(handles=leg, title="Δ MAE, EUR/MWh (blue = graph better)", title_fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 0.985), fontsize=6.8, labelcolor=INK2)
    ax.text(0, -0.01, f"Interconnectors minus self, both fine-tuned monthly with price curves; mean over seeds "
            f"{', '.join(map(str, seeds))}. Test split.", transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top")
    save(fig, t, "r2_zone_gain_map")


# ---- r3: bootstrap forest ------------------------------------------------------------------------------------

def _label(names: list[str]) -> str:
    n = names[0].replace("c3s_gat_", "").replace("c4s_gat_", "").replace("_curves", "").replace("_rolling", "")
    n = n.replace("interconnects", "interconnectors")
    return n + (f" ({len(names)} seeds)" if len(names) > 1 else "")


def r3_bootstrap():
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "bootstrap_*.json"))):
        r = json.load(open(f))
        comp = f"{_label(r['a'])}  −  {_label(r['b'])}, " + ("fit once" if r["static"] else "fine-tuned")
        for sl, x in r["slices"].items():
            rows.append({"comparison": comp, "slice": sl, "delta": x["delta"], "lo": x["ci95"][0], "hi": x["ci95"][1],
                         "file": os.path.basename(f), "n_seeds": max(len(r["a"]), len(r["b"]))})
    if not rows:
        return skip("r3_bootstrap", "no bootstrap results")
    t = pd.DataFrame(rows)
    # with several seeds for the same comparison, keep the seed-averaged one
    t["key"] = t.comparison.str.replace(r" \(\d seeds\)", "", regex=True)
    t = t[t.n_seeds == t.groupby("key").n_seeds.transform("max")]
    comps = list(dict.fromkeys(t.comparison))
    slices = [s for s in ["all", "pre_2025-10", "post_2025-10", "summer", "core_zones"] if s in set(t.slice)]
    sl_color = {"all": INK, "pre_2025-10": BLUE, "post_2025-10": ORANGE, "summer": YELLOW, "core_zones": AQUA}
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(comps) * len(slices) / 2.6 + 1.4))
    y, ticks = 0.0, []
    for c in comps:
        d = t[t.comparison == c].set_index("slice")
        ys = []
        for k, sl in enumerate(slices):
            if sl not in d.index:
                continue
            r = d.loc[sl]
            yy = y + k * 0.16
            ax.plot([r.lo, r.hi], [yy, yy], color=sl_color[sl], lw=1.6, solid_capstyle="round")
            ax.scatter([r.delta], [yy], s=22 if sl == "all" else 12, color=sl_color[sl], zorder=3,
                       edgecolor=SURFACE, linewidth=0.8)
            ys.append(yy)
        ticks.append((np.mean(ys), c))
        y += 0.16 * len(slices) + 0.45
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks([a for a, _ in ticks], [b for _, b in ticks], fontsize=7.4)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Δ test MAE, EUR/MWh (first minus second; < 0 = first model better)")
    ax.set_title("Paired weekly block bootstrap, 95 % intervals", pad=8)
    ax.legend(handles=[D.Line2D([], [], color=sl_color[s], lw=1.6, marker="o", ms=4,
                                label={"all": "all", "pre_2025-10": "before 15-min", "post_2025-10": "after 15-min",
                                       "summer": "summer", "core_zones": "Core zones"}[s]) for s in slices],
              loc="lower right", fontsize=6.8, labelcolor=INK2)
    ax.text(0, -0.13 if len(comps) > 3 else -0.2, "Intervals resample days in 7-day blocks (all zones together); "
            "they cover sampling over days, and seed noise only where several seeds are averaged.",
            transform=ax.transAxes, fontsize=6.5, color=MUTED)
    save(fig, t.drop(columns="key"), "r3_bootstrap")


# ---- r4: by hour of day ----------------------------------------------------------------------------------------

def r4_hour_profile():
    s = samples("test")
    hrs = pd.DatetimeIndex(pd.to_datetime(np.where(s.hours >= 0, s.hours, 0).ravel(), utc=True)) \
        .tz_convert("Europe/Brussels").hour.to_numpy().reshape(s.hours.shape)
    series = [("interconnects", False, BLUE, "-", "interconnectors, fine-tuned"),
              ("self", False, INK2, "-", "no graph, fine-tuned"),
              ("interconnects", True, BLUE, (0, (3, 2)), "interconnectors, fit once"),
              ("self", True, INK2, (0, (3, 2)), "no graph, fit once")]
    rows = []
    for arm, static, *_ in series:
        e = seed_mean_err(arm, "test", static, seeds=[0])
        if e is None:
            continue
        df = pd.DataFrame({"h": hrs.ravel(), "e": e.ravel()}).dropna()
        for h, v in df.groupby("h").e.mean().items():
            rows.append({"arm": arm, "fit": "fit once" if static else "fine-tuned", "hour": h, "mae": v})
    if not rows:
        return skip("r4_hour_profile", "no runs")
    t = pd.DataFrame(rows)
    py = pd.DataFrame({"h": hrs.ravel(), "y": np.where(s.mask, s.y, np.nan).ravel()}).dropna().groupby("h").y.mean()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.1})
    for arm, static, col, ls, lab in series:
        d = t[(t.arm == arm) & (t.fit == ("fit once" if static else "fine-tuned"))].sort_values("hour")
        if len(d):
            a1.plot(d.hour, d.mae, color=col, ls=ls, lw=1.6, label=lab)
    ygrid(a1)
    a1.set_ylabel("test MAE, EUR/MWh")
    a1.set_title("Error by delivery hour (Brussels time, test split, seed 0)", pad=8)
    a1.legend(fontsize=7, labelcolor=INK2, ncol=2, loc="lower center")
    a2.fill_between(py.index, py.values, color=BLUE_150, lw=0)
    a2.plot(py.index, py.values, color=BLUE_550, lw=1.2)
    ygrid(a2)
    a2.set_ylabel("mean price")
    a2.set_xlabel("delivery hour")
    a2.set_xticks(range(0, 24, 3))
    save(fig, t, "r4_hour_profile")


# ---- r5: seeds ----------------------------------------------------------------------------------------------------

def r5_seeds():
    rows = []
    for arm in ARMS:
        for k in range(10):
            f = os.path.join(RESULTS, run_name(arm, k) + ".json")
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            rows.append({"arm": arm, "seed": k, "fit": "fine-tuned", "test_mae": r["metrics"]["test"]["all"]["mae"]})
            st = r.get("info", {}).get("static_metrics")
            if st:
                rows.append({"arm": arm, "seed": k, "fit": "fit once", "test_mae": st["test"]["all"]["mae"]})
    if not rows:
        return skip("r5_seeds", "no runs")
    t = pd.DataFrame(rows)
    lear = pd.read_csv(os.path.join(D.OUT, "baselines_mae.csv")).set_index("method").test_mae
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    arms = [a for a in ARMS if a in set(t.arm)]
    for i, fit in enumerate(["fit once", "fine-tuned"]):
        for j, arm in enumerate(arms):
            d = t[(t.arm == arm) & (t.fit == fit)]
            x = i * (len(arms) + 1) + j
            ax.scatter(np.full(len(d), x) + np.linspace(-0.12, 0.12, len(d)), d.test_mae, s=26, color=ARM_COLOR[arm],
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            if len(d) > 1:
                ax.plot([x - 0.3, x + 0.3], [d.test_mae.mean()] * 2, color=ARM_COLOR[arm], lw=1.6)
            ax.text(x, d.test_mae.max() + 0.08, f"n={len(d)}", ha="center", fontsize=6.5, color=MUTED)
    for name, lab in [("LEAR (static fit on train)", "LEAR fit once"), ("LEAR (refit monthly; trailing 730 days)", "LEAR refit monthly")]:
        if name in lear:
            ax.axhline(lear[name], color=ORANGE, lw=0.8, ls=(0, (3, 3)))
            ax.text(len(arms) * 2 + 0.6, lear[name], lab, fontsize=6.8, color=INK2, va="center")
    ax.set_xticks([i * (len(arms) + 1) + (len(arms) - 1) / 2 for i in range(2)], ["fit once", "fine-tuned monthly"])
    ax.set_xlim(-0.7, len(arms) * 2 + 2.4)
    ygrid(ax)
    ax.set_ylabel("test MAE, EUR/MWh")
    ax.set_title("Seed spread per arm (GAT + price curves)", pad=8)
    ax.legend(handles=[D.Line2D([], [], marker="o", ls="", color=ARM_COLOR[a], label=ARMS[a]) for a in arms],
              fontsize=7, labelcolor=INK2, loc="lower left")
    save(fig, t, "r5_seeds")


# ---- r6: attention of the full arm ------------------------------------------------------------------------------

def r6_attention_full():
    f = os.path.join(RUNS, run_name("full"), "attention.json")
    if not os.path.exists(f):
        return skip("r6_attention_full", "no attention.json")
    a = json.load(open(f))
    t = pd.DataFrame(a["by_zone"]).dropna(subset=["neighbour_share_of_cross"])
    t["lift"] = t.neighbour_share_of_cross / t.uniform_neighbour_share
    t = t.sort_values("lift")
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    y = np.arange(len(t))
    ax.hlines(y, t.uniform_neighbour_share, t.neighbour_share_of_cross, color=AXIS, lw=1.2)
    ax.scatter(t.uniform_neighbour_share, y, s=18, color=SURFACE, edgecolor=INK2, linewidth=1.1, zorder=3, label="uniform attention")
    ax.scatter(t.neighbour_share_of_cross, y, s=20, color=np.where(t.lift >= 1, BLUE, ORANGE), zorder=4, edgecolor=SURFACE,
               linewidth=0.8, label="learned attention")
    ax.set_yticks(y, [z.replace("_", "") for z in t.zone], fontsize=6.8)
    for yy, (_, r) in zip(y, t.iterrows()):
        ax.text(max(r.uniform_neighbour_share, r.neighbour_share_of_cross) + 0.008, yy, f"×{r.lift:.2f}", fontsize=6,
                color=MUTED, va="center")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("share of the zone's cross-zone attention (layer 1) on its physical neighbours")
    ax.set_title(f"Full attention barely finds the grid: mean lift ×{a['lift']:.2f} over uniform", pad=8)
    ax.legend(handles=[D.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, label="if attention were uniform"),
                       D.Line2D([], [], marker="o", ls="", color=BLUE, label="learned, above uniform"),
                       D.Line2D([], [], marker="o", ls="", color=ORANGE, label="learned, below uniform")],
              fontsize=7, labelcolor=INK2, loc="lower right")
    ax.text(0, -0.09, f"c3s_gat_full_curves_rolling, seed 0, model after the initial fit; {a['graphs']:,} validation "
            "graphs, mean over heads.", transform=ax.transAxes, fontsize=6.5, color=MUTED)
    save(fig, t, "r6_attention_full")


# ---- r7: per-zone MAE ------------------------------------------------------------------------------------------------

def r7_zone_mae():
    zt = zone_table()
    if zt is None:
        return skip("r7_zone_mae", "no common seeds")
    t, seeds = zt
    t = t.dropna(subset=["mae_a"]).sort_values("mae_a")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 7.8), sharey=True, gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.06})
    y = np.arange(len(t))
    a1.hlines(y, t[["mae_a", "mae_b"]].min(1), t[["mae_a", "mae_b"]].max(1), color=AXIS, lw=1.2)
    a1.scatter(t.mae_b, y, s=16, color=SURFACE, edgecolor=INK2, linewidth=1.1, zorder=3)
    a1.scatter(t.mae_a, y, s=18, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=4)
    a1.set_yticks(y, [z.replace("_", "") for z in t.zone], fontsize=6.8)
    core = set(D.CORE)
    for lab in a1.get_yticklabels():
        if lab.get_text() in {c.replace("_", "") for c in core}:
            lab.set_fontweight("bold")
    a1.grid(axis="x")
    a1.set_axisbelow(True)
    a1.set_xlabel("test MAE, EUR/MWh")
    a1.set_title("Per-zone test MAE (Core zones in bold)", pad=8)
    a1.legend(handles=[D.Line2D([], [], marker="o", ls="", color=BLUE, label="interconnectors"),
                       D.Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, label="no graph")],
              fontsize=7, labelcolor=INK2, loc="lower right")
    a2.barh(y, t.price_std, color=BLUE_150, height=0.7)
    a2.grid(axis="x")
    a2.set_axisbelow(True)
    a2.set_xlabel("intraday price std")
    a2.set_title("Scale", pad=8)
    a1.text(0, -0.075, f"Both fine-tuned monthly with price curves; mean over seeds {', '.join(map(str, seeds))}.",
            transform=a1.transAxes, fontsize=6.5, color=MUTED)
    save(fig, t, "r7_zone_mae")


# ---- r8: H2 constraint attention ------------------------------------------------------------------------------

H2_RUN = "c4s_gat_interconnects_constraints_rolling"


def constraint_attention(run: str = H2_RUN, every: int = 7, weights: str = "model.pt") -> pd.DataFrame | None:
    """One row per (test day-slot, Core zone, exposed constraint): alpha (mean over heads), the uniform
    weight 1/n_exposed, RAM/Fmax, (RAM-AMR)/Fmax, AMR > 0, |PTDF|. `weights`: model.pt (after the initial
    fit) or model_rolled.pt (after the last monthly fine-tune)."""
    import torch
    import yaml
    from epgnn import train as T
    from epgnn.models import gat_constraints as GC
    from epgnn.models import gat_scope as G
    from epgnn.pipeline import graphs
    wf = os.path.join(RUNS, run, weights)
    if not os.path.exists(wf):
        return None
    rf = os.path.join(RESULTS, f"{run}.json")
    cfg = json.load(open(rf))["config"] if os.path.exists(rf) else \
        yaml.safe_load(open(os.path.join(D.FW, "configs", "experiments", f"{run}.yaml")))
    zt, data, hc, dc = G.prepare(cfg, cfg["fit_splits"])
    ext = GC.Extension(cfg)
    dc = ext.prepare(cfg, zt, data, dc, None)
    zones = zt.zones
    torch.manual_seed(0)                                            # the untrained reference: same seed as the run
    m = ext.build(len(hc), len(dc), graphs.edges_for(cfg.get("graph", "interconnects"), zones), len(zones),
                  graphs.laplacian_pe(zones, cfg.get("pe_k", 8)) if cfg.get("positional", True) else None,
                  hidden=cfg.get("hidden", 64), heads=cfg.get("heads", 4), layers=cfg.get("layers", 2),
                  ffn=cfg.get("ffn", 128), mlp_hidden=cfg.get("mlp_hidden", 128), dropout=cfg.get("dropout", 0.1))
    dev = T.device(cfg.get("device"))
    core_names = [zones[i] for i in m.core_zone.cpu().numpy()]
    days = zt.day_indices("test")[::every]
    out = []
    for label in ("untrained", "trained"):
        if label == "trained":
            m.load_state_dict(torch.load(wf, map_location="cpu"))
        m.to(dev).eval()
        out.append(_collect(m, data, days, cfg, dev, core_names).assign(model=label))
    t = pd.concat(out, ignore_index=True)
    t["lift"] = t.alpha * t.n_exposed
    return t


def _collect(m, data, days, cfg, dev, core_names) -> pd.DataFrame:
    import torch
    from epgnn.models import gat_scope as G
    rows, offset = [], 0
    with torch.no_grad():
        for b in G.GatherBatches(data, days, 8, cfg.get("history_window", 168), shuffle=False):
            row = b["xday"][:, 0, :, 0].round().long().reshape(-1)
            m(b["hist"].to(dev), b["xday"].to(dev), None)
            if m.last_attention is None:
                continue
            alpha, gate = (a.cpu() for a in m.last_attention)               # (n,1,C,Zc,nh), (n,1,Zc)
            r = row[row > 0]
            alpha = alpha[:, 0].mean(-1)                                    # (n, C, Zc)
            x = m.cx[r].float().cpu()                                       # (n, C, 8)
            p = m.cp[r].float().cpu()[..., m.core_col.cpu()]                # (n, C, Zc)
            ok = (m.cv[r].cpu()[..., None] & (p.abs() >= m.ptdf_min)) & gate[:, 0][:, None, :]
            n_exp = ok.sum(1, keepdim=True).expand_as(ok)
            ni, ci, zi = torch.nonzero(ok, as_tuple=True)
            rows.append(pd.DataFrame({
                "slot": (ni + offset).numpy(),
                "zone": np.array(core_names)[zi.numpy()],
                "alpha": alpha[ni, ci, zi].numpy(), "n_exposed": n_exp[ni, ci, zi].numpy(),
                "ram_fmax": x[ni, ci, 0].numpy(), "ram_grid_fmax": x[ni, ci, 1].numpy(),
                "amr_lifted": (x[ni, ci, 2] > 0).numpy(), "abs_ptdf": p[ni, ci, zi].abs().numpy()}))
            offset += len(r)
            m.last_attention = None
    return pd.concat(rows, ignore_index=True)


def r8_constraint_attention(run: str = H2_RUN, out_name: str = "r8_constraint_attention"):
    t = constraint_attention(run)
    if t is None:
        return skip(out_name, f"no runs/{run}/model.pt yet")
    fig = plt.figure(figsize=(7.4, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.42, wspace=0.28)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    # (a) lift by RAM/Fmax, split by whether the minimum-RAM uplift applied
    edges = [-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf]
    mids = ["<0.1"] + [f"{a:.1f}" for a in edges[1:-2]] + [">1"]
    t["ram_bin"] = pd.cut(t.ram_fmax, edges, labels=mids)
    tab_a = t.groupby(["model", "ram_bin", "amr_lifted"], observed=True).agg(lift=("lift", "mean"), n=("lift", "size")).reset_index()
    for model, lifted, col, ls, lab in [("trained", False, BLUE, "-", "trained, no AMR uplift"),
                                        ("trained", True, ORANGE, "-", "trained, RAM lifted by AMR"),
                                        ("untrained", False, MUTED, (0, (3, 2)), "untrained (reference)")]:
        d = tab_a[(tab_a.model == model) & (tab_a.amr_lifted == lifted) & (tab_a.n >= 50)]
        a1.plot([mids.index(b) for b in d.ram_bin], d.lift, color=col, ls=ls, marker="o", ms=3 if model == "trained" else 0,
                lw=1.5 if model == "trained" else 1.1, label=lab)
    a1.axhline(1, color=AXIS, lw=0.9, ls=(0, (3, 3)))
    a1.set_xticks(range(len(mids))[::2], mids[::2])
    a1.set_xlabel("RAM / Fmax of the constraint (D-1, same hour)")
    a1.set_ylabel("attention lift (1 = uniform)")
    a1.set_title("Tight lines get more attention?", fontsize=8.8)
    ygrid(a1)
    a1.legend(fontsize=6.6, labelcolor=INK2)
    # (b) lift by |PTDF|
    pe = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, np.inf]
    pl = [".05", ".075", ".10", ".15", ".20", ">.3"]
    t["p_bin"] = pd.cut(t.abs_ptdf, pe, labels=pl, right=False)
    tab_b = t.groupby(["model", "p_bin"], observed=True).agg(lift=("lift", "mean"), n=("lift", "size")).reset_index()
    for model, col, ls in [("trained", BLUE, "-"), ("untrained", MUTED, (0, (3, 2)))]:
        d = tab_b[tab_b.model == model]
        a2.plot([pl.index(b) for b in d.p_bin.astype(str)], d.lift, color=col, ls=ls, marker="o",
                ms=3 if model == "trained" else 0, lw=1.5 if model == "trained" else 1.1)
    a2.axhline(1, color=AXIS, lw=0.9, ls=(0, (3, 3)))
    a2.set_xticks(range(len(pl)), pl)
    a2.set_xlabel("|zone-to-rest PTDF| (lower edge of bin)")
    a2.set_title("Strongly coupled lines get more attention?", fontsize=8.8)
    ygrid(a2)
    # (c) per zone: mass on the tightest tenth of the exposed constraints
    t = t[t.model == "trained"].copy()
    t["rank"] = t.groupby(["slot", "zone"]).ram_fmax.rank(method="first")
    t["tight10"] = t["rank"] <= np.ceil(0.1 * t.n_exposed)
    per = t[t.tight10].groupby(["slot", "zone"]).agg(mass=("alpha", "sum"), k=("alpha", "size"),
                                                     n=("n_exposed", "first")).reset_index()
    per["uniform"] = per.k / per.n
    z = per.groupby("zone")[["mass", "uniform"]].mean().sort_values("mass")
    y = np.arange(len(z))
    a3.hlines(y, z.uniform, z.mass, color=AXIS, lw=1.2)
    a3.scatter(z.uniform, y, s=18, color=SURFACE, edgecolor=INK2, linewidth=1.1, zorder=3)
    a3.scatter(z.mass, y, s=20, color=np.where(z.mass >= z.uniform, BLUE, ORANGE), edgecolor=SURFACE, linewidth=0.8, zorder=4)
    a3.set_yticks(y, [q.replace("_", "") for q in z.index], fontsize=7)
    a3.grid(axis="x")
    a3.set_axisbelow(True)
    a3.set_xlabel("attention mass on the zone's tightest 10 % of exposed constraints (by RAM/Fmax)")
    a3.set_title("Per Core zone: learned (filled) vs uniform (open)", fontsize=8.8)
    fig.suptitle("H2: what the constraint nodes attend to (D-1 flow-based domain)", x=0.07, ha="left",
                 fontsize=9.5, fontweight="bold", y=0.97)
    fig.text(0.07, 0.015, f"{run}, weights after the initial fit; every 7th test day, {t.slot.nunique():,} day-slots, "
             f"{len(t):,} (zone, constraint) pairs, mean over heads. JAO shadow prices (which constraints bound) "
             "would be the proper label; this uses RAM/Fmax as its proxy.", fontsize=6.3, color=MUTED, wrap=True)
    save(fig, pd.concat([tab_a.assign(panel="a_ram"), tab_b.assign(panel="b_ptdf"),
                         z.reset_index().assign(panel="c_zone")], ignore_index=True), out_name)


FIGURES = [r1_monthly_mae, r2_zone_gain_map, r3_bootstrap, r4_hour_profile, r5_seeds, r6_attention_full, r7_zone_mae, r8_constraint_attention]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in FIGURES:
        if not want or any(f.__name__.startswith(w) for w in want):
            f()
