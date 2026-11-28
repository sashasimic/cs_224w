#!/usr/bin/env python3
"""Time-series figures of the JAO Core flow-based data (9 Jun 2022 - 29 Oct 2026), in the style of
make_data_figures.py. Each figure is written as PNG + PDF + CSV of the plotted values to figures/data/.

  j1_domain_margins     Core-wide, daily: RAM/Fmax (median and 10th percentile), share of constraint-hours
                        that are tight (RAM/Fmax < 0.2), share lifted by the minimum-RAM uplift (AMR), the
                        median minimum-RAM factor (the 70 % rule) and the median loop flow F0core/Fmax
  j2_zone_exposure      per Core zone, weekly: number of exposed constraints (|zone-to-rest PTDF| >= 0.05) and
                        the PTDF-weighted tightness (jao_n_exposed, jao_tightness in zone_features_hourly)
  j3_maxbex             per Core border, weekly mean MaxBex in both directions (the published trade capacity)
  j4_border_use         per Core border and month: scheduled exchange / MaxBex (how much of the capacity
                        the market used), as a heat map
  j5_domain_vs_spreads  daily Core price dispersion (max - min of the 12 zones' daily mean prices) against the
                        Core mean PTDF-weighted tightness: when the domain is tight, Core prices separate

Usage (from the project folder):
    python figures/make_jao_figures.py           # all
    python figures/make_jao_figures.py j1 j3     # some (prefix match)

Inputs: cs_224w-c1_xgb_zone/features/v1/{constraints_hourly,zone_features_hourly}.parquet and
data/model/{edges,nodes}_hourly.parquet. Peak memory about 1.5 GB.
"""
import os
import sys

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_data_figures as D  # noqa: E402  (style, paths, save, split_marks)

TIGHT = 0.2                     # RAM/Fmax below this counts as a tight constraint
CORE_START, CORE_END = pd.Timestamp("2022-06-09"), pd.Timestamp("2026-10-29")
ZONE_ORDER = ["DE_LU", "FR", "NL", "BE", "AT", "PL", "CZ", "SK", "HU", "SI", "HR", "RO"]


def _day(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts.dt.tz_convert("Europe/Brussels").dt.date)


def _time_axis(ax, lo=CORE_START, hi=CORE_END, label=False):
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(mdates.YearLocator(1, month=1, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    D.split_marks(ax, label=label, lo=lo, hi=hi)


def _band(ax, s: pd.Series, color, label, roll=28, lw_raw=0.5):
    ax.plot(s.index, s, color=color, lw=lw_raw, alpha=0.45)
    r = s.rolling(roll, center=True, min_periods=7).median()
    ax.plot(r.index, r, color=color, lw=1.8, label=label)
    return r


# ---- j1 --------------------------------------------------------------------------------------------------

def j1_domain_margins():
    h = pd.read_parquet(os.path.join(D.FEAT, "constraints_hourly.parquet"),
                        columns=["ts_utc", "ram_fmax", "amr_fmax", "min_ram_factor", "f0core", "fmax"])
    h["day"] = _day(h.ts_utc)
    h["tight"] = h.ram_fmax < TIGHT
    h["lifted"] = h.amr_fmax > 0
    h["loop"] = (h.f0core / h.fmax.where(h.fmax > 0)).clip(-2, 2)
    g = h.groupby("day")
    t = pd.DataFrame({"ram_median": g.ram_fmax.median(), "ram_p10": g.ram_fmax.quantile(0.10),
                      "tight_share": g.tight.mean(), "lifted_share": g.lifted.mean(),
                      "minram_median": g.min_ram_factor.median(), "loop_median": g.loop.median(),
                      "loop_p90": g.loop.quantile(0.90)})
    del h
    fig, axes = plt.subplots(4, 1, figsize=(11, 9.2), sharex=True, gridspec_kw={"hspace": 0.32})
    a = axes[0]
    _band(a, t.ram_median, D.BLUE, "median")
    _band(a, t.ram_p10, D.ORANGE, "10th percentile")
    a.axhline(TIGHT, color=D.MUTED, lw=0.7, ls=(0, (2, 2)))
    a.text(CORE_START, TIGHT, f"  tight < {TIGHT}", va="bottom", fontsize=7, color=D.INK2)
    a.set_ylabel("RAM / Fmax")
    a.set_title("Remaining margin of the presolved constraints (daily, 28-day median bold)", pad=16)
    a.legend(fontsize=7, labelcolor=D.INK2, loc="upper left", ncol=2)
    a = axes[1]
    _band(a, 100 * t.tight_share, D.ORANGE, f"RAM/Fmax < {TIGHT}")
    _band(a, 100 * t.lifted_share, D.AQUA, "RAM lifted by minimum-RAM uplift (AMR > 0)")
    a.set_ylabel("% of constraint-hours")
    a.set_title("Tight and policy-lifted constraints")
    a.legend(fontsize=7, labelcolor=D.INK2, loc="upper left")
    a = axes[2]
    a.plot(t.index, 100 * t.minram_median, color=D.BLUE, lw=1.4)
    a.axhline(70, color=D.MUTED, lw=0.7, ls=(0, (2, 2)))
    a.text(CORE_START, 70, "  70 % target (EU 2019/943)", va="bottom", fontsize=7, color=D.INK2)
    a.set_ylabel("min-RAM factor, %")
    a.set_title("Minimum RAM the TSOs had to offer (median over constraints)")
    a = axes[3]
    _band(a, t.loop_median, D.BLUE, "median")
    _band(a, t.loop_p90, D.ORANGE, "90th percentile")
    a.axhline(0, color=D.AXIS, lw=0.7)
    a.set_ylabel("F0core / Fmax")
    a.set_title("Loop and transit flow already on the lines before Core trading")
    a.legend(fontsize=7, labelcolor=D.INK2, loc="upper left", ncol=2)
    for k, a in enumerate(axes):
        D.ygrid(a)
        _time_axis(a, label=(k == 0))
    D.save(fig, t.reset_index().rename(columns={"index": "day"}), "j1_domain_margins")


# ---- j2 --------------------------------------------------------------------------------------------------

def j2_zone_exposure():
    z = pd.read_parquet(os.path.join(D.FEAT, "zone_features_hourly.parquet"),
                        columns=["ts_utc", "zone", "jao_n_exposed", "jao_tightness"],
                        filters=[("zone", "in", D.CORE)])
    z = z[z.ts_utc >= pd.Timestamp(CORE_START, tz="UTC")]
    z["week"] = z.ts_utc.dt.tz_convert(None).dt.to_period("W").dt.start_time
    w = z.groupby(["zone", "week"])[["jao_n_exposed", "jao_tightness"]].mean().reset_index()
    core = w.groupby("week")[["jao_n_exposed", "jao_tightness"]].mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"hspace": 0.28})
    colors = plt.get_cmap("tab20").colors
    for k, (col, title, ylab) in enumerate([
            ("jao_n_exposed", "Constraints each Core zone is exposed to (|zone-to-rest PTDF| >= 0.05), weekly mean",
             "constraints per hour"),
            ("jao_tightness", "PTDF-weighted tightness of those constraints (0 = all slack, 1 = no margin), weekly mean",
             "tightness")]):
        a = axes[k]
        for i, zn in enumerate(ZONE_ORDER):
            s = w[w.zone == zn].set_index("week")[col]
            a.plot(s.index, s, lw=0.9, color=colors[(2 * i) % 20 if i < 10 else (2 * i + 1) % 20], label=zn)
        a.plot(core.index, core[col], color=D.INK, lw=2.0, label="Core mean")
        a.set_ylabel(ylab)
        a.set_title(title, pad=16 if k == 0 else 6)
        D.ygrid(a)
        _time_axis(a, label=(k == 0))
    axes[0].legend(fontsize=7, labelcolor=D.INK2, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    D.save(fig, w, "j2_zone_exposure")


# ---- j3 / j4 helpers -------------------------------------------------------------------------------------

def _core_borders() -> pd.DataFrame:
    e = pd.read_parquet(D.EDGES, columns=["ts_utc", "from_zone", "to_zone", "maxbex", "flow_sched"],
                        filters=[("from_zone", "in", D.CORE), ("to_zone", "in", D.CORE)])
    e = e[e.maxbex.notna() | e.flow_sched.notna()]
    return e


def j3_maxbex():
    e = _core_borders()
    e["week"] = e.ts_utc.dt.tz_convert(None).dt.to_period("W").dt.start_time
    w = e.groupby(["from_zone", "to_zone", "week"]).maxbex.mean().reset_index()
    pairs = (e.groupby(["from_zone", "to_zone"]).maxbex.mean().reset_index())
    pairs["link"] = pairs.apply(lambda r: tuple(sorted((r.from_zone, r.to_zone))), axis=1)
    links = pairs.groupby("link").maxbex.mean().sort_values(ascending=False).index.tolist()
    n = len(links)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.85 * rows), sharex=True, gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    axes = np.atleast_2d(axes)
    for i, (a_, b_) in enumerate(links):
        ax = axes.flat[i]
        for (f, t_), c in (((a_, b_), D.BLUE), ((b_, a_), D.ORANGE)):
            s = w[(w.from_zone == f) & (w.to_zone == t_)].set_index("week").maxbex
            ax.plot(s.index, s / 1000, color=c, lw=1.0, label=f"{f} -> {t_}")
        ax.set_title(f"{a_} - {b_}", fontsize=8.5, pad=3)
        ax.legend(fontsize=6, labelcolor=D.INK2, loc="lower left", handlelength=1.2, frameon=True,
                  facecolor=D.SURFACE, edgecolor="none", framealpha=0.85, borderpad=0.2)
        D.ygrid(ax)
        ax.set_xlim(CORE_START, CORE_END)
        ax.xaxis.set_major_locator(mdates.YearLocator(1, month=1, day=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
        ax.tick_params(labelsize=7)
        if i % cols == 0:
            ax.set_ylabel("GW", fontsize=7.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("MaxBex on the Core borders (published maximum bilateral exchange), weekly mean, both directions",
                 x=0.07, ha="left", y=0.995, fontsize=10, weight="bold")
    fig.subplots_adjust(top=0.95)
    D.save(fig, w, "j3_maxbex")


def j4_border_use():
    e = _core_borders()
    e = e[e.maxbex > 0]
    e["month"] = e.ts_utc.dt.tz_convert(None).dt.to_period("M").dt.start_time
    e["use"] = (e.flow_sched.clip(lower=0) / e.maxbex).clip(upper=1.2)
    m = e.groupby(["from_zone", "to_zone", "month"]).use.mean().reset_index()
    m["pair"] = m.from_zone + " -> " + m.to_zone
    order = m.groupby("pair").use.mean().sort_values(ascending=False).index
    tab = m.pivot(index="pair", columns="month", values="use").reindex(order)
    tab = tab[[c for c in tab.columns if tab[c].notna().sum() > 0]]
    tab = tab[tab.notna().sum(axis=1) > 0]                             # pairs with MaxBex but no scheduled flow
    fig, ax = plt.subplots(figsize=(11.5, 0.2 * len(tab) + 1.6))
    im = ax.imshow(tab.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(tab)), tab.index, fontsize=6.5)
    months = pd.DatetimeIndex(tab.columns)
    ticks = [i for i, d in enumerate(months) if d.month in (1, 7)]
    ax.set_xticks(ticks, [months[i].strftime("%Y-%m") for i in ticks], fontsize=7)
    for name, (a_, _) in D.SPLITS.items():
        if name in ("val", "test"):
            i = np.searchsorted(months, pd.Timestamp(a_))
            ax.axvline(i - 0.5, color=D.ORANGE, lw=1.0)
            ax.text(i - 0.4, -0.8, name, fontsize=7.5, color=D.INK2, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("scheduled exchange / MaxBex", fontsize=7.5)
    ax.set_title("How much of each border's trade capacity the market used (monthly mean of flow / MaxBex, "
                 "directions with flow > 0)", pad=14)
    note = ("Scheduled day-ahead exchanges are published for these 38 directed Core pairs only (pairs without them are left out). "
            "Values near 1: the border is often at its MaxBex.")
    ax.text(0, -0.09, note, transform=ax.transAxes, fontsize=7, color=D.INK2)
    D.save(fig, m, "j4_border_use")


# ---- j5 --------------------------------------------------------------------------------------------------

def j5_domain_vs_spreads():
    z = pd.read_parquet(os.path.join(D.FEAT, "zone_features_hourly.parquet"),
                        columns=["delivery_date", "zone", "jao_tightness"], filters=[("zone", "in", D.CORE)])
    tight = z.groupby(pd.to_datetime(z.delivery_date)).jao_tightness.mean().dropna()
    del z
    n = D.nodes(["delivery_date", "zone", "price", "has_price"], filters=[("zone", "in", D.CORE)])
    n = n[n.has_price]
    n["day"] = pd.to_datetime(n.delivery_date)
    n = n[(n.day >= CORE_START) & (n.day <= CORE_END)]
    daily = n.groupby(["day", "zone"]).price.mean().unstack()
    disp = (daily.max(axis=1) - daily.min(axis=1)).rename("dispersion")
    t = pd.concat([tight.rename("core_tightness"), disp], axis=1).dropna()
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"hspace": 0.3})
    _band(axes[0], t.dispersion, D.ORANGE, "28-day median")
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title("Core price dispersion: max - min of the 12 zones' daily mean prices", pad=16)
    _band(axes[1], t.core_tightness, D.BLUE, "28-day median")
    axes[1].set_ylabel("tightness (0-1)")
    axes[1].set_title("Core mean of the zones' PTDF-weighted constraint tightness (same day's domain)")
    for k, a in enumerate(axes):
        D.ygrid(a)
        _time_axis(a, label=(k == 0))
    r = t.rolling(28, center=True, min_periods=7).median().dropna()
    rho = r.corr(method="spearman").iloc[0, 1]
    rho_raw = t.corr(method="spearman").iloc[0, 1]
    axes[1].text(1.0, -0.2, f"Spearman correlation, daily: {rho_raw:.2f};  28-day medians: {rho:.2f}", ha="right",
                 transform=axes[1].transAxes, fontsize=7.5, color=D.INK2)
    D.save(fig, t.reset_index().rename(columns={"index": "day"}), "j5_domain_vs_spreads")


FIGS = {"j1": j1_domain_margins, "j2": j2_zone_exposure, "j3": j3_maxbex, "j4": j4_border_use,
        "j5": j5_domain_vs_spreads}

if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for k, f in FIGS.items():
        if any(k.startswith(w) or w.startswith(k) for w in want):
            f()
