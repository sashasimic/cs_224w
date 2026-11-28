#!/usr/bin/env python3
"""Data figures for the European day-ahead price GNN: the grid, the data, and the baselines.

Each figure is written as PNG (200 dpi) + PDF + a CSV of the plotted values to figures/data/.

  d1_zone_map           the 45 bidding zones and 90 interconnectors on a map of Europe; Core (flow-based)
                        zones shaded; line width = mean |net physical flow| over the test split
  d2_availability       share of hours with data per zone and month, for price, load forecast, weather,
                        physical flows and the JAO Core domain; split boundaries marked
  d3_price_regimes      monthly mean price of every zone, 2016-2026, with the cross-zone median: the 2021-23
                        crisis, and where the splits fall relative to it
  d4_price_by_zone      test-split price distribution per zone (median, IQR, 5-95 %) and the share of
                        negative-price hours
  d5_spiky_vs_smooth    why prices are hard: a sample fortnight of price vs load forecast, temperature and
                        wind (DE_LU), autocorrelation by variable, and heavy tails of hour-to-hour changes
  d6_constraints        JAO Core domain: presolved constraints per hour over time, RAM/Fmax (with the
                        minimum-RAM uplift where the table has it), and how many days a constraint lives
  d7_baselines          val/test MAE of the naive baselines, LEAR and the trained models

Usage (from the project folder):
    python figures/make_data_figures.py               # all figures
    python figures/make_data_figures.py d1 d5         # some of them (prefix match)

Inputs: data/model/{nodes,edges}_hourly.parquet, cs_224w-c1_xgb_zone/features/v1/{graph.json,
constraints_hourly.parquet, constraint_catalog.parquet}, cs_224w-c1_xgb_zone/results/*.json,
figures/data/baselines_mae.csv, and the zone shapes used by weather/make_maps.py (entsoe-py geojson +
Natural Earth 1:50m in weather/data/shapes/). Peak memory is about 1.5 GB (d2).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "data")
NODES = os.path.join(ROOT, "data", "model", "nodes_hourly.parquet")
EDGES = os.path.join(ROOT, "data", "model", "edges_hourly.parquet")
FW = ROOT if os.path.isdir(os.path.join(ROOT, "epgnn")) else os.path.join(ROOT, "cs_224w-c1_xgb_zone")  # repo root or local layout
FEAT = os.path.join(FW, "features", "v1")
SHAPES_NE = os.path.join(ROOT, "weather", "data", "shapes", "ne_50m_admin_0_map_units.zip")

# ---- style: the reference palette (light), same as weather/make_figures.py -------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots 1-4, fixed order
BLUE_150, BLUE_250, BLUE_550, BLUE_700 = "#b7d3f6", "#86b6ef", "#1c5cab", "#0d366b"
CORE = ["AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]
SPLITS = json.load(open(os.path.join(FW, "configs", "splits.json")))["splits"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2.0,
})


def save(fig, table: pd.DataFrame, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    table.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    plt.close(fig)
    print(f"  {name}", flush=True)


def ygrid(ax):
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def split_marks(ax, label: bool = True, y: float = 1.0, lo=None, hi=None) -> None:
    """Dashed hairlines at the split boundaries, named at the top of the axes (only the part of each split
    inside [lo, hi] is labelled, so a split outside the plotted range gets no label)."""
    lo, hi = pd.Timestamp(lo or "1900-01-01"), pd.Timestamp(hi or "2100-01-01")
    for name, (a, b) in SPLITS.items():
        if name != "pretrain":
            ax.axvline(pd.Timestamp(a), color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=1)
        a, b = max(pd.Timestamp(a), lo), min(pd.Timestamp(b), hi)
        if label and a < b:
            mid = a + (b - a) / 2
            ax.text(mid, y, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, color=INK2)


def nodes(cols: list[str], **kw) -> pd.DataFrame:
    return pd.read_parquet(NODES, columns=cols, **kw)


# ---- d1: map ---------------------------------------------------------------------------------------------

def zone_shapes():
    """Model-zone polygons in EPSG:3035: entsoe-py bidding zones, plus Natural Earth outlines for the
    zones entsoe-py lacks (GB, IE_SEM = Ireland + Northern Ireland, BA, ME, MK). Also returns the
    country outlines used as the background."""
    import entsoe
    import geopandas as gpd
    geo = os.path.join(os.path.dirname(entsoe.__file__), "geo", "geojson")
    zl = json.load(open(os.path.join(FEAT, "graph.json")))["zones"]
    parts = []
    for z in zl:
        f = os.path.join(geo, f"{z}.geojson")
        if os.path.exists(f):
            g = gpd.read_file(f)[["geometry"]]
            g["zone"] = z
            parts.append(g.set_crs(4326, allow_override=True))
    ne = gpd.read_file(SHAPES_NE).to_crs(4326)
    extra = {"GB": ["ENG", "SCT", "WLS"], "IE_SEM": ["IRL", "NIR"], "BA": ["BIS", "BHF"], "ME": ["MNE"], "MK": ["MKD"]}
    for z, units in extra.items():
        g = ne[ne.GU_A3.isin(units)].dissolve()[["geometry"]]
        g["zone"] = z
        parts.append(g)
    zs = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326).dissolve(by="zone").reset_index()
    return zs.to_crs("EPSG:3035"), ne.to_crs("EPSG:3035")


# label/anchor nudges (km in EPSG:3035) where a representative point sits badly for a line hub
ANCHOR_SHIFT = {"NO_4": (60, -150), "SE_1": (-40, -60), "SE_2": (-60, 0), "NO_3": (30, -30), "GR": (-40, 40),
                "IT_SARD": (0, 0), "DK_2": (20, 0), "FI": (-60, -120), "NO_1": (0, -20), "HR": (-60, 30)}


def d1_zone_map():
    zs, ne = zone_shapes()
    g = json.load(open(os.path.join(FEAT, "graph.json")))
    zl = g["zones"]
    pts = zs.set_index("zone").geometry.representative_point()
    xy = {z: (p.x + ANCHOR_SHIFT.get(z, (0, 0))[0] * 1e3, p.y + ANCHOR_SHIFT.get(z, (0, 0))[1] * 1e3) for z, p in pts.items()}

    a, b = SPLITS["test"]
    e = pd.read_parquet(EDGES, columns=["from_zone", "to_zone", "flow_phys"], filters=[("split", "=", "test")])
    m = e.groupby(["from_zone", "to_zone"]).flow_phys.mean()
    links = []
    for i, j in zip(*g["spread_pairs"]):
        u, v = zl[i], zl[j]
        net = abs(np.nan_to_num(m.get((u, v), np.nan)) - np.nan_to_num(m.get((v, u), np.nan)))
        has = np.isfinite(m.get((u, v), np.nan)) or np.isfinite(m.get((v, u), np.nan))
        links.append({"a": u, "b": v, "core": u in CORE and v in CORE, "mean_abs_net_flow_mw": net if has else np.nan})
    lk = pd.DataFrame(links)

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ne.plot(ax=ax, color=NEUTRAL, edgecolor=SURFACE, linewidth=0.4)
    zs["kind"] = np.where(zs.zone.isin(CORE), "core", np.where(zs.zone == "BA", "noprice", "other"))
    zs[zs.kind == "other"].plot(ax=ax, color="#e1e0d9", edgecolor=SURFACE, linewidth=0.6)
    zs[zs.kind == "core"].plot(ax=ax, color=BLUE_150, edgecolor=SURFACE, linewidth=0.6)
    zs[zs.kind == "noprice"].plot(ax=ax, color=NEUTRAL, edgecolor=AXIS, linewidth=0.6, hatch="////")
    wmax = np.nanmax(lk.mean_abs_net_flow_mw)
    for r in lk.itertuples():
        (x0, y0), (x1, y1) = xy[r.a], xy[r.b]
        w = 0.6 + 3.4 * (r.mean_abs_net_flow_mw / wmax if np.isfinite(r.mean_abs_net_flow_mw) else 0)
        ax.plot([x0, x1], [y0, y1], color=BLUE_550 if r.core else INK2, lw=w, solid_capstyle="round", zorder=3,
                alpha=0.95, ls="-" if np.isfinite(r.mean_abs_net_flow_mw) else (0, (2, 2)))
    for z, (x, y) in xy.items():
        ax.scatter([x], [y], s=16, color=SURFACE, edgecolor=INK, linewidth=0.8, zorder=4)
        ax.annotate(z.replace("_", ""), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6.2,
                    color=INK, zorder=5)
    ax.set_xlim(2.55e6, 6.45e6)
    ax.set_ylim(1.35e6, 5.45e6)
    ax.set_axis_off()
    ax.set_title(f"45 bidding zones, {len(lk)} interconnectors", loc="left", pad=8)
    ref = [500, 2000, 4000]
    leg = [Patch(facecolor=BLUE_150, edgecolor="none", label="Core flow-based region (12 zones)"),
           Patch(facecolor="#e1e0d9", edgecolor="none", label="other modelled zone"),
           Patch(facecolor=NEUTRAL, edgecolor=AXIS, hatch="////", label="no day-ahead price (BA)")]
    leg += [Line2D([], [], color=BLUE_550, lw=2, label="Core-Core border"), Line2D([], [], color=INK2, lw=2, label="other border"),
            Line2D([], [], color=INK2, lw=0.8, ls=(0, (2, 2)), label="no flow data in test split")]
    leg += [Line2D([], [], color=MUTED, lw=0.6 + 3.4 * v / wmax, label=f"{v:,} MW mean |net flow|") for v in ref]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.0, 0.985), fontsize=6.8, handlelength=2.2,
              labelcolor=INK2)
    ax.text(0, -0.01, f"Line width: mean |net physical flow| over the test split ({a} to {b}). Lines join "
            "representative points of the zones, not the actual cable routes.", transform=ax.transAxes,
            fontsize=6.5, color=MUTED, va="top")
    save(fig, lk, "d1_zone_map")


# ---- d2: availability --------------------------------------------------------------------------------------

def d2_availability():
    n = nodes(["zone", "delivery_date", "has_price", "load_fc", "temp_pw", "net_import", "n_presolved", "in_core"])
    n["month"] = pd.to_datetime(n.delivery_date.str.slice(0, 7))
    src = {"Day-ahead price (target)": n.has_price.astype(float),
           "Load forecast": n.load_fc.notna().astype(float),
           "Weather (temperature)": n.temp_pw.notna().astype(float),
           "Physical flows (net import)": n.net_import.notna().astype(float),
           "JAO domain (Core zones only)": n.n_presolved.notna().astype(float).where(n.in_core)}
    zones = sorted(n.zone.unique())
    order = [z for z in CORE] + [z for z in zones if z not in CORE]
    months = pd.DatetimeIndex(sorted(n.month.unique()))
    tabs = []
    fig, axes = plt.subplots(1, len(src), figsize=(13.5, 6.4), sharey=True)
    cmap = ListedColormap([NEUTRAL, BLUE_250, BLUE_550])
    for ax, (name, v) in zip(axes, src.items()):
        t = v.groupby([n.zone, n.month]).mean().unstack().reindex(index=order, columns=months)
        tabs.append(t.stack(future_stack=True).rename("share").reset_index().assign(source=name))
        cls = np.where(t.isna(), np.nan, np.where(t >= 0.99, 2, np.where(t > 0.01, 1, 0)))
        ax.imshow(cls, aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                  extent=(matplotlib.dates.date2num(months[0]), matplotlib.dates.date2num(months[-1] + pd.offsets.MonthEnd(1)),
                          len(order), 0))
        ax.xaxis_date()
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(4, month=1, day=1))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        for nm in ("train", "val", "test"):
            ax.axvline(matplotlib.dates.date2num(pd.Timestamp(SPLITS[nm][0])), color=INK, lw=0.6, ls=(0, (3, 3)))
        ax.axhline(len(CORE), color=SURFACE, lw=2)
        ax.set_title(name, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(np.arange(len(order)) + 0.5, [z.replace("_", " ") for z in order], fontsize=6.3)
    axes[0].text(-0.32, len(CORE) / 2, "Core", rotation=90, va="center", ha="center", transform=axes[0].get_yaxis_transform(),
                 fontsize=7.5, color=INK2)
    fig.legend(handles=[Patch(color=NEUTRAL, label="no data"), Patch(color=BLUE_250, label="partial month"),
                        Patch(color=BLUE_550, label="complete (≥ 99 % of hours)"),
                        Line2D([], [], color=INK, lw=0.6, ls=(0, (3, 3)), label="start of train / val / test")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=7.5, labelcolor=INK2)
    fig.suptitle("Data availability per zone and month (share of hours with a value)", x=0.06, ha="left",
                 fontsize=10, weight="bold")
    fig.subplots_adjust(wspace=0.06, bottom=0.12)
    save(fig, pd.concat(tabs, ignore_index=True), "d2_availability")


# ---- d3, d4: prices -----------------------------------------------------------------------------------------

def d3_price_regimes():
    n = nodes(["zone", "delivery_date", "price", "has_price"])
    n = n[n.has_price]
    n["month"] = pd.to_datetime(n.delivery_date.str.slice(0, 7))
    m = n.groupby(["zone", "month"]).price.mean().unstack(0)
    med = m.median(axis=1)
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for z in m.columns:
        ax.plot(m.index, m[z], color=AXIS, lw=0.6, zorder=2)
    ax.plot(med.index, med, color=BLUE, lw=2, zorder=3)
    hi = med.idxmax()
    ax.annotate(f"median {med.max():.0f} EUR/MWh\n({hi:%b %Y})", (hi, med.max()), xytext=(14, -2),
                textcoords="offset points", fontsize=7.5, color=INK2, va="top")
    ax.axhline(0, color=AXIS, lw=0.8)
    ygrid(ax)
    split_marks(ax)
    ax.set_ylabel("monthly mean price, EUR/MWh")
    ax.set_ylim(-20, min(np.nanmax(m.values) * 1.02, 650))
    ax.legend(handles=[Line2D([], [], color=AXIS, lw=0.8, label="one zone"),
                       Line2D([], [], color=BLUE, lw=2, label="median across zones")],
              loc="upper left", fontsize=7.5, labelcolor=INK2)
    ax.set_title("Price regimes: the 2021-23 energy crisis sits inside pretrain and train", pad=16)
    save(fig, m.reset_index().melt(id_vars="month", var_name="zone", value_name="mean_price"), "d3_price_regimes")


def d4_price_by_zone():
    a, b = SPLITS["test"]
    n = nodes(["zone", "delivery_date", "price", "has_price"], filters=[("split", "=", "test")])
    n = n[n.has_price]
    q = n.groupby("zone").price.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    q.columns = ["p05", "p25", "p50", "p75", "p95"]
    q["neg_share"] = n.groupby("zone").price.apply(lambda s: (s < 0).mean())
    q["spike_share"] = n.groupby("zone").price.apply(lambda s: (s > 250).mean())
    q = q.sort_values("p50")
    y = np.arange(len(q))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 7.0), sharey=True, gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.08})
    col = [BLUE if z in CORE else INK2 for z in q.index]
    ax.hlines(y, q.p05, q.p95, color=col, lw=0.9, alpha=0.6)
    ax.hlines(y, q.p25, q.p75, color=col, lw=4.5)
    ax.scatter(q.p50, y, s=18, color=SURFACE, edgecolor=col, linewidth=1.2, zorder=3)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y, [z.replace("_", " ") for z in q.index], fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("price, EUR/MWh")
    ax.set_title(f"Test-split price per zone ({a} to {b})")
    ax.legend(handles=[Line2D([], [], color=INK2, lw=0.9, label="5-95 %"), Line2D([], [], color=INK2, lw=4.5, label="25-75 %"),
                       Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, label="median"),
                       Patch(color=BLUE, label="Core zone"), Patch(color=INK2, label="other zone")],
              loc="lower right", fontsize=7, labelcolor=INK2)
    ax2.barh(y, q.neg_share * 100, height=0.62, color=col)
    for yy, v in zip(y, q.neg_share * 100):
        if v >= 1:
            ax2.text(v + 0.3, yy, f"{v:.0f}", va="center", fontsize=6.3, color=INK2)
    ax2.set_xlabel("hours with price < 0, %")
    ax2.grid(axis="x")
    ax2.set_axisbelow(True)
    ax2.set_title("Negative prices")
    ax2.tick_params(axis="y", length=0)
    save(fig, q.reset_index(), "d4_price_by_zone")


# ---- d5: spiky vs smooth ------------------------------------------------------------------------------------

VARS = [("price", "Day-ahead price", BLUE), ("load_fc", "Load forecast", ORANGE),
        ("temp_pw", "Temperature", AQUA), ("wind_cf", "Wind capacity factor", YELLOW)]


def _acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[: lags + 1]
    return r / r[0]


def d5_spiky_vs_smooth():
    cols = [v for v, _, _ in VARS]
    n = nodes(["zone", "ts_utc", "has_price"] + cols, filters=[("split", "=", "test")])
    n.loc[~n.has_price, "price"] = np.nan
    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(len(VARS) * 2, 2, width_ratios=[1.6, 1], wspace=0.24, hspace=0.9)

    # (a) a sample fortnight in DE_LU: one small multiple per variable, shared time axis (no dual axes)
    d = n[n.zone == "DE_LU"].set_index("ts_utc").sort_index()
    t0 = pd.Timestamp("2025-11-03", tz="UTC")
    w = d.loc[t0: t0 + pd.Timedelta(days=14)]
    units = {"price": "EUR/MWh", "load_fc": "GW", "temp_pw": "°C", "wind_cf": "share"}
    sample = []
    for k, (c, label, color) in enumerate(VARS):
        ax = fig.add_subplot(gs[2 * k:2 * k + 2, 0])
        y = w[c] / 1000 if c == "load_fc" else w[c]
        ax.plot(w.index, y, color=color, lw=1.4)
        ax.set_ylabel(units[c], fontsize=7)
        ax.text(0.005, 0.97, label, transform=ax.transAxes, fontsize=8, weight="bold", color=INK, va="top")
        ygrid(ax)
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        if k < len(VARS) - 1:
            ax.tick_params(labelbottom=False)
        if k == 0:
            ax.set_title("DE_LU, two weeks of November 2025: the price has spikes that its drivers do not", pad=6)
        sample.append(pd.DataFrame({"ts_utc": w.index, "variable": c, "value": y.to_numpy()}))

    # (b) autocorrelation, median over zones
    ax = fig.add_subplot(gs[:4, 1])
    lags = 24 * 8
    acf_rows = []
    for c, label, color in VARS:
        curves = []
        for z, g in n.groupby("zone"):
            x = g.sort_values("ts_utc")[c].to_numpy(dtype=float)
            if np.isnan(x).mean() > 0.05 or np.nanstd(x) == 0:
                continue
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
            curves.append(_acf(x, lags))
        med = np.median(curves, axis=0)
        ax.plot(np.arange(lags + 1), med, color=color, lw=1.8, label=label)
        acf_rows.append(pd.DataFrame({"variable": c, "lag_h": np.arange(lags + 1), "acf_median": med, "n_zones": len(curves)}))
    ax.set_xticks(np.arange(0, lags + 1, 24), [str(d) for d in range(9)])
    ax.axhline(0, color=AXIS, lw=0.8)
    ygrid(ax)
    ax.set_title("Autocorrelation (median over zones, test split)")
    ax.set_xlabel("lag (days)")
    ax.legend(fontsize=6.8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    # (c) heavy tails: excess kurtosis of the hour-to-hour change, one dot per zone
    ax = fig.add_subplot(gs[5:, 1])
    kur_rows = []
    for k, (c, label, color) in enumerate(VARS):
        vals = []
        for z, g in n.groupby("zone"):
            x = g.sort_values("ts_utc")[c].diff().dropna()
            if len(x) > 1000 and x.std() > 0:
                vals.append((z, float(x.kurt())))
        v = np.array([b for _, b in vals])
        jitter = (np.random.default_rng(k).random(len(v)) - 0.5) * 0.36
        ax.scatter(np.log10(np.clip(v, 0.1, None)), k + jitter, s=10, color=color, alpha=0.8, edgecolor=SURFACE, linewidth=0.4)
        ax.scatter([np.log10(np.median(v))], [k], marker="|", s=260, color=INK, linewidth=1.6, zorder=3)
        kur_rows += [{"variable": c, "zone": z, "excess_kurtosis_of_hourly_change": b} for z, b in vals]
    ax.set_yticks(range(len(VARS)), [lbl for _, lbl, _ in VARS], fontsize=7.5)
    ax.invert_yaxis()
    ticks = [0.1, 1, 10, 100, 1000]
    ax.set_xticks(np.log10(ticks), [f"{t:g}" for t in ticks])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("excess kurtosis of hour-to-hour change (log)")
    ax.set_title("Heavy tails: one dot per zone, bar = median")
    ax.text(0, -0.36, "0 = Gaussian; values ≤ 0.1 drawn at 0.1", transform=ax.transAxes, fontsize=6.5, color=MUTED)
    save(fig, pd.concat([pd.concat(sample).assign(panel="sample"), pd.concat(acf_rows).assign(panel="acf"),
                         pd.DataFrame(kur_rows).assign(panel="kurtosis")], ignore_index=True), "d5_spiky_vs_smooth")


# ---- d6: constraints ----------------------------------------------------------------------------------------

def d6_constraints():
    path = os.path.join(FEAT, "constraints_hourly.parquet")
    have = pq.ParquetFile(path).schema_arrow.names
    cols = ["ts_utc", "ram_fmax"] + (["amr_fmax"] if "amr_fmax" in have else [])
    h = pd.read_parquet(path, columns=cols)
    cat = pd.read_parquet(os.path.join(FEAT, "constraint_catalog.parquet"), columns=["constraint_id", "n_days"])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 3.4), gridspec_kw={"width_ratios": [1.6, 1, 1], "wspace": 0.28})

    per = h.groupby("ts_utc").size()
    day = per.groupby(per.index.tz_convert("Europe/Brussels").date).median()
    day.index = pd.to_datetime(day.index)
    roll = day.rolling(28, center=True, min_periods=7).median()
    a1.plot(day.index, day, color=BLUE_250, lw=0.6)
    a1.plot(roll.index, roll, color=BLUE, lw=2)
    ygrid(a1)
    split_marks(a1, label=True, lo=day.index.min(), hi=day.index.max())
    a1.set_xlim(day.index.min(), day.index.max())
    a1.xaxis.set_major_locator(matplotlib.dates.YearLocator(1, month=1, day=1))
    a1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    a1.set_ylabel("presolved constraints per hour")
    a1.legend(handles=[Line2D([], [], color=BLUE_250, lw=0.8, label="daily median"),
                       Line2D([], [], color=BLUE, lw=2, label="28-day median")], fontsize=7, labelcolor=INK2, loc="upper left")
    a1.set_title("JAO Core domain size", pad=16)

    bins = np.linspace(0, 1.6, 65)
    r = h.ram_fmax.clip(upper=1.6)
    if "amr_fmax" in h:
        lifted = h.amr_fmax > 0
        a2.hist([r[~lifted].dropna(), r[lifted].dropna()], bins=bins, stacked=True, color=[BLUE, ORANGE],
                edgecolor=SURFACE, linewidth=0.3, label=["grid margin only", "lifted by minimum-RAM uplift (AMR)"])
        a2.legend(fontsize=6.8, labelcolor=INK2, loc="upper left")
    else:
        a2.hist(r.dropna(), bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
        a2.text(0.98, 0.5, "rebuild features/v1\nto split by AMR", transform=a2.transAxes, fontsize=6.5, color=MUTED,
                va="center", ha="right")
    a2.axvline(0.7, color=INK2, lw=0.7, ls=(0, (3, 3)))
    a2.text(0.71, 0.98, "70 % target", transform=a2.get_xaxis_transform(), fontsize=6.8, color=INK2, va="top")
    a2.axvline(1.0, color=AXIS, lw=0.8)
    a2.text(1.02, 0.85, "> 1: counter-flow\nfrees margin", transform=a2.get_xaxis_transform(), fontsize=6.5, color=MUTED, va="top")
    ygrid(a2)
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    a2.set_xlabel("RAM / Fmax (clipped at 1.6)")
    a2.set_ylabel("constraint-hours")
    a2.set_title("Remaining margin")

    nd = cat.n_days.to_numpy()
    edges = np.unique(np.round(np.logspace(0, np.log10(nd.max() + 1), 30)).astype(int))
    a3.hist(nd, bins=edges, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    a3.set_xscale("log")
    a3.axvline(np.median(nd), color=INK, lw=0.9)
    a3.text(np.median(nd) * 1.1, 0.97, f"median {np.median(nd):.0f} days", transform=a3.get_xaxis_transform(), fontsize=7, color=INK2, va="top")
    ygrid(a3)
    a3.set_xlabel("days on which a constraint is presolved (log)")
    a3.set_ylabel("constraints")
    a3.set_title(f"Constraint lifetimes ({len(cat):,} constraints)")
    tab = pd.concat([day.rename("median_per_hour").rename_axis("day").reset_index().assign(panel="domain_size"),
                     pd.DataFrame({"n_days": nd}).assign(panel="lifetime")], ignore_index=True)
    save(fig, tab, "d6_constraints")


# ---- d7: baselines -----------------------------------------------------------------------------------------

MODEL_RUNS = {"c1_xgb_zone": "XGBoost per zone", "c2_xgb_neighbours": "XGBoost + neighbours",
              "c25_gru_mlp": "GRU + MLP (no graph)", "c3s_gat_interconnects": "GATv2, interconnects",
              "c3s_gat_full": "GATv2, full attention", "c3s_gat_self_curves_lag24": "GATv2 self + D-1 curves (lag24 weather)",
              "c3s_gat_self_curves": "GATv2 self + D-1 curves", "c3s_gat_interconnects_curves": "GATv2 interconnects + D-1 curves",
              "c3s_gat_full_curves": "GATv2 full + D-1 curves",
              "c3s_gat_self_curves_rolling": "GATv2 self + curves, fine-tuned monthly",
              "c3s_gat_interconnects_curves_rolling": "GATv2 interconnects + curves, fine-tuned monthly",
              "c3s_gat_full_curves_rolling": "GATv2 full + curves, fine-tuned monthly",
              "c1_xgb_zone_asof": "XGBoost per zone (safe inputs)", "c2_xgb_neighbours_asof": "XGBoost + neighbours (safe inputs)",
              "c25_gru_mlp_asof": "GRU + MLP (safe inputs)", "c3s_gat_self_asof": "GATv2 self (safe inputs)",
              "c3s_gat_interconnects_asof": "GATv2 interconnects (safe inputs)", "c3s_gat_full_asof": "GATv2 full (safe inputs)",
              "c4s_gat_interconnects_jao_rolling": "GATv2 interconnects + curves + D-1 JAO summaries, fine-tuned",
              "c4s_gat_interconnects_constraints_rolling": "GATv2 interconnects + curves + D-1 constraint nodes, fine-tuned"}


def d7_baselines():
    b = pd.read_csv(os.path.join(OUT, "baselines_mae.csv"))
    rows = [b[["method", "family", "val_mae", "test_mae"]]]
    for run, label in MODEL_RUNS.items():
        f = os.path.join(FW, "results", f"{run}.json")
        if os.path.exists(os.path.join(FW, "results", f"{run}_asof.json")):
            continue                                       # superseded by its rerun on cutoff-safe inputs
        if os.path.exists(f):
            m = json.load(open(f))["metrics"]
            rows.append(pd.DataFrame([{"method": label, "family": "trained", "val_mae": m["val"]["all"]["mae"],
                                       "test_mae": m["test"]["all"]["mae"]}]))
            st = json.load(open(f)).get("info", {}).get("static_metrics")
            if st and not os.path.exists(os.path.join(FW, "results", run.replace("_rolling", "") + ".json")):    # the same run's fit-once predictions
                rows.append(pd.DataFrame([{"method": label.replace(", fine-tuned monthly", ", fit once"),
                                           "family": "trained", "val_mae": st["val"]["all"]["mae"],
                                           "test_mae": st["test"]["all"]["mae"]}]))
    t = pd.concat(rows, ignore_index=True).dropna(subset=["test_mae"]).sort_values("test_mae", ascending=False)
    lear = float(b.loc[b.method.str.startswith("LEAR (static"), "test_mae"].iat[0])
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(t) + 1.3))
    fam = {"naive": INK2, "linear": ORANGE, "trained": BLUE}
    ax.hlines(y, t.val_mae, t.test_mae, color=AXIS, lw=1)
    ax.scatter(t.val_mae, y, s=26, facecolor=SURFACE, edgecolor=[fam[f] for f in t.family], linewidth=1.3, zorder=3)
    ax.scatter(t.test_mae, y, s=30, color=[fam[f] for f in t.family], edgecolor=SURFACE, linewidth=0.8, zorder=4)
    for yy, v in zip(y, t.test_mae):
        ax.text(max(t.test_mae.max(), t.val_mae.max()) + 0.6, yy, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.axvline(lear, color=ORANGE, lw=0.8, ls=(0, (3, 3)))
    ax.text(lear, -0.75, " static LEAR, test", fontsize=7, color=INK2, va="center")
    ax.set_yticks(y, t.method, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("MAE, EUR/MWh (all zones, all priced hours)")
    ax.set_title("Baselines and models: test MAE (filled, value at right) and val MAE (hollow)")
    ax.legend(handles=[Patch(color=INK2, label="naive"), Patch(color=ORANGE, label="linear (LEAR)"), Patch(color=BLUE, label="trained model"),
                       Line2D([], [], marker="o", ls="", color=INK2, label="test"),
                       Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=INK2, label="val")],
              loc="lower left", fontsize=7, labelcolor=INK2, ncol=2)
    save(fig, t, "d7_baselines")


FIGURES = [d1_zone_map, d2_availability, d3_price_regimes, d4_price_by_zone, d5_spiky_vs_smooth, d6_constraints, d7_baselines]

if __name__ == "__main__":
    want = sys.argv[1:]
    for f in FIGURES:
        if not want or any(f.__name__.startswith(w) for w in want):
            f()
