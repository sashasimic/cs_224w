"""Merge weather, ENTSO-E and JAO data into model-ready hourly tables by bidding zone.

Outputs (in data/model/):
    nodes_hourly.parquet   one row per (ts_utc, zone): target price, inputs, safe lags, JAO domain features
    edges_hourly.parquet   one row per (ts_utc, from_zone, to_zone): physical flow, JAO MaxBex and scheduled exchange
    splits/<split>_nodes.parquet, splits/<split>_edges.parquet   the same rows, one file per split (see SPLITS)

Usage (from the project folder):
    python build_model_dataset.py
    python build_model_dataset.py --start "2015-12-31 23:00" --end "2026-09-30 22:00" --out data/model
    python build_model_dataset.py --entsoe-version 1.1 --end "2025-09-30 22:00"   # the earlier release

Conventions: ts_utc is the UTC start of the hour. Same-hour columns are raw values; columns
ending in _lagN are shifted N hours so they are known before day-ahead gate closure (noon D-1).
"""
import argparse
import os

import numpy as np
import pandas as pd

# ENTSO-E area names -> bidding-zone codes used in the weather file.
BZN = {
    "Austria (AT)": "AT", "Bosnia and Herz. (BA)": "BA", "Belgium (BE)": "BE", "Bulgaria (BG)": "BG",
    "Switzerland (CH)": "CH", "Czech Republic (CZ)": "CZ", "DE-LU": "DE_LU", "DK1": "DK_1", "DK2": "DK_2",
    "Estonia (EE)": "EE", "Spain (ES)": "ES", "Finland (FI)": "FI", "France (FR)": "FR", "GB": "GB",
    "Greece (GR)": "GR", "Croatia (HR)": "HR", "Hungary (HU)": "HU", "IE(SEM)": "IE_SEM",
    "IT-North": "IT_NORD", "IT-Centre-North": "IT_CNOR", "IT-Centre-South": "IT_CSUD", "IT-South": "IT_SUD",
    "IT-Calabria": "IT_CALA", "IT-Sicily": "IT_SICI", "IT-Sardinia": "IT_SARD",
    "Lithuania (LT)": "LT", "Latvia (LV)": "LV", "Montenegro (ME)": "ME", "North Macedonia (MK)": "MK",
    "Netherlands (NL)": "NL", "NO1": "NO_1", "NO2": "NO_2", "NO3": "NO_3", "NO4": "NO_4", "NO5": "NO_5",
    "Poland (PL)": "PL", "Portugal (PT)": "PT", "Romania (RO)": "RO", "Serbia (RS)": "RS",
    "SE1": "SE_1", "SE2": "SE_2", "SE3": "SE_3", "SE4": "SE_4", "Slovenia (SI)": "SI", "Slovakia (SK)": "SK",
}
# The joint DE-AT-LU zone (until 30 Sep 2018) is mapped to DE_LU for prices and flows.
PRE_SPLIT = {"DE-AT-LU": "DE_LU"}
# Load fallbacks when the bidding-zone series is missing: sum of country/control-area series.
LOAD_FALLBACK = {"DE_LU": ["Germany (DE)", "Luxembourg (LU)"], "IE_SEM": ["Ireland (IE)", "NIE"]}
JAO_ZONE = {"DE": "DE_LU"}  # ALBE/ALDE (ALEGrO virtual hubs) are dropped
TZ = "Europe/Brussels"
# Splits by Brussels delivery date: (name, first day, last day), inclusive. Pretrain has no JAO constraints.
SPLITS = [("pretrain", "2016-01-01", "2022-06-08"), ("train", "2022-06-09", "2024-03-31"),
          ("val", "2024-04-01", "2025-03-31"), ("test", "2025-04-01", "2026-09-30")]


def read_prices(path):
    p = pd.read_parquet(path, columns=["AreaName", "DeliveryHourUTC", "PriceEUR", "PriceEUR_input", "is_filled"])
    p["zone"] = p.AreaName.map({**BZN, **PRE_SPLIT})
    p = p.dropna(subset=["zone"]).rename(columns={"DeliveryHourUTC": "ts_utc"})
    p = p[p.PriceEUR_input.notna()]  # DE-AT-LU and DE-LU rows only overlap where one is empty
    return p.groupby(["ts_utc", "zone"]).agg(
        price=("PriceEUR", "first"), price_input=("PriceEUR_input", "first"), price_filled=("is_filled", "first"))


def read_load(path):
    l = pd.read_parquet(path, columns=["AreaName", "DeliveryHourUTC", "LoadForecastMW_input"])
    l = l.rename(columns={"DeliveryHourUTC": "ts_utc", "LoadForecastMW_input": "load_fc"})
    main = l.assign(zone=l.AreaName.map(BZN)).dropna(subset=["zone"]).set_index(["ts_utc", "zone"]).load_fc
    for zone, parts in LOAD_FALLBACK.items():
        fb = l[l.AreaName.isin(parts)].groupby("ts_utc").load_fc.sum(min_count=len(parts))
        fb.index = pd.MultiIndex.from_arrays([fb.index, [zone] * len(fb)], names=["ts_utc", "zone"])
        main = main.combine_first(fb)  # keeps the bidding-zone value where present
    return main.rename("load_fc")


def read_flows(path):
    f = pd.read_parquet(path, columns=["OutAreaName", "InAreaName", "DeliveryHourUTC", "FlowMW_input"])
    names = {**BZN, **PRE_SPLIT}
    f["from_zone"], f["to_zone"] = f.OutAreaName.map(names), f.InAreaName.map(names)
    f = f.dropna(subset=["from_zone", "to_zone"])
    f = f[f.from_zone != f.to_zone].rename(columns={"DeliveryHourUTC": "ts_utc", "FlowMW_input": "flow_phys"})
    return f.groupby(["ts_utc", "from_zone", "to_zone"]).flow_phys.sum(min_count=1)


def read_jao(root):
    j = pd.read_parquet(os.path.join(root, "jao_hourly.parquet"), columns=["ts_utc", "zone", "n_presolved", "min_ram_ratio"])
    j = j[~j.zone.isin(["ALBE", "ALDE"])].replace({"zone": JAO_ZONE})
    b = pd.read_parquet(os.path.join(root, "jao_border_hourly.parquet"))
    b = b.replace({"from_zone": JAO_ZONE, "to_zone": JAO_ZONE}).rename(columns={"flow": "flow_sched"})
    return j.set_index(["ts_utc", "zone"]), b.set_index(["ts_utc", "from_zone", "to_zone"])


def split_of(delivery_date):
    """Split label for each Brussels delivery date (ISO strings); null outside all splits."""
    out = pd.Series(pd.NA, index=delivery_date.index, dtype="string")
    for name, first, last in SPLITS:
        out[(delivery_date >= first) & (delivery_date <= last)] = name
    return out


def add_lags(df, key, cols, hours):
    """Shift within each key on a complete hourly grid (rows sorted by key then time)."""
    g = df.groupby(key, sort=False)
    for c in cols:
        for h in hours:
            df[f"{c}_lag{h}"] = g[c].shift(h)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--start", default="2015-12-31 23:00", help="UTC; default is the start of delivery day 1 Jan 2016")
    ap.add_argument("--end", default="2026-09-30 22:00", help="UTC, exclusive; the weather file ends with delivery day 30 Sep 2026")
    ap.add_argument("--out", default="data/model")
    ap.add_argument("--entsoe-version", default="1.2", help="ENTSO-E release; v1.2 runs to Oct 2026, v1.1 to Sep 2025")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ent, jao = os.path.join(a.data, "entsoe"), os.path.join(a.data, "jao_core_hourly_views")
    # Lags need a week of history before the start.
    lo, hi = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    pad = lo - pd.Timedelta(days=8)

    w = pd.read_parquet(os.path.join(a.data, "weather_hourly_bzone_2010_2026.parquet")).rename(columns={"time_utc": "ts_utc"})
    zones = sorted(w.zone.unique())
    w = w.set_index(["ts_utc", "zone"])
    src = lambda kind: os.path.join(ent, f"entsoe_{kind}_v{a.entsoe_version}.parquet")
    prices, load, flows = read_prices(src("prices")), read_load(src("load_forecast")), read_flows(src("physical_flows"))
    jao_nodes, jao_edges = read_jao(jao)

    # ---------------- edges: interconnectors among modelled zones with any physical flow data
    flows = flows[flows.index.get_level_values("from_zone").isin(zones) & flows.index.get_level_values("to_zone").isin(zones)]
    pairs = flows.dropna().reset_index()[["from_zone", "to_zone"]].drop_duplicates()
    pairs = pd.concat([pairs, jao_edges[jao_edges.flow_sched.notna()].reset_index()[["from_zone", "to_zone"]].drop_duplicates()]).drop_duplicates()
    pairs = pairs.sort_values(["from_zone", "to_zone"]).reset_index(drop=True)
    hours = pd.date_range(pad, hi, freq="h", tz="UTC", inclusive="left", name="ts_utc")
    eidx = pd.MultiIndex.from_arrays([np.tile(hours, len(pairs)), np.repeat(pairs.from_zone.to_numpy(), len(hours)),
                                      np.repeat(pairs.to_zone.to_numpy(), len(hours))], names=["ts_utc", "from_zone", "to_zone"])
    edges = pd.DataFrame(index=eidx).join(flows).join(jao_edges).reset_index()
    edges = add_lags(edges, ["from_zone", "to_zone"], ["flow_phys"], [48, 168])
    edges = add_lags(edges, ["from_zone", "to_zone"], ["flow_sched"], [24])

    # Net physical import into each zone from other modelled zones.
    inflow = edges.groupby(["ts_utc", "to_zone"]).flow_phys.sum(min_count=1).rename_axis(["ts_utc", "zone"])
    outflow = edges.groupby(["ts_utc", "from_zone"]).flow_phys.sum(min_count=1).rename_axis(["ts_utc", "zone"])
    net_import = inflow.sub(outflow, fill_value=0).rename("net_import")

    # ---------------- nodes
    nidx = pd.MultiIndex.from_arrays([np.tile(hours, len(zones)), np.repeat(zones, len(hours))], names=["ts_utc", "zone"])
    nodes = pd.DataFrame(index=nidx).join(prices).join(load).join(w).join(net_import).join(jao_nodes).reset_index()
    nodes = add_lags(nodes, "zone", ["price_input"], [24, 168])
    nodes = nodes.rename(columns={"price_input_lag24": "price_lag24", "price_input_lag168": "price_lag168"})
    nodes = add_lags(nodes, "zone", ["temp_pw", "temp_anom", "wind_cf", "solar_cf"], [24])
    nodes = add_lags(nodes, "zone", ["net_import"], [48])
    local = nodes.ts_utc.dt.tz_convert(TZ)
    nodes.insert(2, "delivery_date", local.dt.date.astype("string"))
    nodes.insert(3, "hour_local", local.dt.hour.astype("int8"))
    nodes.insert(4, "dow", local.dt.dayofweek.astype("int8"))
    nodes.insert(5, "month", local.dt.month.astype("int8"))
    nodes["price_filled"] = nodes.price_filled.astype("boolean").fillna(False).astype(bool)
    nodes["has_price"] = nodes.price.notna()
    nodes["in_core"] = nodes.zone.isin(jao_nodes.index.get_level_values("zone").unique())
    nodes = nodes.drop(columns=["price_input"])

    nodes = nodes[(nodes.ts_utc >= lo) & (nodes.ts_utc < hi)]
    edges = edges[(edges.ts_utc >= lo) & (edges.ts_utc < hi)]
    nodes["split"] = split_of(nodes.delivery_date)
    edges["split"] = split_of(edges.ts_utc.dt.tz_convert(TZ).dt.date.astype("string"))
    for df in (nodes, edges):
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
    nodes.to_parquet(os.path.join(a.out, "nodes_hourly.parquet"), index=False, compression="zstd")
    edges.to_parquet(os.path.join(a.out, "edges_hourly.parquet"), index=False, compression="zstd")
    os.makedirs(os.path.join(a.out, "splits"), exist_ok=True)
    for name, first, last in SPLITS:
        n, e = nodes[nodes.split == name], edges[edges.split == name]
        n.to_parquet(os.path.join(a.out, "splits", f"{name}_nodes.parquet"), index=False, compression="zstd")
        e.to_parquet(os.path.join(a.out, "splits", f"{name}_edges.parquet"), index=False, compression="zstd")
        print(f"  {name:8s} {first} to {last}: {n.delivery_date.nunique():,} days, {len(n):,} node rows, "
              f"{int(n.has_price.sum()):,} with price, {len(e):,} edge rows")
    print(f"nodes_hourly.parquet: {len(nodes):,} rows, {nodes.zone.nunique()} zones, {nodes.ts_utc.min()} to {nodes.ts_utc.max()}")
    print(f"edges_hourly.parquet: {len(edges):,} rows, {len(pairs)} directed pairs")
    print("columns:", list(nodes.columns))


if __name__ == "__main__":
    main()
