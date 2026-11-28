"""Extend ENTSO-E prices, load forecasts and physical flows past Sep 2025, aggregated to hourly.

From 1 Oct 2025 SDAC prices (and many load series) are published per 15 minutes. Each hour here is
the duration-weighted mean of its sub-hourly values, and is left empty unless the full 60 minutes
are present (the same rule JAO used for scheduled exchanges).

Outputs (in data/entsoe_ext/):
    prices_hourly.parquet    ts_utc, zone, price              (EUR/MWh)
    load_hourly.parquet      ts_utc, zone, load_fc            (MW, day-ahead forecast)
    flows_hourly.parquet     ts_utc, from_zone, to_zone, flow_phys (MW)
    overlap_check.csv        Sep 2025 comparison against the v1.1 files (if --start is before Oct 2025)

Usage (from the project folder; needs an ENTSO-E Transparency API token):
    export ENTSOE_API_KEY=...            # do not write the token into this file
    python download_entsoe_ext.py                       # Sep 2025 to 29 Oct 2026
    python download_entsoe_ext.py --only prices --zones FR DE_LU
    python download_entsoe_ext.py --dry-run             # list requests, no downloads

Requests are cached per series and month under data/entsoe_ext/cache/, so the script can be
stopped and rerun. Delete a cache file to download it again.
"""
import argparse
import os
import sys
import time

import pandas as pd

ZONES = ["AT", "BA", "BE", "BG", "CH", "CZ", "DE_LU", "DK_1", "DK_2", "EE", "ES", "FI", "FR", "GB", "GR", "HR",
         "HU", "IE_SEM", "IT_CALA", "IT_CNOR", "IT_CSUD", "IT_NORD", "IT_SARD", "IT_SICI", "IT_SUD", "LT", "LV",
         "ME", "MK", "NL", "NO_1", "NO_2", "NO_3", "NO_4", "NO_5", "PL", "PT", "RO", "RS", "SE_1", "SE_2", "SE_3",
         "SE_4", "SI", "SK"]


def month_starts(start, end):
    return list(pd.date_range(start.normalize().replace(day=1), end, freq="MS", tz="UTC"))


def to_hourly(s):
    """Duration-weighted hourly mean; hours without a full 60 minutes of data are dropped."""
    s = s.dropna()
    if s.empty:
        return s
    s = s.tz_convert("UTC").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    step = s.index.to_series().diff().shift(-1)
    native = min(step.median(), pd.Timedelta("1h")) if step.notna().any() else pd.Timedelta("1h")
    step = step.fillna(native).clip(upper=native).dt.total_seconds() / 60  # a gap never stretches a neighbour
    minutes = step.resample("1h").sum()
    value = (s * step).resample("1h").sum() / minutes
    return value[minutes >= 60 - 1e-6]


def fetch(client, kind, key, t0, t1):
    from entsoe.exceptions import NoMatchingDataError
    t0l, t1l = t0.tz_convert("Europe/Brussels"), t1.tz_convert("Europe/Brussels")
    for attempt in range(4):
        try:
            if kind == "prices":
                s = client.query_day_ahead_prices(key, start=t0l, end=t1l)
            elif kind == "load":
                s = client.query_load_forecast(key, start=t0l, end=t1l).iloc[:, 0]
            else:
                s = client.query_crossborder_flows(key[0], key[1], start=t0l, end=t1l)
            return to_hourly(s.astype("float64"))
        except NoMatchingDataError:
            return pd.Series(dtype="float64")
        except Exception as e:  # network errors and rate limits: back off and retry
            wait = 10 * 2 ** attempt
            print(f"  {kind} {key} {t0:%Y-%m}: {type(e).__name__}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {kind} {key} {t0:%Y-%m}")


def cache_path(out, kind, key, t0):
    name = "__".join(key) if isinstance(key, tuple) else key
    return os.path.join(out, "cache", kind, name, f"{t0:%Y-%m}.parquet")


def collect(client, out, kind, keys, start, end, dry):
    frames, n = [], 0
    for key in keys:
        for t0 in month_starts(start, end):
            t1 = min(t0 + pd.offsets.MonthBegin(1), end)
            lo = max(t0, start)
            path = cache_path(out, kind, key, t0)
            if not os.path.exists(path):
                n += 1
                if dry:
                    continue
                s = fetch(client, kind, key, lo, t1)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                s.rename("value").rename_axis("ts_utc").reset_index().to_parquet(path, index=False)
            if dry:
                continue
            df = pd.read_parquet(path)
            if df.empty:
                continue
            if isinstance(key, tuple):
                df["from_zone"], df["to_zone"] = key
            else:
                df["zone"] = key
            frames.append(df)
        if not dry:
            print(f"  {kind} {key}: done", flush=True)
    if dry:
        print(f"{kind}: {n} requests to make")
        return None
    if not frames:
        sys.exit(f"No {kind} data returned; check the token and dates.")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.ts_utc >= start) & (df.ts_utc < end)]
    return df


def overlap_check(out, data, prices):
    """Compare Sep 2025 against the cleaned v1.1 prices to confirm units, currency and timing agree."""
    path = os.path.join(data, "entsoe", "electricity_prices_hourly_v1.1.parquet")
    if not os.path.exists(path) or prices is None:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_model_dataset import read_prices
    old = read_prices(path).price.dropna().rename("v11").reset_index()
    new = prices.rename(columns={"price": "ext"})
    m = old.merge(new, on=["ts_utc", "zone"])
    if m.empty:
        return
    res = m.groupby("zone").apply(lambda g: pd.Series({"hours": len(g), "mae": (g.ext - g.v11).abs().mean(),
                                                     "max_abs_diff": (g.ext - g.v11).abs().max()}))
    res.round(3).to_csv(os.path.join(out, "overlap_check.csv"))
    print("\nOverlap with v1.1 prices (EUR/MWh):")
    print(res.round(3).sort_values("mae", ascending=False).head(10).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-09-01", help="UTC; Sep 2025 overlaps the v1.1 files as a check")
    ap.add_argument("--end", default="2026-10-30", help="UTC, exclusive")
    ap.add_argument("--zones", nargs="+", default=ZONES)
    ap.add_argument("--only", nargs="+", choices=["prices", "load", "flows"], default=["prices", "load", "flows"])
    ap.add_argument("--edges", default="data/model/edges_hourly.parquet", help="directed pairs to download flows for")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/entsoe_ext")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    start, end = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    os.makedirs(a.out, exist_ok=True)

    client = None
    if not a.dry_run:
        key = os.environ.get("ENTSOE_API_KEY")
        if not key:
            sys.exit("Set ENTSOE_API_KEY first (Transparency Platform > My Account > Web API Security Token).")
        from entsoe import EntsoePandasClient
        client = EntsoePandasClient(api_key=key)

    prices = None
    if "prices" in a.only:
        df = collect(client, a.out, "prices", a.zones, start, end, a.dry_run)
        if df is not None:
            prices = df.rename(columns={"value": "price"})[["ts_utc", "zone", "price"]]
            prices.to_parquet(os.path.join(a.out, "prices_hourly.parquet"), index=False)
            print(f"prices_hourly.parquet: {len(prices):,} rows, {prices.zone.nunique()} zones")
    if "load" in a.only:
        df = collect(client, a.out, "load", a.zones, start, end, a.dry_run)
        if df is not None:
            df = df.rename(columns={"value": "load_fc"})[["ts_utc", "zone", "load_fc"]]
            df.to_parquet(os.path.join(a.out, "load_hourly.parquet"), index=False)
            print(f"load_hourly.parquet: {len(df):,} rows, {df.zone.nunique()} zones")
    if "flows" in a.only:
        pairs = pd.read_parquet(a.edges, columns=["from_zone", "to_zone"]).drop_duplicates()
        pairs = [p for p in map(tuple, pairs.to_numpy()) if p[0] in a.zones and p[1] in a.zones]
        df = collect(client, a.out, "flows", pairs, start, end, a.dry_run)
        if df is not None:
            df = df.rename(columns={"value": "flow_phys"})[["ts_utc", "from_zone", "to_zone", "flow_phys"]]
            df.to_parquet(os.path.join(a.out, "flows_hourly.parquet"), index=False)
            print(f"flows_hourly.parquet: {len(df):,} rows, {df.groupby(['from_zone', 'to_zone']).ngroups} pairs")
    overlap_check(a.out, a.data, prices)


if __name__ == "__main__":
    main()
