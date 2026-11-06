#!/usr/bin/env python3
"""Build agg_zones_hourly: one row per (zone, hour) joining ENTSO-E prices, ENTSO-E load forecast and
per-zone weather on the canonical zone code. Writes the parquet and its doc to staging/agg_zones_hourly/.

    scripts/build_agg_zones_hourly.py [--version v1.0]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from epgnn.agg import sources_table, write_doc  # noqa: E402
from epgnn.paths import ROOT, dataset_path  # noqa: E402
from epgnn.zones import DE_SPLIT_TS, ZONES, zone_of  # noqa: E402

NAME = "agg_zones_hourly"
SOURCES = ["entsoe_prices", "entsoe_load_forecast", "weather_hourly"]

PRICE_COLS = {"PriceEUR": "price_eur", "PriceEUR_input": "price_eur_input", "IsValidTarget": "price_valid",
              "row_status": "price_status", "is_filled": "price_filled"}
LOAD_COLS = {"LoadForecastMW": "load_mw", "LoadForecastMW_input": "load_mw_input", "IsValidValue": "load_valid",
             "row_status": "load_status", "is_filled": "load_filled"}
WEATHER_COLS = ["temp_pw", "temp_anom", "wind_cf", "low_wind", "solar_cf"]


def load_entsoe(name: str, cols: dict[str, str], ts_col: str = "DeliveryHourUTC") -> pd.DataFrame:
    df = pd.read_parquet(dataset_path(name, f"{name}.parquet"),
                         columns=["AreaCode", "AreaName", ts_col, "MarketDayCET"] + list(cols))
    df["zone"] = zone_of(df.AreaName, df[ts_col])
    df = df[df.zone.notna()].rename(columns={ts_col: "ts_utc", "MarketDayCET": "market_day_cet", "AreaCode": "eic", **cols})
    assert not df.duplicated(["zone", "ts_utc"]).any(), f"{name}: duplicate zone-hours after mapping"
    return df.drop(columns="AreaName")


def build() -> pd.DataFrame:
    prices = load_entsoe("entsoe_prices", PRICE_COLS)
    load = load_entsoe("entsoe_load_forecast", LOAD_COLS)
    weather = pd.read_parquet(dataset_path("weather_hourly", "weather_hourly.parquet")).rename(columns={"time_utc": "ts_utc"})
    weather = weather[weather.zone.isin(ZONES)]
    assert not weather.duplicated(["zone", "ts_utc"]).any()

    # Rule: the grid is the price grid (every zone x every ENTSO-E hour); load and weather are left-joined onto it
    hours = prices.ts_utc.drop_duplicates().sort_values()
    grid = pd.MultiIndex.from_product([ZONES, hours], names=["zone", "ts_utc"]).to_frame(index=False)
    out = grid.merge(prices, on=["zone", "ts_utc"], how="left")
    assert out.eic.notna().all(), "every zone must have a price row for every hour"
    out = out.merge(load.drop(columns=["eic", "market_day_cet"]), on=["zone", "ts_utc"], how="left")
    out = out.merge(weather, on=["zone", "ts_utc"], how="left")
    out["weather_valid"] = out.temp_pw.notna()
    for c in ("load_valid", "load_filled"):
        out[c] = out[c].fillna(False).astype(bool)
    out["load_status"] = out.load_status.fillna("NO_ROW")
    cols = ["zone", "eic", "ts_utc", "market_day_cet"] + list(PRICE_COLS.values()) + list(LOAD_COLS.values()) \
        + WEATHER_COLS + ["weather_valid"]
    return out[cols].sort_values(["ts_utc", "zone"]).reset_index(drop=True)


def doc(out: pd.DataFrame, version: str) -> str:
    z = out.groupby("zone").agg(price_valid=("price_valid", "mean"), load_valid=("load_valid", "mean"),
                                weather_valid=("weather_valid", "mean"))
    zone_rows = "\n".join(f"| {i} | {r.price_valid:.1%} | {r.load_valid:.1%} | {r.weather_valid:.1%} |" for i, r in z.iterrows())
    return f"""# Zone-hour table: prices, load forecast and weather per bidding zone

| | |
|---|---|
| Dataset id | `{NAME}` |
| Version | **{version}** |
| Bucket | `gs://cs224w-epgnn-datasets/{NAME}/{version}/` |
| Local file(s) | `datasets/{NAME}_{version}.parquet`, `datasets/{NAME}_{version}.md` |
| Source | derived from the datasets listed under *Built from*, by `scripts/build_{NAME}.py` |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("{NAME}", "{NAME}.parquet"))
```

Version history:

| Version | What |
|---|---|
| {version} | first build: {len(out.zone.unique())} zones x {out.ts_utc.nunique():,} hours from entsoe_prices, entsoe_load_forecast, weather_hourly |

## Built from

{sources_table(SOURCES)}

Rebuild with `scripts/build_{NAME}.py`; the zone mapping is `epgnn/zones.py`.

## What it is

One row per **(zone, hour)**: the zone-node inputs and targets, already joined. {len(out):,} rows = {len(out.zone.unique())} zones x {out.ts_utc.nunique():,} hours, {out.ts_utc.min()} to {out.ts_utc.max()} UTC.

Nothing is filtered or re-cleaned: the price, load and weather values are the source datasets' values, with the source quality flags carried along under new names. Use the source docs for what the flags mean.

## Rules used to build it

1. **Zone id.** The canonical `zone` is the weather / JAO code (`DE_LU`, `NO_1`, `IT_NORD`). ENTSO-E areas are mapped by their display name to that code (`epgnn.zones.ENTSOE_NAME_TO_ZONE`); the ENTSO-E EIC code of the row is kept in `eic`.
2. **Which zones.** Every zone that has both an ENTSO-E price series and a weather series: {len(ZONES)} zones. Bosnia (BA) has weather but no ENTSO-E prices and is not included. No other zone-level filtering (GB, IE_SEM, ME, MK, IT_CALA, IT_SICI are in, with whatever coverage the sources have).
3. **Germany.** Before market day 2018-10-01 ENTSO-E reports the combined zone DE-AT-LU; from then on DE-LU. Both are spliced onto `zone = DE_LU`, switching at {DE_SPLIT_TS} UTC (first hour of market day 2018-10-01, Brussels time); `eic` shows which one each row came from. The source rows that are lifetime filler on the other side of the split (DE-LU before, DE-AT-LU after) are dropped. AT is its own zone from Oct 2018; before that its prices are masked in the source.
4. **Grid and joins.** The row grid is the price table's grid: every zone x every ENTSO-E hour ({out.ts_utc.min().date()} to {out.ts_utc.max().date()}). Load forecast and weather are left-joined on (`zone`, `ts_utc`). A missing load row gives `load_status = NO_ROW`, `load_valid = False`; weather is absent after {out[out.weather_valid].ts_utc.max()} UTC (`weather_valid = False`).
5. **Column names.** Source columns are renamed only to avoid clashes between the three sources (see schema); values are untouched. `market_day_cet` is the price table's `MarketDayCET`.

## Schema

| Column | From | Source column |
|---|---|---|
| `zone` | derived | canonical zone code |
| `eic` | entsoe_prices | `AreaCode` |
| `ts_utc` | entsoe_prices | `DeliveryHourUTC` |
| `market_day_cet` | entsoe_prices | `MarketDayCET` |
| `price_eur` | entsoe_prices | `PriceEUR` (target; null when not valid) |
| `price_eur_input` | entsoe_prices | `PriceEUR_input` (history input, short gaps patched) |
| `price_valid` | entsoe_prices | `IsValidTarget` |
| `price_status` | entsoe_prices | `row_status` |
| `price_filled` | entsoe_prices | `is_filled` |
| `load_mw` | entsoe_load_forecast | `LoadForecastMW` |
| `load_mw_input` | entsoe_load_forecast | `LoadForecastMW_input` |
| `load_valid` | entsoe_load_forecast | `IsValidValue` (False when no row) |
| `load_status` | entsoe_load_forecast | `row_status` (`NO_ROW` when no row) |
| `load_filled` | entsoe_load_forecast | `is_filled` |
| `temp_pw`, `temp_anom`, `wind_cf`, `low_wind`, `solar_cf` | weather_hourly | same names |
| `weather_valid` | derived | weather row present for the zone-hour |

## Coverage per zone (share of hours)

| Zone | price valid | load valid | weather |
|---|---|---|---|
{zone_rows}

## Caveats

- Validity differences between zones are inherited from the sources (e.g. AT and DE_LU prices before Oct 2018, Poland 2016-17, GB after 2020, IE_SEM load from 2022); nothing is masked or filled here beyond what the sources did.
- Weather ends on 2026-09-30; the ENTSO-E tables run to 2026-10-31.
- Weather is observed ERA5, not a forecast; using day-D weather as an input is look-ahead (see the weather_hourly doc).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1.0")
    a = ap.parse_args()
    out = build()
    stage = ROOT / "staging" / NAME
    stage.mkdir(parents=True, exist_ok=True)
    out.to_parquet(stage / f"{NAME}.parquet", index=False)
    write_doc(stage, NAME, doc(out, a.version))
    print(f"wrote {stage}/{NAME}.parquet: {len(out):,} rows, {out.zone.nunique()} zones, "
          f"{out.ts_utc.min()} -> {out.ts_utc.max()}; doc written. Next: scripts/datasets.py push {NAME} {a.version}")


if __name__ == "__main__":
    main()
