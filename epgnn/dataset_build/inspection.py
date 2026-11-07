"""Thin notebook helpers. Structural reports cover all splits; targets do not."""
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from .contracts import calendar
from .storage import config, resolve


def generation(config_path):
    cfg = config(config_path)
    return (resolve(cfg, cfg["output_directory"]) / "current").resolve()


def metadata(config_path):
    p = generation(config_path)
    return {k: json.loads((p / f"{k}.json").read_text()) for k in ["build_fingerprint", "feature_registry", "findings", "geography_crosswalk", "cnec_references"]}


def coverage(config_path, zones=None, splits=None, features=None):
    f = pd.read_parquet(generation(config_path) / "coverage.parquet")
    if zones:
        f = f.loc[f.get("zone", pd.Series(index=f.index, dtype="string")).isin(zones) |
                  f.get("from_zone", pd.Series(index=f.index, dtype="string")).isin(zones) |
                  f.get("to_zone", pd.Series(index=f.index, dtype="string")).isin(zones)]
    if splits:
        f = f.loc[f.split.isin(splits)]
    if features:
        f = f.loc[f.feature.isin(features)]
    return f


def select(config_path, zones, start, end, variables, table="nodes"):
    if table not in {"nodes", "edges", "strict"}:
        raise ValueError("Unknown table")
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if a.tzinfo is None or b.tzinfo is None:
        raise ValueError("Selection bounds need explicit timezones")
    if b - a > pd.Timedelta(days=62):
        raise ValueError("Select at most 62 days for bounded interactive inspection")
    p = generation(config_path) / table
    dataset = ds.dataset(p, format="parquet")
    filt = (ds.field("ts_utc") >= a.to_pydatetime()) & (ds.field("ts_utc") < b.to_pydatetime())
    if zones:
        filt &= ds.field("zone" if table != "edges" else "from_zone").isin(zones)
    keys = ["zone", "ts_utc"] if table != "edges" else ["from_zone", "to_zone", "ts_utc"]
    cols = list(dict.fromkeys(keys + ["split", "forecast_origin_utc"] + variables))
    frame = dataset.to_table(columns=cols, filter=filt).to_pandas()
    target_columns = [c for c in variables if c == "price" or c.startswith("price_source_Price")]
    if target_columns and (frame.split == "test").any():
        raise ValueError("Detailed test-target inspection is disabled; use coverage reports")
    return frame.sort_values(keys).reset_index(drop=True)


def dst_examples():
    splits = {"train": ["2024-01-01", "2024-12-31"]}
    frames = []
    for day in ["2024-01-15", "2024-07-15", "2024-03-31", "2024-10-27", "2024-04-01", "2024-10-28"]:
        a = pd.Timestamp(day, tz="Europe/Brussels")
        b = a + pd.DateOffset(days=1)
        f = calendar(pd.DataFrame({"ts_utc": pd.date_range(a, b, inclusive="left", freq="h").tz_convert("UTC")}), splits)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def reference_comparison(config_path):
    return pd.DataFrame(json.loads((generation(config_path) / "reference_comparison.json").read_text()))
