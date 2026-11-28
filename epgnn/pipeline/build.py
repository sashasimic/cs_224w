"""Build features/v1/ from the pinned tables and the JAO views.

    python -m epgnn.pipeline.build                 # everything
    python -m epgnn.pipeline.build --skip-jao      # without constraints (JAO columns left empty)
    python -m epgnn.pipeline.build --jao-dir PATH  # JAO views elsewhere (default ../data/jao_core_hourly_views)

Outputs in features/v1/
  zone_features_hourly.parquet   one row per (zone, hour): zone_nodes_hourly + engineered columns
  edge_features_hourly.parquet   one row per (from_zone, to_zone, hour): zone_edges_hourly + engineered columns
  constraints_hourly.parquet     one row per (hour, constraint): RAM, Fmax, RAM/Fmax, FRM, F0core, 12 PTDFs
  constraint_catalog.parquet     one row per constraint: names, direction, contingency, TSO, coverage
  graph.json                     zones, interconnector edges, spread pairs, Laplacian PE, components
  feature_dictionary.csv         every column: group, timing (gate / past / outcome / unverified / static), source, meaning
  feature_summary.csv            coverage and range of every numeric column, per split
  build.json                     inputs, row counts, timings
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.data import tables
from epgnn.paths import dataset_version
from epgnn.pipeline import constraints, edge_features, graphs, registry, zone_features
from epgnn.pipeline.tensors import FEATURES_DIR
from epgnn.splits import split_of


def _summary(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    split = split_of(df.delivery_date) if "delivery_date" in df else split_of(
        df.ts_utc.dt.tz_convert("Europe/Brussels").dt.strftime("%Y-%m-%d"))
    rows = []
    num = [c for c in df.columns if c not in key_cols and pd.api.types.is_numeric_dtype(df[c])]
    for sp in ["pretrain", "train", "val", "test"]:
        part = df[split == sp]
        for c in num:
            v = part[c].astype("float64")
            rows.append({"split": sp, "column": c, "coverage": round(float(v.notna().mean()), 4),
                         "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())})
    return pd.DataFrame(rows)


def _edges(out: Path, log: dict) -> None:
    print("edges: engineering features ...", flush=True)
    ef = edge_features.build(tables.edges())
    ef.to_parquet(out / "edge_features_hourly.parquet", index=False, compression="zstd")
    _summary(ef, ["ts_utc", "from_zone", "to_zone", "split"]).assign(table="edge").to_csv(
        out / "feature_summary_edge.csv", index=False)
    log["edges"] = {"rows": len(ef), "columns": len(ef.columns), "pairs": int(len(ef[["from_zone", "to_zone"]].drop_duplicates()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jao-dir", type=Path, default=constraints.JAO_DIR)
    ap.add_argument("--skip-jao", action="store_true")
    ap.add_argument("--rebuild-constraints", action="store_true", help="re-read the JAO views even if constraints_hourly exists")
    ap.add_argument("--skip-edges", action="store_true", help="do not write edge_features_hourly (17 M rows)")
    ap.add_argument("--out", type=Path, default=FEATURES_DIR)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log: dict = {"inputs": {"zone_nodes_hourly": dataset_version("zone_nodes_hourly"),
                            "zone_edges_hourly": dataset_version("zone_edges_hourly"),
                            "jao_dir": None if args.skip_jao else str(args.jao_dir)}, "seconds": {}}
    t0 = time.time()

    exposure = None
    if not args.skip_jao:
        if (out / "constraints_hourly.parquet").exists() and not args.rebuild_constraints:
            print("constraints: reusing constraints_hourly.parquet", flush=True)
            hourly = pd.read_parquet(out / "constraints_hourly.parquet")
            cat = pd.read_parquet(out / "constraint_catalog.parquet")
            syn = cat.constraint_id.astype(str)[constraints.is_synthetic(cat.cne_name)]
            if len(syn):  # a constraints_hourly built before the synthetic filter: drop those nodes here too
                print(f"  dropping {len(syn)} synthetic constraint id(s) from the reused files; "
                      "--rebuild-constraints also fixes the ids of elements without an EIC", flush=True)
                hourly = hourly[~hourly.constraint_id.isin(set(syn))].reset_index(drop=True)
                cat = cat[~cat.constraint_id.astype(str).isin(set(syn))].reset_index(drop=True)
                hourly.to_parquet(out / "constraints_hourly.parquet", index=False, compression="zstd")
            cat = constraints.with_ptdf_summary(cat, hourly)
            cat.to_parquet(out / "constraint_catalog.parquet", index=False)
        else:
            print("constraints: reading JAO views (a few minutes) ...", flush=True)
            hourly, cat = constraints.load(args.jao_dir)
            hourly.to_parquet(out / "constraints_hourly.parquet", index=False, compression="zstd")
            cat.to_parquet(out / "constraint_catalog.parquet", index=False)
        exposure = constraints.zone_exposure(hourly)
        log["constraints"] = {"rows": len(hourly), "constraints": len(cat), "ptdf_reference": "rest",
                              "synthetic_dropped": getattr(constraints.load, "synthetic_dropped", None),
                              "duplicates_dropped": getattr(constraints.load, "duplicates_dropped", None),
                              "per_hour_median": float(hourly.groupby("ts_utc").size().median())}
        del hourly
        log["seconds"]["constraints"] = round(time.time() - t0, 1)
        print(f"  {log['constraints']}", flush=True)

    t1 = time.time()
    print("zones: engineering features ...", flush=True)
    zf = zone_features.build(tables.nodes(), exposure)
    zf.to_parquet(out / "zone_features_hourly.parquet", index=False, compression="zstd")
    _summary(zf, ["ts_utc", "zone", "delivery_date", "split"]).assign(table="zone").to_csv(
        out / "feature_summary_zone.csv", index=False)
    log["zones"] = {"rows": len(zf), "columns": len(zf.columns), "zones": int(zf.zone.nunique())}
    log["seconds"]["zones"] = round(time.time() - t1, 1)
    del zf

    t2 = time.time()
    if args.skip_edges:
        print("edges: skipped (--skip-edges)", flush=True)
    else:
        _edges(out, log)
        log["seconds"]["edges"] = round(time.time() - t2, 1)


    zl = graphs.zones()
    g = {"zones": zl, "interconnect_edges": graphs.interconnect_edges(zl).tolist(),
         "spread_pairs": graphs.spread_pairs(zl).tolist(), "laplacian_pe_k8": graphs.laplacian_pe(zl, 8).round(6).tolist(),
         "components": graphs.components(zl), "degree": dict(zip(zl, graphs.adjacency(zl).sum(1).astype(int).tolist())),
         "dropped_pairs": graphs.dropped_pairs().values.tolist(), "active_since": graphs.ACTIVE_SINCE}
    (out / "graph.json").write_text(json.dumps(g))
    pd.DataFrame([vars(f) for f in registry.all_features()]).to_csv(out / "feature_dictionary.csv", index=False)
    log["seconds"]["total"] = round(time.time() - t0, 1)
    (out / "build.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
