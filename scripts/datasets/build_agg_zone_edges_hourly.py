#!/usr/bin/env python3
"""Build agg_zone_edges_hourly: one row per (directed zone pair, hour) from ENTSO-E physical flows, for every
pair whose two ends are agg zones. Writes the parquet and its doc to staging/agg_zone_edges_hourly/.

    scripts/datasets/build_agg_zone_edges_hourly.py [--version v1.0]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from epgnn.agg import sources_table, write_doc  # noqa: E402
from epgnn.paths import ROOT, dataset_path  # noqa: E402
from epgnn.zones import DE_SPLIT_TS, zone_of  # noqa: E402

NAME = "agg_zone_edges_hourly"
SOURCES = ["entsoe_physical_flows"]
FLOW_COLS = {"FlowMW": "flow_mw", "FlowMW_input": "flow_mw_input", "IsValidValue": "flow_valid",
             "row_status": "flow_status", "is_filled": "flow_filled"}


def build() -> pd.DataFrame:
    f = pd.read_parquet(dataset_path("entsoe_physical_flows", "entsoe_physical_flows.parquet"),
                        columns=["OutAreaCode", "OutAreaName", "InAreaCode", "InAreaName", "DeliveryHourUTC", "MarketDayCET"] + list(FLOW_COLS))
    f["zone_from"] = zone_of(f.OutAreaName, f.DeliveryHourUTC)
    f["zone_to"] = zone_of(f.InAreaName, f.DeliveryHourUTC)
    # Rule: keep every directed pair whose two ends are agg zones (German splice applied to both ends); nothing else
    f = f[f.zone_from.notna() & f.zone_to.notna()]
    f = f.rename(columns={"DeliveryHourUTC": "ts_utc", "MarketDayCET": "market_day_cet",
                          "OutAreaCode": "eic_from", "InAreaCode": "eic_to", **FLOW_COLS})
    assert not f.duplicated(["zone_from", "zone_to", "ts_utc"]).any(), "duplicate pair-hours after mapping"
    cols = ["zone_from", "zone_to", "eic_from", "eic_to", "ts_utc", "market_day_cet"] + list(FLOW_COLS.values())
    return f[cols].sort_values(["ts_utc", "zone_from", "zone_to"]).reset_index(drop=True)


def doc(out: pd.DataFrame, version: str) -> str:
    live = out[out.flow_status != "OUTSIDE_ZONE_LIFETIME"]
    pairs = live.groupby(["zone_from", "zone_to"]).agg(first=("ts_utc", "min"), last=("ts_utc", "max"), valid=("flow_valid", "mean"))
    pair_rows = "\n".join(f"| {a} | {b} | {r['first'].date()} | {r['last'].date()} | {r.valid:.1%} |" for (a, b), r in pairs.iterrows())
    n_pairs = out.groupby(["zone_from", "zone_to"]).ngroups
    n_und = len({frozenset(p) for p in pairs.index})
    return f"""# Interconnector table: hourly physical flows per directed zone pair

| | |
|---|---|
| Dataset id | `{NAME}` |
| Version | **{version}** |
| Bucket | `gs://cs224w-epgnn-datasets/{NAME}/{version}/` |
| Local file(s) | `datasets/{NAME}_{version}.parquet`, `datasets/{NAME}_{version}.md` |
| Source | derived from the dataset listed under *Built from*, by `scripts/datasets/build_{NAME}.py` |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("{NAME}", "{NAME}.parquet"))
edges = df[["zone_from", "zone_to"]].drop_duplicates()        # the directed edge list
```

Version history:

| Version | What |
|---|---|
| {version} | first build: {n_pairs} directed pairs ({n_und} interconnectors) x {out.ts_utc.nunique():,} hours from entsoe_physical_flows |

## Built from

{sources_table(SOURCES)}

Rebuild with `scripts/datasets/build_{NAME}.py`; the zone mapping is `epgnn/zones.py`.

## What it is

One row per **(zone_from -> zone_to, hour)** for every directed pair ENTSO-E reports between two agg zones: the zone-zone edges of the graph and their hourly flow feature. {len(out):,} rows = {n_pairs} directed pairs x {out.ts_utc.nunique():,} hours, {out.ts_utc.min()} to {out.ts_utc.max()} UTC.

The distinct (`zone_from`, `zone_to`) pairs **are** the edge list; there is no separate edge file. ENTSO-E reports each direction of a border as its own series, so both directions are present as separate rows and no netting is done. Which pairs a model uses (e.g. requiring both directions, a minimum coverage, or `first` <= the forecast day) is a modelling choice made at load time from the per-pair coverage below.

## Rules used to build it

1. **Zone id.** Both ends are mapped from the ENTSO-E display names to the canonical zone code (`epgnn.zones.ENTSOE_NAME_TO_ZONE`); the EIC codes are kept in `eic_from` / `eic_to`.
2. **Which pairs.** Every directed pair whose two ends are agg zones (the {len(set(out.zone_from) | set(out.zone_to))} zones that have both ENTSO-E prices and weather). Pairs with a non-zone end (country aggregates, control areas, virtual Italian areas, cable areas like NO2NSL / IT-SACODC, UA, GB-only neighbours) are not included. No coverage filter.
3. **Germany.** DE-AT-LU pairs before {DE_SPLIT_TS} UTC and DE-LU pairs from then on are both mapped to `DE_LU`, so a border like NL -> DE_LU is one continuous series across the Oct 2018 split; `eic_from` / `eic_to` show which source area each row came from.
4. **Grid.** The source's full grid is kept: every pair has a row for every hour, with `flow_status = OUTSIDE_ZONE_LIFETIME` outside the period the pair is reported. Nothing is filled or masked beyond the source's own rules.
5. **Column names.** Source columns renamed with a `flow_` prefix (see schema); values untouched.

## Schema

| Column | From | Source column |
|---|---|---|
| `zone_from`, `zone_to` | derived | canonical zone codes of `OutArea`, `InArea` (flow direction from -> to) |
| `eic_from`, `eic_to` | entsoe_physical_flows | `OutAreaCode`, `InAreaCode` |
| `ts_utc` | entsoe_physical_flows | `DeliveryHourUTC` |
| `market_day_cet` | entsoe_physical_flows | `MarketDayCET` |
| `flow_mw` | entsoe_physical_flows | `FlowMW` (measured, MW, >= 0 in the stated direction) |
| `flow_mw_input` | entsoe_physical_flows | `FlowMW_input` (short gaps patched) |
| `flow_valid` | entsoe_physical_flows | `IsValidValue` |
| `flow_status` | entsoe_physical_flows | `row_status` |
| `flow_filled` | entsoe_physical_flows | `is_filled` |

## Directed pairs (period reported, share of valid hours in it)

| from | to | first | last | valid |
|---|---|---|---|---|
{pair_rows}

## Caveats

- Flows are **realised** values for the hour, published after the fact; for a day-ahead forecast they are inputs only up to the cutoff (lagged). Capacities (NTC) are not in this table.
- Both directions can be positive in the same hour on some borders (source finding D); decide the netting convention in the model.
- Pairs that start late or end early (cables, zone changes) are visible in the table above; nothing is pruned here.
- Because of the German splice, before Oct 2018 `DE_LU` stands for the combined DE-AT-LU zone, so its edges in that period include Austria's borders (`DE_LU` <-> HU, SI, IT_NORD end on 2018-09-30 and continue as `AT` <-> HU, SI, IT_NORD). A model that starts in 2019 never sees this; one that uses 2016-18 should treat those three pairs as AT's.
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
    print(f"wrote {stage}/{NAME}.parquet: {len(out):,} rows, {out.groupby(['zone_from', 'zone_to']).ngroups} directed pairs, "
          f"{out.ts_utc.min()} -> {out.ts_utc.max()}; doc written. Next: scripts/datasets/datasets.py push {NAME} {a.version}")


if __name__ == "__main__":
    main()
