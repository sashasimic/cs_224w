#!/usr/bin/env python3
"""Aggregate JAO flow-based constraints (one row per hour x CNEC) into per-zone hourly features.

Input:  the JAO bundle (jao_cnec_hourly.parquet + cnec_provenance.parquet, same row order).
Output: staging/jao_zone_hourly/jao_zone_hourly.parquet, one row per (zone, hour).

The rules are documented in the dataset doc (jao_zone_hourly.md); this file is the executable version.

    scripts/build_jao_zone_hourly.py [--src staging/jao_cwe_core_hourly_views] [--out staging/jao_zone_hourly]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Rule 1: a constraint is synthetic (not a physical line) if its source name matches this.
SYNTHETIC = re.compile(r"LTA|IMPORT|EXPORT|EC_|EXTERNAL|CONSTRAINT|NET POSITION|NETPOSITION|MAXBEX")
# Rule 3: a constraint is relevant to a zone if its zone-to-zone PTDF with some neighbour,
# |ptdf_zone - ptdf_neighbour|, is >= this. Published PTDFs are zone-to-slack, so their absolute
# size depends on the slack convention (Germany's is ~0 by construction); differences do not.
SPREAD_MIN = 0.10
# Rule 5: top3 uses this many constraints.
TOP_K = 3

# Zones with a PTDF column in jao_cnec_hourly. DE_AT_LU is the pre-2018-10-01 combined zone.
ZONES = ["AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK", "DE_AT_LU"]
OUT_ZONES = [z for z in ZONES if z != "DE_AT_LU"]
# Interconnected neighbours inside the flow-based domain (AC borders between the zones above).
NEIGHBOURS = {
    "AT": ["DE_LU", "DE_AT_LU", "CZ", "HU", "SI"],
    "BE": ["FR", "NL", "DE_LU", "DE_AT_LU"],
    "CZ": ["DE_LU", "AT", "PL", "SK"],
    "DE_LU": ["AT", "BE", "CZ", "FR", "NL", "PL"],
    "DE_AT_LU": ["BE", "FR", "NL", "CZ"],
    "FR": ["BE", "DE_LU", "DE_AT_LU"],
    "HR": ["HU", "SI"],
    "HU": ["AT", "HR", "RO", "SI", "SK"],
    "NL": ["BE", "DE_LU", "DE_AT_LU"],
    "PL": ["DE_LU", "CZ", "SK"],
    "RO": ["HU"],
    "SI": ["AT", "HR", "HU"],
    "SK": ["CZ", "HU", "PL"],
}


def load(src: Path) -> pd.DataFrame:
    prov = pq.read_table(src / "cnec_provenance.parquet",
                         columns=["cnec_key", "CriticalBranchName", "CneName"]).to_pandas()
    cnec = pq.read_table(src / "jao_cnec_hourly.parquet",
                         columns=["ts_utc", "cnec_key", "ram_fmax", "regime"] + [f"ptdf_{z}" for z in ZONES]).to_pandas()
    assert len(prov) == len(cnec) and (prov.cnec_key.values == cnec.cnec_key.values).all(), \
        "provenance rows must align positionally with jao_cnec_hourly"
    # Rule 1: physical vs synthetic, from the source constraint name (CWE: CriticalBranchName, Core: CneName)
    name = prov.CriticalBranchName.fillna(prov.CneName).fillna("").str.upper()
    cnec["physical"] = ~name.str.contains(SYNTHETIC)
    del prov
    cnec["regime"] = cnec.regime.replace({"CORE": "Core"})
    return cnec.drop(columns="cnec_key")


def zone_features(c: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Rules 3-5 for one zone, over physical rows with a valid ratio."""
    # Rule 3: spread = largest zone-to-zone PTDF with a neighbour; NaN when the zone or all its
    # neighbours are outside the domain that hour
    spread = pd.concat([(c[f"ptdf_{zone}"] - c[f"ptdf_{n}"]).abs() for n in NEIGHBOURS[zone]], axis=1).max(axis=1)
    keep = spread >= SPREAD_MIN
    s = c.loc[keep, ["ts_utc", "ram_fmax", "tight"]].copy()
    # Rule 4: weight = that spread
    s["w"] = spread[keep]
    s["wt"] = s.w * s.tight
    g = s.groupby("ts_utc", sort=False)
    f = pd.DataFrame({
        "n_rel": g.size(),
        "min_ratio": g.ram_fmax.min(),
        "wmean_tight": g.wt.sum() / g.w.sum(),
    })
    top = s.sort_values(["ts_utc", "wt"], ascending=[True, False]).groupby("ts_utc", sort=False).head(TOP_K)
    f["top3"] = top.groupby("ts_utc").wt.sum()
    f["zone"] = zone
    return f.reset_index()


def build(src: Path) -> pd.DataFrame:
    c = load(src)
    hours = c.groupby("ts_utc").agg(regime=("regime", "first"), n_phys_hour=("physical", "sum"))
    # Rule 2: in_domain = the zone has PTDF coefficients in that hour's domain (any row, any kind)
    in_dom = c.groupby("ts_utc")[[f"ptdf_{z}" for z in ZONES]].apply(lambda d: d.notna().any())
    in_dom.columns = [col.removeprefix("ptdf_") for col in in_dom.columns]

    # Rules 1-2 applied: physical rows with a valid ratio; ratio clipped to [0, 1]
    c = c[c.physical & c.ram_fmax.notna()].copy()
    c["tight"] = 1.0 - c.ram_fmax.clip(0.0, 1.0)
    feats = pd.concat([zone_features(c, z) for z in ZONES], ignore_index=True)
    del c

    # Rule 6: DE_AT_LU (before the 2018-10-01 split) is reported under DE_LU; AT is not in the domain then
    feats["zone"] = feats.zone.replace({"DE_AT_LU": "DE_LU"})
    in_dom["DE_LU"] = in_dom.DE_LU | in_dom.DE_AT_LU
    in_dom = in_dom.drop(columns="DE_AT_LU")
    assert not feats.duplicated(["ts_utc", "zone"]).any()

    # Rule 7: full hourly grid x 12 zones; hours absent from JAO and zones outside the domain are flagged, not invented
    grid_hours = pd.date_range(hours.index.min(), hours.index.max(), freq="h", tz="UTC")
    grid = pd.MultiIndex.from_product([grid_hours, OUT_ZONES], names=["ts_utc", "zone"]).to_frame(index=False)
    out = grid.merge(hours, left_on="ts_utc", right_index=True, how="left")
    out["jao_hour"] = out.regime.notna()
    out["n_phys_hour"] = out.n_phys_hour.fillna(0).astype("int32")
    out["in_domain"] = in_dom.stack().reindex(pd.MultiIndex.from_frame(out[["ts_utc", "zone"]])).eq(True).values
    out = out.merge(feats, on=["ts_utc", "zone"], how="left")
    out["n_rel"] = out.n_rel.fillna(0).astype("int16")
    out["cnec_avail"] = out.n_rel > 0
    out["regime"] = out.regime.astype("string")
    cols = ["ts_utc", "zone", "regime", "jao_hour", "in_domain", "cnec_avail", "n_phys_hour",
            "n_rel", "min_ratio", "wmean_tight", "top3"]
    return out[cols].sort_values(["ts_utc", "zone"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="staging/jao_cwe_core_hourly_views", type=Path)
    ap.add_argument("--out", default="staging/jao_zone_hourly", type=Path)
    a = ap.parse_args()
    out = build(a.src)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out / "jao_zone_hourly.parquet", index=False)
    print(f"wrote {a.out / 'jao_zone_hourly.parquet'}: {len(out):,} rows, "
          f"{out.ts_utc.min()} -> {out.ts_utc.max()}, cnec_avail {out.cnec_avail.mean():.1%}")
    print(out.groupby("zone").agg(hours_avail=("cnec_avail", "sum"), first=("ts_utc", lambda s: s[out.loc[s.index, "cnec_avail"]].min()),
                                  wmean_med=("wmean_tight", "median"), min_ratio_med=("min_ratio", "median")).to_string())


if __name__ == "__main__":
    main()
