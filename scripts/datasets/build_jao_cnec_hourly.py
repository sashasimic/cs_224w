#!/usr/bin/env python3
"""Build jao_cnec_hourly: one row per physical constraint (CNEC) per hour, the constraint-node table
for the proposal's zone-constraint graph.

Input:  the JAO bundle (jao_cnec_hourly.parquet + cnec_provenance.parquet, same row order).
Output: staging/jao_cnec_hourly/jao_cnec_hourly.parquet

The rules are documented in the dataset doc (jao_cnec_hourly.md); this file is the executable version.

    scripts/datasets/build_jao_cnec_hourly.py [--src staging/jao_cwe_core_hourly_views] [--out staging/jao_cnec_hourly]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from epgnn.jao import SOURCE_ZONES, SYNTHETIC, ZONES  # noqa: E402


def build(src: Path) -> pd.DataFrame:
    prov = pq.read_table(src / "cnec_provenance.parquet",
                         columns=["cnec_key", "CriticalBranchName", "CneName", "Tso",
                                  "RemainingAvailableMargin", "Ram", "Fmax"]).to_pandas()
    c = pq.read_table(src / "jao_cnec_hourly.parquet",
                      columns=["ts_utc", "cnec_key", "ram_fmax", "regime"] + [f"ptdf_{z}" for z in SOURCE_ZONES]).to_pandas()
    assert len(prov) == len(c) and (prov.cnec_key.values == c.cnec_key.values).all(), \
        "provenance rows must align positionally with jao_cnec_hourly"

    # Rule 1: physical constraints only. CWE names the line in CriticalBranchName, Core in CneName.
    c["cne_name"] = prov.CriticalBranchName.fillna(prov.CneName).fillna("").values
    c["tso"] = prov.Tso.fillna("").values
    # Rule 2: RAM and Fmax in MW from the source fields (CWE: RemainingAvailableMargin, Core: Ram)
    c["ram"] = pd.to_numeric(prov.RemainingAvailableMargin.fillna(prov.Ram), errors="coerce").values
    c["fmax"] = pd.to_numeric(prov.Fmax, errors="coerce").values
    del prov
    c = c[~c.cne_name.str.upper().str.contains(SYNTHETIC) & c.ram_fmax.notna()].copy()

    # Rule 3: DE_AT_LU (before the 2018-10-01 split) is reported under DE_LU
    c["ptdf_DE_LU"] = c.ptdf_DE_LU.fillna(c.ptdf_DE_AT_LU)
    c = c.drop(columns="ptdf_DE_AT_LU")

    c["regime"] = c.regime.replace({"CORE": "Core"})
    # Rule 4: market_day_cet, same definition as the ENTSO-E tables (calendar day in Europe/Brussels)
    c["market_day_cet"] = c.ts_utc.dt.tz_convert("Europe/Brussels").dt.strftime("%Y-%m-%d")
    cols = ["ts_utc", "market_day_cet", "regime", "cnec_key", "cne_name", "tso", "ram", "fmax", "ram_fmax"] \
        + [f"ptdf_{z}" for z in ZONES]
    return c[cols].sort_values(["ts_utc", "cnec_key"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="staging/jao_cwe_core_hourly_views", type=Path)
    ap.add_argument("--out", default="staging/jao_cnec_hourly", type=Path)
    a = ap.parse_args()
    out = build(a.src)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out / "jao_cnec_hourly.parquet", index=False)
    print(f"wrote {a.out / 'jao_cnec_hourly.parquet'}: {len(out):,} rows, "
          f"{out.ts_utc.min()} -> {out.ts_utc.max()}, {out.ts_utc.nunique():,} hours")
    per_hour = out.groupby(["regime", "ts_utc"]).size().groupby("regime").describe()[["mean", "min", "50%", "max"]]
    print("constraints per hour:\n", per_hour.round(0).to_string())
    print("ptdf non-null share:\n", out[[f"ptdf_{z}" for z in ZONES]].notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
