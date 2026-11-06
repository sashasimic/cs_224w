# Zone-node table: hourly features and target per bidding zone

| | |
|---|---|
| Dataset id | `zone_nodes_hourly` |
| Version | **v1.0** |
| Bucket | `gs://cs224w-epgnn-datasets/zone_nodes_hourly/v1.0/` |
| Local file(s) | `datasets/zone_nodes_hourly_v1.0.md`, `datasets/zone_nodes_hourly_v1.0.parquet` |
| Source | built by Sophie-An's `build_model_dataset.py` from ENTSO-E v1.2 (prices, load forecast, physical flows), weather_hourly v1.1 (PECD/ERA5, 45 zones) and the Core-only JAO views (`jao_core_hourly_views`) |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("zone_nodes_hourly", "zone_nodes_hourly.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | first build: 4.24 M rows, key `ts_utc, zone`, delivery days 1 Jan 2016 – 30 Sep 2026; 45 zones; `split` column |

Companion dataset: `zone_edges_hourly` (same build, same `split` column). Constraint nodes come from `jao_cnec_hourly`.

## Build notes (from the build README)

Built by `build_model_dataset.py` (project folder) from `data/weather_hourly_bzone_2010_2026.parquet`, `data/entsoe/entsoe_*_v1.2.parquet` and `data/jao_core_hourly_views/`.

The data cover delivery days 1 Jan 2016 to 30 Sep 2026. ENTSO-E v1.2 runs to 31 Oct 2026 and JAO to 29 Oct 2026, but the weather file ends on 30 Sep 2026. All timestamps are `ts_utc`, the UTC start of the hour. The zones are the 45 bidding zones of the weather file.

| File | Rows | Key | Use |
| --- | ---: | --- | --- |
| `nodes_hourly.parquet` | 4.24 M | `ts_utc, zone` | Node features and the target |
| `edges_hourly.parquet` | 17.0 M | `ts_utc, from_zone, to_zone` | Edge features on 180 directed interconnectors |
| `splits/<split>_nodes.parquet`, `splits/<split>_edges.parquet` | | same | The same rows, one file per split (`pretrain`, `train`, `val`, `test`) |

Both tables have a `split` column (see Study design and splits).

Constraint nodes (checkpoint 4) come straight from `jao_core_hourly_views/jao_cnec_hourly.parquet`. It uses the same `ts_utc`, and its `ptdf_DE` refers to `DE_LU`.

## Nodes

| Column | Meaning | Use for day D? |
| --- | --- | --- |
| `delivery_date`, `hour_local`, `dow`, `month` | Brussels delivery calendar (23- and 25-hour days at DST changes) | Yes |
| `price` | Day-ahead price, EUR/MWh. **Target.** Null where invalid or the zone did not exist | Target only |
| `has_price` | `price` is not null; use as the loss mask | |
| `price_filled` | The lag inputs for this hour came from a gap fill | |
| `price_lag24`, `price_lag168` | Price 1 day and 1 week earlier (short gaps filled) | Yes |
| `load_fc` | ENTSO-E day-ahead load forecast, MW | Yes (see caveats) |
| `temp_pw`, `temp_anom`, `wind_cf`, `low_wind`, `solar_cf` | Same-hour weather | Only as a perfect forecast (ablation B) |
| `*_lag24` weather | Weather at the same hour the day before | Yes (main setting A) |
| `net_import` | Physical imports minus exports with the other modelled zones, MW | No |
| `net_import_lag48` | The same, two days earlier | Yes |
| `n_presolved`, `min_ram_ratio` | JAO Core domain: number of presolved constraints affecting the zone, and their smallest RAM/Fmax | Yes, if published before gate closure (unverified) |
| `in_core` | Zone is in the Core flow-based region (JAO columns are filled from 9 Jun 2022) | |
| `split` | `pretrain`, `train`, `val` or `test`, by delivery date | |

## How the zones were mapped

- **DE_LU** gets the DE-AT-LU price and flows before 1 Oct 2018. **AT** has no price before that date.
- **DE_LU** load before the split is Germany + Luxembourg.
- **IE_SEM** load falls back to Ireland + Northern Ireland when the bidding-zone series is missing.
- The JAO hub **DE** is DE_LU. The ALEGrO virtual hubs (ALBE, ALDE) are dropped from the node table.
- Italian virtual and pre-2021 zones (Rossano, Brindisi, Foggia, Priolo, the North-XX zones, SACOAC and SACODC) are dropped.

## Known gaps

- **No price at all:** BA.
- **Price only part of the period:**
  - GB: until Dec 2020.
  - ME and MK: from 2023.
  - AT: from Oct 2018.
  - IT_CALA: from 2021.
  - HR: from Oct 2017.
  - BG: from Nov 2016.
  - RS: from Dec 2016.
- **Load missing:**
  - GB: after mid-2021.
  - IE_SEM: mostly missing after mid-2021, and entirely missing from Oct 2025. This is a gap in the ENTSO-E source.
- **JAO:**
  - Core only, from 9 Jun 2022.
  - Five domain days are missing in 2026. About 98.6% of Core zone-hours from Oct 2025 have domain data.
  - Shadow prices and published net-position bounds are not yet acquired.
- **Load forecast vintage:** ENTSO-E update times are later than the delivery hours, so revisions after gate closure cannot be ruled out.
- **Southern Norway:** NO_1, NO_2 and NO_5 have identical weather. The source data has only one zone covering all three.
- **15-minute prices:** from 1 Oct 2025, each hourly price is the mean of four 15-minute prices, done in ENTSO-E v1.2. Italy and IE_SEM have 1–2 masked days.
- **v1.2 revisions to earlier data:** compared with v1.1, prices are identical. RS load changed on two days in July 2022. A few hundred hours of Balkan physical flows changed, which moves `net_import` for 8 zones in 863 hours.
- **Italy:** IT_CALA became a bidding zone only in 2021, so weather rows exist before then but there are no matching prices.

## Study design and splits

The main model covers the Core flow-based period. The longer ENTSO-E and weather history before Core, which has no constraints, is used for the baseline and for pretraining. All data end on 30 Sep 2026, because PECD had not published October 2026 by 5 Nov 2026.

| Stage | Delivery days | Days | Priced zone-hours | Graph |
| --- | --- | ---: | ---: | --- |
| Pretrain / baseline | 1 Jan 2016 – 8 Jun 2022 | 2,351 | 2.25 M | Zones + interconnectors, no constraint nodes |
| Train (Core) | 9 Jun 2022 – 31 Mar 2024 | 662 | 667 k | + JAO constraint nodes, maxbex |
| Validation | 1 Apr 2024 – 31 Mar 2025 | 365 | 377 k | Same |
| Test | 1 Apr 2025 – 30 Sep 2026 | 548 | 565 k | Same |

- **Files:** `splits/` holds one nodes and one edges file per split; the `split` column in the full tables is the same labelling. The split dates are set in `SPLITS` at the top of `build_model_dataset.py`.
- **Lags cross boundaries:** the first week of each split has lag columns taken from the end of the previous split. That is past data, so it is not leakage, and it means each split can be used on its own.
- **Order:** train, then validation, then test, so hyperparameters and early stopping never see data from after the test start.
- **Baseline:** the same GNN without constraint nodes, trained on Jan 2016 – Mar 2024 and scored on the same test set. The gap to the Core model is what the JAO constraints add.
- **Before Oct 2018:** DE_LU carries the joint DE-AT-LU price and AT has no price, so mask AT out of the loss there.
- **Scaling:** fit normalisation statistics on training rows only (pretrain + Core train).
- **Shift inside the test set:** prices move to 15-minute MTUs on 1 Oct 2025. Hourly values are averages of the quarter-hours, but the price distribution may shift. Report test errors before and after that date separately.
- **Heat stress:** validation includes summer 2024; test includes summers 2025 and 2026. Report heatwave zone-days separately.
- **Adding October 2026 later:** once PECD publishes it, run `python3 download_pecd_all.py --years 2026 --months 10` in `weather/`, re-run the weather processing and the 2010–2026 export, then `python3 build_model_dataset.py --end "2026-10-31 23:00"`.

Rebuild (about 1 minute and 3 GB of RAM):

    python3 build_model_dataset.py

## Leakage rule

For delivery day D, forecasts are made at noon on D−1. Use only:
- the calendar columns,
- `load_fc`,
- the `_lag` columns,
- and the JAO domain columns.

Same-hour weather counts as a perfect forecast, so use it only for the ablation. `net_import`, `flow_phys` and `flow_sched` are outcomes and must never be inputs for the same day. The 24-hour weather lag includes the afternoon of D−1, which hasn't happened yet at gate closure. That makes it a near-perfect short-range forecast; use `_lag48` (shift it yourself) for a strictly observed version.

## Checked against the agg_* datasets (2026-11-06)

Compared row by row with `agg_zones_hourly` v1.0 and `agg_zone_edges_hourly` v1.0, built independently from the same pinned sources:

- `price`, `has_price`, `price_lag24`, `price_lag168` and all weather columns are identical (max difference 1e-4, float32 rounding).
- `load_fc` equals `agg_zones_hourly.load_mw_input` for every zone except `DE_LU`, where this table uses the ENTSO-E country series Germany + Luxembourg for the whole period (the agg table uses the DE-AT-LU bidding-zone series before Oct 2018, which includes Austria, and DE-LU after).
- `net_import` equals imports minus exports recomputed from `agg_zone_edges_hourly.flow_mw_input`, except for HR, ME and RS where this table also counts the BA borders (BA is a node here, not in the agg tables).
- `n_presolved` and `min_ram_ratio` are **not zone-specific**: they take the same value for all 12 Core zones in an hour (e.g. 116 and 0.1996 on 2026-01-15 17:00 UTC for FR, DE_LU, AT and SK alike), because the relevance test is "PTDF != 0" and practically every constraint has a non-zero PTDF for every zone. Read them as Core-wide totals. The count includes the synthetic (external/equality) constraints; `jao_cnec_hourly` has 114 physical rows in that hour.
- JAO columns are Core only (from 2022-06-08 22:00 UTC); CWE constraints (2015-2022, BE/FR/NL/DE_LU/AT) are not used here, but are in `jao_cnec_hourly`.
- The grid is on Brussels delivery days: it starts at 2015-12-31 23:00 UTC and ends 2026-09-30 21:00 UTC (weather-limited); the agg tables run 2016-01-01 00:00 to 2026-10-31 22:00 UTC.

## The `split` column

The split is a date rule, kept here as a label for convenience (`pretrain` / `train` / `val` / `test` by delivery day, boundaries in the table above). The authoritative definition is `epgnn/splits.py`; loaders should filter by date from there rather than rely on the column, so the split can change without rebuilding this table. No per-split files are kept.
