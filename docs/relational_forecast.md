# Relational electricity data and CORE constraint graphs

The catalog gives SQL and graph models the same source rows and forecast clock. Its forecasting task is to predict every hourly day-ahead price for Brussels delivery day D at noon on the preceding local calendar date.

The [five-seed validation report](core_constraint_validation.md) summarizes the completed comparison.

## Build and run

From the repository root, with the audited dataset and original JAO export available:

```bash
uv pip install --python .venv/bin/python 'duckdb>=1.3,<2'
.venv/bin/python -m epgnn.relational_forecast build
.venv/bin/python -m epgnn.relational_forecast validate
.venv/bin/python -m unittest discover -s tests/relational_forecast -v
.venv/bin/python -m epgnn.relational_forecast smoke
.venv/bin/python -m epgnn.relational_forecast run
```

The default config pins the audited dataset and the existing 49-market benchmark arrays. The experiment reuses their chronological price lags and training-derived physical topology. It obtains the flat JAO inputs and constraint graphs from the new catalog. See `configs/experiments/relational_forecast.json` for paths and settings.

Open `staging/relational_forecast/current/forecast.duckdb` read-only. The catalog and a validation report are published atomically after checking the pinned source hashes and relational integrity. Parquet views refer to resolved local source paths. On another checkout, rebuild the catalog using a locally built audited dataset. The experiment requires ROCm. It saves epoch checkpoints, validation predictions, metrics and comparisons under `runs/relational_forecast/`. Rerun the same command to resume with the same settings and source fingerprint.

To regenerate the tracked report, pass the completed validation directory to `scripts/reports/build_core_constraint_report.py` using `.venv/bin/python`. The report builder checks the prediction grid and masks for every job before aggregating the scores.

## Tables and keys

| Relation | Key | Data and links |
|---|---|---|
| `area` | `zone` | Canonical area and whether it has a price grid. Includes unresolved and virtual identities. |
| `source_area_map` | `source, area_code` in this release | Raw area identifier and name, canonical area, compatibility and validity dates. JAO has its own `jao_area_map` with regime and dated source tokens. |
| `delivery_hour` | `ts_utc` | UTC hour, Brussels local delivery date, UTC offset, previous-day local-noon origin and split. The two autumn repeated hours have distinct keys. |
| `forecast_target`, `strict_forecast` | `zone, ts_utc` | Canonical target scaffold, masks and audit-approved predictor view. |
| `price`, `load_forecast` | `AreaCode, effective_utc` within source | Source values, quality and fill flags, and dated area mapping. Unmapped or outside-lifetime records retain their source ID. |
| `weather` | `zone, effective_utc` | Published zonal series in source units. Germany-only compatibility remains unresolved in the underlying audited dataset. |
| `border_hour` | `release_id, kind, source_from, source_to, ts_utc` | Directional physical flows, CORE schedules, CWE BEC and MaxBex in MW. Published zeros and missing values stay distinct. |
| `jao_domain_hour`, `jao_zone_summary` | `regime, ts_utc`, with `zone` for summaries | Publication coverage and zone counts or minimum RAM ratios. |
| `cnec_observation` | `cnec_key` | One observed presolved row, linked to the domain by regime and source hour. RAM and Fmax are in MW. The key encodes the original hash and row. |
| `cnec_ptdf` | `cnec_key, zone` | Signed coefficients, including published zeros. Missing coefficients have no row. Both German bidding-zone identities retain their own columns in the pinned export. |
| `source_release` | `release_id` | Source artifact paths, versions, checksums and any available archive timing. |

Analytical timestamps in this release are hourly or calendar boundaries. Original source Parquet retains the upstream timestamp types and precision. Archive or update times are distinct from publication times. The catalog keeps publication time null where the source provides no verified history.

The forecast target's price is an outcome. Historical price lags in the benchmark use the guarded prior-day auction convention and exact same-zone weekly price. Their historical revision vintages are unverified. The strict feature view follows the audited registry. The CORE experiment is conditional because the JAO files do not establish what version was published before each origin.

## Graph experiment

For each delivery hour, use the CORE domain at exactly 48 elapsed hours earlier, provided the source interval ended by the forecast origin. A constraint observation is an hourly node. Its signed, finite, nonzero PTDFs define attention edges to affected priced zones. The architecture sends messages zone → constraint → zone and has a local price-feature path. Virtual hubs stay in the catalog. The model consumes the 12 CORE priced-zone coefficients. There is no persistent constraint embedding, synthetic-name filter or shadow-price feature.

The four variants are the interconnector GAT with price and calendar features, the same GAT with flat JAO, a bipartite GAT with flat JAO, and the same bipartite GAT with the zones on the constraint edges permuted within each hour. Flat summaries are real in both bipartite variants. The shuffle preserves the hourly coefficient values and constraint-node data, so the real-versus-shuffled comparison tests the attribution to zones. Flat-versus-bipartite also changes the architecture and the detail of the available data.

Fit dates are June 9, 2022 through March 31, 2024. The validation period is April 1, 2024 through March 31, 2025, with all 49 benchmark markets. Determine whether the edges help primarily using the CORE-covered subset. Compare against weekly persistence on rows where the weekly lag exists. Resample Brussels calendar weeks jointly across zones when estimating uncertainty for paired model differences. Validation selected the epoch checkpoints. The test period was kept out of this experiment.

## External data extensions

Add a dated many-to-many mapping from country or region to bidding-zone identity. Key macroeconomic facts by geography, indicator, reference month and release vintage. For each origin, filter on verified publication time, then use the latest known vintage for each reference period. Record the unit, seasonal adjustment and series definition so a rebasing does not become a false jump.

Eurostat's industrial production vintages are the first candidate to test as a demand-context variable. Use manufacturing separately from energy production if series coverage allows. Monthly unemployment is a slower comparison. A monthly value is a single release-time fact even if multiple deliveries use it. [Eurostat vintage metadata](https://ec.europa.eu/eurostat/cache/metadata/EN/euroind_vtg_esms.htm)

For air quality, keep station and pollutant readings at source resolution. Store station location and station type, use a dated station-to-zone mapping, and distinguish provisional readings from retrospectively verified data. Trainable inputs need an observation interval that ended before the forecast origin and verified data availability. [EEA air-quality data](https://www.eea.europa.eu/en/about/contact-us/faqs/where-can-i-access-the-latest-air-quality-data-in-europe/)

A relational GNN can start from the forecast's zone and hour and follow these dated relations. Build each neighborhood under the seed's cutoff, including reverse edges, so later observations cannot reach earlier forecasts through shared zone or country nodes. Compare with a tabular model that gets SQL aggregates from the same eligible rows. [Stanford RelBench](https://star-project.stanford.edu/relbench/)
