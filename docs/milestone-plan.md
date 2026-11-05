# Milestone plan: simplest end-to-end version

Written 2026-11-01, revised the same day after an independent review (section 8). Milestone due **Tuesday 2026-11-10**.

Goal: by the milestone, have one pipeline that goes from the parquet files to a trained graph model, a set of baselines, and an evaluation table, all runnable with one command. Everything not required for that is deferred. The deferrals are listed in section 6 so nothing is lost.

## 1. Scope

| Keep for the milestone | Defer |
|---|---|
| Zone nodes with interconnector edges (ablation rungs 1–3 of the proposal) | Constraint (CNEC) nodes, PTDF edges, shadow prices (rungs 4–5) |
| 38 bidding zones, Jan 2019 – Sep 2025 | 2016–2018 history; Germany pre-split; Sicily; Ireland |
| Prices, load forecasts, physical flows, country-level weather as inputs | Per-zone weather, weather forecasts, heat slices beyond a basic heatwave flag |
| Targets on the CET market day; the two DST days per year dropped | 23/25-hour outputs with masks |
| Naive, local-GRU, flat-neighbour, and GNN models | HGT, learned temporal aggregation, US transfer |
| MAE, spread MAE, separation F1 | Full slice analysis (renewable surplus, congested hours) |

This is the proposal's own fallback: *"in those cases we drop the constraint nodes and zones exchange messages only over their interconnectors, so one model covers every case."* The milestone builds exactly that case; constraint nodes are added later without changing the pipeline.

## 2. Data definition

### 2.1 Zones (38)

All candidate zones have 100% valid prices from 2019-01-01 and ≥96.7% valid load forecasts in every period (lowest: Estonia, 2022). Verified against the v1.1 parquets.

| Group | Zones |
|---|---|
| West/Central | FR, BE, NL, DE-LU, AT, CH, CZ, SK, PL, HU, SI, HR |
| Iberia | ES, PT |
| Nordic | NO1, NO2, NO3, NO4, NO5, SE1, SE2, SE3, SE4, DK1, DK2, FI |
| Baltic | EE, LV, LT |
| South-east | RO, BG, RS, GR |
| Italy | IT-North, IT-Centre-North, IT-Centre-South, IT-South, IT-Sardinia |

Dropped: **IT-Sicily** (no interconnector to any zone in the set; its link runs through IT-Calabria, which has prices only from 2021), **Ireland** (load forecast missing 2022 onward), GB, Ukraine, Balkans-2023 zones, virtual/cable areas.

Why Jan 2019 rather than 2016: DE-LU and AT exist as separate zones only from Oct 2018, DE-LU load has 840 missing hours in Q4 2018, Italian internal flows start in 2018, and PL/HR/BG/RS prices are complete from 2018. A 2019 start makes every zone and edge clean with zero special-casing. Cost: training shrinks from 6 years to 3. The Germany splice (DE-AT-LU → DE-LU) can extend training later.

### 2.2 Edges (68 + 3)

An unordered zone pair is an edge if flows are reported in **both directions** with ≥98% valid hours over 2019–2025. That gives **68 edges**, including BE–DE-LU (ALEGrO; ENTSO-E reports the pair from Oct 2018) and GR–IT-South. Three more pairs are cables commissioned during the period; `edges.csv` records a `valid_from` date for each:

| Edge | Valid from | Coverage 2019–25 |
|---|---|---|
| DK1 – NL (COBRA) | Sep 2019 | 90% |
| DE-LU – NO2 (NordLink) | Mar 2021 | 75% |
| HU – SI | 2022 | 48% |

**Message passing** uses the static graph of all 71 edges every day (142 directed edges for `GATv2Conv`, plus self-loops). **Flow features and spread metrics** use only edges with `valid_from ≤ D` (section 2.4, 4). DE-LU is the hub (degree 11); PT and IT-Sardinia are leaves (degree 1).

### 2.3 Training table

Following lecture 12: one explicit table of examples, with every input derived through a time filter from the example's forecast time.

One row per `(zone, D)` where D is a **CET market day** (`MarketDayCET` in the parquets) from 2019-01-09 to 2025-09-30.

| Field | Definition |
|---|---|
| `zone` | one of the 38 |
| `D` | target market day (CET) |
| `t_cut` | forecast time = D-1 10:00 UTC. The day-ahead auction closes at 12:00 CET = 11:00 UTC in winter, 10:00 UTC in summer; 10:00 UTC is at or before the real cutoff year-round. |
| `y` | 24 prices `PriceEUR` for the hours with `MarketDayCET == D` |
| `y_mask` | `IsValidTarget` per hour |
| `split` | train if D ≤ 2021-12-31; val if 2022; test if D ≥ 2023-01-01 |

Days with 23 or 25 hours (the two DST changes per year, 13 days in 2019-01-09 – 2025-09-30) are **dropped**. Days where D-1 has 23 or 25 hours are kept: the history window is defined by hour count, not by day (section 2.4).

Counts (verified from `MarketDayCET`): train 1,082 days, val 363, test 999; × 38 zones ≈ 93k zone-days. The history tensor is about 0.6 GB in float32, so it fits in memory and trains on a laptop.

### 2.4 Inputs per example

**History window:** the last **168 hours with `MarketDayCET ≤ D-1`**. Prices for local day D-1 are published at ~12:45 CET on D-2, so every hour in the window is known at `t_cut`. A DST day inside the window just shifts where the window starts.

Per hour in the window, per zone, the GRU input vector:

| Feature | Source column | Known at `t_cut`? | Rule |
|---|---|---|---|
| price | `PriceEUR_input` | yes (published D-2) | use as is |
| load forecast | `LoadForecastMW_input` | yes | fill 0 after scaling where null; `load_avail` flag |
| net import | `FlowMW_input`, Σ in − Σ out over the zone's **active** edges (`valid_from ≤ D`) | hours with start ≤ D-1 08:00 UTC (ENTSO-E publishes up to 1 h after the hour) | later hours → 0, `flow_avail` = 0; a missing direction on an active edge → that edge contributes 0 and `flow_avail` = fraction of active edges present |
| temperature | `temp_pw`, `weather_hourly` | hours with start < `t_cut` (approximation: ERA5 is a reanalysis with days of latency; in operation these would be observations or short-range forecasts) | later hours → 0, `wx_avail` = 0 |
| wind cf, solar cf | `wind_cf`, `solar_cf` | as temperature | as temperature |
| hour of day, day of week | derived | always | sin/cos |

**Day-ahead inputs for D** (known at `t_cut`):

| Feature | Source | Note |
|---|---|---|
| load forecast, 24 hours of D | `LoadForecastMW_input` | ENTSO-E publishes day-ahead load forecasts no later than 2 h before day-ahead gate closure, i.e. by 10:00 CET on D-1. `IsAvailabilityVerified` is false in the data; the write-up states inputs are archived historical versions. Null → 0 after scaling, `load_avail` flag. |
| calendar features for D | derived | day of week, month |

**Not used as D inputs:** weather for D (observed, not forecast), flows for D, wind/solar generation forecasts (not in the delivered data).

`build_table.py` asserts that `samples.parquet` has no NaN in any input column and that every input hour is strictly before `t_cut` or belongs to a market day ≤ D-1.

**Rung-2 features (model 2 only):** for each zone and each history hour, the **mean over active neighbours** of price and load forecast, and for day D the mean neighbour load forecast. These are extra channels built in `build_table.py`; the model that consumes them has no graph.

### 2.5 Scaling

- Prices: $\tilde y = \operatorname{asinh}(y / 50)$, applied per hour, no fitted parameters. Handles negative prices and compresses the 2022 spikes (a 500 €/MWh price maps to 3.0, not z ≈ 25 as a train-fitted z-score would). Metrics are computed in EUR/MWh after inverting. Per-zone z-score on train is kept as a one-line ablation.
- Load, flows, temperature: per-zone z-score fitted on train.
- Capacity factors: already in [0, 1].

## 3. Models

All share the same training table, dataloader, loss and evaluation. Each is one config file.

| # | Name | Ladder rung | What it is |
|---|---|---|---|
| 0a | `naive_d1` | — | $\hat y_h$ = price at hour h on market day D-1 |
| 0b | `naive_d7` | — | hour h on market day D-7 |
| 1 | `local_gru` | local | shared GRU over the 168-h history; its output is concatenated with an MLP embedding of D's 24 load forecasts and calendar features; MLP → 24 |
| 2 | `flat_neighbours` | flat neighbours | as 1, with the rung-2 neighbour-mean channels added to the GRU input and the D block. No graph in the model. |
| 3 | `gnn_interconnector` | interconnector graph | as 1, then K = 2 layers of `GATv2Conv` over the 38-node graph, then MLP → 24 |
| 3b | `gnn_mean1` (optional) | — | as 1, then one layer of mean aggregation over neighbours' embeddings (GraphSAGE-mean). Separates "any graph" from "attention + 2 hops". |

The GNN in PyG terms:

```
z  = concat( GRU(history[168, F]),  MLP(dayahead[24, F']) )   # [38, H]
h1 = GATv2Conv(z,  edge_index)                                 # [38, H]
h2 = GATv2Conv(h1, edge_index)                                 # [38, H]
y  = MLP(h2)                                                   # [38, 24]
```

Each day is one `Data` object with 38 nodes; a batch is a `Batch` of days. H = 64, 4 attention heads, dropout 0.1, Adam 1e-3, early stopping on validation MAE. Starting values, not tuned. Every model is run with **3 seeds**; the table reports mean ± std.

**Loss:** L1 on $\tilde y$, plus λ × L1 on **spreads in EUR** over the edges active on D: invert the asinh, then $|(\hat y_u - \hat y_v) - (y_u - y_v)| / 50$. λ = 0 for the first runs, then λ = 1 as one extra run. Hours with `y_mask = 0` are excluded.

Model 2 is the "expert data scientist" column from lecture 12; model 3 is the "GNN" column. Their difference is the H1 test.

## 4. Evaluation

Computed on validation (2022) and test (2023–Sep 2025), in EUR/MWh.

| Metric | Definition |
|---|---|
| MAE | mean over valid zone-hours |
| spread MAE | over **active** edges and hours: $|(\hat y_u - \hat y_v) - (y_u - y_v)|$ |
| separation F1 | event = $|y_u - y_v| > 5$ €/MWh on an active edge-hour; F1 of predicted vs actual events. Reported on **coupled borders only**: edges touching CH (outside SDAC) and RS (not coupled) are excluded, since those borders are "separated" nearly every hour. GR (coupled Dec 2020) and BG (May 2021) are included; note this in the write-up. |

Slices for the milestone: **all hours** and **heatwave days** (`heatwave = 1` in `weather_daily` for the zone's country on D). Per-zone MAE as an appendix table.

One results table: 5 models × 2 splits × 3 metrics (mean ± std over seeds), plus the heatwave slice for MAE. That is the milestone deliverable.

## 5. Repository layout

```
cs_224w/
├── configs/
│   ├── zones.csv            # zone id, AreaCode, weather country, group
│   ├── edges.csv            # u, v, valid_from, keep, reason
│   └── models/*.yaml        # one per model in section 3
├── epgnn/
│   ├── build_table.py       # parquets → samples.parquet; the ONLY file with the cutoff logic
│   ├── dataset.py           # samples → PyG Data per day
│   ├── models.py            # naive, local_gru, flat_neighbours, gnn
│   ├── train.py             # one model, one config, one seed → checkpoint + val metrics
│   ├── evaluate.py          # checkpoints → metrics table
│   └── metrics.py
├── scripts/run_all.sh       # builds table, trains all models × 3 seeds, writes results.md
├── datasets/                # ignored
├── docs/
└── lectures/                # ignored
```

`build_table.py` reads the flows parquet with pyarrow filter pushdown on the 142 directed `(OutAreaCode, InAreaCode)` pairs, not a full 36M-row load into pandas.

## 6. Data adjustments needed

None requires new downloads; the five parquets in `datasets/` are sufficient.

1. **Zone mapping file.** The three ENTSO-E tables use identical `AreaCode`s for all 38 zones (verified). `build_table.py` filters on `AreaCode`, so the load table's country and control-area rows are ignored automatically.
2. **Weather country mapping.** `weather_hourly.zone` is a country code: NO1–5 → `NO`, SE1–4 → `SE`, DK1/DK2 → `DK`, IT-* → `IT`, DE-LU → `DE` (the weather doc says DE and LU were merged), everything else → its own code. All 38 map to a code present in the file (verified). Nordic and Italian zones share weather for now; fix later by removing the country-merge step in the weather pipeline and re-running per bidding zone.
3. **Weather columns.** The delivered `weather_hourly.parquet` has a single `wind_cf` and `solar_cf` (the doc describes rooftop/utility and onshore/offshore variants that were combined). Column names in this plan come from the parquets; the delivered schema is in the dataset doc (`datasets/weather_v1.0.md` after `pull`).
4. **Weather daily 2026-10-01 row.** Outside the study period; filter by date.
5. **Use `_input` columns for history**, `PriceEUR` + `IsValidTarget` for targets. Never let a filled value become a target.
6. **Load-forecast gaps > 48 h** are null in `LoadForecastMW_input` (EE 2022 ≈ 290 h; DE-LU 2022 = 48 h). Handled by `load_avail` + zero-fill in `build_table.py`.
7. **Net import** derived in `build_table.py` over active edges only (section 2.4).
8. **JAO.** Not needed. Before adding constraint nodes: resolve the publication-time question (the final domain is published ~10:30 on D, after the cutoff), obtain shadow prices separately, download Jan–Sep 2025.
9. **Availability caveat.** `IsAvailabilityVerified` is false for load and flows. The write-up states that inputs are archived historical versions, which may include later revisions.

## 7. What the milestone write-up can claim

- A working spatio-temporal graph pipeline over 38 European bidding zones and 71 interconnectors, 2019–2025, with the information cutoff enforced in one file and checked by assertions.
- First results on H1 (graph vs flat neighbour features) on 2022 validation and 2023–2025 test, with seed variance.
- A spread-aware loss and the separation-event metric, with first numbers.
- The constraint-node extension (H2) and per-zone weather (H3) as next steps, with the data issues already diagnosed.

## 8. Check log

**Verified against the v1.1 parquets, 2026-11-01:**

- All 38 zones have 100% valid prices from 2019-01-01 in train, val and test.
- Lowest load-forecast coverage in any zone-period is 96.7% (EE, 2022); all others ≥ 98%.
- 71 zone pairs have flows in both directions; 68 at ≥ 98% coverage; the 3 exceptions are cables with known start dates.
- IT-Sicily has no edge to any zone in the set.
- Price, load and flow tables use the same `AreaCode` for every zone.
- All 38 weather country codes exist in `weather_hourly`; no gaps 2015–2026.
- `PriceEUR_input` is never null where `IsValidTarget` is true.
- A hand-built example (FR, D = 2022-07-19): the last history hour is on local day D-1 even in summer, so it was published before `t_cut`.

**Independent review, 2026-11-01**, and what changed:

1. UTC-day targets included 1–2 hours cleared by the auction on D, leaking the load forecast for those hours and making `naive_d1` use unpublished prices. → Targets moved to the CET market day; DST days dropped. (§2.3, §2.4, §3)
2. Net import over the static edge set zeroed the flow feature for six zones before their cables existed. → Active edges only. (§2.4)
3. No rule for null load forecasts. → `load_avail` + zero-fill + NaN assertion. (§2.4, §6)
4. `flat_neighbours` as a mean over GRU embeddings was a 1-hop GNN, not a flat baseline. → Raw neighbour-mean channels; embedding-mean kept as optional 3b. (§3)
5. Flows publish up to 1 h after the hour. → Flow history ends at D-1 08:00 UTC. (§2.4)
6. Reviewer thought ALEGrO and GR–IT-South were missing from the edge list. Checked: both are in the 68. DE-LU degree corrected to 11 (with NordLink). (§2.2)
7. Spread metrics on inactive edges and non-coupled borders; spread loss in z-space. → Active edges; CH/RS excluded from F1; spread loss in EUR. (§3, §4)
8. Train-fitted z-score puts 2022 at z ≈ 8–25. → `asinh(y/50)` default, z-score as ablation. (§2.5)
9. Column names not in the dataset docs. Partly moot (names come from the parquets, which are in `datasets/`); ERA5-latency caveat added. (§2.4, §6)
10. Weather join was scheduled after the table that needs it. → Build it together with the training table.
11. Serial critical path, 36M-row flows read, single seed. → Stub `Data` for parallel model work; pyarrow pushdown; 3 seeds. (§3, §5)
