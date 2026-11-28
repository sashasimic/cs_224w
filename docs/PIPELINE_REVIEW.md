# Data pipeline review, baselines and target MAE (26 Nov 2026)

How the data moves from raw files to model input, the shape at each stage, what is wrong or missing, and the baselines a result has to beat. All numbers were computed on this laptop on 26 Nov 2026 from `features/v1/` and the JAO views. MAE is pooled over all priced zone-hours, in EUR/MWh. Scripts: the scratchpad copies `baselines.py`, `lear_weather.py` and `jao_audit.py` (they can be moved into `cs_224w-c1_xgb_zone/scripts/` if wanted).

---

## 1. Headline

1. **The reference baseline is too weak.** The framework scores everything against weekly persistence (D-7, test 29.0). Same-hour D-1 persistence alone gets **23.8**. A plain linear LEAR model (Lago et al. 2021), using only the zone's own price lags, load forecast and weekday, gets **20.67 test / 19.18 val**. That is level with every GNN so far (test 20.40–20.64, val 19.71–20.03) and better than all of them on val. Reported `rel_mae` values (~0.70) therefore look much better than they are.
2. **Weather can be worth at most about 0.4–0.8 MAE**, at least in a linear model. LEAR with *perfect* same-day weather improves by 0.44 test / 0.80 val. Lagged weather (what the pipeline uses) makes LEAR slightly *worse* (+0.15 to +0.31). Seed noise is about ±0.1, so a weather claim needs several seeds and a paired test (section 6).
3. **The models cannot see the D-1 price curve.** D-1's 24 prices are all public at gate closure, and LEAR's strongest inputs are D-1's 24 prices for every target hour. In `gat_scope`, each slot sees only its own hour's `price_lag24`, and the GRU history is `price_lag24` up to the cutoff, which ends at **D-2** 10:00 UTC. This is the most likely reason the neural models only tie LEAR.
4. **Constraint data** is usable but needs four fixes: a PTDF reference that doesn't change the result, explicit removal of synthetic constraints, a decision on the publication-timing question, and handling of the policy-adjusted RAM (section 4).

---

## 2. The pipeline, stage by stage

### Stage 0: raw sources (`data/`)

| Source | Rows | Key | Notes |
|---|---:|---|---|
| `entsoe/entsoe_prices_v1.2` | | AreaName, DeliveryHourUTC | 62 areas. `PriceEUR` (gap-filled) and `PriceEUR_input` (raw); 15-minute prices averaged to the hour from 1 Oct 2025 |
| `entsoe/entsoe_load_forecast_v1.2` | | AreaName, DeliveryHourUTC | 68 areas, day-ahead total load forecast (MW) |
| `entsoe/entsoe_physical_flows_v1.2` | | Out/InAreaName, hour | Realised cross-border flows (an outcome) |
| `weather_hourly_bzone_2010_2026` | 6.6 M | time_utc, zone | ERA5/PECD reanalysis: `temp_pw`, `temp_anom`, `wind_cf`, `solar_cf`, `low_wind`. **Observed values, not forecasts** |
| `jao_core_hourly_views/jao_cnec_hourly` | 4,630,925 | ts_utc, cnec_key | Presolved Core constraints: `ram_fmax`, `presolved` (always True), 14 hub PTDFs, `shadow_price` (**all null**) |
| `jao_core_hourly_views/cnec_provenance` | 4,630,925 | cnec_key | ~110 text columns: names, EIC, TSO, hubs, contingency, Ram, Fmax, Frm, F0core, **Amr, MinRamFactor, LtaMargin**, PTDFs to CH and to the virtual hubs of the DC links (`Ptdf_DE_NO2_BigHub`, …) |
| `jao_core_hourly_views/jao_border_hourly` | 5,081,604 | ts_utc, from, to | `maxbex` (132 directed pairs), `flow` = scheduled exchange (38 pairs, an outcome) |
| `jao_core_hourly_views/jao_hourly` | 538,958 | ts_utc, zone | `n_presolved`, `min_ram_ratio`. `net_position`, `n_binding` and `cong_component` are **all null** |

Core JAO covers 9 Jun 2022 – 29 Oct 2026. Five domain days are missing: 16–17 Aug and 25–27 Sep 2026, all in the test split.

### Stage 1: `build_model_dataset.py` → `data/model/` (pinned as `zone_*_hourly` v1.0)

**`nodes_hourly`**: 4,240,035 rows = 45 zones × 94,223 UTC hours (a complete grid, 1 Jan 2016 – 30 Sep 2026).
- Keys: `ts_utc`, `zone`, `delivery_date` (Brussels), `hour_local`, `dow`, `month`, `split`.
- Target: `price` (filled), `has_price`, `price_filled`.
- Lags are row shifts on the UTC grid:
  - `price_lag24` and `price_lag168`, built from the *unfilled* `price_input`.
  - `temp_pw`, `temp_anom`, `wind_cf` and `solar_cf` with `_lag24`.
  - `net_import` with `_lag48`.
- Same-hour values: `load_fc`, weather (perfect-forecast ablation only) and `net_import` (an outcome).
- JAO (Core zones, same hour of day D): `n_presolved`, `min_ram_ratio`, `in_core`.

**`edges_hourly`**: 16,960,140 rows = 180 directed pairs × 94,223 hours.
- `flow_phys` with `_lag48` and `_lag168`.
- `maxbex`: Core borders only, same hour of day D.
- `flow_sched` with `_lag24`.

### Stage 2: `python -m epgnn.pipeline.build` → `features/v1/`

**`zone_features_hourly`**: 4,240,035 × 57. The node table plus:
- calendar sin/cos and `is_weekend`
- `price_wk_change`, `price_d1_{mean,min,max,std}`, `europe_price_lag24`, `price_lag24_vs_europe`, `nb_price_lag24`
- `load_fc_lag168`, `load_fc_wk_change`, `load_fc_ramp`, `load_fc_day_rel`
- weather `_lag48` and `heat_stress = max(temp_anom_lag24, 0)`
- per-zone JAO exposure: `jao_n_exposed`, `jao_min_ram_exposed`, `jao_tightness` and their `_lag24` versions. These are NaN outside the 12 Core zones and before Jun 2022.

**Constraint path** (`epgnn/pipeline/constraints.py`):

```
jao_cnec_hourly (4.63 M) ──┐  aligned on cnec_key
cnec_provenance (4.63 M) ──┘
   │  constraint_id = sha1(CneEic | Direction | ContName)[:12]
   │  dedupe (ts_utc, constraint_id), keeping the lowest RAM/Fmax  → drops ~360 k rows (see 4.2)
   ▼
constraints_hourly   4,271,471 rows; one per (ts_utc, constraint_id)
                     ram_fmax, ram, fmax, frm, f0core, ptdf_{12 Core zones}   (float32)
                     93 / 110 / 138 constraints per hour (5th / 50th / 95th percentile)
constraint_catalog   6,319 rows; one per constraint: names, EIC, direction, contingency, TSO,
                     element type, first/last ts, n_hours, n_days, mean RAM/Fmax, main_zone, max |PTDF|
   │
   ├─ zone_exposure()        → (ts_utc, zone) for 12 zones × 3 columns → merged into zone_features
   └─ constraint_graph(day)  → per delivery day (computed on demand, not stored):
          ids      (C,)            C ≈ 230 constraints active on the day
          x        (C, 25, 5)      [ram_fmax, ram, fmax, frm, f0core], NaN where not presolved that hour
          present  (C, 25)         bool
          ptdf     (C, 25, 12)
          edges    (2, E)          zone → constraint where |PTDF averaged over the day| ≥ 0.05
          weight   (E,)            signed mean PTDF
```

No trained model uses constraint nodes yet. `epgnn/data/graph.constraint_graph` (the collaborator's stub) still raises `NotImplementedError`. The pipeline's version is the working one.

**`graph.json`**: 45 zones, 180 directed interconnector edges, 90 spread pairs, an 8-dimensional Laplacian PE, and the connected components.

### Stage 3: `epgnn/pipeline/tensors.load(cols)` → `ZoneTensors`

| Array | Shape | Contents |
|---|---|---|
| `x` | (T = 94,223, Z = 45, F) | float32, raw (unscaled) |
| `price`, `has_price`, `base` | (T, Z) | `base` = `price_lag168`, else `price_lag24`, else 0 |
| `slots` | (D, 25) | hour index of each slot of a delivery day, −1 for padding (23/24/25-hour days) |
| `cutoff` | (D,) | hour index of D-1 10:00 UTC |

### Stage 4: samples for `gat_scope`

For each batch of days, every zone gets:
- **history** (168, 8): `price_lag24, price_lag168, load_fc, net_import, temp_pw, temp_anom, wind_cf, solar_cf` for the hours ending at the cutoff.
- **day inputs** (25, 14) (`own_cyclic`): `price_lag24, price_lag168, load_fc`, weather `_lag24` ×4, `net_import_lag48`, and 6 cyclic calendar columns.
- **target** (25,): `asinh(p/50) − asinh(base/50)`, plus the mask.

The model:
1. GRU → (Z, 64), copied to the 25 slots.
2. Concatenate the day inputs and the PE → Linear → (B·25, Z, 64).
3. Two GATv2 blocks; **each (day, slot) is a separate graph, and slots never exchange messages.**
4. Head → (B, Z, 25).

Loss: masked L1 + 0.5 × spread L1.

### Stage 5: `epgnn/metrics.score`

MAE and RMSE over masked hours, `rel_mae` against D-7 persistence, spread MAE and F1 over the 90 pairs, and slices (before/after 15-minute prices, summer, Core, Iberian cap).

---

## 3. Pipeline problems (not constraint-specific), most important first

1. **The D-1 price curve is not visible (stage 4).** The `price_lag24` history column ends at D-2 10:00 UTC. D-1's full curve is published on D-2 (around 12:45 CET), so it is a legal input. Fix: give each sample D-1's 24 prices (and D-2's, D-7's) as a per-day vector shared by all slots, the way LEAR does. Alternatively, extend the window of the `gate` columns through the D slots. `own_cyclic_plus` adds only the D-1 mean and std. Expect the biggest gain from this fix.
   **Implemented (26 Nov 2026), opt-in:**
   - `price_curves: true` in `gat_scope` appends the D-1, D-2 and D-7 curves (72 columns, by local hour, asinh, z-scored on the fit days) to every slot's day inputs.
   - Configs: `c3s_gat_{interconnects,full,self}_curves.yaml`. The model has 100,033 parameters in every arm.
   - Tests in `tests/test_gat_scope.py::TestPriceCurves`. Checked on real data, including DST days.
   - The existing configs are unchanged.
2. **The weather `_lag24` columns are partly future data.** For a day-D hour at t ≥ 10:00 UTC, t−24 falls on D-1 after the cutoff, which is not yet observed. That is about 14 of the 24 hours. These columns are labelled `gate` in `registry.py` and are used by XGB, the GRU+MLP and the GAT (`OWN`, `own_cyclic`), and `heat_stress` is built on them. In LEAR this barely changes anything (section 5), but it is wrong for H3. Fix: for each hour use the latest observation at or before the cutoff (lag48 for the afternoon hours), or better, use real forecasts (item 3).
   **Implemented (26 Nov 2026) in the pipeline and `gat_scope`:**
   - `zone_features` adds `temp_pw_asof`, `temp_anom_asof`, `wind_cf_asof` and `solar_cf_asof`. Each is lag 24 where that D-1 hour is already observed at the cutoff (up to 10:00 UTC of day D, i.e. 12–13 hours of the day) and lag 48 otherwise. `heat_stress` is now built on the as-of anomaly.
   - The `_lag24` weather columns are registered as `past`: legal in the history window, not as day-D inputs.
   - `gat_scope` has cutoff-safe day sets, `own_cyclic_asof` and `own_cyclic_plus_asof`. It now warns when a run's day inputs include `past` or `outcome` columns, and refuses them with `strict_inputs: true`.
   - The old `own_cyclic` sets stay unchanged, so earlier runs can be reproduced.
   - Tests: `tests/test_zone_features.py`, covering both DST changes.
   - **Not changed:** `epgnn/data/features.py` (`OWN`, which XGB and the GRU+MLP use) still lists the `_lag24` weather. It belongs to collaborator A, so those baselines still carry the leak until A swaps in the `_asof` columns.
3. **The market's real day-ahead information is missing.** Traders and TSOs work from **ENTSO-E day-ahead wind and solar generation forecasts** (Transparency doc type A69, in MW) and the day-ahead load forecast. Capacity factors from reanalysis are a weak stand-in: the observed values cannot be inputs, and they ignore installed capacity, which has grown fast. Fuel and carbon prices (TTF gas, EUA) set the price level and are known daily. Pulling A69 forecasts is the single biggest data improvement for H3. Ask before downloading.
4. **`rel_mae` uses the weakest naive baseline.** Report it against the similar-day naive (the standard in the EPF literature) or against LEAR.
5. **`net_import`** is built with `inflow.sub(outflow, fill_value=0)` and `sum(min_count=1)`. If one direction or one border is missing in an hour, the result is a partial sum with no warning. Add a coverage count per zone and hour, or mask such hours.
6. **`europe_price_lag24`** averages whichever zones have prices, and that set changes over time (AT from Oct 2018, GB until Dec 2020). This creates level shifts. Use a fixed zone set, or a median.
7. **DST:** lags are row shifts on the UTC grid, so around clock changes `price_lag24` is not the same local hour, and the `price_d1_*` statistics on 23- and 25-hour days mix two days. This is small; the same-hour D-7 MAE is 29.03 by local hour against 29.01 by row shift.
8. **The test split is two-thirds post-15-minute era** (from 1 Oct 2025). Persistence MAE is 23.7 before and 31.7 after. Report both slices every time, and keep in mind that the target changed from an hourly auction price to the mean of four 15-minute prices.
9. **Three links in the "interconnects" graph don't physically exist today** (found while drawing `figures/data/d1_zone_map`):
   - DE_LU–HU, DE_LU–SI and DE_LU–IT_NORD have physical flows only from 2016 to 30 Sep 2018. They are Austria's borders, inherited because the old DE-AT-LU zone is mapped to DE_LU.
   - `build_model_dataset.py` keeps any pair that ever had a flow, so these 6 directed edges are in `graph.json`, the Laplacian PE, the spread loss and the spread metric. DE_LU's degree is 14 instead of 11.
   - (DE_LU–HU and DE_LU–SI also have JAO `maxbex`, but MaxBex is published for non-adjacent Core pairs too.)
   - Fix: drop pairs with no physical flow after 1 Oct 2018 when building the edge list.
   - **Implemented** in `epgnn/pipeline/graphs.py`: `interconnect_edges` now uses `active_pairs()`, which keeps pairs with any `flow_phys` or `flow_sched` on or after `ACTIVE_SINCE = 2018-10-01`. Result: 174 directed edges (87 links), DE_LU degree 11, graph still one component. `dropped_pairs()` lists what was removed and the build log records it. Applies to adjacency, `spread_pairs`, the Laplacian PE, `graph.json` and `nb_price_lag24` after the next features/v1 rebuild. Tested in `tests/test_graphs.py`.
   - Not changed: collaborator code that reads `tables.edge_pairs()` directly (the spread metric in `metrics.py`, xgb neighbour means, `gru_mlp`, `data/graph.py`) still sees the 6 phantom edges. The c3s interconnects/full results so far were trained with them.
10. **Known holes** to state in the paper: IE_SEM load after mid-2021, GB prices only until 2020, BA never priced, AT before Oct 2018, and the 5 missing JAO days in the test split.

---

## 4. Constraint data in detail

Measured on `constraints_hourly` and the provenance table (`jao_audit.py`):

| Fact | Value |
|---|---|
| Rows per hour (presolved) | median 110 (5th–95th percentile 93–138) |
| `presolved` | always True. This is the presolved domain (non-redundant constraints), **not** the binding ones |
| `shadow_price`, `n_binding`, `net_position` | all null: there is no information on which constraint bound |
| RAM/Fmax quantiles (1/10/50/90/99 %) | 0.20 / 0.43 / 0.78 / 1.08 / 1.37 |
| RAM/Fmax > 1 | 17 % of rows; **all** have F_LTN < 0, a counter-flow, so this is not policy |
| `MinRamFactor` | 70 on 51 % of physical rows (the Clean Energy Package's 70 % minimum-RAM rule); 31–60 elsewhere (derogations) |
| `Amr > 0` | 18.8 % of physical rows; median uplift 8 % of Fmax |
| Cross-border elements (HubFrom ≠ HubTo) | 53 % |
| Base-case constraints (no contingency) | 11 % |
| Synthetic rows in the raw view (equality/external constraints of ALEGrO) | 8 % |
| Max zone-to-zone PTDF ≥ 0.05 | 98.7 % (as expected: this is Core's selection rule) |
| Day-D constraints also present on D-1 | median 78 % (day-to-day Jaccard 0.63) |
| corr(`jao_tightness`, its `_lag24`) | 0.69; for `jao_min_ram_exposed`, 0.49 |

Issues:

4.1 **The exposure test depends on the PTDF reference.** Hub PTDFs are zone-to-slack. Adding the same constant to all 12 leaves every flow unchanged, because net positions sum to zero. So `|PTDF_z| ≥ 0.05`, used for `jao_n_exposed`, `jao_tightness` and the graph edges, is not physically meaningful. Subtracting the mean over hubs changes the exposed share a lot: AT goes from 43 % to 20 %, DE_LU from 35 % to 17 %, RO from 20 % to 32 %. Fix: use zone-to-zone PTDFs (`PTDF_a − PTDF_b` for bordering pairs; `jao.NEIGHBOURS` is already there for this), or PTDFs minus their mean over the hubs, for both the exposure features and the zone→constraint edges. A GNN edge weight should also be invariant to that constant.
   **Implemented (26 Nov 2026).**
   - All zone-level uses (`zone_exposure`, `constraint_graph` edges/weights/ptdf, catalog `main_zone`/`max_abs_ptdf`) now use the **zone-to-rest PTDF**, `PTDF_z − mean of the other 11 hubs` (`constraints.rest_ptdf`). That is the flow change per MW exported from z and taken up evenly by the rest of Core.
   - `reference="raw"` reproduces the old behaviour, and `constraints_hourly` still stores the published PTDFs.
   - Tests: `tests/test_constraints.py`.
   - Size of the change, old vs new, per zone: the `jao_n_exposed` correlation is −0.02 to 0.44, `jao_tightness` 0.28–0.68 and `jao_min_ram_exposed` 0.21–0.69. The old features were therefore mostly slack artefacts.
   - Mean exposed constraints per hour: AT 48 → 25, DE_LU 39 → 24, RO 22 → 39.

4.2 **Synthetic constraints are removed only by accident.** `jao.SYNTHETIC` exists but `constraints.load` never applies it. The 8 % synthetic rows (e.g. "Equality Constraint Core hubs 01", "External Constraint DE_AL_import") hash to a single `constraint_id`, and the "keep the tightest variant" dedupe then deletes all but one per hour (about 360 k rows dropped). What remains is one "External Constraint BE_AL_export" node per hour, with Fmax 1000 and RAM/Fmax 1. Fix: filter explicitly (by name regex or by empty `ElementType`) before hashing, and log how many real variants the dedupe drops. `build.json` reports `duplicates_dropped: 0` because that count is not set on the reuse path.
   **Implemented (26 Nov 2026).**
   - `constraints.is_synthetic` applies `jao.SYNTHETIC` and drops those rows in `load()` before ids are assigned. On the Core data the rule matches exactly 58 names (372,684 rows, 8.0 %):
     - ALEGrO equality constraints
     - external import/export limits on ALEGrO, on the hybrid-coupling DC links and on NL
     - the `Default_<zone>_Import/Export` rows

     All of them have `CneEic = "NA"` and an empty `ElementType`. No real line matches; an empty `ElementType` alone would be the wrong rule, since 17 k real-line rows also have it.
   - A second bug, found along the way: 2,567 **real** line rows also have `CneEic = "NA"`, and the id hash used "NA" as the key, so different lines merged. The id now falls back to the line name when the EIC is "NA" or blank.
   - `build.json` now logs `synthetic_dropped` and `duplicates_dropped`.
   - If `build.py` reuses an older `constraints_hourly`, it drops the synthetic ids too. `--rebuild-constraints` is needed for the id fix.
   - Tests: `tests/test_constraints.py::TestSyntheticConstraints`, including an end-to-end `load()` on a fake JAO folder.

4.3 **RAM is partly a regulatory number.**
   - On every physical row the published margin decomposes exactly (within 1 MW): **RAM = Fmax − FRM − F_LTN + AMR − IVA**. Here F_LTN = `FtotalLtn` = F0core + the flow of long-term nominations. FUAF and `LtaMargin` do not enter it; `LtaMargin` is always 0.
   - AMR, the minimum-RAM uplift, is positive on 18.8 % of rows. On those rows, RAM/Fmax measures the policy target, not the grid.
   - *Correction to an earlier draft of this review:* RAM/Fmax > 1 is **not** policy. All 17 % of those rows have F_LTN < 0, a counter-flow that frees margin.
   - `jao_tightness` was an unnormalised sum, so it grew with the number of constraints. **Now (26 Nov 2026)** it is a PTDF-weighted mean scarcity, Σ|P|·(1 − min(RAM/Fmax, 1)) / Σ|P|, over the constraints with a valid RAM/Fmax. It lies in [0, 1] and doesn't change if every constraint is duplicated; the count stays in `jao_n_exposed`. `jao_tightness_grid` is computed the same way.

   **Implemented (26 Nov 2026).**
   - New node features in `constraints_hourly` (`NODE_FEATURES`, so `constraint_graph` gives `x` of shape (C, 25, 11)):
     - `fltn`, `amr`, `iva`, `min_ram_factor` (as a fraction)
     - `ram_grid_fmax = (RAM − AMR)/Fmax`: the margin the grid leaves before the uplift; 1.2 % of rows are < 0
     - `amr_fmax = AMR/Fmax`
   - New zone features, each with a `_lag24` version:
     - `jao_amr_share_exposed`: the share of a zone's exposed constraints that were lifted
     - `jao_min_ram_grid_exposed`
     - `jao_tightness_grid`: tightness computed on the grid margin
   - Tests: `tests/test_constraints.py::TestPolicyAdjustedRam`.

4.4 **Timing needs a decision and should be reported both ways.** The day-D domain gives a materially different signal from D-1's (correlation 0.5–0.7, and 22 % of constraints change). Core publishes the final day-ahead flow-based parameters before the SDAC gate closure on D-1, so the day-D domain is probably legal at a 10:00 UTC cutoff (11:00 CET in winter). Verify this against JAO's publication times before relying on it. Run H2 with both the D and the `_lag24` versions. If the gain appears only with the D version, the timing has to be settled.

4.5 **Unused information that fits the GNN well:**
- `HubFrom`/`HubTo`, TSO, `ElementType` (PST, transformer, tie-line) and base case vs contingency, as categorical node features.
- **PTDFs to CH and to the virtual hubs of the HVDC links** (`Ptdf_DE_NO2_BigHub`, `PL_SE4_SwePol`, `PL_LT`, `RO_BG`, `NL_DK1_COBRA`, …). These connect Core constraints to non-Core zones (NO_2, SE_4, LT, BG, DK_1, CH), so constraint nodes would touch more than 12 of the 45 zones.
- ALBE/ALDE are dropped, which removes the effect of the ALEGrO DC link.

4.6 **There are no labels for H2.** Without shadow prices or active constraints, you cannot show that the model attends to the constraints that actually bound. JAO publishes the Core active constraints with their shadow prices after clearing. Those are outcomes, so never inputs, but they are the natural auxiliary target and attention check for H2 (for example, does attention weight correlate with shadow price > 0).

4.7 **Coverage:** Core only, from Jun 2022 (pretraining sees no constraints). Nordic flow-based (live since late Oct 2024) is not included. CWE 2015–2022 is in the bucket (`jao_cnec_hourly` v1.0) but not used here.

---

## 5. Baselines (computed here, same masked hours as `epgnn.metrics`)

| Baseline | Val MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|
| D-7 same hour (current reference, = c1) | 30.16 | 29.03 | 46.0 |
| D-1 daily mean, flat over the day | 30.53 | 33.18 | 48.4 |
| Median of the last 4 same weekdays | 26.32 | 26.28 | 40.0 |
| Similar day (Mon/Sat/Sun: D-7, else D-1); the EPF-literature naive | 24.28 | 24.00 | 39.6 |
| D-1 same hour | 23.92 | 23.79 | 39.2 |
| Mean of D-1 and D-7 | 22.98 | 22.41 | 35.2 |
| **LEAR, fitted once on train (2022-06 – 2024-03)** | **19.18** | **20.67** | **32.6** |
| **LEAR, refitted monthly on the trailing 730 days** | **18.25** | **18.43** | **29.7** |
| *for comparison:* c1 XGBoost per zone | 20.69 | 21.24 | |
| c2.5 GRU+MLP | 20.03 | 20.40 | 34.1 |
| c3s GAT interconnects (old inputs) | 19.80 | 20.54 | 32.6 |
| c3s GAT full (old inputs) | 19.71 | 20.64 | 32.7 |
| c3s GAT self, safe inputs (`_asof`) | 20.48 | 20.86 | 33.0 |
| c3s GAT interconnects, safe inputs | 20.07 | 20.27 | 32.1 |
| c3s GAT full, safe inputs | 19.95 | 20.65 | 32.7 |

The "safe inputs" rows are reruns of the same c3s configs (no D-1 curves, fitted once) with cutoff-safe `_asof` weather and the 174 active edges (configs `c3s_gat_*_asof.yaml`). The c1, c2 and c2.5 reruns (`c1_xgb_zone_asof`, `c2_xgb_neighbours_asof`, `c25_gru_mlp_asof`, through the `models/asof.py` adapter) are queued behind the H2 runs.

**LEAR** here follows Lago et al. (2021):
- One LASSO per zone and target hour (45 × 24 = 1,080), with the penalty chosen by AIC (`LassoLarsIC`).
- Inputs: the zone's own prices for all 24 hours of D-1, D-2, D-3 and D-7, after median/MAD scaling and asinh; the load forecast for all 24 hours of D and of D-7; weekday dummies.
- No weather, no neighbours, no constraints.
- 1.6 % of hours (zones with gaps in the inputs) fall back to the similar-day naive.

LEAR by zone on test runs from about 10 (NO_4, NO_5) to about 36 (the Baltics). Every zone beats similar-day except ES, PT, GR, MK and RS, where similar-day is better by 1–2 EUR/MWh.

**Weather in LEAR** (fitted on train, 72 extra inputs: temp_pw, wind_cf and solar_cf for all 24 hours):

| Weather input | Val | Test | Test, summer |
|---|---:|---:|---:|
| none | 19.18 | 20.67 | 20.27 |
| D-2 observed (lag48, strictly legal) | 19.38 | 20.98 | 20.67 |
| D-1 observed (lag24, partly future data) | 19.20 | 20.82 | 20.51 |
| day D actual (perfect forecast, the ceiling) | 18.38 | 20.23 | 20.33 |

Lagged weather adds nothing beyond what lagged prices already carry. Even perfect weather is worth only 0.44 MAE on test, and nothing in test summers. Most of what weather can add probably comes from wind, through generation, rather than from temperature.

### Other baselines worth knowing

- **ENTSO-E does not forecast prices.** Day-ahead prices are the outcome of the SDAC auction (the Euphemia algorithm), cleared from bids. What ENTSO-E and the TSOs publish are the inputs the market uses: day-ahead load and wind/solar forecasts, flow-based domains and NTCs, outages, plus long-term adequacy studies (ERAA, seasonal outlooks). The honest "what the market knew" baseline is therefore **LEAR (or your model) fed with ENTSO-E's own day-ahead forecasts**.
- **Standard EPF benchmarks:** the similar-day naive, LEAR and the DNN from the `epftoolbox` of Lago, Marcjasz, De Schutter and Weron (*Applied Energy*, 2021). Their DNN is typically a few percent to about 10 % better than LEAR on single markets. It is the natural non-graph deep baseline, and an ensemble of LEAR over several calibration windows is a cheap, strong one.
- **Market-implied prices:** EEX day and weekend futures are market forecasts, but they are not free. They can be mentioned as out of scope.

---

## 6. Target MAE, and how to make weather claims

- **The bar is LEAR, not persistence.** There are two versions, and they answer different questions:
  - *Static LEAR* (fitted once on train, like every neural model here): 19.18 val / 20.67 test. This is the like-for-like bar. Right now the GRU+MLP beats it on test by 0.27 and loses on val by 0.85.
  - *Monthly-refitted LEAR* (trailing 730 days): 18.25 val / 18.43 test. Refitting is worth 0.9 on val and **2.2 on test**, more than any input change measured so far, and more than perfect weather (0.44). The 18-month test split drifts (the 15-minute era, the 2025 price level), and a model frozen in March 2024 or March 2025 pays for it.
- **A reasonable target:**
  - Against static LEAR, about 5 % better: **19.6 or lower on test, 18.2 or lower on val**. Fixing item 3.1 should get most of the way there.
  - Any claim that the model is better than the state of the art has to beat refitted LEAR: **18.4 on test**. A neural model fitted once is unlikely to get there. To compete, either fine-tune it on a rolling window too (for example monthly, a few epochs on the trailing data), or report the static comparison and say plainly that recalibration is a separate 2-point effect.
  - **Implemented:** `rolling` in `gat_scope` fine-tunes the model monthly on the trailing 730 days (2 epochs, lr 1e-4) over val+test. A rolling run also scores its static predictions (`info.static_metrics`), so one run gives both comparisons. Configs: `c3s_gat_{self,interconnects,full}_curves_rolling.yaml`.
  - Graph and weather effects (H1–H3) are differences between arms trained the same way, so they stay valid against static LEAR. Just don't present a static model as beating LEAR without saying which LEAR.
- **Noise floor:** about ±0.1 across seeds (controlled study), and the GAT arms differ by 0.1. A claimed effect needs at least 3 seeds per arm and a paired weekly block bootstrap of the hour-level MAE difference (next step 3 in HANDOFF). Report the CI, not only the point estimate.
- **For H3**, measure weather in three steps:
  1. no weather, against lag48 weather (realistic, legal)
  2. against ENTSO-E wind/solar forecasts (realistic, what the market sees)
  3. against perfect same-day weather (the ceiling)

  On LEAR the ceiling is only 0.4–0.8 MAE, so lagged weather that gives more than that in a neural model is a warning sign of leakage or tuning noise. "Heat stress raises prices and spreads" is better tested conditionally than through overall MAE: errors and price/spread levels on heat-anomaly days against matched normal days, using the perfect-forecast arm to bound the effect.

---

## 7. Results after the fixes (overnight run, 28 Nov 2026)

Run on the rebuilt features/v1, which includes:
- the rest-of-Core PTDF reference;
- synthetic constraints dropped (372,684 rows);
- the policy RAM features;
- tightness as a weighted mean;
- cutoff-safe `_asof` weather;
- 174 interconnector edges, with the phantom DE_LU links removed.

Inputs: `own_cyclic_asof` plus the D-1, D-2 and D-7 price curves, with `strict_inputs: true`. All runs use one seed (0) and 100,033 parameters, and early stopping picked epoch 33 in every arm. Monthly fine-tuning: trailing 730 days, 2 epochs at lr 1e-4.

| Model | Val | Test | Test before Oct 2025 | Test after Oct 2025 | Test, Core zones |
|---|---:|---:|---:|---:|---:|
| LEAR, fitted once | 19.18 | 20.67 | | | |
| LEAR, refitted monthly | 18.25 | 18.43 | | | |
| GAT self + curves, fitted once | 18.35 | 19.21 | 16.61 | 20.52 | 21.12 |
| GAT full + curves, fitted once | 18.02 | 18.79 | 16.18 | 20.09 | 20.19 |
| **GAT interconnects + curves, fitted once** | **17.85** | **18.46** | 15.77 | 19.81 | 20.15 |
| GAT self + curves, fine-tuned monthly | 18.07 | 17.88 | 15.45 | 19.10 | 19.18 |
| GAT full + curves, fine-tuned monthly | 17.71 | 17.57 | 15.11 | 18.80 | 18.61 |
| **GAT interconnects + curves, fine-tuned monthly** | **17.62** | **17.35** | 14.91 | 18.58 | 18.48 |

- **The D-1 curve fix (3.1) was the big win.** Without a graph, the curves take the GAT from 20.5 to 19.2 on test, which beats LEAR fitted once by 1.5 and meets the target.
- **The graph helps, on both val and test, fitted once or fine-tuned.** Compared with self, interconnects is −0.50 val / −0.75 test fitted once and −0.45 / −0.53 fine-tuned. That is about 5× the seed noise, but this is still one seed: run 3 seeds and the block bootstrap before claiming it. Full attention lands in between; the physical topology beats unrestricted attention by 0.2–0.3.
- **Fine-tuning monthly is worth about 1.1–1.3 on test.** All three fine-tuned arms beat refitted LEAR (18.43); the best is 1.08 below it. The gain is similar before and after the 15-minute switch.
- **The weather leak barely mattered here.** The self arm on the old partly-future `_lag24` weather scored 18.36 / 19.22, against 18.35 / 19.21 with cutoff-safe weather, consistent with the LEAR weather study. Older results that used `_lag24` are therefore not badly inflated, but they should still not be reported.
- **The attention diagnostic is not informative for the interconnects arm:** its neighbour share is 1.0 by construction, since attention only covers neighbours and self. For the full arm, the lift over uniform attention is only 1.14, so the full model does not learn to concentrate on physical neighbours. That is mild support for H1 (the topology must be given, not learned) and worth a figure.
- Results files: `results/c3s_gat_{self,interconnects,full}_curves_rolling.json`. The fitted-once numbers are in `info.static_metrics`, with predictions in `runs/<name>/pred_<split>_static.npy`. `c3s_gat_self_curves_lag24.json` is the leaky reference.
- **c3s reruns on safe inputs (no curves, fitted once):** the safe weather and dropped phantom edges change test MAE by −0.27 (interconnects) and +0.01 (full), so the old c3s numbers were not inflated. Without curves the graph helps too: interconnects − self is −0.59 [−0.72, −0.46] on test (Core −0.82), and interconnects − full −0.38 [−0.58, −0.15] (seed 0, `results/bootstrap_ic_vs_self_asof.json`). The D-1 curves are worth 1.65 (self), 1.81 (interconnects) and 1.86 (full) on test, fitted once.
- c1, c2 and c2.5 still use the old features; their safe-input reruns are queued.

**Next:**
1. Seeds 1 and 2 for the three fine-tuned arms, about 2–3 h each on this laptop. Better on Sherlock.
2. The weekly block bootstrap on interconnects vs self.
3. A constraint arm (H2) on the rebuilt constraint features.

**Weekly block bootstrap** (`python -m epgnn.bootstrap A B [--static]`, 7-day moving blocks, 2,000 resamples, test split, seed 0). Negative means the first model is better:

| Comparison | Δ MAE | 95 % CI | Core zones Δ |
|---|---:|---:|---:|
| interconnects − self, fine-tuned | −0.53 | [−0.64, −0.42] | −0.71 |
| interconnects − self, fitted once | −0.75 | [−0.89, −0.61] | −0.97 |
| interconnects − full, fine-tuned | −0.21 | [−0.33, −0.09] | |

Every slice (before and after the 15-minute switch, summer, Core) has the same sign. These CIs cover sampling noise over days only, not seed noise. Seeds 1 and 2 of self and interconnects (fine-tuned) were started on 28 Nov and give the seed-averaged comparison:

`python -m epgnn.bootstrap I_s0,I_s1,I_s2 S_s0,S_s1,S_s2`, with the `c3s_gat_*_curves_rolling[_sN]` run names.

### H2 arms (queued 28 Nov)

The question: does the flow-based domain add anything beyond the zone graph? The control is `c3s_gat_interconnects_curves_rolling` (seeds 0–2). Both arms use the **D-1 domain of the same UTC hour**: the day-D domain is published near the cutoff and its timing is unverified, so neither arm uses it.

- `c4s_gat_interconnects_jao_rolling`: the six zone-level `jao_*_lag24` summaries as extra day inputs. This is the cheap version of H2.
- `c4s_gat_interconnects_constraints_rolling` (`epgnn/models/gat_constraints.py`): constraints as a second node type.
  - Each Core zone attends over the constraints it is exposed to (|zone-to-rest PTDF| ≥ 0.05), GATv2-style, with the signed PTDF in both the score and the message.
  - The pooled message is added to the zone state before the interconnector GAT, so the information then spreads to neighbours.
  - Constraint features are scale-free: RAM/Fmax, (RAM−AMR)/Fmax, AMR/Fmax, the min-RAM factor, FRM/F0core/F_LTN over Fmax, and log Fmax.
  - The model has 111,201 parameters (100,033 + 11,168). The store of 38,317 hours × at most 182 constraints takes 266 MB in fp16.
  - Non-Core zones, days before 9 Jun 2022 and days without a published domain get no message.
  - Tests in `tests/test_gat_constraints.py` check that the lag is 24 h, that the day-D domain is refused, that only Core zones receive messages, and that unexposed constraints are ignored.
- For the attention check, the model keeps `last_attention` (alpha over constraints per Core zone and slot). With JAO shadow prices as labels, this is the natural test of whether attention goes to the constraints that bind.

### Result figures (`python figures/make_result_figures.py`, outputs in figures/data/)

| Figure | Shows |
|---|---|
| `r1_monthly_mae` | MAE by month for the three scopes (fine-tuned), and the monthly gain from fine-tuning. The gain grows through the test period, from about 0.5 in 2025 to 1.5–2.5 in spring and summer 2026: drift. |
| `r2_zone_gain_map` | Per-zone Δ MAE of interconnects − self on the map. The gain is largest in the meshed centre and the Baltics (DE_LU, PL, LT, CZ, SK, FI around −0.9 to −1.3). It is near zero or slightly positive on the edges: NO4, NO5, IE_SEM, GR. |
| `r3_bootstrap` | Forest plot of every `results/bootstrap_*.json`, by slice. It keeps the seed-averaged version when one exists. |
| `r4_hour_profile` | MAE by delivery hour. Fine-tuning helps most at midday (solar hours, up to about 4 EUR/MWh), and the graph helps at every hour. |
| `r5_seeds` | Every seed per arm, fitted once and fine-tuned, against both LEAR lines. |
| `r6_attention_full` | Full arm, per zone: the neighbour share of attention against the uniform share. The lift is above 1 in western Europe (ES, PT, BE, CH, FR) and below 1 in the south-east and the Baltics. |
| `r7_zone_mae` | Per-zone MAE of interconnects and self, with each zone's intraday price spread for scale. |
| `r8_constraint_attention` | H2: where the constraint nodes look, recomputed from the saved weights on every 7th test day. It shows the attention lift by RAM/Fmax (split by AMR uplift) and by |PTDF|, and, per Core zone, the attention mass on its tightest 10 % of constraints. The grey line is the same model untrained: features enter the scores directly, so even random weights tilt slightly toward tight lines, and only the gap to it is learned. RAM/Fmax stands in for JAO shadow prices, which we do not have. Renders once `runs/c4s_gat_interconnects_constraints_rolling/model.pt` exists. |

### H1 with three seeds (28 Nov, 10:34)

Seeds 0, 1 and 2 of interconnects and self (+ curves). Test MAE (mean over seeds, with the range):

| | Fitted once | Fine-tuned monthly |
|---|---:|---:|
| self | 19.02 (18.77–19.21) | 17.84 (17.76–17.88) |
| interconnects | 18.55 (18.46–18.60) | 17.37 (17.35–17.40) |

The seed-averaged block bootstrap (`results/bootstrap_ic_vs_self_{rolling,static}_3seeds.json`) gives **interconnects − self = −0.46 [−0.57, −0.36]** fine-tuned and **−0.47 [−0.59, −0.35]** fitted once. The sign is the same in every slice, and the effect is largest on the Core zones (−0.63 / −0.60) and smallest in summer (−0.39 / −0.30). Seed spread within an arm is ≤ 0.13 fine-tuned and ≤ 0.44 fitted once, so fine-tuning also makes runs more reproducible. **H1 holds:** the physical interconnector graph improves day-ahead forecasts by about 0.47 EUR/MWh (2.6 %) over the same model without message passing.
