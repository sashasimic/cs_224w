# Graph neural networks for European day-ahead electricity prices: project overview

CS224W final project. Status as of 27 Nov 2026. This file is the one-stop summary: data sources, the decisions behind the pipeline, the pipeline itself, the models, and the results so far. The detailed audit, with every number's provenance, is in [PIPELINE_REVIEW.md](PIPELINE_REVIEW.md). Working notes for picking the project up are in [HANDOFF.md](HANDOFF.md).

All MAE values are in EUR/MWh, pooled over every priced zone-hour of the split. Unless a row says otherwise, results are for seed 0.

---

## 1. Task and hypotheses

**Task.** Forecast all 24 hourly day-ahead prices of delivery day D for 45 European bidding zones at once. Every input must be known at the **cutoff, D-1 10:00 UTC**, before the SDAC auction's gate closure (12:00 CET).

**Why a graph.** Zones are coupled by interconnectors and, in the Core region, by the flow-based capacity domain, which sets how much power can cross each border. Prices in neighbouring zones converge when interconnectors have spare capacity and split apart when they are congested. A graph model can pass information along those physical links.

**Hypotheses.**

| | Hypothesis | Test | Status |
|---|---|---|---|
| H1 | The physical interconnector graph improves forecasts over the same model without message passing | GATv2 with three attention scopes: self, interconnects, full | **Supported** (3 seeds, section 7.3) |
| H2 | Published flow-based constraints add information beyond the zone graph | Constraint nodes linked to zones by PTDF; and a cheaper zone-feature version | **Running** (section 7.5) |
| H3 | Heat stress raises prices, and cross-border spreads most of all | Weather ablations | Not started; groundwork in section 4.3 |

---

## 2. Data sources

![Data pipeline](figures/data/m1_data_pipeline.png)

| Source | Coverage | What we use | Notes |
|---|---|---|---|
| **ENTSO-E Transparency**, v1.2 extracts | 2016 – Sep 2026, 62 areas | Day-ahead prices (the target), day-ahead total load forecast, realised physical cross-border flows | From 1 Oct 2025 prices are 15-minute; we average them to the hour. ENTSO-E does not forecast prices: prices are the auction outcome. |
| **ERA5 / PECD weather** per bidding zone | 2010 – 2026 | Population-weighted temperature, temperature anomaly, wind and solar capacity factors | **Reanalysis, i.e. observed values, not forecasts.** Same-day values are only allowed as a "perfect forecast" ablation. |
| **JAO Core flow-based domain** | 9 Jun 2022 – 29 Oct 2026, 12 Core zones | Presolved critical network elements (CNECs) per hour: RAM, Fmax, FRM, F0core, AMR, min-RAM factor, 14 hub PTDFs; MaxBex per border | Five domain days in the test split are missing. Shadow prices and binding flags are all null in our extract. |
| **Topology** | | 87 interconnector links (174 directed edges) between the 45 zones | Pairs with physical or scheduled flow on or after 1 Oct 2018 |

**The 45 zones:** AT, BA, BE, BG, CH, CZ, DE_LU, DK_1–2, EE, ES, FI, FR, GB, GR, HR, HU, IE_SEM, the 7 Italian zones, LT, LV, ME, MK, NL, NO_1–5, PL, PT, RO, RS, SE_1–4, SI, SK. The 12 Core zones are AT, BE, CZ, DE_LU, FR, HR, HU, NL, PL, RO, SI and SK.

**Known gaps, stated in every result:**
- BA never has a price, so it is a node without a loss.
- GB has prices only until Dec 2020.
- AT is priced from Oct 2018. Before that, DE_LU carries the joint DE-AT-LU price.
- IE_SEM load is mostly missing after mid-2021.
- From Oct 2025 the target is the mean of four 15-minute prices instead of an hourly auction price. Two-thirds of the test split falls in this era.

The data figures in `figures/data/` (from `figures/make_data_figures.py`) show the zone map (d1), data availability (d2), price regimes (d3, d4, d5), the constraint data (d6) and the baselines (d7).

---

## 3. Splits and the cutoff rule

| Split | Delivery days | Role |
|---|---|---|
| pretrain | 2016-01-01 – 2022-06-08 | Fitting. No constraint data exists in this period. |
| train | 2022-06-09 – 2024-03-31 | Fitting. The flow-based era starts here. |
| val | 2024-04-01 – 2025-03-31 | Early stopping and model choice |
| test | 2025-04-01 – 2026-09-30 | Reported once. Two slices: before and after the 15-minute switch on 1 Oct 2025. |

Every input column has a timing label in `features/v1/feature_dictionary.csv`:
- **calendar:** always known.
- **gate:** known at the cutoff, such as the load forecast for day D and D-1 prices.
- **past:** usable only as history up to the cutoff.
- **outcome:** never an input for day D (price, net imports, flows).
- **unverified:** the day-D flow-based domain. It is probably published before the cutoff, but this hasn't been checked, so no model uses it.

With `strict_inputs: true`, a run refuses any day-D input labelled past or outcome.

---

## 4. Decisions made, and why

### 4.1 Framework

- **We work in the team's `cs_224w-c1_xgb_zone` framework**, whose data is local. The original plan was to run inside Arsen's controlled-study repo. That repo needs a dataset (`forecast_dataset_v1.0`) that is neither on the laptop nor in the GCS bucket, and it can't be rebuilt.
- **New work goes into new modules.** Collaborators own particular files. `gat_scope.py`, `gat_constraints.py`, `asof.py`, `bootstrap.py` and `epgnn/pipeline/` are new files. The baselines run on cutoff-safe inputs through an adapter (`asof.py`), which swaps two data functions while a model trains and leaves their code untouched.

### 4.2 Leakage fixes (each found in the pipeline review)

| Problem | Decision |
|---|---|
| The weather `_lag24` columns are partly future data: for afternoon hours of D, t − 24 h falls after the cutoff. | New `_asof` columns: lag 24 where that hour is observed by the cutoff, otherwise lag 48. `_lag24` weather is now "past" only. In practice the leak was small (19.22 vs 19.21 test MAE), but all new runs use `_asof`. |
| Three "interconnectors" (DE_LU–HU, DE_LU–SI, DE_LU–IT_NORD) are really Austria's borders, inherited from the pre-2018 DE-AT-LU zone. | Keep only pairs with flow on or after 1 Oct 2018: 174 directed edges instead of 180. |
| The day-D flow-based domain has unverified publication timing. | Use the D-1 domain for the same hour (lag 24 h), which is certainly public. |

### 4.3 Constraint data (JAO Core)

| Problem | Decision |
|---|---|
| Published PTDFs are zone-to-slack. Adding a constant to all of them changes no flow, so a threshold such as \|PTDF\| ≥ 0.05 depended on the arbitrary slack. | Use the **zone-to-rest PTDF** (the zone's PTDF minus the mean of the other 11 Core hubs). It doesn't depend on the slack. Exposure features and graph edges changed substantially (AT 48 → 25 exposed constraints per hour, RO 22 → 39). |
| 8 % of rows are synthetic (ALEGrO equality constraints, external import/export limits) and only disappeared by accident in a dedupe. In addition, 2,567 real lines had the EIC "NA" and were merged with each other. | Drop the 58 synthetic names explicitly. When the EIC is missing, build the id from the line name. |
| RAM is partly regulatory: RAM = Fmax − FRM − F_LTN + AMR − IVA, and on 18.8 % of rows the minimum-RAM uplift (AMR) inflates it. | Add grid-only margin features: (RAM − AMR)/Fmax, AMR/Fmax and the min-RAM factor. |
| `jao_tightness` was a sum, so it grew with the number of constraints. | Make it a PTDF-weighted mean scarcity in [0, 1]. |
| There are no shadow prices, so no labels for which constraints actually bind. | Use RAM/Fmax as a proxy in the attention analysis, and say so. |

### 4.4 Modelling

- **Give the model the D-1 price curve.** In the original setup each hourly slot saw only its own hour's D-1 price, and the GRU history ended at D-2. Appending the full D-1, D-2 and D-7 curves (72 columns) was the largest single improvement: −1.65 to −1.86 test MAE.
- **Compare against LEAR, not persistence.** The framework's `rel_mae` was relative to weekly persistence (29.0). LEAR (Lago et al. 2021) gets 20.67 fitted once and 18.43 refitted monthly, and is the bar the literature uses.
- **Monthly rolling fine-tuning.** Refitting LEAR monthly is worth 2.2 MAE on test, more than any input change, because the 18-month test split drifts. The neural models therefore get the same treatment. Before each val/test month they fine-tune for 2 epochs at lr 1e-4 on the trailing 730 days, all published before that month. Every rolling run also saves the fit-once predictions, so both comparisons come from one run.
- **Statistics.** H1 and H2 are differences between arms trained the same way, measured with a paired weekly moving-block bootstrap (7-day blocks, 2,000 resamples). With several seeds, each cell's error is averaged over seeds.

---

## 5. Data pipeline

| Stage | Command | Output | Shape |
|---|---|---|---|
| 0 | (downloads; see HANDOFF) | `data/entsoe/`, `data/weather_hourly_bzone_2010_2026.parquet`, `data/jao_core_hourly_views/` | |
| 1 | `python build_model_dataset.py` | `data/model/nodes_hourly.parquet` (= `zone_nodes_hourly` v1.0) | 45 zones × 94,223 UTC hours = 4.24 M rows, a complete grid |
| | | `data/model/edges_hourly.parquet` (= `zone_edges_hourly` v1.0) | 180 directed pairs × 94,223 hours = 17.0 M rows |
| 2 | `python -m epgnn.pipeline.build` (in `cs_224w-c1_xgb_zone`) | `features/v1/zone_features_hourly.parquet` | 4.24 M × 57 columns |
| | | `features/v1/constraints_hourly.parquet`, `constraint_catalog.parquet` | 4.27 M rows (about 110 constraints per hour); 6,319 distinct constraints |
| | | `features/v1/graph.json`, `feature_dictionary.csv` | 45 zones, 174 edges, 87 spread pairs, 8-dim Laplacian PE |
| 3 | `epgnn.pipeline.tensors.load` | `ZoneTensors` | x (94,223 × 45 × F), price/mask/base (T × Z), slots (days × 25), cutoff index per day |
| 4 | inside `gat_scope` | per (day, zone) sample | history 168 h × 8; day inputs 25 slots × 86; target 25 |
| 5 | `python -m epgnn.run configs/experiments/X.yaml` | `results/X.json`, `runs/X/pred_{val,test}[_static].npy` | |
| 6 | `python -m epgnn.bootstrap A B [--static]` | `results/bootstrap_*.json` | |

**Main feature groups in `zone_features`:**
- Price: D-1 and D-7 same hour; D-1 mean, min, max and std; the Europe-wide and neighbour-mean D-1 price.
- Load: the forecast, its change from last week, ramp and daily shape.
- Weather: `_asof` and `_lag48` versions, plus heat stress.
- Calendar: sin/cos encodings.
- Core JAO exposure per zone, as `_lag24` versions: number of exposed constraints, minimum RAM/Fmax, weighted tightness, AMR share, and the grid-only minimum margin and tightness.

**Target.** `asinh(p/50) − asinh(base/50)`, the variance-stabilised change from the same hour last week (or from D-1 if D-7 is missing). Predictions are mapped back to prices before scoring.

**Scoring.** `epgnn.metrics.score` computes MAE, RMSE, spread MAE over the 87 interconnected pairs, and slices (before/after the 15-minute switch, summer, Core zones).

---

## 6. Models

### 6.1 Baselines

| Model | Description |
|---|---|
| Naive | D-7 same hour; D-1 same hour; similar day (D-7 on Mon/Sat/Sun, else D-1); mean of D-1 and D-7 |
| **LEAR** | One LASSO per zone and hour (1,080 models), penalty chosen by AIC. Inputs: the zone's own D-1, D-2, D-3 and D-7 price curves, the load forecast for D and D-7, and weekday dummies. Fitted once on train, or refitted monthly on the trailing 730 days. |
| c1 XGBoost per zone | Gradient boosting on the zone's own features |
| c2 XGBoost + neighbours | c1 plus neighbour-mean features |
| c2.5 GRU + MLP | The neural model with no graph |

### 6.2 GATv2 attention-scope model (`epgnn/models/gat_scope.py`)

![GATv2 attention-scope model](figures/data/m2_gat_scope_model.png)

- **Encoder.** A GRU (8 → 64) reads each zone's last 168 hours up to the cutoff. Its final state is copied to all 25 hourly slots of day D (25 covers DST days). It is concatenated with the slot's 86 day inputs and the zone's 8-dim Laplacian positional encoding, then projected to 64.
- **Message passing.** Two pre-LayerNorm blocks: GATv2 attention (4 heads × 16) plus a feed-forward layer (64 → 128 → 64). Each (day, slot) is its own graph of 45 zones; slots never exchange messages.
- **Head.** An MLP (64 → 128 → 128 → 1) predicts the asinh change.
- **Loss.** Masked L1 plus 0.5 × L1 on cross-border price spreads.
- **Parameter count.** 100,033 in every arm. Only the attention mask differs:
  - **self:** each zone attends only to itself (no messages).
  - **interconnects:** each zone attends to its physical neighbours and itself.
  - **full:** each zone attends to all 45 zones.
- **Implementation.** The GATv2 layer is a dense re-implementation, tested equal to PyG's `GATv2Conv`. It runs on the Mac's MPS backend.
- **Training.** Fitted on pretrain + train with Adam (lr 1e-3), batches of 16 days, early stopping on val (patience 8, at most 50 epochs). Early stopping picked epoch 33 in every arm. Then optional monthly rolling fine-tuning (section 4.4).

### 6.3 H2: constraint arm (`epgnn/models/gat_constraints.py`)

![H2 constraint arm](figures/data/m3_constraint_arm.png)

- **Constraint nodes.** For each slot hour, the constraints are those in the D-1 domain for the same hour: about 110 per hour, at most 182.
- **Features.** Each constraint carries 8 scale-free features: RAM/Fmax, (RAM − AMR)/Fmax, AMR/Fmax, the min-RAM factor, FRM, F0core and F_LTN over Fmax, and log Fmax.
- **Attention.** Each Core zone attends, GATv2-style, over the constraints it is exposed to (\|zone-to-rest PTDF\| ≥ 0.05). The signed PTDF enters both the attention score and the message, so the model can tell whether exporting from the zone loads a line or relieves it.
- **Where it plugs in.** The message is added to the zone state *before* the interconnector GAT blocks, so constraint information then spreads to non-Core neighbours.
- **When it is off.** Non-Core zones, pretrain days and days without a published domain receive nothing.
- **Size and control.** It adds 11,168 parameters (111,201 in total). The control is the interconnects arm with curves, fine-tuned monthly.
- **Cheaper H2 version (`c4s_gat_interconnects_jao_rolling`).** The six zone-level `jao_*_lag24` summaries are added as extra day inputs, with no constraint nodes.

---

## 7. Preliminary results

### 7.1 Baselines and models (test MAE)

![Baselines and models](figures/data/d7_baselines.png)

| Model | Val | Test | Test RMSE |
|---|---:|---:|---:|
| D-7 same hour | 30.16 | 29.03 | 46.0 |
| Similar day (EPF-literature naive) | 24.28 | 24.00 | 39.6 |
| D-1 same hour | 23.92 | 23.79 | 39.2 |
| Mean of D-1 and D-7 | 22.98 | 22.41 | 35.2 |
| c1 XGBoost per zone (old inputs)\* | 20.69 | 21.24 | |
| c2 XGBoost + neighbours (old inputs)\* | 20.36 | 21.03 | |
| c2.5 GRU + MLP (old inputs)\* | 20.03 | 20.40 | 34.1 |
| GAT self, safe inputs, no curves, fitted once | 20.48 | 20.86 | 33.0 |
| GAT full, safe inputs, no curves, fitted once | 19.95 | 20.65 | 32.7 |
| GAT interconnects, safe inputs, no curves, fitted once | 20.07 | 20.27 | 32.1 |
| **LEAR, fitted once** | 19.18 | 20.67 | 32.6 |
| GAT self + curves, fitted once | 18.35 | 19.21 | |
| GAT full + curves, fitted once | 18.02 | 18.79 | |
| GAT interconnects + curves, fitted once | 17.85 | 18.46 | |
| **LEAR, refitted monthly** | 18.25 | 18.43 | 29.7 |
| GAT self + curves, fine-tuned monthly | 18.07 | 17.88 | |
| GAT full + curves, fine-tuned monthly | 17.71 | 17.57 | |
| **GAT interconnects + curves, fine-tuned monthly** | **17.62** | **17.35** | |

\* These use the partly leaky `_lag24` weather and the 6 phantom edges. Their reruns on safe inputs are queued (section 8).

**Takeaways:**
1. **The D-1 price curves matter most.** Adding them improves every GAT arm by 1.65 to 1.86 on test, and takes it from tying LEAR to beating it.
2. **Fine-tuning monthly adds 1.1 to 1.3 on test.** All three fine-tuned GAT arms beat monthly-refitted LEAR. The best, interconnects, is 1.08 (5.9 %) below it.
3. **The gain from fine-tuning grows over the test period**, from about 0.5 in 2025 to 1.5–2.5 in spring and summer 2026 (`r1_monthly_mae`). This is drift: a model frozen in March 2024 gets steadily staler. Fine-tuning helps most at midday, in the solar hours (`r4_hour_profile`).
4. **The weather leak barely mattered.** The same self arm scored 19.22 test MAE on the leaky `_lag24` weather and 19.21 on the safe `_asof` weather. This matches the LEAR weather study, where even *perfect* same-day weather is worth only 0.44 test MAE. Lagged weather carries little beyond lagged prices.

### 7.2 H1, seed 0

The weekly block bootstrap of the test MAE difference: negative means the first model is better.

| Comparison | Δ MAE | 95 % CI | Core zones Δ |
|---|---:|---:|---:|
| interconnects − self, fine-tuned | −0.53 | [−0.64, −0.42] | −0.71 |
| interconnects − self, fitted once | −0.75 | [−0.89, −0.61] | −0.97 |
| interconnects − full, fine-tuned | −0.21 | [−0.33, −0.09] | |
| interconnects − self, no curves, safe inputs | −0.59 | [−0.72, −0.46] | −0.82 |
| interconnects − full, no curves, safe inputs | −0.38 | [−0.58, −0.15] | −0.50 |

### 7.3 H1 with three seeds: the headline result

![Bootstrap forest plot](figures/data/r3_bootstrap.png)

| Test MAE, mean of 3 seeds (range) | Fitted once | Fine-tuned monthly |
|---|---:|---:|
| self | 19.02 (18.77–19.21) | 17.84 (17.76–17.88) |
| interconnects | 18.55 (18.46–18.60) | 17.37 (17.35–17.40) |

- **interconnects − self = −0.46 [−0.57, −0.36] fine-tuned, and −0.47 [−0.59, −0.35] fitted once.** No bootstrap resample came out ≥ 0.
- **Every slice has the same sign:**

  | Slice | Fine-tuned | Fitted once |
  |---|---:|---:|
  | Before the 15-minute switch | −0.48 | −0.50 |
  | After the 15-minute switch | −0.46 | −0.46 |
  | Summer | −0.39 | −0.30 |
  | Core zones | −0.63 | −0.60 |

- **Fine-tuning also makes results more reproducible.** The seed range is ≤ 0.13 fine-tuned against ≤ 0.44 fitted once.
- **Conclusion:** the physical interconnector graph improves day-ahead price forecasts by about 0.47 EUR/MWh (2.6 %) over the identical model without message passing, and by 0.2–0.4 over unrestricted attention across all zones.

### 7.4 Where the graph helps, and what attention learns

![Per-zone gain map](figures/data/r2_zone_gain_map.png)

- **By zone (`r2_zone_gain_map`, `r7_zone_mae`):**
  - Largest gains in the meshed centre and the Baltics: DE_LU, PL, LT, CZ, SK and FI improve by 0.9 to 1.3.
  - Near zero or slightly worse at the periphery (NO_4, NO_5, IE_SEM, GR), where zones are radial or weakly coupled.
- **Full-attention arm (`r6_attention_full`).** The share of attention on physical neighbours is only 1.14× what uniform attention would give. Given every zone, the model does not rediscover the grid; and it is 0.2–0.4 worse than the arm that is given the topology.
  - The lift is above 1 in western Europe (ES, PT, BE, CH, FR) and below 1 in the south-east and the Baltics.
  - This supports H1's framing: the topology helps when it is *imposed*, not when it has to be learned.
- **Seeds (`r5_seeds`)** shows every seed of both arms against both LEAR lines.

### 7.5 H2: in progress

Both H2 arms are running now; the queue is in `runlogs/queue.sh` and its log in `runlogs/queue.log`.
- **Constraint arm (`c4s_gat_interconnects_constraints_rolling`).** Fitting on pretrain + train finished. Monthly fine-tuning is in progress at about 9 minutes per month. No metrics yet.
- **JAO-feature arm (`c4s_gat_interconnects_jao_rolling`).** Queued next.

Each arm will be compared with the interconnects control by the same bootstrap, fine-tuned and fitted once. `r8_constraint_attention` will then show whether attention concentrates on tight constraints (low RAM/Fmax, the stand-in for binding), split by AMR uplift and by \|PTDF\|, with the untrained model as a reference.

Caveats for H2:
- Constraint data starts in Jun 2022, so the pretrain period gives this arm no constraint signal.
- Only the 12 Core zones receive messages directly.
- Without shadow prices, "attends to the binding constraints" can only be checked through the RAM/Fmax proxy.

---

## 8. Open items

1. **Running now** (in order; expected done about 1:30–2:30 am):
   - the H2 constraint arm;
   - the H2 JAO-feature arm;
   - the bootstraps of both arms against the control;
   - the safe-input reruns of c2.5, c1 and c2;
   - redraws of figures d7, r3 and r8.
2. **More seeds.** Seeds 1 and 2 for the full arm and for both H2 arms, if H2 shows an effect.
3. **H3.** Test heat stress conditionally rather than through overall MAE: errors and spreads on heat-anomaly days against matched normal days. Bound the effect with the perfect-weather arm. The most valuable new data would be ENTSO-E day-ahead wind and solar forecasts (document type A69), which are what the market actually sees (requires a download; to be agreed first).
4. **H2 timing.** Verify when JAO publishes the day-D domain. If it is before 10:00 UTC on D-1, run a day-D variant.
5. **Labels for H2.** JAO publishes Core active constraints with shadow prices after clearing. Those are outcomes, so never inputs, but they would be the proper attention check.
6. **Data holes to state in the paper:** see section 2.

---

## 9. Where things are

| What | Path (relative to `final project/`) |
|---|---|
| Framework, configs, models | `cs_224w-c1_xgb_zone/` (`epgnn/`, `configs/experiments/`) |
| Feature pipeline | `cs_224w-c1_xgb_zone/epgnn/pipeline/` → `features/v1/` |
| Models | `epgnn/models/gat_scope.py`, `gat_constraints.py`, `asof.py` |
| Bootstrap | `epgnn/bootstrap.py` |
| Results and predictions | `cs_224w-c1_xgb_zone/results/*.json`, `runs/<name>/` |
| Figures | `figures/data/` (scripts: `make_data_figures.py`, `make_result_figures.py`, `make_model_schematics.py`) |
| Detailed audit | `PIPELINE_REVIEW.md` |
| Run queue and logs | `runlogs/` |
| Tests | `cs_224w-c1_xgb_zone/tests/` |
