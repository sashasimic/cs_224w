# Pipeline comparison: milestone evidence and reproducibility appendix

The partner's stronger XGBoost result reproduces. The largest measured feature
addition in a fixed-learner France sequence is **the previous delivery day's
auction price**, not the target transformation alone or post-origin weather.
This is an exploratory, conditional benchmark; it does not certify historical
publication availability or demonstrate a graph-model gain.

The two-page [milestone draft](../reports/milestone/milestone.pdf) follows the
Approach / Experiments / Future Work structure in the user's CS224N template.
That directory contained a blank template, not a completed prior report. Author
details remain a marked placeholder. Course-specific submission requirements
have not been supplied or inferred from the CS224N administrative checklist.

## What differs between the pipelines

| Component | Our audited weekly baseline | Partner `c1_xgb_zone` |
|---|---|---|
| Source table | New canonical grid, missingness preserved | Original `zone_nodes_hourly_v1.0` |
| Master calendar | Jan 2016–Oct 2026; October unassigned | Through Sep 2026, weather-limited |
| Historical Germany | DE_AT_LU and DE_LU separate | DE_AT_LU history spliced under DE_LU |
| Targets | Original valid price, never filled | `price` gated by `has_price`; `price_filled` flags input lag filling, not target filling |
| Inputs | Hour, weekday, month, UTC offset, dated CWE/CORE flags, price_lag168 | price_lag24, price_lag168, load_fc, temp_pw_lag24, temp_anom_lag24, wind_cf_lag24, solar_cf_lag24, net_import_lag48, hour, weekday, month |
| Regional flags | Dated CWE/CORE membership | Not used by this XGBoost configuration; original table's in_core is static membership |
| Input gaps | Native XGBoost NaNs | Retains upstream input fills |
| Learning target | Raw price initially; later asinh(price/50) | asinh(price/50) |
| Objective | Squared error initially; later absolute error | Absolute error |
| Depth / maximum rounds | 5 / 1,000 | 6 / 600 |
| Minimum child weight / lambda | 20 / 5 | 1 / 1 |
| Column / row sampling | 1 / 0.8 | 0.8 / 0.8 |
| Learning rate / seed | 0.05 / 224 | 0.05 / 0 |
| Stopping | Validation only, patience 50 | Validation only, patience 50 |
| Conditional asinh stopping metric | MAE of transformed targets | MAE of transformed targets |
| Final scoring | Inverse transform, EUR/MWh, paired persistence | Same units and paired target masks |

The partner fits independent zone models, not one pooled pan-European model.
Neither this configuration nor our weekly baseline uses neighbours or numeric
JAO constraints. An aggregate protocol comparison cannot identify a single
cause. In particular, adopting only the transformation/loss did not recreate
the full partner result.

## Exact data and partitions

The audited build fingerprint is
`6f5f86c54cbcc7545f4cef01602c863b50e2cb6a436f7ff508d33b1f7f3aaa1f`.
Its immutable generation is reached through `staging/forecast_dataset_v1.0/current`.
It contains 5,887,954 node rows, 49,382,840 directed-edge rows, 62 price-area
identities, 520 directed pairs and 94,967 UTC hours, January 1, 2016 00:00 through
October 31, 2026 22:00 UTC. It is a full grid, not 62 complete valid price series.
The legacy node reference contains 4,240,035 rows and 45 zones, December 31,
2015 23:00 through September 30, 2026 21:00 UTC. Bounds differ because local
delivery dates and source coverage differ.

Source releases are `entsoe_prices@v1.2`, `entsoe_load_forecast@v1.2` and
`entsoe_physical_flows@v1.2`. The supplied weather member is
`weather_hourly_bzone_2010_2026.parquet`, a PECD/ERA5 reanalysis export through
September 2026; requested `weather_hourly@v1.1` release identity remains
unverified despite recording the supplied file's hash. Original JAO exports
resolve to generation `20261105T080957-border-6e149c6f` in
`arsen/initial/jao_analysis_bundle/jao_analysis_output/combined_views`.
Their detailed 17,672,685 CNEC observations remain separate. Observation names
are not asserted to be persistent physical constraint identities.

Price units are EUR/MWh, load/flows MW, and temperature degrees Celsius. No
normalization is fitted. No historical German-only load/weather substitution is
introduced by the audited grid. The original legacy splice remains documented
in the partner reproduction rather than being silently repaired.

| Split, by Brussels delivery date | Inclusive dates | Use in these experiments |
|---|---|---|
| pretrain | 2016-01-01–2022-06-08 | Fit |
| train | 2022-06-09–2024-03-31 | Fit |
| val | 2024-04-01–2025-03-31 | Early stopping and descriptive diagnostics |
| test | 2025-04-01–2026-09-30 | Exploratory held-out-period reporting |
| unassigned | October 2026 | Structural/coverage audit only |

All France diagnostics fit 72,310 valid targets and score 8,760 validation and
13,152 test hours. Every feature profile retains these same target rows, including
rows whose features are masked to NaN. All comparison predictions are finite.

## Matched ten-zone results

All figures are MAE in EUR/MWh; lower is better. Ten zones are BE, FR, NL,
DE_LU, AT, ES, PT, PL, CZ, CH. Each row scores the same 87,600 validation and
131,520 test zone-hours. Target values agree up to about 0.00003 EUR/MWh of
legacy float32 rounding. Aggregation pools errors over zone-hours; country
grouping does not invent a national average target price.

| Model | Validation | Test |
|---|---:|---:|
| Weekly persistence | 31.238496 | 29.146392 |
| Our raw-price, squared-error weekly XGBoost | 29.650764 | 29.593087 |
| Same inputs and parameters, asinh/L1 | 29.439272 | 28.811180 |
| Reproduced full partner pipeline | 21.040087 | 21.166311 |

The full partner reproduction, separately across **43 scored zones**, obtains
20.672483 validation and 21.253918 test versus published 20.691822 / 21.243275.
There are 44 fitted models (including GB historical data), with BA/GB unscored
in evaluation. See [the country breakdown](partner_pipeline_reproduction.md).
Do not compare a 43-zone aggregate directly with our ten-zone average as if
the evaluation population were matched.

## Controlled France sequence

`configs/experiments/pipeline_diagnostics.json` fixes the partner learner,
the original v1.0 table, all target masks, and a single seed. The learner uses
`hist`, depth 6, maximum 600 rounds, learning rate 0.05, row and column sampling
0.8, child weight 1, lambda 1, seed 0, four CPU threads, absolute-error objective,
and asinh(price/50). Validation transformed MAE selects the iteration with
patience 50. The saved model predicts transformed units; invert with
`50 * sinh(prediction)` before reporting price errors. No test-driven parameter
search is performed by this diagnostic script.

| Profile | Validation MAE | Test MAE | Selected rounds |
|---|---:|---:|---:|
| Weekly + dated era flags + UTC offset | 34.668605 | 36.317608 | 236 |
| Weekly + hour/weekday/month, partner order | 31.550977 | 32.092127 | 99 |
| + guarded prior-day auction price | 20.386696 | 22.198833 | 356 |
| + load forecast | 20.390248 | 22.211521 | 418 |
| + lagged net imports | 20.470899 | 22.139388 | 366 |
| + four lag-24 weather inputs, past intervals only | 20.366738 | 22.059441 | 466 |
| Same full input schema, four weather inputs at lag 48 | 20.246292 | 22.119126 | 488 |
| + all original lag-24 weather, including post-origin | 20.409917 | 22.041409 | 408 |

The lag-48 weather row is an alternative to both lag-24 weather profiles, not
another cumulative feature addition. The last row uses the same guarded price
as the preceding rows; it is deliberately distinct from the exact unmodified
partner reproduction (France test 22.089316).

Two controls narrow the explanation:

- Switching only our audited seven-feature asinh/L1 model to partner tree
  parameters worsens France test MAE from 34.826143 to 36.317608. Hyperparameters
  alone do not explain the stronger full pipeline.
- At these same parameters, the corresponding seven-feature legacy-table model
  scores 36.317607778689 versus audited 36.317608288244, a difference of
  0.00000051. France's raw load source has 94,967 rows, zero filled inputs and
  zero availability-verified rows; raw price has zero filled/invalid rows.
  Input filling or tiny target precision differences do not explain France's gap.

The calendar/order change improves 36.32 to 32.09, but removes several columns
and changes column sampling, so it is not a causal test of any single flag.
Adding guarded prior-day prices next reduces test MAE **30.8%**, from 32.09 to
22.20, the dominant measured feature addition in this sequence. Later load,
flow and weather increments are small and mixed. This does not establish their
universal importance in other countries, feature combinations or learners.

## Timing: a correction and remaining uncertainty

The origin is constructed as the preceding calendar date at literal Brussels
noon, then converted to UTC. Winter and summer origins therefore differ in UTC.
Observation valid time, interval completion and publication time are distinct.

**Price correction.** Previous-day day-ahead prices are normally auctioned on
D−2, so a D−1 evening delivery timestamp does not make that auction price
unavailable at D−1 noon. The original audited registry applied a conservative
observation-time rule to these prices; it is not a valid reason to reject every
price_lag24 input. We record this erratum here and in the temperature notes,
without altering immutable build contents or claiming verified archived revisions.
The [official auction timeline](https://www.nordpoolgroup.com/en/the-power-market/Day-ahead-market/)
supports the general publication convention. Exact elapsed lag-24 can land in
day D at the end of the 25-hour autumn day. The diagnostic masks that case:
six pretrain, two train, one validation and one test hour. It does not fill it.

**Weather.** Half of each evaluation split's original weather-lag-24 source
intervals finish after the origin: 4,380 validation and 6,576 test hours. The
past-time experiment masks the four weather inputs there, preserving targets.
Unmasking improves test MAE by only 0.018032, while worsening validation by
0.043179. This is evidence against post-origin weather explaining most of this
France gain; it is not evidence that retrospective weather is operationally safe.
[ERA5](https://www.ecmwf.int/en/forecasts/datasets/complete-era5-global-atmospheric-reanalysis)
is reanalysis, and even lag-48 past observations lack verified historical
publication availability. We have no archived temperature forecast vintages.

**Load/flows/JAO.** Effective-time lags and update fields do not identify the
original publication or subsequent revision history. France's load source marks
no rows availability-verified. These features remain conditional; same-day prices
and realised outcomes are excluded as predictors. No numeric JAO feature is used
in this controlled sequence. The separate JAO experiment is described below.
A historical forecast vintage means the exact saved forecast as it appeared at
that time, before later updates. A current archive of revised values cannot by
itself show what our forecaster would have seen on a past day.

The legacy history-window nanosecond/microsecond defect, fixed-UTC winter cutoff,
and NaN-times-zero torch masking defect remain modelling blockers for affected
paths. Inspection showed that the partner XGBoost flatten/day-input path does
not consume those history tensors or torch losses. They are not explanations
for this XGBoost result. Partner code is unchanged.

## Descriptive uncertainty and earlier related experiments

Validation intervals use a paired calendar-week block bootstrap (Monday–Sunday),
53 blocks, 2,000 resamples, seed 224. The statistic is before-MAE minus after-MAE,
computed on identical target keys. Partial weeks retain their actual hour counts.
Models are not retrained, so this is conditional loss uncertainty, not training
uncertainty or a causal feature effect. No test interval is used for model selection.

| Validation contrast | MAE improvement | 95% interval |
|---|---:|---:|
| Weekly/calendar → + prior-day price | 11.164280 | [8.552848, 14.130501] |
| + price → + load | −0.003551 | [−0.236621, 0.236168] |
| + load → + flows | −0.080651 | [−0.232209, 0.070413] |
| + flows → + past-time weather | 0.104161 | [−0.001801, 0.210755] |
| Past-time weather → all lag-24 weather | −0.043179 | [−0.152428, 0.061570] |
| Lag-48 weather → all lag-24 weather | −0.163626 | [−0.381921, 0.042794] |

Earlier France raw-price/MSE experiments, with only weekly/calendar baseline
features, scored 35.615341 test MAE; adding the last completed current temperature
gave 35.302898, and perfect target-hour realised temperature gave 35.084997.
Persistence was 32.442539. Those are a different learner/input context, not
conflicting replications of this richer-input sequence. See
[temperature experiments](temperature_experiments.md).

Coarse JAO summaries at lag 48 (`n_presolved`, `min_ram_ratio`, publication-domain
presence) changed seven covered countries' mean test MAE from 30.958999 to
30.937099, while slightly worsening validation. That conditional, approximately
0.07% test change does not establish that detailed constraints or graphs are
unhelpful. Missing domains and published zeros remain distinct. See
[constraint-summary experiment](boosting_constraints.md).

The test period was inspected during earlier exploratory runs. The milestone
reports that fact; it does not present repeated test comparisons as a fresh blind
confirmatory evaluation. Controlled diagnostics currently cover France and one
seed. Multi-country replications and publication-aware features are next work,
before evaluating graph gains against the strong recent-price baseline.

## Reproduction and saved evidence

All commands run from the repository root on `arsen_test`; no upload or commit
is required. Python needs the repository's model/inspection dependencies. PDF
compilation additionally uses `pdflatex` and Poppler's `pdfinfo`.

```bash
# Activate the repository environment; the system Python lacks the model stack.
source .venv/bin/activate

# Reuse the exact partner predictions after checking pinned source/config/hashes.
python scripts/experiments/partner_xgb_reaggregate.py \
  --partner-root /srv/cs224w/Project/arsen/partner_pipeline_reproduction \
  --expected-commit 757da793c2e2e726cc6e750f2d42f1dd836ca39c \
  --output-root results --threads 4 --reuse-predictions

# Audited seven-feature hyperparameter-only control (reuses matching completed run).
python -m epgnn.boosting \
  --config configs/experiments/diagnostic_fr_audited_weekly_partner_params.json

# Eight fixed France profiles; verifies and reuses completed diagnostic artifacts.
python scripts/experiments/pipeline_diagnostics.py \
  --config configs/experiments/pipeline_diagnostics.json

# Recreate tables/figure from saved metrics, compile twice, assert exactly two pages.
python scripts/experiments/build_milestone_report.py

python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s tests/dataset_build -p 'test_*.py'
```

Diagnostic fingerprint:
`543a551ca557013472cc4609dec1ed04d3896f34a7b11e648e6eb44751f72376`.
Models, predictions, validation learning curves, feature importance and artifact
checksums are in `runs/pipeline_diagnostics_france/<fingerprint>/`. Source/config
identity changes produce a new generation; incomplete outputs are not published.
Saved/reloaded validation predictions are exactly equal. Source and output SHA-256
hashes are checked before reuse. No original inputs or partner tracked source
were changed.

Relevant source SHA-256:

- Legacy nodes: `589d3d5739397d76a86d0f2d669195c2156e481e40ea23463250046f24cb7a67`.
- Weather: `ce44b58d872836c59f4b1eb78f34b5e067c4d515d2c13ca1d1b6b22bab4082cb`.

Runtime: Python 3.14.8, NumPy 2.5.4, pandas 3.1.1, PyArrow 26.0.1, XGBoost 3.4.1.
The published partner result did not record all original package versions;
the small reproduction difference is retained rather than tuned away.

Machine-readable evidence:

- `results/pipeline_diagnostics_france.json` / `.csv`: profiles, metrics, rounds,
  timing counts, bootstrap results, code/config/source identities.
- `results/milestone_pipeline_audit.json`: matched ten-zone table, France source
  and precision controls, supported and unsupported claims.
- `results/diagnostic_fr_audited_weekly_partner_params.json`: audited control.
- `results/partner_vs_audited_weekly_country_comparison.json` / `.csv`: matched
  country results and key/mask/value checks.
- `results/partner_c1_xgb_zone_by_country.json`: unchanged pipeline reproduction,
  original aggregate reconciliation, country and zone breakdowns.

The report's table, key numbers and figure are generated from these saved metrics.
The separate source manifest retains full raw-source and reference provenance.

Verification completed: 25 top-level experiment tests and 17 dataset tests pass
in the repository environment. Reusing the France generation checks all saved
artifact hashes without refitting. The PDF is compiled twice, checked to contain
exactly two pages and visually inspected after rasterizing both pages.
