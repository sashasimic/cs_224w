# Reproducing and reaggregating the partner XGBoost pipeline

The `c1_xgb_zone` branch did not contain per-country loss metrics. It contained
aggregate metrics and per-zone tree counts; no existing prediction files for this
run were present in the shared workspace. We ran its original pipeline once and
then reaggregated the saved predictions. The pipeline already fits separate
zone models, so no training changes were required to produce these breakdowns.

## Reproduction protocol

- Branch snapshot: `c1_xgb_zone`, commit `757da793c2e2e726cc6e750f2d42f1dd836ca39c`.
- Published result references training commit `068b30f`; the reproduction runs
  the requested branch's latest pinned source, without editing it.
- Exact `configs/experiments/c1_xgb_zone.yaml` model/feature/training settings.
  Only the output name is overridden to avoid replacing published results.
- Original `zone_nodes_hourly_v1.0` and `zone_edges_hourly_v1.0` inputs, checked
  against the branch manifest's sizes and MD5s, with SHA-256 also recorded.
- Original feature builder, samples, asinh target, objective, validation stopping
  and scoring functions. No feature admission or imputation changes are applied.
- Four OpenMP threads bound runtime resources; model hyperparameters are unchanged.
- One fit through the original pipeline; later reaggregation reuses prediction
  arrays without fitting or making new model predictions.

The reproduction snapshot lives in
`/srv/cs224w/Project/arsen/partner_pipeline_reproduction`, detached at the pinned
commit. The shared checkout stays on `arsen_test`; the partner branch reference
and its tracked files remain unchanged. Original Parquets are linked into the
snapshot and are not modified.

```bash
# First execution in a fresh, pinned partner worktree:
python scripts/experiments/partner_xgb_reaggregate.py \
  --partner-root /srv/cs224w/Project/arsen/partner_pipeline_reproduction \
  --expected-commit 757da793c2e2e726cc6e750f2d42f1dd836ca39c \
  --output-root results --threads 4

# With the saved reproduction already present, use this instead:
python scripts/experiments/partner_xgb_reaggregate.py \
  --partner-root /srv/cs224w/Project/arsen/partner_pipeline_reproduction \
  --expected-commit 757da793c2e2e726cc6e750f2d42f1dd836ca39c \
  --output-root results --threads 4 --reuse-predictions
```

The script verifies the commit, unchanged tracked source, pinned source files,
and, on reuse, saved artifact hashes/config/source identities before importing
the partner package from that worktree. It records import location, code hashes,
versions, input identities and protocol with the scores. A changed reproduction
is rejected rather than silently refitted or overwritten.

## Aggregate reproduction

| Split | Published MAE | Reproduced MAE | Valid zone-hours |
|---|---:|---:|---:|
| val | 20.69182 | 20.67248 | 376,680 |
| test | 21.24327 | 21.25392 | 565,320 |

MAE is EUR/MWh. Reproduction test MAE **21.25** closely matches the published
**21.24**; the measured difference is **0.01064 EUR/MWh**. We do not force an exact
match or tune against that difference. Full original package versions were not
recorded in the published report; the reproduction's versions are saved. Its
pipeline execution and initial reaggregation took about **37.2 seconds**.

There are **44 fitted zone models**, including GB with historical training data,
and **43 scored bidding zones** in each evaluation split. BA and GB have no valid
validation/test targets and are explicitly unscored. The country breakdown
combines subzones into **29 scored country/combined-market groups**.

## Familiar country/zone results

All figures below are test MAE in EUR/MWh. The comparison with our audited
weekly-only asinh/L1 run uses exactly the same zone-hour keys, masks and target
values (up to at most about 0.00003 EUR/MWh of float32 rounding). It is not a
single-feature ablation: input features, hyperparameters and source-table
construction differ, and the partner retains v1.0's historical German splice.

| Country / zone | Partner pipeline | Our weekly-only asinh/L1 | Weekly persistence |
|---|---:|---:|---:|
| France | 22.09 | 34.83 | 32.44 |
| Belgium | 23.66 | 30.01 | 31.12 |
| Netherlands | 23.10 | 30.68 | 31.04 |
| Germany–Luxembourg | 24.58 | 32.34 | 33.91 |
| Austria | 20.04 | 26.24 | 27.24 |
| Spain | 18.22 | 26.82 | 25.77 |
| Portugal | 17.92 | 26.57 | 25.98 |
| Poland | 25.07 | 30.51 | 33.24 |
| Czechia | 22.32 | 28.60 | 29.40 |
| Switzerland | 14.66 | 21.51 | 21.33 |

France is **22.09** against persistence's **32.44**, versus **34.83** for our
weekly-only asinh/L1 model. The partner's advantage on France is therefore not
explained merely by reporting an average over other countries. The full partner
pipeline is materially stronger on these matched evaluation hours.

## Country and bidding-zone aggregation

Country summaries combine error sums over bidding-zone hours: Italy's seven
zones, Norway's five, Sweden's four and Denmark's two. MAE is weighted by valid
hours; RMSE is calculated by pooling squared errors. These are errors across
regional targets, not the error of a newly constructed national average price.
DE_LU and IE_SEM remain explicitly combined markets rather than being split
between countries using invented weights. Individual zone scores remain available.

Country and zone error sums, RMSEs and target counts reconcile with the original
pipeline aggregate. Float64 reaggregation is compared with its float32 reductions
using a small explicit numerical tolerance. No model-specific target filtering
is added. All valid target predictions and persistence values are finite.

## Availability interpretation

These are reproductions of the partner's existing protocol, **not a strict-noon
availability certification**. Its richer inputs include price_lag24, load
forecasts, yesterday's temperature/wind/solar and lagged net imports. Day-ahead
prices for the previous delivery day are normally known before our origin;
a later delivery timestamp alone does not make them unavailable. Exact elapsed
24-hour lags still require a fall-DST guard. Realised weather_lag24 for yesterday
afternoon is after the noon origin, and load/weather publication vintages remain
unverified. Reproduction preserves those choices rather than silently repairing
or relabelling them. Their contribution cannot be inferred from the aggregate
score alone.

## Saved results

- `results/partner_c1_xgb_zone_by_country.json`: full pipeline report, original
  and reproduced aggregates, per-zone scores, country aggregates and provenance.
- `results/partner_c1_xgb_zone_country_aggregates.csv` / `.json`: all country and
  combined-market scores, including explicit no-target cases.
- `results/partner_c1_xgb_zone_by_zone.csv`: all 45 zone rows per split, 43 scored.
- `results/partner_vs_audited_weekly_country_comparison.csv` / `.json`: matched
  comparison for the ten countries in our previous experiments.
- `partner_pipeline_reproduction/runs/partner_c1_xgb_zone_reproduction/`:
  original prediction arrays, keyed prediction Parquets, reaggregation report,
  per-zone scores and checksums. The original pipeline does not save fitted
  XGBoost models; no model-saving changes were injected into it.

The real-data integration checks verify sample/prediction shapes, unique UTC
keys, finite valid values, aggregate reconciliation and unchanged source hashes.
Two unit tests protect weighted country/RMSE aggregation, combined-market
identity and no-target handling. Reusing saved predictions was also exercised.
