# Country XGBoost comparison

Run the fixed experiments with:

```bash
python -m epgnn.boosting_batch --config configs/experiments/c1_xgb_countries_weekly_conditional.json
python -m unittest discover -s tests -p 'test_boosting*.py' -v
```

Nine new, separate country models were fitted; Belgium's existing conditional
model was verified and reused. These are per-zone models, rather than one pooled
model across countries. Each uses the same six calendar/region features plus its
own price_lag168. Model parameters, the 1,000-round maximum, 50-round early
stopping patience, training roles and dataset fingerprint match the Belgium run.
Only validation chooses the final boosting round. One configuration was fitted
per new country, with no test-driven parameter or country selection.

Training uses available valid targets in pretrain + train, from 2016-01-01
through 2024-03-31. Austria and Germany–Luxembourg start on 2018-10-01 on their
own identities: DE_AT_LU is not substituted into either zone. Poland retains
its original missing/invalid training targets. Feature NaNs are handled natively;
there is no complete-case filtering or target filling. Validation runs from
2024-04-01 through 2025-03-31; test runs from 2025-04-01 through 2026-09-30.
October 2026 stays unassigned.

Both XGBoost and weekly persistence remain conditional because archived
historical publication vintages are unverified. No weather, load, neighbouring
prices or numeric JAO features were added.

## Test results

MAE is EUR/MWh. Improvement is 100 × (1 − model MAE / persistence MAE), so
positive values favor XGBoost. All ten countries have 13,152 common valid test
hours and 8,760 common valid validation hours. The model and comparator use
identical target values and finite masks for every country/split pair.

| Country / zone | XGBoost MAE | Weekly persistence MAE | Improvement | Selected rounds |
|---|---:|---:|---:|---:|
| Belgium (BE) | 30.41 | 31.12 | +2.28% | 45 |
| France (FR) | 35.62 | 32.44 | -9.78% | 127 |
| Netherlands (NL) | 30.87 | 31.04 | +0.53% | 36 |
| Germany–Luxembourg (DE_LU) | 32.72 | 33.91 | +3.52% | 33 |
| Austria (AT) | 27.41 | 27.24 | -0.62% | 33 |
| Spain (ES) | 29.33 | 25.77 | -13.83% | 41 |
| Portugal (PT) | 28.51 | 25.98 | -9.76% | 45 |
| Poland (PL) | 30.79 | 33.24 | +7.37% | 33 |
| Czechia (CZ) | 28.90 | 29.40 | +1.72% | 38 |
| Switzerland (CH) | 21.37 | 21.33 | -0.21% | 49 |

XGBoost improves test MAE in **5 of 10 countries**.
Across 131,520 country-hours, its hour-weighted test MAE is
**29.59**, versus **29.15** for persistence:
XGBoost is **1.53% worse** overall on MAE. Its
pooled test RMSE is lower: **42.51** versus **45.30**.
Pooled RMSE is calculated from hour-weighted squared RMSEs, not the average of
country RMSEs.

On validation, hour-weighted MAE is **29.65** versus
**31.24**, an improvement of **5.08%**.
France is worse than persistence on both validation and test. These results do
not establish a consistent advantage for this small feature set across countries.
All country results are retained, including the countries where XGBoost is worse.

## Artifacts and verification

- `configs/experiments/c1_xgb_<zone>_weekly_conditional.json`: separate reviewable configs.
- `results/c1_xgb_countries_weekly_conditional.json`: aggregate scores, per-country
  scores, parameters, fingerprints, country report hashes and evaluation protocol.
- `results/c1_xgb_countries_weekly_conditional.csv`: validation and test comparison rows.
- `results/c1_xgb_<zone>_weekly_conditional.json`: each model's metrics and provenance.
- `runs/c1_xgb_<zone>_weekly_conditional/`: saved model, predictions, learning curve,
  config, feature registry and checksums.

Nine runner/aggregation tests pass. Each newly saved model was reloaded and its
validation predictions checked for exact equality. All saved artifact hashes,
zone/hour uniqueness, split labels, target masks, finite predictions and equality
between the XGBoost weekly input and the persistence comparator were checked for
all 20 country/split prediction files. Belgium's saved artifacts are unchanged.
Repeating matching country runs reuses saved outputs without test reevaluation.
