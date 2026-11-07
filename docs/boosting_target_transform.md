# Country models with the partner's target transformation and loss

## Transformation

The partner's default XGBoost target on branch `c1_xgb_zone` is

```text
z = asinh(price_EUR_per_MWh / 50)
price_prediction_EUR_per_MWh = 50 * sinh(predicted_z)
```

This is the inverse hyperbolic sine, with a fixed 50 EUR/MWh scale. It is roughly
linear near zero and logarithmic for large magnitudes, preserving zero and
negative prices. For example, -500 maps to about -2.998, zero to zero, and +500
to +2.998. It compresses extreme prices before fitting, rather than dropping,
clipping, filling or changing their valid target values. The scale is fixed;
no normalization statistics are estimated from validation or test.

The partner uses `reg:absoluteerror` (L1 loss) in this transformed target space.
This is not identical to minimizing MAE in EUR/MWh. Its default validation
metric/early stopping is also transformed-target MAE. After selecting rounds,
we invert all predictions and report the usual MAE/RMSE in EUR/MWh. The run
metadata and learning curves explicitly record their metric units.

The partner also supports a separate transformed residual target, but that is
not the default in its published `c1_xgb_zone` run and is not used here.

## Controlled country experiment

```bash
python -m epgnn.boosting_batch --config configs/experiments/c1_xgb_countries_weekly_asinh_l1_conditional.json
python -m unittest discover -s tests -v
```

We fit one independent model per country/zone, including France, using the same
ten-country selection as the previous comparison. Previous country models were
also separate models, not one pooled European model. Only **target transformation
and loss** change here. To avoid confounding these changes with more inputs or
other hyperparameters, this run retains the original audited feature set:
hour_local, dow, month, utc_offset_hours, in_cwe, in_core and price_lag168.

All other tree parameters remain unchanged: max_depth 5, at most 1,000 rounds,
learning_rate 0.05, min_child_weight 20, subsample 0.8, colsample_bytree 1,
reg_lambda 5, random_state 224, CPU hist with 4 threads and patience 50.
No constraints, current temperature or perfect weather inputs are added.
This is the partner's transformation/loss applied to our controlled baseline;
it is not an exact replication of their richer feature set or different tree
hyperparameters, and it does not reproduce their multi-zone 21.24 score.

The dataset fingerprint, target values, zone-hour keys and split dates are
unchanged. Training uses valid pretrain + train labels through 2024-03-31,
validation runs from 2024-04-01 through 2025-03-31, and test runs from 2025-04-01
through 2026-09-30. Each new country model evaluates test once. Only validation
selects rounds. No parameter sweep or test-driven country selection is performed.
Historical geography remains separate, missing input histories stay NaN, invalid
labels are excluded only as fit/evaluation labels, and October stays unassigned.
The weekly price input and all runs retain their conditional availability status.

## Test results

MAE is EUR/MWh. Each method is compared on the same 13,152 target/persistence
hours per country. The previous raw-price/squared-error model is retained as a
control; both its archived artifacts and its target/mask values were verified.

| Country | Previous raw-price XGBoost | Asinh + L1 XGBoost | Weekly persistence |
|---|---:|---:|---:|
| Belgium | 30.41 | 30.01 | 31.12 |
| France | 35.62 | 34.83 | 32.44 |
| Netherlands | 30.87 | 30.68 | 31.04 |
| Germany–Luxembourg | 32.72 | 32.34 | 33.91 |
| Austria | 27.41 | 26.24 | 27.24 |
| Spain | 29.33 | 26.82 | 25.77 |
| Portugal | 28.51 | 26.57 | 25.98 |
| Poland | 30.79 | 30.51 | 33.24 |
| Czechia | 28.90 | 28.60 | 29.40 |
| Switzerland | 21.37 | 21.51 | 21.33 |

Across the same ten countries, hour-weighted test MAE improves from
**29.59 to 28.81 EUR/MWh**, or **2.64%**.
Nine of ten countries improve relative to their previous raw-price model.
The new model beats persistence in **6 of ten countries**;
persistence's aggregate test MAE is **29.15**, giving the new recipe an
aggregate advantage of **1.15%**. Aggregate test RMSE is
**42.11** versus persistence's **45.30**.
France improves from 35.62 to 34.83, but persistence still has the lower MAE there
(32.44). This change does not close the gap to the partner's aggregate result.

On validation, aggregate MAE changes from **29.65 to 29.44**,
versus persistence's **31.24**. These EUR/MWh metrics are separate
from the transformed-unit MAE used to select boosting rounds.

## Artifacts and checks

- `results/c1_xgb_countries_asinh_l1_comparison.json` and `.csv`: matched comparison
  with previous raw-price models and percentage changes.
- `results/c1_xgb_countries_weekly_asinh_l1_conditional.json` and `.csv`: aggregate
  and per-country comparison against weekly persistence, with model provenance.
- `configs/experiments/c1_xgb_<zone>_weekly_asinh_l1_conditional.json`: full configs.
- `runs/c1_xgb_<zone>_weekly_asinh_l1_conditional/`: saved model, learning curve,
  predictions, feature registry, config, source fingerprints and checksums.

**The saved model.ubj predicts transformed units.** Apply the config's inverse
transform, `50*sinh(prediction)`, when loading it directly. Saved prediction
Parquets already contain `prediction` in EUR/MWh and additionally retain
`prediction_transformed` so inversion can be audited.

Twenty-one top-level tests pass, including transform/inverse round trips for
negative, zero, extreme and missing prices, invalid-scale rejection, and errors
computed in original price units. Existing tests protect chronological roles,
missing-value masks, feature admission, temperature oracle separation and
country aggregation. Saved models reload with exact validation predictions.
All saved inversions and EUR/MWh metrics were recomputed and checked, and all
old/new target keys, validity masks, weekly inputs, parameters other than the
objective, runtime versions and output artifact hashes were verified.
