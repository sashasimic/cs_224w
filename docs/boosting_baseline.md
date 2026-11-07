# First strict XGBoost run

Run from the repository root:

```bash
python -m epgnn.boosting --config configs/experiments/c1_xgb_be_strict.json
python -m unittest discover -s tests -p 'test_boosting.py' -v
```

This is **one Belgium (BE) model**, using the audited strict feature view. It
uses hour_local, dow, month, utc_offset_hours, in_cwe and in_core, exactly the
currently admitted node inputs. Targets remain unfilled EUR/MWh prices, including
negative prices. No weather, load, JAO numeric features or archived price lags
are admitted to this model. There is no scaling, target transform or imputation;
XGBoost handles feature NaNs natively.

The runner uses the new construction artifacts directly, independently of the
legacy `epgnn.run`/sample/scoring path and its recorded modelling blockers. It
does not change that path or claim that its XGBoost stub is implemented.

The experiment pins dataset fingerprint
`6f5f86c54cbcc7545f4cef01602c863b50e2cb6a436f7ff508d33b1f7f3aaa1f`.
Artifact checksums and feature admission are checked before fitting. Only
`pretrain` and `train` supply fit labels: **72,310 valid hourly targets**.
Early stopping uses **8,760 validation targets**, with patience 50 and a maximum
of 1,000 rounds. Exactly one model configuration was fitted; validation selected
**one round**. The unchanged test period is evaluated once on **13,152 hours**.
All **745 unassigned BE rows** remain outside fitting and evaluation, including
the two UTC hours in September belonging to October locally.

| Split | Strict XGBoost MAE | Strict XGBoost RMSE | Conditional weekly persistence MAE | Common hours |
|---|---:|---:|---:|---:|
| Validation | 37.82 | 50.18 | 32.26 | 8,760 |
| Test | 40.89 | 57.41 | 31.12 | 13,152 |

Errors are EUR/MWh. This calendar/region-only model underperforms weekly
persistence. The persistence comparison uses the original `price_lag168` on
exact zone/UTC keys, with the same finite target/prediction mask for both
methods. It remains explicitly **conditional** because the archived price
vintages lack verified historical publication times. It is not presented as an
admitted strict input. All valid BE validation/test targets have this comparator;
missing histories elsewhere would be counted and excluded only from the paired
comparison, not the model's full-target score.

`results/c1_xgb_be_strict.json` contains metrics, fixed temporal evaluation
slices, feature importance, config, runtime versions, dataset and runner hashes,
git state, early stopping choice and exclusions. The git workspace was dirty
because the dataset-construction and modelling changes were not committed.
The precise runner SHA-256 is recorded independently of the commit.

`runs/c1_xgb_be_strict/` contains model.ubj, prediction Parquets with original
targets and validity/comparison masks, learning_curve.json, the selected feature
registry, input artifact hashes, config.json, result.json and complete.json.
The model was reloaded and its validation predictions checked for exact equality.
Evaluation predictions include rows with missing labels when present; their
labels remain missing and never enter loss calculations.

Publication is atomic. Repeating the command verifies and reuses the completed
run without reevaluating test. A changed code/config/runtime requires a new run
name. The unit tests cover feature admission, chronological roles, retention of
rows with missing features, NaN targets, and matched persistence masks. No test
target distributions or tuning sweeps were produced.

## Conditional XGBoost with weekly price history

```bash
python -m epgnn.boosting --config configs/experiments/c1_xgb_be_weekly_conditional.json
```

The second run adds **only price_lag168** to the first model's six features.
It preserves the same dataset fingerprint, BE targets, chronological splits,
model parameters, objective, random seed, early-stopping patience and evaluation
masks. Weather, load and numeric JAO inputs remain unused. Exactly one additional
configuration was fitted, with validation selecting **45 boosting rounds**.

| Method | Validation MAE | Test MAE |
|---|---:|---:|
| Original strict calendar/region XGBoost | 37.82 | 40.89 |
| Conditional weekly persistence | 32.26 | 31.12 |
| Conditional calendar/region + weekly-price XGBoost | **30.73** | **30.41** |

MAE is EUR/MWh. The new model improves on weekly persistence by **4.74% on
validation** and **2.28% on test**. Its test RMSE is **45.64 EUR/MWh**, compared
with **49.47** for persistence. The paired comparisons use exactly the original
8,760 validation and 13,152 test zone-hours, target values, validity masks and
weekly persistence values, checked against the first run's prediction files.
Weekly price history accounts for approximately 85.25% of the new model's
reported normalized gain-based feature importance.

The configuration must explicitly declare
`"conditional_features": ["price_lag168"]`. The runner accepts this one
conditional feature only; the default strict whitelist remains unchanged.
Conditional values come from the original master through zone/UTC key joins.
The runner checks exact t-168h observations and interval completion before the
forecast origin. Historical publication availability remains unverified, so the
input and model retain their **conditional** label. The dataset registry and
strict feature view are not reclassified or modified.

The first 168 pretraining rows have missing weekly history; they are retained
with NaNs handled natively by XGBoost. Missing targets remain missing. Validation
and test have complete weekly histories. No rows were dropped for missing
features, and the fit target count remains 72,310.

Model, config, learning curve, predictions, feature registry and artifact hashes
are saved under `runs/c1_xgb_be_weekly_conditional/`; the metrics and provenance
record is `results/c1_xgb_be_weekly_conditional.json`. Saved prediction files also
retain the conditional input, checked to equal the persistence comparator
exactly. Model reload reproduces validation predictions exactly. All artifacts
of the earlier strict run retain their original hashes.

Seven runner contract tests now pass, including explicit conditional opt-in,
rejection of other conditional/outcome columns, exact historical timing checks,
and retention of missing feature histories. As with the first run, repeating a
completed run with matching code/config/runtime verifies its artifacts and
returns cached results without refitting or reevaluating test. The runner code
changed to add this mode; reproducing the earlier code under a changed runner
requires a new run name, as described above.
