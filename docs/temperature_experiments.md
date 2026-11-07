# France temperature experiments

Run the two fixed configurations from `arsen_test`:

```bash
python -m epgnn.boosting --config configs/experiments/c1_xgb_fr_weekly_temp_current_conditional.json
python -m epgnn.boosting --config configs/experiments/c1_xgb_fr_weekly_temp_perfect_oracle.json
python -m unittest discover -s tests -p 'test_boosting*.py' -v
python -m unittest discover -s tests -p 'test_temperature_experiments.py' -v
```

Both models add exactly one temperature input to the saved France weekly-price
XGBoost baseline: six calendar/region inputs plus price_lag168. No constraints,
load forecasts, wind or solar inputs are added. Dataset fingerprint, model
parameters, objective, seed, runtime versions, training labels and split dates
match that baseline. Only validation selects the number of rounds, and one
configuration is fitted per temperature experiment. Existing models are unchanged.

## Definitions and limits

**Current-temperature proxy** (`temp_pw_current_origin`, conditional): retrieve
population-weighted temperature from the exact source timestamp one hour before
the D-1 Brussels-noon origin. This is **11:00 Europe/Brussels on D-1**, interpreted
as the last hourly interval ending at noon. Freeze that one value for every
hour of delivery day D. Winter/summer UTC offsets and DST are derived from the
local calendar. No D-1 afternoon or day-D temperature is supplied. Missing exact
observations remain NaN; no nearest-time substitution or filling is used.

This uses the retrospectively reconstructed PECD/ERA5 series as a proxy for
current weather. Its actual historical publication latency is unverified, so
it is not a verified real-time measurement feed or an archived weather forecast.

**Perfect-weather diagnostic** (`temp_pw_perfect`, explicit oracle): retrieve
actual/reanalysis population-weighted temperature for each target delivery hour.
This deliberately supplies future weather. The run is labelled
`oracle_perfect_weather`; its experiment registry keeps the feature **excluded**
from ordinary forecasting. A separate `oracle_features` opt-in is required;
conditional/strict feature opt-ins cannot silently admit it. It is a diagnostic
for the information in future temperature, not a deployable forecasting score
or a mathematical guarantee of an upper performance bound.

The original weather file covers 2010 through September 2026 and is verified
against the dataset source-manifest hash. Reading it directly provides the
2015 temperature needed for the first January 2016 forecast origin. No external
weather forecasts were fetched. Only temperature in degrees Celsius is used;
upstream temperature-anomaly climatology is not included. The original source
registry and strict view remain unchanged.

## Results

MAE and RMSE are EUR/MWh. All methods use the same 8,760 validation and 13,152
test hours and identical target values, masks and weekly persistence values.
Temperature is complete on these evaluation hours. Training retains all 72,310
valid target rows, with missing input histories handled natively by XGBoost.
Unassigned October rows stay excluded.

| Method | Validation MAE | Test MAE | Test RMSE | Selected rounds |
|---|---:|---:|---:|---:|
| Weekly-price XGBoost baseline | 33.62 | 35.62 | 44.90 | 127 |
| + Current-temperature proxy | 31.06 | 35.30 | 44.54 | 116 |
| + Perfect delivery-hour temperature (oracle) | 30.70 | 35.08 | 44.76 | 322 |
| Weekly persistence (conditional) | 33.15 | 32.44 | 45.38 | — |

The current proxy improves test MAE by about **0.88%**, and perfect temperature
by **1.49%**, versus the weekly-price XGBoost baseline. Validation gains are
larger: **7.61%** and **8.70%**. Neither temperature model beats weekly persistence
on test. These two simple temperature inputs do not explain the full gap to the
partner's stronger published XGBoost result.

## Partner setup comparison and timing correction

The inspected `c1_xgb_zone` branch at commit
`757da793c2e2e726cc6e750f2d42f1dd836ca39c` reports test MAE **21.24** across
565,320 zone-hours, with no separate France score in its report. It uses a richer
feature set: price_lag24, price_lag168, load forecasts, 24-hour-lagged temperature,
temperature anomaly, wind and solar, 48-hour-lagged net imports, and calendar
inputs. It uses **reg:absoluteerror on asinh(price/50)**, with default early
stopping on transformed-target error. Our controlled runs retain squared-error
training on raw EUR/MWh prices, with validation MAE selecting rounds.

These are not matched models: country coverage, features, target transformation,
objective and source tables differ. The reported aggregate 21.24 cannot be
compared directly with France's 35.62.

**Correction to earlier commentary:** a later delivery timestamp does not prove
a day-ahead price is unavailable. EPEX's day-ahead auction closes at noon on the
day before delivery and normally publishes results afterward. Prices for D-1
are therefore normally already published on D-2, before our D-1 noon origin.
Price_lag24 can be useful available information even for D-1 evening delivery
hours; archived revisions still require a separate vintage check. The timestamp
check finding that half of validation's lag24 delivery intervals lie after the
origin is **not evidence of leakage** for day-ahead auction prices.
An exact elapsed 24-hour lag also needs a fall-DST guard: the last hour of a
25-hour day can reference a price from the same delivery day, which would not
normally be published by the preceding noon. One such France validation hour is
present. A proper availability rule must use market delivery dates and actual
publication timing, rather than blanket admission or rejection based on lag names.

Sources: [EPEX trading schedule](https://www.epexspot.com/sites/default/files/2024-12/_Trading%20brochure%202024%20December%20%281%29.pdf),
[Nord Pool day-ahead process](https://www.nordpoolgroup.com/en/the-power-market/Day-ahead-market/).
This distinction also limits the old dataset guide's overly conservative
interval-completion language for price lags; valid delivery time is not a
publication timestamp. These experiments do not reclassify the stored registry.
Realised weather for D-1 afternoon, unlike the already-auctioned prices for those
hours, is still future information at noon. The partner's weather_lag24 and
load-forecast vintages need their own availability review.

A useful next matched comparison is adding legitimately available recent
auction prices and separately testing the MAE objective/target transform.
Those changes were not mixed into these two temperature experiments.

## Artifacts and verification

- `results/fr_temperature_experiments.json` and `.csv`: matched comparisons,
  temperature definitions and partner setup notes with the timing correction.
- `results/c1_xgb_fr_weekly_temp_current_conditional.json` and
  `results/c1_xgb_fr_weekly_temp_perfect_oracle.json`: full run metrics/provenance.
- Corresponding `runs/<name>/` folders: models, predictions, learning curves,
  selected registries, temperature source timestamps, configs and checksums.

Thirteen existing boosting/aggregation tests and four temperature tests pass.
The temperature tests cover DST/current-time freezing, future temperature changes
not affecting current inputs, exact zone/hour lookup, missing versus negative
values, verified history before the output window, and separate oracle admission.
Every newly saved model was reloaded with exact validation-prediction equality.
Old and new artifact hashes, evaluation keys, targets, masks, baseline inputs,
source temperatures and mode labels were checked. The user-authorized shared
checkout was restored to `arsen_test`; the temporary worktree was removed without
changing the partner branch or original data.
