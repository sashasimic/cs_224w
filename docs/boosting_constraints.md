# Adding historical constraint summaries

```bash
python -m epgnn.boosting_batch --config configs/experiments/c1_xgb_countries_weekly_jao48_conditional.json
python -m unittest discover -s tests -p 'test_boosting*.py' -v
```

Seven separate country models add three explicit conditional inputs to the
existing calendar/region + weekly-price experiment:

- `n_presolved_lag48`: number of presolved constraints affecting the zone.
- `min_ram_ratio_lag48`: minimum source RAM/Fmax ratio for the zone.
- `jao_domain_present_lag48`: original domain-presence indicator, masked to the
  zone's historical CWE/CORE membership.

These are zone summaries of original JAO observations. They are not detailed
CNEC graph inputs, market-clearing binding labels, physical interconnector
ratings or a jointly feasible import capacity.

The effective source-domain hour is exactly 48 elapsed hours before the target
hour, joined on the same canonical zone and UTC timestamp. Its interval ends
before the forecast origin. This avoids using a delivery-day domain as though it
were already observed at the previous day's noon. Historical publication time
remains unverified: a lag establishes historical domain timing, not publication
availability. Inputs and runs remain **conditional**. Same-day domains were not
used. The original dataset feature registry and strict view are unchanged;
additional feature definitions are recorded in each experiment's local registry.

CWE and CORE summaries retain their original units and geography. Missing counts
and ratios stay NaN, published/source zeros remain zeros, and a missing domain
publication has a false presence indicator rather than invented constraint
values. Unknown coverage remains missing. No forward/backward filling, region
substitution or complete-case filtering is performed. Czechia and Poland have
no CWE summary history; their pre-CORE feature values remain missing. Austria
and Germany–Luxembourg never inherit DE_AT_LU's observations. Verified original
JAO records before the node-master calendar supply the first 48 hours of eligible
history, including December 2015 for January 2016 targets. Their hashes are
included in model input provenance.

Spain, Portugal and Switzerland have no zone constraint summaries in this
source; their earlier models are retained without fabricating extra features.
The other seven countries get exactly one new configuration each. Dataset,
parameters, objective, seed, fit labels and splits match the existing weekly-price
models. Only validation selects the final number of boosting rounds. Training
uses pretrain + train through 2024-03-31, validation uses 2024-04-01–2025-03-31,
and test uses 2025-04-01–2026-09-30. October stays unassigned.

## Incremental test results

MAE is EUR/MWh. Both models use identical valid target hours and weekly
persistence values. Each country contributes 13,152 test hours and 8,760 validation
hours. Rows with missing constraint values remain in the model and comparison.

| Country | Weekly-price XGBoost | + JAO summaries | Improvement from constraints |
|---|---:|---:|---:|
| Belgium | 30.41 | 30.46 | -0.15% |
| France | 35.62 | 34.94 | +1.89% |
| Netherlands | 30.87 | 30.87 | +0.02% |
| Germany–Luxembourg | 32.72 | 32.63 | +0.25% |
| Austria | 27.41 | 27.46 | -0.17% |
| Poland | 30.79 | 31.03 | -0.77% |
| Czechia | 28.90 | 29.17 | -0.95% |

Across the **same seven countries**, test MAE changes from **30.96
to 30.94**, an improvement of **0.07%**.
Three countries improve and four worsen. France improves by about 1.89%, but its
34.94 MAE remains above weekly persistence's 32.44. On validation, aggregate MAE
changes from **30.65 to 30.66**, a slight deterioration.
These summaries add little aggregate benefit in this fixed experiment.

The seven-country averages should not be compared directly with the previous
ten-country average: Spain, Portugal and Switzerland are excluded from this
incremental comparison because their JAO zone features are absent.

## Artifacts and checks

- `results/c1_xgb_countries_jao48_incremental.json` and `.csv`: original versus
  constraint-feature scores on matched hours, source coverage and fixed
  coverage-subgroup scores.
- `results/c1_xgb_countries_weekly_jao48_conditional.json` and `.csv`: country and
  aggregate scores against weekly persistence.
- `configs/experiments/c1_xgb_<zone>_weekly_jao48_conditional.json`: complete configs.
- `runs/c1_xgb_<zone>_weekly_jao48_conditional/`: model, predictions, learning
  curve, experiment registry, source-domain timestamps and input/output hashes.

Thirteen runner/aggregation tests pass, including conditional admission,
pre-window history/checksums, missing versus zero, structural regional absences,
exact lag timing, and DE_AT_LU separation. Every saved model was reloaded with
exact validation-prediction equality. All old/new model artifacts were checked
against their hashes; target keys, labels, validity masks, weekly lag inputs and
comparison masks match the original models. Used constraint-domain timestamps
were checked against exact t-48h joins and the forecast origin. Earlier runs and
original source files remain unchanged.
