# Relational and adaptive zone graph experiments

The suite compares two ways to use market relationships. The relational study compares learned aggregation and cross-zone messages with pooled SQL-based XGBoost. The adaptive study tests whether learned links improve on physical links and uniform pooling, then asks whether historical directional CORE margins improve learned routing.

See the [verified 45-run validation report](combined_graph_validation.md) for the score table and paired confidence intervals.

## Run from the repository root

```bash
.venv/bin/python -m epgnn.combined_graph_study prepare
.venv/bin/python -m unittest discover -s tests/combined_graph_study -v
.venv/bin/python -m epgnn.combined_graph_study smoke
.venv/bin/python -m epgnn.combined_graph_study run
.venv/bin/python -m epgnn.combined_graph_study validate
.venv/bin/python -m epgnn.combined_graph_study report
```

The config is `configs/experiments/combined_graph_study.json`. It pins the audited catalog and benchmark array identities through the existing CORE config. The preparation stage streams source rows into time-indexed arrays, separates presence from value validity, and publishes an immutable hash-checked cache. SQL aggregates use the same windows and masks as the relational GNN. The XGBoost fits run on CPU. Use `--family xgb`, `--family relational`, or `--family adaptive` for separate runs. The GPU families require the local ROCm device. `--seed` can select one of the five configured seeds for a training command.

Runs under `runs/combined_graph_study/` retain checkpoints, keyed predictions, source lineage, and run settings. Repeating a command resumes neural calibration or refitting at the last saved epoch and reuses completed jobs. The adaptive jobs also save their average directed validation weights. The report aggregates five seeds and verifies the exact prediction keys, outcome masks and checksums first.

## Forecast-time windows

The origin is noon Europe/Brussels on the calendar day before delivery. Exact UTC hours preserve spring and autumn DST. All models share guarded daily and weekly price features and calendar inputs. The relational inputs add 168 completed price hours and 48 completed hours of weather, archived load forecasts, historical physical flows, and JAO summaries ending 24 hours before origin. Where recorded, publication must also precede origin. The dated area crosswalk applies at the source hour. The unresolved German-only weather proxy is omitted. Source availability and missing values remain separate.

The flow relation uses directed physical observations for mapped benchmark pairs. Each row joins the same source and destination for both relational models. The local control receives the same incident rows with only its own zone state on its endpoint copy. The full model can pass the other zone state through the border. SQL computes each field's count, mean, minimum, maximum and latest available value by time bucket, with border summaries and their neighbor-price aggregates split by direction.

Adaptive margins use the prior CORE domain at exactly 48 elapsed hours before each delivery hour. Values are historical summaries of positive `PTDF_source - PTDF_destination` above 0.005. Missing domains and unavailable pairs contribute zero to the learned margin term. A published pair with zero qualifying constraints remains an observed zero count. The shuffled control jointly permutes the active CORE zone labels at source and destination within each hour. Archive vintages remain conditional, and candidate link weights represent fitted predictive relationships.

## Evaluation

Both studies select epoch or tree counts with January–March 2024 calibration, then refit through March. The relational family starts its fit in 2016, and the adaptive family starts with CORE delivery on June 9, 2022. Final validation covers April 2024–March 2025 and the fixed 49-market roster. Compare variants within each family using the paired calendar-week intervals. The report includes all markets, the CORE-covered subset, France, weekly persistence skill, and spread MAE on fit-defined physical borders. The test period is outside this suite.
