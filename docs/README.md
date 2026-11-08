# Documentation

**Start with the [complete experiment record](experiments.md)**
([browser version](../reports/experiments/experiments.html)). It contains every
completed programme, all 20 controlled variants, all nine airport variants,
earlier diagnostics and the current status of both GRU families.

## Current reading

| Document | Use |
|---|---|
| [Experiment record](experiments.md) | Main results, comparison populations, limitations and next work |
| [Milestone v3](../reports/milestone_v3/README.md) | Current two-page report: persistence, two XGBoosts, static/residual-GRU MLP, self-loop/real GAT |
| [Leakage audit](leakage_audit.md) | Source publication/vintage problems and verified timing/preprocessing checks |
| [Metrics and daily deployment](metrics_and_live_forecasting.md) | Relative errors and a prospective forecasting service |
| [Setup](setup.md) | Virtual environment, ROCm and source locations |
| [Code walkthrough](code_structure.md) | Package responsibilities and reproducibility |
| [Dataset construction](forecast_dataset.md) | Immutable build contract, provenance and audit/build/validate commands |
| [Investigation notebook](../reports/investigation/README.md) | Executed visual report and editable country/week diagnostics |

**GRU correction, November 9:** original controlled GRUs fail the fused-backend
CPU check on shuffled training-shaped inputs. Saved scores remain archived;
the old claim that temporal learning hurt performance is withdrawn. The corrected
partner-inspired GRU passes native-GPU/CPU checks. The
[experiment record](experiments.md#our-grus-versus-the-partner-inspired-gru)
explains the distinction and links both machine-readable audits.

## Detailed experiment protocols and saved reports

Use these for exact features, launch commands, configurations and provenance.
The consolidated record takes precedence over superseded interpretations.

| Programme | Protocol | Results |
|---|---|---|
| First canonical suite | [Five-model guide](vibing_suite.md) | [Single-seed scores](vibing_suite_results.md) |
| Five-seed controls | [Study design](controlled_study.md) | [Comparison](controlled_study_comparison.md), [original score export](controlled_study_results.md) |
| Airport observations | [Method](airport_temperature.md), [smoke gates](airport_temperature_smoke.md) | [Reviewed results](airport_temperature_results.md) |
| Residual GRU adaptation | [Method and backend correction](partner_gru.md) | [Comparison](partner_gru_comparison.md), [full learning curve](partner_gru_results.md) |

## Earlier investigations

| Record | Question |
|---|---|
| [Belgium baseline](boosting_baseline.md) | Strict calendar inputs versus weekly history |
| [Ten-country baseline](boosting_countries.md) | Independent weekly-price XGBoost by country |
| [Target transformation](boosting_target_transform.md) | asinh(price/50) and L1 loss |
| [Constraint summaries](boosting_constraints.md) | Flat lag-48 CWE/CORE features |
| [France temperature](temperature_experiments.md) | Current proxy, perfect-weather oracle and price-timing erratum |
| [Partner XGBoost reproduction](partner_pipeline_reproduction.md) | Original pipeline and country reaggregation |
| [France pipeline comparison](pipeline_comparison.md) | Recent-price effect and matched feature sequence |
| [Legacy pipeline](legacy_pipeline.md) | Original code, stubs and modelling blockers |

These files retain historical branch names and commands as reproduction records.
The immutable dataset guide also retains its original conservative price-lag
rule; the later price-publication correction is explained in the leakage and
pipeline notes. Original model metrics and source hashes are not rewritten.

## Milestone snapshots and artifacts

| Snapshot | Status |
|---|---|
| [Milestone v3](../reports/milestone_v3/README.md) | Current seven-method comparison; verified partner-inspired GRU, affected original GRUs excluded |
| [Milestone v1](../reports/milestone_v1/README.md) | Frozen first-suite/France report |
| [Milestone draft](../reports/milestone/README.md) | Historical report/build workflow |
| [Milestone v2](../reports/milestone_v2/README.md) | Five-seed report; its negative GRU interpretation is superseded by the backend audit |
| [Raw results index](../results/README.md) | Metric files, verification records and provenance |

PDF snapshots are retained unchanged. For current conclusions, use the
[consolidated record](experiments.md). Original inputs, training code and large
local run artifacts were not changed by documentation cleanup.
