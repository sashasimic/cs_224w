# Documentation

Start with the root [README](../README.md), then choose the relevant workflow.

## Current workflow

Read the [forecast leakage audit](leakage_audit.md) alongside the model results:
effective-time lags passed, but weather product latency and historical vintages
prevent an operational availability certification.

| Guide | Contents |
|---|---|
| [Setup](setup.md) | Virtual environment, ROCm, source locations and checks |
| [Five-model suite](vibing_suite.md) | Features, splits, GPU gates, tmux, checkpoint resume and outputs |
| [Expanded controlled study](controlled_study.md) | MLP/pooled controls, five seeds, three rewired graphs, feature and GRU experiments; separate 100-job launch |
| [Controlled-study comparison](controlled_study_comparison.md) | Completed 100-job evidence, earlier experiments, market/regime breakdowns and limitations |
| [Milestone v2](../reports/milestone_v2/README.md) | Revised two-page interpretation, with frozen v1 preserved separately |
| [Full results](vibing_suite_results.md) | Verified 49-market comparison, runtime and limitations |
| [Two-page milestone](../reports/milestone/README.md) | Full-suite results combined with the earlier France diagnostics; PDF, source and build command |
| [Code walkthrough](code_structure.md) | Active entry points and responsibilities |
| [Dataset construction](forecast_dataset.md) | Source pins, geography, missingness and audit/build/validate commands |
| [Original pipeline](legacy_pipeline.md) | Legacy entry points, stubs and known limitations |

## Earlier experiments and provenance

These guides describe earlier runs, not additional results from the five-model suite.

| Guide | Question |
|---|---|
| [First baseline](boosting_baseline.md) | What do strict calendar inputs and conditional weekly prices achieve? |
| [Ten-country baseline](boosting_countries.md) | How does weekly-price XGBoost compare across countries? |
| [Target transform](boosting_target_transform.md) | What changes with asinh targets and L1 loss? |
| [Constraint summaries](boosting_constraints.md) | Do lagged JAO summaries improve the weekly baseline? |
| [Temperature](temperature_experiments.md) | What do current/perfect temperature diagnostics show? Includes the price-publication correction. |
| [Partner reproduction](partner_pipeline_reproduction.md) | Can the partner result be reproduced and reaggregated? |
| [Pipeline comparison](pipeline_comparison.md) | Which pipeline differences explain the France gap? |

[Milestone PDF and source](../reports/milestone/README.md) ·
[Saved metrics](../results/README.md) ·
[Inspection notebook](../notebooks/forecast_dataset_inspection.ipynb)

The dataset construction guide is part of an immutable build fingerprint. Its
original conservative price-lag interpretation is corrected in the linked
temperature and pipeline-comparison notes; publication time differs from delivery
time. Keeping the original build documentation intact preserves its audit trail.

The [airport temperature experiment](airport_temperature.md) defines the new
45-fit observation comparison and its source, timing, ROCm and smoke gates.
The [completed airport results](airport_temperature_results.md) include the
five-seed comparison, country scores and a review of indirect pooled-model gains.
