# Saved results

Files here are tracked summaries. Large models, keyed predictions, logs and
checkpoints are kept under the ignored `runs/` directory.

## Latest full evaluation

`controlled_study_smoke*` files are preflight evidence for the expanded study;
its five-seed full summaries will be written as `controlled_study_full*` after
all 100 jobs complete. The established full evaluation below is preserved.

| File | Contents |
|---|---|
| [vibing_suite_full.json](vibing_suite_full.json) | All five methods, feature lists, per-zone scores, curves and aggregates |
| [vibing_suite_full_by_zone.csv](vibing_suite_full_by_zone.csv) | Zone metrics, target counts and paired persistence comparisons |
| [vibing_suite_full_aggregate.csv](vibing_suite_full_aggregate.csv) | Country/market and overall metrics, including JAO-publication subsets |
| [vibing_suite_full_verification.json](vibing_suite_full_verification.json) | Summary identity and completed artifact/key/mask checks |

Read [the results report](../docs/vibing_suite_results.md) first. It explains the
score populations, conditional availability, seed limitations and runtimes.
Inspect the original run with:

```bash
.venv/bin/python -m epgnn.experiment_suite status --saved
```

`vibing_suite_france*` and `vibing_suite_gpu_smoke.json` are preflight evidence,
not additional full evaluations. The France neural checks use short training
windows and should not be compared as fully trained models.

## Earlier experiments

| Prefix | Guide |
|---|---|
| `c1_xgb_*weekly_conditional*` | [Weekly-price baselines](../docs/boosting_countries.md) |
| `c1_xgb_*asinh_l1*` | [Target transformation](../docs/boosting_target_transform.md) |
| `c1_xgb_*jao48*` | [Constraint summaries](../docs/boosting_constraints.md) |
| `fr_temperature_*`, `c1_xgb_fr_*temp*` | [Temperature diagnostics](../docs/temperature_experiments.md) |
| `partner_*` | [Partner reproduction](../docs/partner_pipeline_reproduction.md) |
| `pipeline_diagnostics_*`, `milestone_pipeline_audit.json` | [Pipeline comparison](../docs/pipeline_comparison.md) |

Existing filenames are preserved because scripts/configs and provenance refer
to them. Compare runs using matched target populations, not merely similar names.

## Airport observation study

`airport_temperature_smoke*` files are short France correctness checks.
`airport_temperature_full*` files are published only after the 45-job full
comparison and verification. The shared `airport_temperature_coverage.csv`,
`source_findings.csv`, `geography.csv`, feature registry and source manifest
make spatial proxies and missingness auditable. See the
[experiment guide](../docs/airport_temperature.md).

All 45 full airport fits completed with exit code 0 and verified predictions.
The [reviewed report](../docs/airport_temperature_results.md) explains the
geographic subgroup and market-contribution tables as post-hoc diagnostics.

## Partner-inspired residual GRU

`partner_gru_full.json` and `_by_zone.csv` contain the verified single-seed
49-market experiment. `_verification.json` records artifact, target and metric
checks; `_comparison.csv` and `_review_sources.json` accompany the
[comparison report](../docs/partner_gru_comparison.md). ERA5 inputs remain
conditional retrospective reanalysis. `partner_gru_backend_audit.json` documents
why the first fused-RNN run was rejected; published full scores use native GPU
operations checked against CPU float64. Smoke files remain pipeline checks.
