# Saved results

Files here are tracked summaries. Large models, keyed predictions, logs and
checkpoints are kept under the ignored `runs/` directory.

## Latest full evaluation

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
