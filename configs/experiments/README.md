# Experiment configurations

## Current suite

[`vibing_suite.json`](vibing_suite.json) configures the active five-model suite:
recent-price, richer and neighbour XGBoost, plus real/shuffled hourly GAT.

```bash
.venv/bin/python -m epgnn.experiment_suite gpu-smoke
.venv/bin/python -m epgnn.experiment_suite france
.venv/bin/python -m epgnn.experiment_suite full
```

See [the suite guide](../../docs/vibing_suite.md) for tmux, checkpoints and results.
`configs/splits.json` defines the shared original date boundaries; suite arrays
use the audited dataset's corresponding split labels. Source identity is pinned
by `dataset_directory` and `dataset_fingerprint` in the suite config.

## Earlier audited JSON experiments

| Pattern | Entry point / purpose |
|---|---|
| `c1_xgb_COUNTRY_*.json` | `python -m epgnn.boosting --config PATH`; separate-country baselines |
| `c1_xgb_countries_*.json` | `python -m epgnn.boosting_batch --config PATH`; country batches |
| `diagnostic_fr_audited_weekly_partner_params.json` | Audited France hyperparameter control |
| `pipeline_diagnostics.json` | `scripts/experiments/pipeline_diagnostics.py --config PATH`; fixed France feature sequence |

The JSONs retain their original names and data pins for reproduction. Their
saved results are indexed in [results/README.md](../../results/README.md).

## Original YAML scaffold

`c1_persistence.yaml`, `c1_xgb_zone.yaml`, `c2_*`, `c3_*` and `c4*` target the
original `python -m epgnn.run PATH` workflow. Several model implementations there
remain stubs. These YAMLs do not run the active suite; see
[legacy pipeline notes](../../docs/legacy_pipeline.md) before using them.
