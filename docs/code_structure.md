# Code walkthrough

## Current packages

The [experiment record](experiments.md) is the result overview. Training code is
preserved under its recorded fingerprints; documentation/report maintenance
does not relabel old checkpoints as new implementations.

| Package | Responsibility |
|---|---|
| `epgnn/dataset_build/` | Audited source joins, historical geography and time contract |
| `epgnn/experiment_suite/` | First five-model programme and shared data/scoring |
| `epgnn/controlled_study/` | Five-seed static, graph and original temporal controls |
| `epgnn/airport_study/` | Airport-observation variants |
| `epgnn/partner_gru/` | Residual GRU with native GPU RNN operations and CPU checks |
| `epgnn/reporting/` | Read-only result validation and notebook investigations |
| `scripts/reports/` | Consolidated document and executed notebook builders |
| `scripts/experiments/check_*gru_backend.py` | Read-only same-weight CPU/fused/native GPU comparisons |

The original controlled GRU training path is affected by the fused backend on
shuffled inputs. Its implementation is retained for reproduction; it should not
be used for new scientific fits before a separately fingerprinted correction.

## First five-model suite

`python -m epgnn.experiment_suite` enters `__main__.py`. Commands are `gpu-smoke`,
`france`, `full` and `status`; use `status --saved` for the completed historical run.

| Module | Responsibility |
|---|---|
| `data.py` | Read the audited grid zone by zone; guard price lags, join historical JAO, construct training topology and neighbour inputs; write memory-mapped tensors |
| `neural.py` | Train-only imputation/scaling, two-layer `PriceGAT`, GPU batches, epoch training, inference and resumable checkpoints |
| `gpu.py` | Real/shuffled operator checks and interrupted-versus-continuous training verification |
| `evaluation.py` | Unique keyed prediction tables, common-mask persistence scores, pooled country/overall MAE and RMSE |
| `artifacts.py` | Atomic JSON publication |
| `runner.py` | Fingerprints, checked cache loading, XGBoost fits, launch locks, France gate and sequential experiment orchestration |

Training flows from **audited Parquets → checked tensor cache → model fit → keyed
predictions → scores and checksums**. CLI functions dispatch work; they do not
parse arguments when imported. The shell launcher adds persistent logs and an
exit-code file for use inside tmux.

The runner still exposes `gpu_smoke`, `evaluate`, `grouped_scores` and `write_json`
imports for compatibility with existing callers. Implementations live in their
focused modules above.

## Dataset and earlier baselines

`epgnn/dataset_build/` owns the source/geography/time contract and atomic dataset
construction. `epgnn/boosting.py` provides the earlier strict/conditional boosting
runner and reusable scoring/history helpers. `boosting_batch.py` coordinates
country batches. The suite reuses those contracts rather than the original
history/sample tensors.

Original modules under `epgnn/data/` and `epgnn/models/` belong to the separate
legacy pipeline; [legacy notes](legacy_pipeline.md) explain which are implemented.
The suite's GAT is in `experiment_suite/neural.py`.

## Reproducibility during maintenance

Source hashes deliberately change after formatting or module extraction. Old
models and metrics keep their old fingerprints and remain reproducible from the
recorded Git implementation. Inspection with `status --saved` is read-only and
does not authorize reusing an old model under changed code. Fresh code must pass
its own GPU/France gates before a new full run.

For this cleanup, existing training, feature and scoring function ASTs were
compared before/after extraction and formatting; the status/CLI inspection path
was the intentional behavior addition. Data, trained weights and metric summaries
remain unchanged. Regression tests cover timing, graph shuffling, inactive nodes,
preprocessing, missing-target losses and country aggregation.

The named compatibility check loads both original full-run graph checkpoints and
compares their first test batch with the saved predictions, without retraining:

```bash
.venv/bin/python scripts/experiments/check_saved_graph_models.py
```

To format the active suite and its tests, use Black 26.10.1:

```bash
uvx --from black==26.10.1 black epgnn/experiment_suite tests/test_experiment_suite.py
```

Keep changes to fingerprinted dataset builders and historical experiment scripts
separate from routine readability cleanup. Their hashes are part of prior builds.
