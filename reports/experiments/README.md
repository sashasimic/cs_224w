# Complete experiment record

[Read in a browser](experiments.html) · [Markdown document](../../docs/experiments.md)

For a concise seven-method report, read [milestone v3](../milestone_v3/README.md).

This is the current all-programme report. It covers the 100-job controlled
study, 45-job airport study, first five-model suite, accepted residual GRU,
earlier country/France diagnostics and rejected or affected runs. Cohorts,
metrics, seeds and source-availability limitations are labelled separately.

Rebuild from the repository root:

```bash
.venv/bin/python scripts/reports/build_experiment_summary.py
```

The builder checks the verified multi-seed result summaries, reconciles their
scores with seed/zone tables, and requires the native/replay GRU audit evidence.
It updates marked result tables in the authored Markdown and exports the HTML.
It fits no models, uses no GPU and imports neither Torch nor XGBoost.
[sources.json](sources.json) records input, code and output hashes.

The original controlled GRUs are affected by a fused-backend discrepancy on
shuffled training-shaped inputs. Their historical numbers are retained, but
the former conclusion that GRU histories hurt performance is withdrawn.
The accepted residual GRU is a different recipe with verified native-GPU
operations. No corrected rerun of the original ten models was launched here.
