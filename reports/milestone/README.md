# CS224W milestone draft

This is a historical report/build workflow. Use the
[current experiment record](../../docs/experiments.md) for the full programme
and the November 9 correction to the old GRU conclusions. The existing PDF is
retained as a snapshot.

[Read the two-page PDF](milestone.pdf), edit [the LaTeX source](milestone.tex), or
see the [full results](../../docs/vibing_suite_results.md) and
[pipeline comparison appendix](../../docs/pipeline_comparison.md).

The draft follows the Approach / Experiments / Future Work structure of the
user's blank CS224N template. `neurips_2019.sty` is copied from that directory
with its course footer changed to CS224W. Author details remain a placeholder;
CS224W-specific submission requirements have not been provided.

Rebuild from the repository root:

```bash
.venv/bin/python scripts/experiments/build_milestone_report.py
```

This reads saved JSON results, verifies the full-summary checksum and cohort
counts, generates the 49-market `suite_results.tex` and `suite_numbers.tex`,
and records input hashes/fingerprints in [sources.json](sources.json). It also
retains the earlier ten-zone table, France numbers and standalone diagnostic
figure. The PDF compiles twice and must contain exactly two pages. The script
does not train models. Dependencies: the Python model/
inspection environment, `pdflatex` and `pdfinfo`. Preview figures are available
as [PDF](figures/france_ablation.pdf) and [PNG](figures/france_ablation.png).

The report combines the completed five-model evaluation across 49 market
identities with the earlier France diagnostics. Real-graph GAT beats its shuffled
control by 3.32% and recent-price XGBoost by 4.75% on common test hours in one seed.
The paired MAE table and all-target RMSE use explicitly different populations.
Historical publication vintages, multiple-seed confirmation, temporal encoders
and detailed constraint nodes remain open. Author details are still placeholders.

The expanded controlled study replaces this report's primary comparison only
after all 100 jobs complete. Its publisher writes `controlled_study.tex` and a
matching source manifest; the builder validates the completed summary before
including the multi-seed MLP, graph and temporal comparison. Until then the PDF
continues to show the verified five-model evaluation above.
