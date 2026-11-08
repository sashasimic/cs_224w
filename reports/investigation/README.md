# Guided forecasting investigation

Open [the self-contained HTML report](forecasting_results.html) for the readable
version, or [the executed notebook](../../notebooks/forecasting_results_investigation.ipynb)
to edit the selectors and inspect the Python. The HTML embeds its figures and
hides code inputs; the notebook keeps the code and saved outputs.

The seven chapters cover data/timing, model inventory and pipeline development,
matched baselines, graph/feature/history controls, airport observations, country
investigations and conclusions. Appendices preserve negative results, earlier
single-seed evidence and provenance. No archived experiment is deleted.

Change `ZONES`, `COUNTRIES`, `START`, `END`, `SEED` and `MODELS` near the top of
the notebook. Local prediction windows are validation-only, at most 31 days.
The default illustrated week and seed were not selected for large errors or good
performance. `LOAD_LOCAL_DETAILS=False` uses tracked summaries only; enable it
for local prediction, price-variability and airport-feature panels. Missing local
files produce explicit notices. Distribution inspection accepts at most five
zones and excludes test targets. White cells in the coverage heatmap mean no
valid price targets; October 2026 is unassigned, with airport data ending earlier.

From the repository root, using the project environment with inspection extras:

```bash
.venv/bin/python scripts/reports/build_investigation.py
.venv/bin/python scripts/reports/build_investigation.py --with-details
```

The first command creates a portable summary report; the second renders the
local investigations included in the committed report. Either command rebuilds
the executed notebook and HTML. There is no training, download or GPU use.
Notebook outputs explicitly verify that Torch and XGBoost were not imported.
The builder uses the same virtual environment for its kernel.

To verify execution with the notebooks directory as the kernel working directory
without replacing the published report:

```bash
.venv/bin/python scripts/reports/build_investigation.py --cwd notebooks --output tmp/investigation_check.ipynb
.venv/bin/python -m unittest discover -s tests -p test_reporting.py -v
```

`forecasting_results.json` records artifact/code hashes, study fingerprints and
execution settings. Helpers reconcile all study-level MAE/RMSE/paired MAE against
seed/zone summaries. The headline models have identical local full-target
signatures. Headlines use matched persistence hours; all-target MAE/RMSE and
coverage remain separately labelled. Seed SD is not predictive uncertainty.
Richer-input/topology results and airport receipt provenance remain conditional;
post-hoc diagnostics and the previously explored test period are labelled.
