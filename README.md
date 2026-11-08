# Weather, Interconnections, and Energy Prices

Stanford CS224W project on forecasting European day-ahead electricity prices.
At **12:00 Europe/Brussels on D−1**, predict every hourly price for delivery day D,
including 23- and 25-hour daylight-saving days.

## Latest controlled study

**Availability audit:** the [leakage analysis](docs/leakage_audit.md) identifies
an ERA5 publication-delay mismatch and unresolved load/JAO/price vintages. The
scores below are retrospective conditional benchmarks, not certified operational
noon forecasts; no direct target/scaler leakage was found in the reviewed paths.

All 100 five-seed jobs completed. On matched test hours, richer real GAT averages
20.97 MAE versus pooled MLP 21.19 and pooled XGBoost 21.06. Gains are modest and
vary by market; the GRU variants performed worse. See the
[full comparison](docs/controlled_study_comparison.md) and
[milestone v2](reports/milestone_v2/milestone.pdf).

## First full suite (single seed)

The five-model suite has completed evaluation on **49 market identities**.
Test MAE below uses the same 633,885 zone-hours with valid targets and weekly
price history. Lower is better; units are EUR/MWh.

| Model | Test MAE |
|---|---:|
| Weekly persistence | 29.13 |
| Recent-price XGBoost | 21.87 |
| Richer XGBoost | 22.13 |
| Neighbour XGBoost | 21.87 |
| Real-graph GAT | **20.83** |
| Shuffled-graph GAT | 21.54 |

The real graph improves MAE by 3.32% over the shuffled control in this seed.
These are **conditional, exploratory results**: historical load/weather/JAO
publication vintages are unverified, and test results have been inspected before.
The weather source is reanalysis, not archived weather forecasts.

Read the [verified results](docs/vibing_suite_results.md) for validation scores,
RMSE, runtimes and limitations. The [experiment guide](docs/vibing_suite.md)
describes features, training and checkpointing.

The [expanded controlled study](docs/controlled_study.md) adds a parameter-matched
pooled MLP, pooled XGBoost, self-loop and multiple shuffled graphs, feature
ablations, and GRU temporal controls across five seeds (100 jobs). Its full run
is launched separately; smoke evidence does not replace the results above.

## Inspect or run experiments

Run commands from the repository root using its virtual environment. This
machine uses the RX 9070 XT with ROCm; see [setup](docs/setup.md) for installation.

```bash
# Inspect the completed run referenced by the saved full result.
.venv/bin/python -m epgnn.experiment_suite status --saved

# Before a fresh run: verify the GPU, then all five methods on France.
.venv/bin/python -m epgnn.experiment_suite gpu-smoke
.venv/bin/python -m epgnn.experiment_suite france

# Full evaluation in a persistent host-terminal session.
tmux new-session -d -s vibing-full './scripts/experiments/run_vibing_suite.sh'
tail -f runs/vibing_suite_launch/full.log
```

The suite runs three per-zone XGBoost variants and two pooled spatial GATs.
The GATs have two layers and real/shuffled interconnector topology; they do not
use a GRU or detailed constraint nodes. Configuration is in
[configs/experiments/vibing_suite.json](configs/experiments/vibing_suite.json).

Runs are tied to code, configuration and source fingerprints. **Code cleanup
creates a new fingerprint.** The completed evaluation remains attached to its
original implementation at commit `a48d599`, with summaries saved in `f50dc6e`.
`status --saved` reads that historical run explicitly; fresh training requires
preflights for the current code. Existing results are never relabelled as new runs.

## Data and evaluation

The audited master has 5,887,954 node rows over 62 price-area identities,
January 2016–October 2026. It retains missing values and negative prices,
keeps DE_AT_LU separate from DE_LU, and records structural absences and unresolved
geography. Of these identities, 49 have valid fit/validation/test targets; the
other 13 are reported separately. October 2026 stays unassigned.

| Partition | Brussels delivery dates | Role |
|---|---|---|
| Pretrain | 2016-01-01–2022-06-08 | Fit |
| Train | 2022-06-09–2024-03-31 | Fit |
| Validation | 2024-04-01–2025-03-31 | Early stopping |
| Test | 2025-04-01–2026-09-30 | Exploratory evaluation |

Price lags use explicit UTC/zone joins. Weekly persistence copies the same zone's
price 168 elapsed hours earlier. Model scores retain all valid targets, with
persistence comparisons using common finite-history hours. Targets are never
filled; neural preprocessing fits training data only.

See [dataset construction](docs/forecast_dataset.md), the
[price-publication clarification](docs/temperature_experiments.md), and
[inspection notebook](notebooks/forecast_dataset_inspection.ipynb).

## Repository map

| Location | Purpose |
|---|---|
| `epgnn/dataset_build/` | Audit, construct and validate the canonical dataset |
| `epgnn/experiment_suite/` | Active five-model suite: data, neural models, GPU checks, scoring and orchestration |
| `epgnn/boosting.py`, `boosting_batch.py` | Earlier audited boosting experiments |
| `epgnn/data/`, `models/`, `run.py` | Original pipeline and scaffold; see [legacy notes](docs/legacy_pipeline.md) |
| `configs/` | Source pins, split dates and experiment configurations |
| `scripts/experiments/` | Host launcher, partner reproduction, diagnostics and report generation |
| `results/` | Tracked metric summaries; see [results index](results/README.md) |
| `runs/` | Local predictions, models, checkpoints and logs; ignored by Git |
| `staging/` | Local audited generations and tensor caches; ignored by Git |
| `datasets/` | Original downloaded tables and tracked schema documentation |
| `docs/`, `reports/`, `notebooks/` | Guides, milestone PDF/source and inspection workflow |

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests/dataset_build -p 'test_*.py'
```

[Documentation index](docs/README.md) · [Code walkthrough](docs/code_structure.md) ·
[Experiment configuration index](configs/experiments/README.md) ·
[Updated milestone draft](reports/milestone/milestone.pdf)

## Airport temperature observations

The [airport experiment](docs/airport_temperature.md) replaces reanalysis inputs
with actual reports from each country's busiest airport. It compares price-only,
latest temperature and temperature-history inputs for pooled XGBoost, MLP and
GAT across five seeds. Publication/revision provenance remains conditional;
spatial proxies and missingness are explicit. Previous results are preserved.

All 45 airport fits completed and passed verification. See the
[reviewed results](docs/airport_temperature_results.md): gains are small and vary
by market; most of pooled XGBoost's headline gain comes from an unmapped market.

## Guided report notebook

Read the [illustrated project investigation](reports/investigation/forecasting_results.html)
or open the [executed notebook](notebooks/forecasting_results_investigation.ipynb).
It covers data, model comparisons, graphs, weather and editable validation-week
diagnostics. [Usage and reproduction](reports/investigation/README.md).
