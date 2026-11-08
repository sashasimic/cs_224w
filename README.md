# Weather, Interconnections, and Energy Prices

Stanford CS224W project on European day-ahead electricity prices. At
**12:00 Europe/Brussels on D−1**, predict every hourly price of delivery day D.

Start with the **[complete experiment record](docs/experiments.md)** or its
[browser version](reports/experiments/experiments.html). It brings all completed
programmes, scores, negative results and numerical/availability caveats together.
The [documentation index](docs/README.md) separates current guidance from
historical protocols and frozen milestones.

The latest concise report is [milestone v3 (two-page PDF)](reports/milestone_v3/milestone.pdf),
with [source, exact values and rebuild instructions](reports/milestone_v3/README.md).

## Current findings

Strong recent-price inputs and pooled training close much of the initial gap
between trees and graph models. In the five-seed controlled study, paired test
MAE is 21.06 EUR/MWh for pooled richer XGBoost, 21.19 for pooled richer MLP and
20.97 for richer real GAT, versus 29.13 for weekly persistence. Graph gains are
modest and vary by market. Airport-temperature gains are small and inconsistent.

**Temporal-model correction:** our original ten GRU checkpoints reproduce their
saved inference scores, but fail CPU agreement on shuffled training-shaped
batches with the fused ROCm backend. Their 24.50/23.59 GRU+MLP/GAT scores are
affected historical results, not evidence that temporal learning hurts.
The corrected partner-inspired residual GRU uses native GPU operations and
passes CPU checks: one-seed test MAE 21.07, RMSE 47.02. See the
[GRU comparison and audit](docs/experiments.md#our-grus-versus-the-partner-inspired-gru).

These are exploratory retrospective results. ERA5 is reanalysis, not archived
operational weather forecasts; historical load/JAO/price vintages and airport
receipt times remain unresolved. Test results have been inspected repeatedly.
The [leakage audit](docs/leakage_audit.md) explains the practical limitations.

## Data and evaluation

The audited grid contains 5,887,954 node rows over 62 historical market identities
from January 2016 through October 2026. Forty-nine have labels in all required
partitions. Missing values, negative prices and DE_AT_LU/DE_LU identities are
preserved. Missing weather never removes a target.

| Partition, Brussels delivery date | Dates | Use |
|---|---|---|
| Pretrain + train | January 2016–March 2024 | Fit models/preprocessing |
| Validation | April 2024–March 2025 | Early stopping |
| Test | April 2025–September 2026 | Exploratory evaluation |
| Unassigned | October 2026 | Coverage checks only |

All-target scores use 428,427 validation and 634,578 test labels. Comparisons with
weekly persistence use 428,213 and 633,885 common finite-history labels.
Persistence copies the same market's price 168 elapsed hours earlier. MAE/RMSE
are EUR/MWh; relative MAE compares model error with persistence on common hours.

## Run or inspect

Use the project virtual environment. ROCm setup and GPU access are described in
[setup](docs/setup.md). No bare system-Python interpreter is required.

```bash
# Rebuild the consolidated document from saved results; no fitting/GPU use.
.venv/bin/python scripts/reports/build_experiment_summary.py

# Inspect the completed first suite without creating a new generation.
.venv/bin/python -m epgnn.experiment_suite status --saved

# Check contracts.
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests/dataset_build -p 'test_*.py'
```

For training commands and GPU gates, choose the appropriate
[experiment configuration](configs/experiments/README.md). Changed source/config
hashes create a new generation; historical models retain their original identity.
The [investigation notebook](reports/investigation/README.md) provides editable
country/week diagnostics, with affected GRU results explicitly labelled.

## Repository map

| Location | Purpose |
|---|---|
| `epgnn/dataset_build/` | Canonical source, geography and timing contract |
| `epgnn/experiment_suite/` | First five-model suite and reusable data/scoring |
| `epgnn/controlled_study/` | Five-seed static and original temporal controls |
| `epgnn/airport_study/` | Airport-observation comparison |
| `epgnn/partner_gru/` | Corrected residual GRU adaptation |
| `epgnn/boosting.py`, `boosting_batch.py` | Earlier audited country baselines |
| `epgnn/reporting/`, `scripts/reports/` | Read-only report/notebook construction |
| `configs/`, `scripts/experiments/` | Pinned recipes, launchers and audits |
| `results/` | Tracked metrics/provenance; [index](results/README.md) |
| `runs/`, `staging/` | Local checkpoints, predictions and data caches; ignored |
| `docs/`, `reports/`, `notebooks/` | Current report, detailed protocols and snapshots |

Original `epgnn/data/`, `models/` and `run.py` form a separate legacy pipeline;
some YAML/model paths remain scaffolds. See [legacy notes](docs/legacy_pipeline.md).
Original data and historical training implementations remain unchanged.
