# Airport-observation temperature experiment

This study tests inexpensive actual-temperature proxies after the availability
limitations found in the [leakage audit](leakage_audit.md). It compares pooled
XGBoost, parameter-matched MLP and real-topology GAT with the same feature groups
and five seeds: 45 fits, plus nine short France-supervised smoke fits.

## Fixed design

The reference airport is the largest in each country by **2023 annual passengers**.
The [mapping](../configs/experiments/airport_stations_2023.csv) records the official
ranking query, counts and ICAO identifiers. Eurostat supplies 28 countries;
Montenegro's airport operator supplies Podgorica versus Tivat. Selection uses
neither observation coverage nor electricity-price prediction scores. Ranking
statistics are a fixed retrospective study-design choice, not a historically
selected airport portfolio at each forecast origin.

All regional zones share the national airport proxy. DE_LU uses Frankfurt and
IE_SEM uses Dublin with explicit combined-market flags. DE_AT_LU is preserved
separately without a German-only temperature substitution. Unresolved EIC
identities and countries without a pinned airport keep missing temperature.
All 62 identities remain in the canonical graph; the existing 49-market scored
cohort and target masks are preserved. Six scored EIC identities currently lack
resolved national mappings and therefore have no local airport feature. This
also limits aggregate effect sizes; country and coverage tables are essential.

The three feature groups are:

| Group | Inputs |
|---|---|
| Price-only | Local hour, weekday, month, guarded price_lag24 and price_lag168 |
| Latest temperature | Price-only + observed temperature in Celsius + age in hours |
| Temperature history | Latest + trailing 24-hour mean + number of available hourly samples |

At literal D−1 noon Europe/Brussels, the latest report must be at least two hours
old and at most six hours old. Temperature and age are missing if it is stale or
invalid. History uses (origin−26h, origin−2h], choosing the latest report in each
hourly bin; at least 18 populated bins are needed for a mean. SPECI bursts do
not get extra weight. The count is zero for an observed archive window with no
usable reports and missing for an unmapped or structurally inactive identity.
All features are frozen across every hour of delivery day D, including DST days.

Original receipt times are enforced when present, but the IEM CSV used here does
not supply them. A two-hour buffer is an assumption, not historical publication
proof. Corrections marked COR and conflicting timestamps are quarantined because
original versions cannot be recovered from this export. This is a **conditional
observation experiment**, not a certified operational forecast. Missing reports
are never replaced with ERA5, model-filled weather or another airport.

The fit period is January 2016–March 2024; validation April 2024–March 2025; test
April 2025–September 2026. October remains unassigned. XGBoost uses native NaNs;
neural imputation, missingness indicators and standardization fit training data
only. Existing asinh(price/50), L1 loss, hyperparameters, validation early stopping
and seed/batch schedules are retained. No load, ERA5, JAO, realized-flow or
country-ID predictors are included. The graph remains the existing training-only
physical-pair union, masked by historical node lifetimes. This study does not
add self-loop or shuffled-graph controls and makes no new topology-causality claim.

## Run and resume

Run named modules from the repository root; no inline interpreter commands:

```bash
.venv/bin/python -u -m epgnn.airport_study fetch
.venv/bin/python -u -m epgnn.airport_study audit
.venv/bin/python -m unittest discover -s tests -p 'test_airport_study.py' -v
.venv/bin/python -u -m epgnn.airport_study smoke
tmux new-session -d -s airport-temperature './scripts/experiments/run_airport_temperature.sh'
.venv/bin/python -m epgnn.airport_study status
tail -n 30 runs/airport_temperature_launch/full.log
```

Every command accepts `--config`. Fetch is restartable per station and preserves
raw CSVs compressed with deterministic gzip metadata, exact URLs, retrieval times
and SHA-256 hashes. The endpoint is queried sequentially with more than a second
between requests, respecting its published throttle. There are no uploads.

Smoke first tests MLP/GAT GPU forward/backward, masked loss, actual-feature
checkpoint round trips and epoch-boundary resume. It then trains all nine
prototypes with France supervision while retaining the full graph. Both tree
and neural smoke fits use the same last 90 fit days and first 14 validation days;
full validation/test predictions exist only to exercise scoring, not to select
scientific conclusions. Full execution requires matching verified smoke evidence.
Restart the launcher to resume completed jobs and neural epoch checkpoints;
source/configuration changes produce a new identity and require new smoke gates.

## Outputs and acceptance

Raw observations live under `staging/airport_observations_v1`; normalized station
records, origins, audited geography, feature registry and a memory-mapped feature
cache live under `staging/airport_temperature/<fingerprint>`. The cache copies
only price/calendar features from the immutable parent. Other parent predictors
are NaN and never selected. Large predictions and checkpoints live under
`runs/airport_temperature/<fingerprint>/{smoke,full}`; original studies stay intact.

`results/airport_temperature_full*` contains summary scores, per-zone errors,
regime diagnostics, GPU evidence, verification and report source identities.
`airport_temperature_coverage.csv` includes every canonical identity plus a
`scored_zone` flag, split and month. Weather gaps never remove valid targets.
The primary MAE/RMSE retain all labels; paired MAE additionally requires weekly
persistence history. Country RMSE pools squared errors before taking the root.
Five-seed summaries average scores, not predictions. Validation-week bootstrap
comparisons condition on fitted models and share sampled weeks across countries.
The old test period has already been explored and is not a fresh holdout.

The full runner verifies saved artifact hashes, unique keys, exact canonical
price values/masks and recomputed errors before publishing
`docs/airport_temperature_results.md`. It includes previous conditional results
as context, with the new price-only fits as the matched feature controls.
Recheck or regenerate a completed full run with:

```bash
.venv/bin/python -m epgnn.airport_study verify
.venv/bin/python -m epgnn.airport_study report
```

A null result says this national-airport proxy did not help these models. It
does not rule out regional heatwaves, population-weighted temperature or genuine
archived weather forecasts.
