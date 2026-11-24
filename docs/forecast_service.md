# Local day-ahead forecast service

The service uses the 49-zone roster from the benchmark panel and the matching
ENTSO-E EICs from the dataset crosswalk. The roster retains archive identities
such as the Italian virtual zones, Ukraine IPS, Kosovo, Albania and NO2NSL.
An EIC with an empty live response remains on the roster and appears in coverage
and preflight diagnostics.

Weekly persistence uses the price exactly 168 elapsed hours earlier. The core
lineup also has pooled and zone-specific XGBoost, per-zone and per-hour LEAR with
four daily curves and calendar features, a cross-market PCA factor model with
separate forecasts for shared and local movements, a pooled GRU that predicts an
entire delivery day, and a nonnegative blend fitted on walk-forward predictions.
The ROCm host can run the two frozen TimesFM modes as separate private research
comparisons. The private modes use the same 336-hour context and the same
checkpoint. [TimesFM non-commercial, non-production license](https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/LICENSE)

## Prepare on the ROCm host

Run from the repository root with the project model dependencies and the pinned
TimesFM checkout, local checkpoint, and Python packages in the sibling
`cs_224w_timesfm` worktree. The commands below use the original benchmark price
panel. Supply a different path with `--archive` for an equivalent pinned copy.

```bash
archive=staging/vibing_suite/eca18fe9668fb919a5be8d7b858578b57f20cab12919998523d735b6ec9b59a4
.venv/bin/python -m epgnn.forecast_service train --archive "$archive"
.venv/bin/python -m epgnn.forecast_service rocm --archive "$archive"
.venv/bin/python -m epgnn.forecast_service rehearse
.venv/bin/python -m epgnn.forecast_service private-backtest --archive "$archive"
.venv/bin/python -m unittest discover -s tests/forecast_service -v
```

Training runs three processes with four threads per process. Per-zone trees are
sharded within that pool. Three chronological 14-day folds fit the models on the
preceding 730 delivery days, omitting the day before each validation boundary.
The online artifact fits use the hyperparameters selected across the three folds.
Each fold after the first uses only earlier validation folds for its parameter
and blend selection. The final blend uses the walk-forward predictions. Archive
receipt and revision vintages are unverified, so the backtests are exploratory.
The live snapshots admit only prices actually received before cutoff and
deterministic calendar features.

The ROCm load test exercises two processes on a 25-hour horizon with context in
all 49 zones. It uses the device's total VRAM counter at 50 ms intervals. Two
workers qualify when the peak is under 12 GiB and both finish inside the forecast
budget. If needed, the same test retries serially. On a host with multiple GPUs,
pass the relevant sysfs counter with `--vram-file`.

The fixture rehearsal uses the real workers and an isolated SQLite database
with synthetic receipts and clocks. It checks the 23- and 25-hour delivery
days, time-bounded fallback, a late commit, missing input, three concurrent CPU
workers, and restart against the original snapshot. It records evidence for the
live pilot gate.

## Arm the live pilot

Install the user units with `bash scripts/forecast_service/install-user-timers.sh`.
With user lingering enabled, the timers continue to run after logout while the
host is awake. The timers stay idle until a pilot is armed.

Place a Transparency Platform REST API token in the local file
`~/.config/epgnn-forecast/entsoe.token` readable only by the service user.
The CLI also accepts `ENTSOE_TOKEN_FILE` or `ENTSOE_API_TOKEN`. The service reads
the file shown above. The ENTSO-E account needs REST API access.
[ENTSO-E token instructions](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token)

```bash
export ENTSOE_TOKEN_FILE="$HOME/.config/epgnn-forecast/entsoe.token"
.venv/bin/python -m epgnn.forecast_service arm --start YYYY-MM-DD
.venv/bin/python -m epgnn.forecast_service status
systemctl --user list-timers 'epgnn-*'
```

Use the first delivery date for `--start` and allow time for the 49-zone
preflight before 11:52 Brussels on the prior day. The preflight records the
response and complete-hour count for every EIC. An explicit no-data response is
recorded as empty. Authentication, identity mismatches or invalid responses
prevent arming. The two-day live pilot must record a terminal status for every
scheduled forecast, pass snapshot and commit-time checks, and obtain outcomes on
both days. At least weekly and pooled XGBoost must produce timely primary
forecasts on both days. After the pilot passes, the outcome timer opens the
earliest schedulable 14 consecutive delivery days and copies the immutable
roster, artifact hashes, runtime and scoring rules into the experiment record.

## Timing and scoring

- Acquisition starts at 11:52 Brussels and requests one shared A44 price panel
  per zone. Only failed network or exchange-rate requests are retried.
  The snapshot freezes at 11:55, filtering on when each response was received.
  It includes past delivery prices and current-day target availability flags.
  The feed uses primary or unnumbered auction series. For Austria and
  Germany–Luxembourg, this keeps the noon coupled auction distinct from
  EXAA’s earlier auction. Target prices stay outside the feature matrix.
  All predictions are indexed by UTC delivery hour, including both repeated
  autumn hours. [EXAA auction schedule](https://www.exaa.at/en/energytrading/handel-mit-exaa/)
- Model jobs commit before noon. Each commit binds its model artifact hash to the
  snapshot hash, and a retry reuses the same delivery-day/model key. Weekly
  persistence fills unavailable predictions when it has a price. Missing,
  degraded, interrupted and late work remains in the operational records.
- The outcome timer retrieves prices separately after noon and retries incomplete
  zones through delivery plus three days. The first complete hourly EUR/MWh
  observation is retained. Quarter-hour observations need all four intervals
  for the hour. The hourly target is their arithmetic mean.
  [ENTSO-E energy price feed](https://transparencyplatform.zendesk.com/hc/en-us/articles/16647234190100-Energy-Prices-12-1-D)
- ENTSO-E publishes the Ukraine IPS price in hryvnia. The service converts
  it to EUR/MWh using the National Bank of Ukraine official rate for the
  delivery date in Kyiv and retains the exchange-rate response with the price
  receipt. Hours with missing rates remain unpriced until the rate is available.
  [National Bank of Ukraine exchange-rate API](https://bank.gov.ua/admin_uploads/article/Instr_API_KURS_VAL_data_Full_eng.pdf)
- The report treats a complete price received by cutoff as already published.
  It counts uncovered hours as unknown at cutoff when the zone follows the
  documented Single Day-Ahead Coupling schedule and a valid source response
  contains no target prices for those hours. Missing captures, partial prices,
  and markets without verified schedule evidence are reported with unverified
  publication timing.
  [SDAC geographic scope](https://www.entsoe.eu/network_codes/cacm/implementation/sdac/)
  · [SDAC publication timing](https://www.entsoe.eu/documents/nc/NC%20CACM//SDAC%202021/210526_Coms_note_new_operational_timings_updated.pdf)

The default state directory is `runs/forecast_service` with the transactional
ledger, raw-receipt hashes, frozen snapshots, attempts and model artifacts.
`backtest.json` contains exploratory core results, and `backtest-private.json`
contains the private comparisons. `gpu.json`, `rehearsal.json` and
`preflight.json` contain the corresponding checks. Once the schedule is active,
the outcome timer writes the pilot and experiment reports in CSV and JSON.

```bash
.venv/bin/python -m epgnn.forecast_service report --phase experiment
.venv/bin/python -m epgnn.forecast_service report --phase experiment --private
```

MAE is in EUR/MWh. Matched skill is `1 − model MAE / weekly MAE`, calculated on
exactly the same observed rows with timely predictions from both models. The
report includes scores weighted by market-hour, by country or joint market,
by bidding zone, and with equal weights across countries or joint markets.
Coverage, fallback counts, latency and attempt statuses accompany the scores.
