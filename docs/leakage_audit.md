# Forecast leakage audit

## Verdict

**The 100-job study is not a verified operational noon forecasting benchmark.**
No direct same-day target injection or cross-split preprocessing leak was found
in the reviewed training paths. However, source-availability problems are material:
the documented ERA5-derived lag-48 weather cannot satisfy ERA5's publication delay,
and load/JAO/price revision vintages are not verified. Repeated test inspection
also prevents treating the results as a fresh confirmatory evaluation.

The existing scores remain valid measurements of the supplied retrospective
features. They do not establish performance using only information actually
published by 12:00 Europe/Brussels on D−1. Equal feature exposure across models
does not guarantee that their relative ranking would survive realistic inputs.

This audit concerns the exact full-study generation
`fd19205863b9d81b0f5d46535d9c970579838565b4b782d2b1550c79fdb1dff6`.
The script verifies current source hashes against trained-run provenance and
reads the actual tensor cache and saved checkpoints. No models were retrained,
and no input, score or checkpoint was modified.

## Findings by mechanism

| Mechanism | Finding | Classification |
|---|---|---|
| Targets supplied directly to predictors | Feature sets contain guarded past prices, calendar and conditional explanatory variables; current target prices and realised same-day flows are excluded | No direct target injection found |
| Training versus validation/test | Fit ends at 2024-03-31 21:00 UTC; validation starts 22:00 UTC. Validation ends 2025-03-31 21:00 UTC; test starts 22:00 UTC. These are Brussels midnight boundaries | Ordered, disjoint labels |
| Normalization/imputation | All 80 neural best checkpoints reproduce statistics computed from active training-period feature rows only | Passed |
| Temporal normalization | GRU checkpoint history statistics match training-origin histories | Passed |
| Price lags | Reconstructed all canonical rows: 0 mismatches for 4,439,350 finite guarded lag-24 values and 4,431,105 finite lag-168 values | Exact identity/time joins; publication revisions unresolved |
| GRU windows | 3,957 origins; 0 source-interval completions after origin; 763 missing source-time slots stay missing | Passed effective-time check |
| Spatial graph | Cached 250 edges equal the physical pairs observed in pretrain/train exactly | No test-derived edge selection found |
| Cross-time graph mixing | Changing the future graph in a two-time batch cannot alter the earlier graph's output in the architecture fixture | Passed |
| Weather lag 48 | Every finite test cell in each of four weather columns is only 12–36 hours old at origin | Inadmissible under documented ERA5/ERA5T release timing |
| Load forecasts | 0 availability-verified rows among 976,531 finite validation/test master rows; 804,754 latest update times are after origin | Strong late-vintage risk; original availability not established |
| JAO summaries | Same-zone lag-48 domain joins pass interval checks; publication/revision history is absent | Unresolved availability risk |
| Evaluation universe | Eligibility requires labels in fit, validation and test | Test-aware coverage/cohort selection, not direct label-value injection |
| Research decisions | Test results were repeatedly inspected before later studies were designed | Adaptive test exposure; no pristine test claim |

Counts for source screens refer to the master/cache rows examined, including graph
context, not a claim that every cell is used by every job or has a valid label.
The legacy registry's conservative delivery-time price rule must not be mistaken
for a genuine publication-time check.

## 1. Weather is the clearest operational problem

Four predictors are realised/reanalysis values at target hour minus 48 hours:
temperature, temperature anomaly, wind capacity factor and solar capacity factor.
For **each column**, the test cache has **565,536 finite values**, all only
**12–36 hours before origin** in valid time. They are in the past, but their
published data product is not necessarily available in the past.

ECMWF states that preliminary **ERA5T is released approximately five days late**;
final ERA5 follows about two to three months later. See
[ERA5T dataset](https://www.ecmwf.int/en/forecasts/dataset/era5t-era5-initial-release-data)
and [ERA5T release explanation](https://www.ecmwf.int/en/about/media-centre/news/2026/era5t-reanalysis-data).
PECD transformations/export releases can introduce additional delay. The export's
precise version/lineage is itself unverified, so the exact earliest release of
each value cannot be reconstructed. Under its documented ERA5-derived lineage,
lag 48 is not sufficient and these columns must be excluded from a strict run.

**65 of 100 jobs use weather.** The headline richer MLP, pooled XGBoost, real/self/
shuffled GAT and GRU comparisons all use it. This is publication-time leakage or
unavailable retrospective information, not the simpler error of supplying the
target day's realised weather. The distinction explains why an effective-time
guard and future-target mutation test can pass while an operational backtest fails.

Temperature anomalies and capacity factors additionally depend on upstream
climatology/calibration definitions. The supplied export lacks enough information
to establish whether those transformations use future years. This is unresolved,
not a confirmed full-period climatology leak.

**A longer lag alone is not a complete repair.** A seven-day lag might clear an
ERA5T latency screen, but using final revised ERA5 rather than its initial vintage
still does not reproduce what existed then. Operational forecasts or observations
need actual issue/release timestamps and vintage-preserving archives.

## 2. Load is forecast-labelled, not demonstrably an as-of forecast

The builder reads original `LoadForecastMW`, not an upstream filled `*_input` or
realised load column. But a current archive of forecast values is not equivalent
to the original forecast vintage.

In validation/test master rows with finite load:

- **976,531** have a source update timestamp; **0** are availability-verified.
- **804,754 (about 82.4%)** have a latest update after origin.
- **804,635** have even their earliest recorded update after origin.
- Example: AT and BE delivery `2024-03-31 22:00 UTC`, origin
  `2024-03-31 10:00 UTC`, have both update fields around `2025-09-28 00:57 UTC`.

These fields may represent later revisions or archive/source-update semantics.
They do **not** prove the forecast number differs from an earlier published value,
and should not be relabelled as original issue timestamps without documentation.
They do show that a noon-available historical vintage cannot be demonstrated.
Load should remain excluded from a strict evaluation until the appropriate
publication/vintage selection is implemented.

## 3. Constraints and price revisions remain unresolved

JAO predictors use only lagged `n_presolved`, `min_ram_ratio` and domain presence.
Same-day congestion components, realised net positions, scheduled exchanges,
MaxBex and target prices are not accidentally included by the richer feature list.
No constraint nodes or PTDF edges exist in this study.

The implemented domain joins enforce same-zone source time t−48 and completed
intervals before origin. They preserve missing versus zero and regional membership.
However, archived revised domains are not automatically the original published
domains. No issue/revision timestamps establish the values available at origin.
This is an unresolved provenance limitation, not proof that every JAO cell leaks.

Previous-day auction prices are different from observations: delivery may occur
after noon while the auction price was already published on D−2. EPEX's
[auction procedure](https://www.epexspot.com/sites/default/files/2022-07/22-07-12_TradingBrochure.pdf)
documents prior-day auction/results timing. The lag-24 autumn-DST guard correctly
removes values whose delivery date reaches D itself. Lag 168 is likewise an exact
same-identity join. Neither lag proves the absence of subsequent price corrections;
an archived auction release is needed for strict vintage certification.

Using actual prices from earlier test days for later rolling forecast origins
is legitimate when those auctions were already published. The contract forecasts
one delivery day at a time, not the whole test year from one fixed origin.

## 4. What the implementation checks rule out—and what they do not

All 80 neural checkpoints match independently recomputed training-only feature
medians/means/scales. The fixed asinh scale of 50 is not fitted. GRU statistics
use training-origin histories. Validation labels select early stopping. Test
labels are used for scoring and, at later rolling origins, legitimate past-price
history; they are not optimizer labels. Loading a full array into memory is not
itself evidence that future entries affect the predictor.
Validation scores therefore reflect early-stopping selection and are not an
independent untouched performance estimate.

Counterfactual fixtures change post-origin prices without changing earlier GRU
inputs or training statistics, and poison a future graph in the batch without
changing the earlier prediction. These exercise actual history/model functions;
they are not a full poisoned-data retraining experiment. The previous full-artifact
verification independently checked all 100 jobs' keys, masks and reported scores.

Training-only physical topology avoids test-derived connections. It is still a
retrospective union: using a pair first observed late in training for earlier
training samples is not a dated commissioning reconstruction. It does not, by
itself, establish that a held-out forecast used a future connection. Node lifetime
masks and same-time graph offsets prevent historical identity substitution and
cross-time message passing. Source geography errors are a separate model-risk
category and cannot be certified away by the leakage checks.

## 5. Research protocol and the meaning of our conclusions

The 49-market cohort was chosen using nonzero target coverage in all three
partitions. That conditions the reported population on test coverage; it is not
direct use of test price values to predict prices. Freeze a deployment universe
from training-time metadata for a stronger prospective evaluation.

More importantly, the same test period has informed successive experiment choices.
The 100-job matrix was fixed before that particular full run, but the research
program had already inspected the period. Five seeds, confidence intervals and
identical masks do not undo adaptive test exposure. The week bootstrap conditions
on the fitted models; it does not account for researcher selection or publication
latency. No clean, blind test or general significance claim is justified.

The richer-input graph advantage may be useful retrospective evidence, but sharing
the same unavailable inputs does not prove its operational value. Some reassuring
existing evidence survives the weather/load/JAO concern: **recent-price-only GAT
scores 20.9947 versus recent-price-only MLP 21.2540**, averaged across five seeds.
Those feature sets exclude all three explanatory-source groups, but still rely
on historical price revisions, static training topology and the exposed test
period. There are no recent-price-only self-loop/rewiring controls in this matrix,
so the full topology attribution is not independently settled by that comparison.

## Recommended repair sequence

1. Freeze and label the current scores as retrospective/conditional. Preserve v1
   and v2 rather than silently replacing their data or reinterpreting their scores.
2. Use the existing recent-price-only results as the initial lower-provenance-risk
   baseline. Add recent-only pooled XGBoost, self-loop and rewired graph controls
   under the same protocol; do not call them vintage-certified without auction data.
3. Obtain vintage-preserving load and weather forecasts with issue time at or
   before origin. Select JAO versions by actual publication time. Reconstruct any
   weather anomalies from a documented, permissible reference period.
4. Run strict as-of admission on every feature, including release latency and
   revisions, not merely a valid-time lag. Define geography/topology and cohort
   from information available by the model fit date.
5. Predeclare the next comparisons and evaluate a genuinely new chronological
   holdout once. Keep test-driven exploration separate from that final claim.

No corrective training or new data downloads were performed in this audit.

## Reproduce

```bash
.venv/bin/python scripts/experiments/audit_forecast_leakage.py
```

The script writes [machine-readable evidence](../results/forecast_leakage_audit.json).
It requires a writable temporary directory for PyTorch imports but no GPU or
training. Source hashes must match trained-run provenance. Relevant implementation:
`experiment_suite/data.py`, `controlled_study/training.py`, `history.py`, `models.py`,
the pooled/per-zone tree runners, and the original dataset construction adapters.
