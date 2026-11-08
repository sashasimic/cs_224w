# European electricity-price forecasting: experiment record

This is the current consolidated account of the completed experiments, including
negative results and rejected runs. It replaces scattered summaries as the
starting point for reading results. Detailed protocols, original metrics and
milestone snapshots remain linked for reproduction. Tables are rebuilt from
saved results; no training or model selection occurs during report generation.

## What we have learned

Recent auction prices explain most of the initial improvement over weekly
persistence. Once models share strong recent-price inputs and pool training
across markets, XGBoost, MLP and static GAT are much closer than the first
per-zone comparison suggested. Graph improvements are modest and vary by market.

Our original GRU experiments need correction: their saved scores are reproducible,
but the fused ROCm GRU disagrees with CPU on shuffled training-shaped batches.
Those scores cannot establish that temporal learning is worse. The newer
partner-inspired GRU uses a verified native GPU path and achieves about
21 EUR/MWh test MAE, close to strong static models. Its recipe differs in several
ways, so the difference cannot be attributed to one change.

Weather and constraint results are mixed. We have retrospective ERA5-lineage
weather and airport observations, **not historical operational weather forecasts**.
JAO constraints entered as flat summaries; no completed experiment used CNECs as
extra graph nodes. None of these results certifies a production-time noon forecast.

## Data, forecast contract and metrics

At 12:00 Europe/Brussels on D−1, predict each hourly price of delivery day D.
Origins use the previous local calendar date at literal noon, then convert to
UTC. DST days retain 23 or 25 delivery hours. Prices are EUR/MWh: a prediction
error of 21 EUR/MWh means a 21-euro price error per megawatt-hour of energy.

The canonical master has 94,967 UTC hours, 62 historical market identities and
5,887,954 node rows, January 2016–October 2026. It preserves missingness,
negative prices and DE_AT_LU separately from DE_LU. Forty-nine identities have
labels in every required partition and form the later evaluation cohort.
Targets are never filled and missing weather never removes target rows.

| Partition, Brussels delivery date | Dates | Purpose |
|---|---|---|
| Pretrain + train | January 2016–March 2024 | Fit models and learned preprocessing |
| Validation | April 2024–March 2025 | Early stopping and validation diagnostics |
| Test | April 2025–September 2026 | Previously explored evaluation period |
| Unassigned | October 2026 | Coverage/structural audit only |

MAE is mean absolute price error, not classification accuracy. RMSE weights
large errors more strongly. Primary all-target evaluation uses 428,427 validation
and 634,578 test labels. The weekly comparison additionally requires history:
428,213 validation and 633,885 test labels. **Paired MAE and all-target MAE are
different populations.** Multi-seed entries average scores, not predictions.

Weekly persistence copies the same market's price exactly 168 elapsed hours
earlier. Relative MAE is model MAE divided by persistence MAE on those same
hours; 0.72 means 28% lower error. Percentage error relative to the actual price
is unstable around zero and negative prices, so we use persistence-relative MAE.
Seed SD measures fit variability; validation-week bootstrap intervals condition
on the fitted models and are not comprehensive uncertainty estimates.

ENTSO-E prices/load/flows are pinned to v1.2. The supplied weather export is
ERA5-lineage; its requested v1.1 release identity remains unresolved. Original
CWE/CORE JAO exports retain generation and file hashes. Current data limitations
and the price-publication correction are in the [leakage audit](leakage_audit.md)
and [pipeline comparison](pipeline_comparison.md). An observation's timestamp is
not its publication time. All conditional columns remain conditional here.

## Experiment inventory

| Programme | Completed scope | What it tested | Detailed record |
|---|---|---|---|
| Initial audited baselines | Belgium, then 10 markets; one seed | Calendar/region inputs, weekly prices, raw targets versus asinh/L1 | [Baseline](boosting_baseline.md), [countries](boosting_countries.md), [transformation](boosting_target_transform.md) |
| Legacy persistence reference | 43 scored legacy zones | Earlier persistence on the original pipeline cohort | [Saved result](../results/c1_persistence.json) |
| JAO summaries | Seven covered markets; one seed | Three lag-48 flat constraint summaries | [Constraints](boosting_constraints.md) |
| Early temperature diagnostics | France; two additions to one baseline | Current-origin ERA5 proxy and explicit perfect-weather oracle | [Temperature](temperature_experiments.md) |
| Partner XGBoost and France sequence | 43 scored legacy zones; eight France profiles | Reproduction, reaggregation and pipeline differences | [Reproduction](partner_pipeline_reproduction.md), [sequence](pipeline_comparison.md) |
| First canonical suite | Five experiment jobs; one seed | Per-zone XGBoost and real/shuffled GAT | [Suite](vibing_suite_results.md) |
| Controlled study | 20 variants × five seeds = 100 jobs | Pooled controls, feature/topology ablations, original GRU+MLP/GAT | [Design](controlled_study.md), [results](controlled_study_comparison.md) |
| Airport observations | Nine variants × five seeds = 45 jobs | Price-only, latest temperature and trailing temperature history | [Airport review](airport_temperature_results.md) |
| Partner-inspired residual GRU | One accepted canonical full fit; seed 0 | Multivariate histories, persistence residual and residual-contrast loss | [Method](partner_gru.md), [results](partner_gru_comparison.md) |

The two five-seed studies share two variant names: 27 distinct names across
145 jobs, not 145 different model designs. A per-zone XGBoost job fits separate
models for the scored markets. The first suite and the newer residual GRU are
additional programmes. Smoke fits, bootstrap resamples and checkpoint reloads
are correctness checks, not additional scientific experiments.

## Controlled study: all 20 variants

All models use the same canonical cohort and splits. XGBoost rows distinguish
independent per-zone models from one pooled model. MLP/GAT share weights across
markets. Neural models fit preprocessing on training rows only and include
missingness indicators. Their target is asinh(price/50), inverted for scoring.
The seed range is 224–228. Rich inputs mean recent prices/calendar plus load,
four lag-48 weather variables and three lag-48 JAO summaries.

Rows are grouped by model class and feature/topology controls, not sorted by
test performance. ± is sample SD across five seeds. Bold marks the smallest
eligible mean in each paired-MAE column, excluding affected GRU runs. It does
not select a new preferred model after looking at test results.

<!-- generated:controlled -->

| Variant | Validation paired MAE | Test paired MAE | Test all-target MAE | Test all-target RMSE |
| --- | --- | --- | --- | --- |
| gat_jao_real | 21.1308 ± 0.0635 | 21.1144 ± 0.0653 | 21.1084 | 35.1345 |
| gat_load_real | 21.0773 ± 0.1322 | 21.0854 ± 0.1336 | 21.0813 | 36.2013 |
| gat_recent_real | 21.1218 ± 0.0903 | 20.9947 ± 0.0914 | 20.9886 | 35.4822 |
| gat_rich_real | 20.9830 ± 0.1074 | 20.9737 ± 0.1265 | 20.9690 | 34.9183 |
| gat_rich_self | 21.2668 ± 0.0908 | 21.0637 ± 0.0805 | 21.0609 | 35.1248 |
| gat_rich_shuffled_rewire901 | 21.1467 ± 0.0600 | 21.4291 ± 0.0746 | 21.4268 | 36.9055 |
| gat_rich_shuffled_rewire902 | 21.2067 ± 0.0866 | 21.5178 ± 0.1153 | 21.5144 | 36.6751 |
| gat_rich_shuffled_rewire903 | 21.2761 ± 0.1023 | 21.4100 ± 0.0937 | 21.4066 | 36.0114 |
| gat_weather_real | 21.0556 ± 0.1140 | **20.9041 ± 0.0878** | 20.9010 | 34.9309 |
| gru_gat_rich_real † | 23.6200 ± 0.5239 | 23.5875 ± 0.5139 | 23.5816 | 37.5797 |
| gru_mlp_rich_none † | 24.3701 ± 1.6481 | 24.4997 ± 1.2160 | 24.5032 | 38.9431 |
| mlp_jao_none | 21.5113 ± 0.0923 | 21.1934 ± 0.0747 | 21.1891 | 35.5591 |
| mlp_load_none | 21.5116 ± 0.1460 | 21.2952 ± 0.1602 | 21.2922 | 36.5797 |
| mlp_recent_none | 21.4864 ± 0.1124 | 21.2540 ± 0.0809 | 21.2507 | 35.9355 |
| mlp_rich_none | 21.4429 ± 0.2253 | 21.1853 ± 0.2059 | 21.1824 | 35.9725 |
| mlp_weather_none | 21.4195 ± 0.1181 | 21.1424 ± 0.0819 | 21.1400 | 35.6213 |
| pooled_xgb_rich_none | **20.7362 ± 0.0145** | 21.0584 ± 0.0135 | 21.0554 | 37.5695 |
| xgb_neighbours_none | 20.8485 ± 0.0224 | 21.8438 ± 0.0993 | 21.8397 | 39.5353 |
| xgb_recent_none | 21.2580 ± 0.0416 | 21.8689 ± 0.0417 | 21.8668 | 39.8388 |
| xgb_rich_none | 21.2348 ± 0.0535 | 22.1096 ± 0.0279 | 22.1084 | 41.4415 |

† Affected fused-GRU training path; retain the numbers as history and exclude them from architecture conclusions. See the next section.

<!-- end:controlled -->

Real rich GAT improves paired validation MAE over parameter-matched rich MLP by
0.4599 EUR/MWh (paired-week 95% interval 0.2004–0.8605), and over self-loop GAT
by 0.2838 (0.1300–0.4601). One shuffled-control validation interval includes zero.
Pooled rich XGBoost has better validation MAE than rich GAT and similar test MAE.
Weather-only GAT has the lowest observed test mean, but richer GAT was preferred
on validation. These are exploratory comparisons with repeated test exposure.
The old GRU-versus-GRU bootstrap contrast is also affected by the backend issue.

## Our GRUs versus the partner-inspired GRU

Our original GRU+MLP and GRU+GAT were actually trained across five seeds. They
were not placeholders. Their archived paired test MAEs are 24.4997 and 23.5875.
The November 9 audit now classifies both families as **affected historical runs**.

| Choice | Original controlled GRU+MLP / GRU+GAT | Accepted partner-inspired GRU+MLP |
|---|---|---|
| Sequence | 168 completed same-identity hourly prices before origin | 168 completed hours of eight explanatory channels |
| History channels | asinh(price/50), normalized value + missingness indicator | Guarded price lags, load, derived net inflow, temperature/anomaly/wind/solar |
| Encoder | Shared GRU, 32 hidden units | Shared GRU, 64 hidden units |
| Day inputs | Rich static features with missingness indicators | 11 day features; no extra missingness indicators |
| Prediction | Direct asinh(price/50) | Residual from transformed weekly/daily persistence base |
| Head | Parameter-matched ELU MLP, or spatial GAT | Two 128-unit ReLU MLP layers |
| Loss | L1 transformed price | L1 transformed residual + 0.5 × cross-zone residual-contrast L1 |
| Batch | 64 shuffled target hours; up to 64 unique origins | 16 delivery days; variable 23/24/25 target slots |
| Epochs / patience | At most 30 / five | At most 50 / eight; selected epoch 29 of 37 |
| Seeds | Five, 224–228 | One, seed 0 |
| RNN backend | Fused ROCm; shuffled-input reference failed | Native GPU; CPU reference and replay checks passed |

The base fallback in the new GRU reconstructs predictions only: weekly price,
else guarded daily price, else zero. It never fills a missing target. Both
recipes use training-only preprocessing and preserve historical zone identities.
Its net inflow is a completeness-checked proxy over training graph pairs, not a
claimed national net-import total. Its spread objective couples zones during
training; the MLP makes no inference-time graph messages.

<!-- generated:gru -->

| Family | Checkpoints | Largest fused inference difference | Shuffled-input max-difference range | Largest native GPU difference |
| --- | --- | --- | --- | --- |
| gru_mlp | 5 | 0.000552 | 26.67–155.62 | 0.000552 |
| gru_gat | 5 | 0.000548 | 57.10–211.47 | 0.000834 |

Differences are EUR/MWh versus the same weights on CPU float64. **10/10 checkpoints failed the fused reference gate** across four batches each (three chronological evaluation batches and one shuffled training-input batch). Tolerances: max ≤0.01, mean ≤0.0001 EUR/MWh. All native and saved-inference replay gates passed.

| Recipe | Seeds/status | Validation paired MAE | Test paired MAE | Test all-target MAE | Test all-target RMSE |
| --- | --- | --- | --- | --- | --- |
| gru_mlp_rich_none | 5; affected | 24.3701 | 24.4997 | 24.5032 | 38.9431 |
| gru_gat_rich_real | 5; affected | 23.6200 | 23.5875 | 23.5816 | 37.5797 |
| partner_residual_gru_conditional | 1; native checked | 20.3192 | 20.9817 | 21.0677 | 47.0222 |

<!-- end:gru -->

**Backend interpretation:** saved predictions can replay correctly while the
training calculation differs from the intended GRU. Our small consecutive
inference batches passed, but the shuffled training-input batches failed the
fused path. Native GPU matched CPU. This audit runs forward passes with dropout
disabled; it does not inspect every batch or verify backward derivatives. We
have not retrained the original ten models. Their old scores remain available
for provenance and cannot establish a temporal-architecture disadvantage.

Likewise, the first partner-style fused run is rejected. Its corrected native
run passed CPU checks and full saved-prediction verification. Switching an old
checkpoint to native inference alone would not correct its training history.
Compare the [original-family audit](../results/controlled_gru_backend_audit.json)
and [partner-family audit](../results/partner_gru_backend_audit.json).

The accepted residual GRU has all-target test MAE 21.0677 and RMSE 47.0222.
It is close to static XGBoost/GAT on MAE but worse on RMSE; its validation MAE is
lower. One seed and several simultaneous recipe changes prevent causal claims
about the residual target, weather, spread objective or GRU architecture.

## Airport temperature: all nine variants

One airport per country is selected by 2023 passenger volume. Regional zones
share that country's station; combined-market proxies and unresolved mappings
are explicit. Latest temperature must be at least two and at most six hours old
at origin. The optional 24-hour mean requires 18 sampled hours. These features
are frozen across the delivery day. Models use recent prices/calendar and
airport features, without ERA5, load, JAO or realised-flow predictors.

<!-- generated:airport -->

| Variant | Validation paired MAE | Test paired MAE | Test all-target MAE | Test all-target RMSE |
| --- | --- | --- | --- | --- |
| gat_airport_history_real | 21.0717 ± 0.0610 | **20.9551 ± 0.0433** | 20.9516 | 34.7894 |
| gat_airport_latest_real | 21.0668 ± 0.1372 | 20.9584 ± 0.1506 | 20.9536 | 35.0515 |
| gat_recent_real | 21.1468 ± 0.0884 | 20.9900 ± 0.0755 | 20.9838 | 35.3302 |
| mlp_airport_history_none | 21.4481 ± 0.1095 | 21.1466 ± 0.1179 | 21.1432 | 35.4303 |
| mlp_airport_latest_none | 21.4524 ± 0.0601 | 21.1247 ± 0.0656 | 21.1216 | 35.3044 |
| mlp_recent_none | 21.4864 ± 0.1124 | 21.2540 ± 0.0809 | 21.2507 | 35.9355 |
| pooled_xgb_airport_history_none | **20.8117 ± 0.0178** | 21.0892 ± 0.0163 | 21.0841 | 37.8838 |
| pooled_xgb_airport_latest_none | 20.8264 ± 0.0220 | 21.0927 ± 0.0269 | 21.0889 | 37.7774 |
| pooled_xgb_recent_none | 20.8421 ± 0.0088 | 21.4360 ± 0.0106 | 21.4299 | 43.5951 |

Five seeds per row. Bold is descriptive within this programme; latest/history are alternative airport feature groups.

<!-- end:airport -->

Gains are small and inconsistent across markets. Latest-temperature validation
intervals include zero for XGBoost and MLP; GAT's interval is narrowly positive,
unadjusted for multiple comparisons. About 87.3% of pooled XGBoost's headline
test improvement comes from an unmapped market whose airport features are all
missing. Joint fitting and missing-value routes can change its predictions;
that improvement cannot be credited to a local temperature reading. France
worsens for all three models. Original receipt/revision times are absent, so
this is still conditional observation evidence, not a proven as-of feed.

## First canonical suite: all five jobs

This earlier single-seed result motivated the pooled MLP and topology controls.
It must not replace the five-seed evidence. There was no GRU in this suite.

<!-- generated:suite -->

| Model | Validation paired MAE | Test paired MAE | Test all-target MAE | Test all-target RMSE |
| --- | --- | --- | --- | --- |
| recent_xgb | 21.1839 | 21.8654 | 21.8675 | 39.9540 |
| rich_xgb | 21.2751 | 22.1323 | 22.1319 | 40.7710 |
| neighbour_xgb | 20.8312 | 21.8724 | 21.8686 | 40.2669 |
| real_gat | 20.8074 | 20.8279 | 20.8235 | 34.3965 |
| shuffled_gat | 21.2261 | 21.5421 | 21.5422 | 37.1272 |

<!-- end:suite -->

The first real GAT's 3.32% advantage over shuffled GAT is a one-seed contrast.
The comparison with independent per-zone trees also changes pooling, which is
why the later pooled XGBoost and MLP controls matter.

## Earlier baselines, transformation and constraints

The strict Belgium calendar/region-only XGBoost scored 40.89 test MAE versus
31.12 for persistence. Adding weekly history reduced it to 30.41. Expanding the
same small feature set to ten countries produced mixed results. The next table
keeps the same ten-country target cohort; these are not 49-market aggregates.

<!-- generated:countries -->

| Market | Raw weekly XGB | Asinh/L1 weekly XGB | Partner pipeline | Persistence |
| --- | --- | --- | --- | --- |
| BE | 30.4092 | 30.0128 | 23.6578 | 31.1176 |
| FR | 35.6153 | 34.8261 | 22.0893 | 32.4425 |
| NL | 30.8736 | 30.6762 | 23.1043 | 31.0367 |
| DE_LU | 32.7167 | 32.3378 | 24.5843 | 33.9098 |
| AT | 27.4135 | 26.2354 | 20.0383 | 27.2449 |
| ES | 29.3343 | 26.8236 | 18.2179 | 25.7698 |
| PT | 28.5127 | 26.5687 | 17.9178 | 25.9775 |
| PL | 30.7896 | 30.5143 | 25.0742 | 33.2402 |
| CZ | 28.8951 | 28.6029 | 22.3162 | 29.3997 |
| CH | 21.3709 | 21.5140 | 14.6630 | 21.3252 |

<!-- end:countries -->

Across those ten countries, raw-price squared-error XGBoost scored 29.59,
asinh/L1 scored 28.81 and persistence scored 29.15 test MAE. Transformation/loss
helped, but did not explain the stronger partner pipeline. Adding three flat
JAO summaries to the original raw-price recipe changed seven covered countries
from 30.9590 to 30.9371, a 0.07% test improvement and a slight validation decline.

<!-- generated:jao -->

| Market | Weekly raw XGB | + JAO summaries | Change in MAE |
| --- | --- | --- | --- |
| BE | 30.4092 | 30.4560 | +0.0468 |
| FR | 35.6153 | 34.9439 | -0.6715 |
| NL | 30.8736 | 30.8687 | -0.0049 |
| DE_LU | 32.7167 | 32.6348 | -0.0819 |
| AT | 27.4135 | 27.4611 | +0.0477 |
| PL | 30.7896 | 31.0265 | +0.2370 |
| CZ | 28.8951 | 29.1687 | +0.2735 |

<!-- end:jao -->

JAO inputs are n_presolved, minimum RAM/Fmax ratio and publication-domain
presence at lag 48. Missing publications differ from published zeros; CWE and
CORE retain dated membership. No experiment treated MaxBex as jointly feasible
import capacity, and no detailed constraint-node model was trained. The CNEC
and heterogeneous-graph YAML configurations remain scaffolds.

## France temperature and pipeline investigations

These temperature diagnostics extend the original weekly/raw-price France
model, not the later rich/pooled models. The current-origin proxy is one
completed ERA5-lineage temperature reading frozen across day D. The perfect
temperature case deliberately includes the realised target-hour temperature
and is labelled an **oracle**, excluded from forecasting rankings.

<!-- generated:temperature -->

| Profile | Status | Validation MAE | Test MAE | Test RMSE |
| --- | --- | --- | --- | --- |
| weekly_price | conditional | 33.6239 | 35.6153 | 44.8970 |
| current_temperature | conditional | 31.0638 | 35.3029 | 44.5365 |
| perfect_temperature | oracle_perfect_weather | 30.6972 | 35.0850 | 44.7562 |

<!-- end:temperature -->

Neither temperature addition beats France persistence's 32.4425 test MAE.
This does not imply that operational regional weather forecasts are unhelpful.

The partner XGBoost reproduction scored 21.2539 test MAE across 43 legacy zones,
close to its published 21.2433. It scored 22.0893 on France. It uses a richer
feature set, legacy geography/fills and different tree parameters. It cannot be
ranked against a 49-market or ten-market aggregate as if the populations match.
The partner's published GRU+MLP score of 20.4014 belongs to that separate
protocol; we adapted its method rather than reproducing that numerical result.
The separate legacy persistence artifact scores 30.1463 validation and 29.0087
test MAE over 376,680/565,320 zone-hours. It is historical context on that legacy
cohort, not a replacement for the canonical persistence comparator of 29.1334.

The fixed-learner France sequence below tests the richer pipeline, including
explicitly retrospective weather controls. Rows are cumulative through flows;
the three weather rows are alternative profiles. The lag-24 all-weather case
contains post-origin weather and cannot be an operational forecast score.

<!-- generated:france -->

| Profile | Validation MAE | Test MAE | Selected rounds |
| --- | --- | --- | --- |
| weekly_flags | 34.6686 | 36.3176 | 236 |
| weekly_calendar | 31.5510 | 32.0921 | 99 |
| recent_price | 20.3867 | 22.1988 | 356 |
| load | 20.3902 | 22.2115 | 418 |
| flows | 20.4709 | 22.1394 | 366 |
| weather_past24 | 20.3667 | 22.0594 | 466 |
| weather48 | 20.2463 | 22.1191 | 488 |
| weather_all24 | 20.4099 | 22.0414 | 408 |

<!-- end:france -->

Adding guarded prior-day auction prices improved test MAE from 32.09 to 22.20,
the dominant measured feature addition in this sequence. Prior-day delivery
prices are normally auctioned earlier; a delivery interval after origin is not
itself proof of price leakage. The fall-DST case where elapsed lag24 refers to
the target delivery day is masked. Calendar-column and sampling changes confound
the preceding profile comparison. Load/flow/weather additions are small and
mixed; those source vintages still need independent publication checks.

## What remains usable, and what comes next

| Evidence | Current interpretation |
|---|---|
| Static XGBoost/MLP/GAT and airport comparisons | Verified retrospective scores; conditional availability and repeated-test limits apply |
| Original controlled GRU+MLP / GRU+GAT | Archived scores; affected fused training path; exclude from architecture conclusions pending corrected fits |
| First partner-style fused GRU | Rejected; preserved for backend diagnosis |
| Corrected partner-inspired GRU | Accepted numerical checks; one seed, conditional ERA5/vintages, no isolated weather effect |
| Perfect target-hour temperature | Explicit oracle diagnostic, not forecasting evidence |
| CNEC/heterogeneous constraint-node configurations | Planned scaffold, no completed fit |
| France/synthetic smoke runs | Correctness gates, not scientific performance estimates |

The next controlled temporal test should rerun our original GRU+MLP and
GRU+GAT recipes with the verified native backend, the same seeds and masks.
Separately, repeat the residual recipe across seeds and ablate histories,
residual target and contrast loss one change at a time. These runs have not
been launched as part of documentation cleanup.

For production work, preserve daily issue-time snapshots, record predictions
before outcomes arrive, compare against weekly persistence and strong pooled
static baselines, and monitor coverage and inference reliability. This would
measure prospective performance without reusing the explored test period as a
new confirmation set. Operational temperature requires an as-of observation or
forecast archive; retrospective ERA5 is not such an archive.

## Reproduction and retained records

```bash
# Rebuild this document's tables and browser-readable version from saved scores.
.venv/bin/python scripts/reports/build_experiment_summary.py

# Read-only GPU/CPU audit of the ten original temporal checkpoints.
.venv/bin/python -u scripts/experiments/check_controlled_gru_backend.py

# Independent saved-prediction checks for the corrected residual GRU.
.venv/bin/python scripts/experiments/review_partner_gru.py
```

The browser version is [reports/experiments/experiments.html](../reports/experiments/experiments.html).
Its source manifest records the result and code hashes used for this compilation.
The [documentation index](README.md) separates current guidance from historical
protocols; [results/README.md](../results/README.md) indexes raw scores and source
evidence. Original Parquets, tensor caches, models and training code are unchanged.
The original milestone PDFs remain snapshots; the old negative GRU conclusion
is superseded by this backend audit. The existing investigation notebook retains
the archived numbers with the same warning.
