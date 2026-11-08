# Multi-seed controls: comparison with earlier experiments

**November 9 correction:** the original GRU families fail CPU agreement on
shuffled training-shaped inputs with the fused ROCm backend. Their archived
scores are numerically reproducible but no longer support temporal-model or
GRU-graph conclusions. Static controls are outside this finding. See the
[current experiment record](experiments.md#our-grus-versus-the-partner-inspired-gru).

All 100 jobs completed and passed independent artifact, target-mask and score verification. Models use the same 49-market cohort and chronological splits. Scores below use common finite weekly-history hours, averaged across training seeds (not ensembled predictions).

## Main controls

| Variant | Validation MAE ± SD | Test MAE ± SD |
|---|---:|---:|
| gat_jao_real | 21.1308 ± 0.0635 | 21.1144 ± 0.0653 |
| gat_load_real | 21.0773 ± 0.1322 | 21.0854 ± 0.1336 |
| gat_recent_real | 21.1218 ± 0.0903 | 20.9947 ± 0.0914 |
| gat_rich_real | 20.9830 ± 0.1074 | 20.9737 ± 0.1265 |
| gat_rich_self | 21.2668 ± 0.0908 | 21.0637 ± 0.0805 |
| gat_rich_shuffled_rewire901 | 21.1467 ± 0.0600 | 21.4291 ± 0.0746 |
| gat_rich_shuffled_rewire902 | 21.2067 ± 0.0866 | 21.5178 ± 0.1153 |
| gat_rich_shuffled_rewire903 | 21.2761 ± 0.1023 | 21.4100 ± 0.0937 |
| gat_weather_real | 21.0556 ± 0.1140 | 20.9041 ± 0.0878 |
| gru_gat_rich_real | 23.6200 ± 0.5239 | 23.5875 ± 0.5139 |
| gru_mlp_rich_none | 24.3701 ± 1.6481 | 24.4997 ± 1.2160 |
| mlp_jao_none | 21.5113 ± 0.0923 | 21.1934 ± 0.0747 |
| mlp_load_none | 21.5116 ± 0.1460 | 21.2952 ± 0.1602 |
| mlp_recent_none | 21.4864 ± 0.1124 | 21.2540 ± 0.0809 |
| mlp_rich_none | 21.4429 ± 0.2253 | 21.1853 ± 0.2059 |
| mlp_weather_none | 21.4195 ± 0.1181 | 21.1424 ± 0.0819 |
| pooled_xgb_rich_none | 20.7362 ± 0.0145 | 21.0584 ± 0.0135 |
| xgb_neighbours_none | 20.8485 ± 0.0224 | 21.8438 ± 0.0993 |
| xgb_recent_none | 21.2580 ± 0.0416 | 21.8689 ± 0.0417 |
| xgb_rich_none | 21.2348 ± 0.0535 | 22.1096 ± 0.0279 |

## What changes the interpretation

- Real GAT has a modest advantage over the parameter-matched pooled MLP and self-loop GAT. Pooled XGBoost is nearly tied on test and has better validation MAE than richer GAT. These controls reduce the gap that can be attributed specifically to topology; the original report already identified pooling as a confound.
- Real GAT beats the three shuffled graph means on test. One validation block-bootstrap interval against a shuffled graph spans zero; evidence is not uniformly decisive across rewiring controls.
- Weather-only GAT has the smallest observed test MAE, but richer GAT is preferred by validation. Do not select the weather-only model retrospectively using test.
- The specified GRU scores are retained as affected historical measurements. A backend discrepancy was found on shuffled training-shaped inputs; corrected training is required before interpreting either their static-model comparison or GRU-GAT versus GRU-MLP contrast.
- Gains are not universal: France is substantially worse under richer GAT than MLP or pooled XGBoost. These market-level patterns warrant further validation, not per-country selection using test.

## Earlier cohorts and models

| Experiment | Test MAE | Interpretation |
|---|---:|---|
| Earlier 49-market real GAT, one seed | 20.8279 | Paired hours; earlier sampler/seed implementation, not a multi-seed estimate |
| Earlier shuffled GAT, one seed | 21.5421 | Paired hours; one rewiring, versus three graphs/five seeds now |
| Earlier recent-price per-zone XGBoost | 21.8654 | Paired hours; same cohort, with pooling now an additional control |
| Earlier richer per-zone XGBoost | 22.1323 | Paired hours; new per-zone mean 22.1096 versus pooled mean 21.0584 |
| France: weekly/calendar → recent-price | 32.0921 → 22.1988 | One-country diagnostic, different population |
| Partner reproduction, all 43 zones | 21.2539 | Full-target MAE; different geography/features/population, not directly ranked |

## Familiar country/market comparisons

| Market | Pooled MLP | Real GAT | Pooled XGBoost |
|---|---:|---:|---:|
| BE | 22.969 | 22.522 | 22.467 |
| FR | 21.998 | 25.480 | 21.943 |
| NL | 23.300 | 22.962 | 22.752 |
| DE_LU | 24.908 | 23.642 | 24.441 |
| AT | 20.421 | 19.552 | 19.785 |
| ES | 17.522 | 17.818 | 17.399 |
| PT | 17.285 | 17.297 | 17.180 |
| PL | 25.184 | 24.257 | 24.477 |
| CZ | 22.231 | 22.080 | 21.524 |
| CH | 15.125 | 15.094 | 14.596 |

## Seasons and training-defined price regimes

All-valid-target MAE means; this population differs from the paired table. Tails use training-only 5th/95th percentiles. These diagnostic slices overlap and are not model-selection criteria.

| Slice | MLP | Real GAT | Pooled XGBoost |
|---|---:|---:|---:|
| DJF | 19.671 | 19.593 | 19.359 |
| MAM | 21.671 | 21.204 | 21.509 |
| JJA | 19.044 | 19.140 | 18.746 |
| SON | 24.961 | 24.495 | 25.279 |
| negative | 26.375 | 24.940 | 27.285 |
| lower_tail | 26.607 | 25.598 | 27.199 |
| upper_tail | 50.103 | 49.089 | 54.597 |

## Uncertainty and limits

Paired validation-week intervals condition on fitted models, with weeks resampled jointly across countries. Seed variability is distinct from temporal loss uncertainty and graph rewiring variability. There is no fresh blind test claim, multiple-comparison significance claim or verified historical publication availability.

- mlp_rich_none_to_gat_rich_real: validation MAE improvement 0.4599, interval [0.2004, 0.8605].
- gat_rich_self_to_gat_rich_real: validation MAE improvement 0.2838, interval [0.1300, 0.4601].
- gru_mlp_rich_none_to_gru_gat_rich_real: validation MAE improvement 0.7501, interval [0.5567, 0.9432].
- gat_rich_shuffled_rewire901_to_gat_rich_real: validation MAE improvement 0.1637, interval [-0.0401, 0.3348].
- gat_rich_shuffled_rewire902_to_gat_rich_real: validation MAE improvement 0.2237, interval [0.0529, 0.3829].
- gat_rich_shuffled_rewire903_to_gat_rich_real: validation MAE improvement 0.2931, interval [0.1451, 0.4497].
- mlp_recent_none_to_mlp_load_none: validation MAE improvement -0.0252, interval [-0.1127, 0.0262].
- mlp_recent_none_to_mlp_weather_none: validation MAE improvement 0.0670, interval [-0.0804, 0.1845].
- mlp_recent_none_to_mlp_jao_none: validation MAE improvement -0.0249, interval [-0.2436, 0.0987].
- mlp_recent_none_to_mlp_rich_none: validation MAE improvement 0.0435, interval [-0.3047, 0.2566].
- gat_recent_real_to_gat_load_real: validation MAE improvement 0.0445, interval [-0.0514, 0.1658].
- gat_recent_real_to_gat_weather_real: validation MAE improvement 0.0662, interval [0.0014, 0.1332].
- gat_recent_real_to_gat_jao_real: validation MAE improvement -0.0090, interval [-0.0690, 0.0471].
- gat_recent_real_to_gat_rich_real: validation MAE improvement 0.1388, interval [0.0143, 0.2681].

[Milestone v2](../reports/milestone_v2/milestone.pdf) · [Milestone v1](../reports/milestone_v1/milestone.pdf) · [Raw multi-seed scores](../results/controlled_study_full_scores.csv)
