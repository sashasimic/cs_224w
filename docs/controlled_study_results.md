# Controlled multi-seed study

**Superseded GRU interpretation, November 9:** GRU rows and GRU-related intervals
below are archived measurements from an affected fused training path. They do
not establish a temporal-architecture disadvantage or GRU graph advantage. See
the [current report and CPU audit](experiments.md#our-grus-versus-the-partner-inspired-gru).
Static-model scores and original metric files remain unchanged.

All 100 prespecified jobs completed. Scores are averaged across seeds, not ensembled predictions.

| Variant | Validation paired MAE | Test paired MAE | Test SD |
|---|---:|---:|---:|
| gat_jao_real | 21.1308 | 21.1144 | 0.0653 |
| gat_load_real | 21.0773 | 21.0854 | 0.1336 |
| gat_recent_real | 21.1218 | 20.9947 | 0.0914 |
| gat_rich_real | 20.9830 | 20.9737 | 0.1265 |
| gat_rich_self | 21.2668 | 21.0637 | 0.0805 |
| gat_rich_shuffled_rewire901 | 21.1467 | 21.4291 | 0.0746 |
| gat_rich_shuffled_rewire902 | 21.2067 | 21.5178 | 0.1153 |
| gat_rich_shuffled_rewire903 | 21.2761 | 21.4100 | 0.0937 |
| gat_weather_real | 21.0556 | 20.9041 | 0.0878 |
| gru_gat_rich_real | 23.6200 | 23.5875 | 0.5139 |
| gru_mlp_rich_none | 24.3701 | 24.4997 | 1.2160 |
| mlp_jao_none | 21.5113 | 21.1934 | 0.0747 |
| mlp_load_none | 21.5116 | 21.2952 | 0.1602 |
| mlp_recent_none | 21.4864 | 21.2540 | 0.0809 |
| mlp_rich_none | 21.4429 | 21.1853 | 0.2059 |
| mlp_weather_none | 21.4195 | 21.1424 | 0.0819 |
| pooled_xgb_rich_none | 20.7362 | 21.0584 | 0.0135 |
| xgb_neighbours_none | 20.8485 | 21.8438 | 0.0993 |
| xgb_recent_none | 21.2580 | 21.8689 | 0.0417 |
| xgb_rich_none | 21.2348 | 22.1096 | 0.0279 |

## Validation uncertainty

Paired calendar-week blocks are resampled jointly across countries; intervals condition on fitted models.

- mlp_rich_none_to_gat_rich_real: improvement 0.4599, 95% interval [0.2004, 0.8605].
- gat_rich_self_to_gat_rich_real: improvement 0.2838, 95% interval [0.1300, 0.4601].
- gru_mlp_rich_none_to_gru_gat_rich_real: improvement 0.7501, 95% interval [0.5567, 0.9432].
- gat_rich_shuffled_rewire901_to_gat_rich_real: improvement 0.1637, 95% interval [-0.0401, 0.3348].
- gat_rich_shuffled_rewire902_to_gat_rich_real: improvement 0.2237, 95% interval [0.0529, 0.3829].
- gat_rich_shuffled_rewire903_to_gat_rich_real: improvement 0.2931, 95% interval [0.1451, 0.4497].
- mlp_recent_none_to_mlp_load_none: improvement -0.0252, 95% interval [-0.1127, 0.0262].
- mlp_recent_none_to_mlp_weather_none: improvement 0.0670, 95% interval [-0.0804, 0.1845].
- mlp_recent_none_to_mlp_jao_none: improvement -0.0249, 95% interval [-0.2436, 0.0987].
- mlp_recent_none_to_mlp_rich_none: improvement 0.0435, 95% interval [-0.3047, 0.2566].
- gat_recent_real_to_gat_load_real: improvement 0.0445, 95% interval [-0.0514, 0.1658].
- gat_recent_real_to_gat_weather_real: improvement 0.0662, 95% interval [0.0014, 0.1332].
- gat_recent_real_to_gat_jao_real: improvement -0.0090, 95% interval [-0.0690, 0.0471].
- gat_recent_real_to_gat_rich_real: improvement 0.1388, 95% interval [0.0143, 0.2681].

## Limits

- Exploratory test period inspected previously
- Conditional source publication vintages
- Weekly bootstrap conditions on fitted models; seed/rewiring summaries are separate
- Means are scores, not ensembles

Regime thresholds use each zone's training-only 5th/95th percentiles. Country grouping retains unresolved identities. Raw predictions, complete job metadata and hashes remain under runs/controlled_study/.
