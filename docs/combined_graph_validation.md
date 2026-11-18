# Combined graph validation

45 jobs used the same 429,240 zone-hour keys from April 2024 through March 2025. MAE is in EUR/MWh and averaged over five seeds. Skill is the reduction in MAE versus weekly persistence on matched rows. JAO, weather, and archived load inputs are conditional.

| Model | Fit begins | All MAE | CORE MAE | France MAE | All skill | CORE skill | Spread MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| sql_price | 2016-01-01 | 20.717 ± 0.000 | 23.650 | 20.656 | 31.4% | 30.2% | 16.513 |
| sql_full | 2016-01-01 | 20.681 ± 0.000 | 23.674 | 20.730 | 31.5% | 30.2% | 16.585 |
| relational_local | 2016-01-01 | 22.252 ± 0.294 | 25.101 | 21.642 | 26.3% | 26.0% | 17.457 |
| relational_cross | 2016-01-01 | 22.235 ± 0.309 | 25.081 | 21.620 | 26.3% | 26.0% | 17.446 |
| physical | 2022-06-09 | 21.662 ± 0.246 | 24.769 | 21.520 | 28.2% | 26.9% | 16.879 |
| uniform | 2022-06-09 | 22.045 ± 0.148 | 25.353 | 21.193 | 26.9% | 25.2% | 17.107 |
| learned | 2022-06-09 | 22.041 ± 0.254 | 25.191 | 21.133 | 27.0% | 25.7% | 17.290 |
| margin | 2022-06-09 | 22.040 ± 0.255 | 25.190 | 21.134 | 27.0% | 25.7% | 17.288 |
| shuffled_margin | 2022-06-09 | 22.041 ± 0.254 | 25.193 | 21.131 | 27.0% | 25.7% | 17.290 |

## Paired contrasts

Positive values favor the first model. Intervals resample Brussels calendar weeks jointly across markets.

| Cohort | Model | Control | MAE improvement | 95% interval |
|---|---|---|---:|---|
| all | relational_cross | sql_full | -1.5533 | [-2.4879, -0.9631] |
| all | relational_cross | relational_local | +0.0173 | [-0.0261, +0.0430] |
| all | learned | physical | -0.3781 | [-0.4648, -0.2968] |
| all | learned | uniform | +0.0042 | [-0.0873, +0.0805] |
| all | margin | learned | +0.0008 | [-0.0013, +0.0031] |
| all | margin | shuffled_margin | +0.0009 | [-0.0002, +0.0021] |
| core | relational_cross | sql_full | -1.4075 | [-1.8566, -0.9386] |
| core | relational_cross | relational_local | +0.0196 | [+0.0082, +0.0303] |
| core | learned | physical | -0.4220 | [-0.5827, -0.2702] |
| core | learned | uniform | +0.1622 | [+0.0776, +0.2303] |
| core | margin | learned | +0.0016 | [-0.0023, +0.0054] |
| core | margin | shuffled_margin | +0.0033 | [+0.0014, +0.0053] |

Peak allocated GPU tensors were 1.74 GiB.
Machine-readable summaries include mean weights and top-five neighbor stability across seeds.
