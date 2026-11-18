# CORE constraint GAT validation results

[Schema and experimental protocol](relational_forecast.md)

Four variants and 5 seeds used the same 429,240 zone-hour keys from April 2024 through March 2025. Values are mean validation MAE across seeds in EUR/MWh. Skill uses the same valid hours as weekly persistence. JAO publication vintages remain unverified.

| Variant | All markets MAE | CORE-covered MAE | France MAE | CORE skill vs. persistence |
|---|---:|---:|---:|---:|
| Interconnector GAT | 21.64 ± 0.12 | 24.57 ± 0.07 | 22.16 | 27.5% |
| Interconnector GAT + flat JAO | 21.66 ± 0.14 | 25.08 ± 0.27 | 27.51 | 26.0% |
| Bipartite GAT + flat JAO | 22.28 ± 0.20 | 24.62 ± 0.20 | 22.92 | 27.4% |
| Bipartite GAT + flat JAO, shuffled links | 22.31 ± 0.19 | 24.65 ± 0.20 | 23.34 | 27.3% |

## Paired CORE comparisons

Positive differences favor the first model. Intervals use a paired Brussels calendar-week bootstrap and condition on the fitted models.

- Bipartite GAT + flat JAO versus Interconnector GAT + flat JAO +0.454 EUR/MWh with 95% interval [-0.015, +0.870].
- Bipartite GAT + flat JAO versus Bipartite GAT + flat JAO, shuffled links +0.026 EUR/MWh with 95% interval [-0.087, +0.135].
- Interconnector GAT + flat JAO versus Interconnector GAT -0.504 EUR/MWh with 95% interval [-0.827, -0.181].

The CORE-covered cohort has 105,120 valid targets. Across all markets there are 428,427 valid targets. Weekly persistence MAE on its matched hours is 33.90 on CORE coverage and 30.18 across all markets. Peak GPU allocation was 0.37 GiB.

Detailed [zone scores](../results/core_constraint_validation.csv) and [comparison metadata](../results/core_constraint_validation.json) are available.
