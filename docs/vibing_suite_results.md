# Full suite results — November 8, 2026

All five experiments completed successfully on `vibing`. The host launcher finished at
01:29:33 Europe/Zurich with exit code zero, approximately 6 minutes 22 seconds after launch.

## Matched persistence comparison

MAE is EUR/MWh, lower is better. These comparisons use the same 428,213 validation
and 633,885 test zone-hours with valid labels and finite weekly history across 49
eligible market identities. Other valid targets are retained in full model scoring.

| Method | Validation MAE | Test MAE |
|---|---:|---:|
| Weekly persistence | 30.1822 | 29.1334 |
| Recent-price XGBoost | 21.1839 | 21.8654 |
| Richer XGBoost | 21.2751 | 22.1323 |
| Neighbour XGBoost | 20.8312 | 21.8724 |
| Real-graph GAT | 20.8074 | 20.8279 |
| Shuffled-graph GAT | 21.2261 | 21.5421 |

Real GAT improves common-hour test MAE by 3.32% versus shuffled GAT,
4.75% versus recent-price XGBoost, and
28.51% versus persistence. The richer feature bundle does not
improve aggregate XGBoost performance here; the neighbour variant nearly matches
recent-price XGBoost on test and improves validation. This does not isolate the
effect of any individual weather/load/JAO feature.

## All valid target hours and runtime

Full model scores retain 428,427 validation and 634,578 test zone-hours, including
hours without weekly persistence. These are a slightly different population from
the paired comparison above.

| Method | Full test MAE | Full test RMSE | Runtime (seconds) |
|---|---:|---:|---:|
| Recent-price XGBoost | 21.8675 | 39.9540 | 17.9 |
| Richer XGBoost | 22.1319 | 40.7710 | 26.3 |
| Neighbour XGBoost | 21.8686 | 40.2669 | 27.4 |
| Real-graph GAT | 20.8235 | 34.3965 | 164.5 |
| Shuffled-graph GAT | 21.5422 | 37.1272 | 141.6 |

Real GAT trained for 30 epochs and selected epoch 28. Shuffled GAT stopped
after 26 epochs and selected epoch 21. Both used batch size 64 and approximately
0.185 GiB peak tensor allocation (excluding ROCm context and other processes).

## Interpretation and verification

The real graph beats the degree-preserving shuffled control on validation and
test in this seed. This supports further graph experiments; it does not establish
statistical significance, universal country gains, or superiority over every
possible tabular learner. Neural models pool zones, whereas XGBoost fits per zone.
Historical publication vintages remain unverified, and test has previously been
inspected. Results are conditional, exploratory and single-seed.

Artifact hashes were checked for all five completed experiments. All ten saved
prediction tables have unique zone/UTC keys and finite predictions; target keys,
values and masks match across methods. MAE recomputed directly from predictions
matches the saved aggregates within 1e-10 EUR/MWh. No model was retrained or tuned
during this verification.

Files: [full JSON](../results/vibing_suite_full.json),
[zone metrics](../results/vibing_suite_full_by_zone.csv),
[country/overall metrics](../results/vibing_suite_full_aggregate.csv).
These contain all-target and JAO-publication subsets, with explicit counts.
Combined markets and unresolved ENTSO-E identities remain distinct; 13 identities
lacking targets in a required partition remain explicitly unscored.

Large models, checkpoints and keyed predictions remain under
`runs/vibing_suite/fd694619ba8ce3bc23ebb2db54b50b559fb6fdff37ddc79799ca98544b4655f8/full/`.
See [the suite guide](vibing_suite.md) for provenance, settings and reuse commands.
