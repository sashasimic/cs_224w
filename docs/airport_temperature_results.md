# Airport temperature experiment — full

Fingerprint: `8be91e76858aa68c606def353b59030b5b96c6ef94904da15287b663779e4d6b`. 45 verified fits; five seeds across the unchanged 49-market cohort.

Each country's largest airport by 2023 annual passengers is fixed before fitting. Temperature is a local proxy, not a national or bidding-zone average. Regional zones share the national airport; DE_LU and IE_SEM are explicitly combined-market proxies. DE_AT_LU and unresolved EIC identities retain missing airport inputs.

At literal D−1 noon Europe/Brussels, use observations at least two hours old and at most six hours old for latest temperature. When receipt timestamps exist they must also precede origin. The IEM export used here lacks original receipt and revision timestamps: **all weather variants remain conditional**. The same inputs are frozen across every delivery-day hour, including DST days.

Features: calendar and guarded 24h/168h prices; optionally latest temperature in Celsius plus age; optionally trailing 24h hourly-sampled mean plus coverage count (mean requires 18 hours). No load, ERA5, JAO or realized-flow predictors. XGBoost is pooled, without country IDs; MLP and GAT share the same feature groups. Target asinh(price/50), L1 objective, train-only neural preprocessing, and validation early stopping match the controlled study.

Fit: January 2016–March 2024; validation: April 2024–March 2025; test: April 2025–September 2026. October 2026 remains unassigned. Missing weather never removes a target. The test period was explored before this experiment; results are not a fresh blind evaluation.

## Reviewed interpretation

The full run finished on 8 November 2026 at 20:48 Europe/Zurich, with exit code 0, after approximately 64 minutes. All 45 fits passed saved-artifact verification. The following subgroup and contribution checks are descriptive post-hoc analyses; they do not change the evaluation cohort, models or hyperparameters.

**The airport proxy gives small, inconsistent improvements.** For latest temperature, validation-week intervals include zero for pooled XGBoost and MLP. GAT's interval is narrowly positive, but these exploratory intervals are unadjusted for multiple comparisons. Adding the trailing mean does not establish a clear advantage over the latest reading.

Test MAE below uses all valid labels within each fixed geography subgroup. Each number is a five-seed mean, weighted by target hours within a seed.

| Model | Subgroup | Price-only | + Latest | + History |
|---|---|---:|---:|---:|
| pooled_xgb | all | 21.4299 | 21.0889 | 21.0841 |
| pooled_xgb | airport_mapped | 20.7837 | 20.7375 | 20.7284 |
| pooled_xgb | airport_unmapped | 26.7047 | 23.9568 | 23.9882 |
| mlp | all | 21.2507 | 21.1216 | 21.1432 |
| mlp | airport_mapped | 21.1845 | 21.1000 | 21.1153 |
| mlp | airport_unmapped | 21.7908 | 21.2985 | 21.3712 |
| gat | all | 20.9838 | 20.9536 | 20.9516 |
| gat | airport_mapped | 20.9365 | 20.9280 | 20.9253 |
| gat | airport_unmapped | 21.3700 | 21.1626 | 21.1661 |

**Why the pooled XGBoost headline needs care:** `ENTSOE:10Y1001A1001A893` accounts for 0.2978 EUR/MWh of its 0.3410 overall test improvement with latest temperature—about 87.3%. This market has no resolved airport mapping, and all four of its local airport inputs were verified to be missing throughout the cached calendar. Its changed predictions therefore cannot be attributed to a local temperature reading: adding features changes the jointly fitted pooled model, including its learned missing-value routes. Unmapped-node GAT predictions can additionally change through messages from mapped neighbours.

France worsens with this airport proxy for all three models. Germany/Luxembourg shows small improvements with latest temperature; effects vary across the other markets. These results do not establish that national or regional temperature is useless, nor that a single airport adequately represents a decentralised country.

GAT remains ahead of the matched pooled MLP overall with and without airport features. France remains a counterexample. The earlier rich-feature results used additional conditional sources and are context, not a direct test of observed temperature versus ERA5. No training was rerun or selected using these post-hoc checks.

The review tables are saved as `airport_temperature_full_geographic_subgroups.csv` and `airport_temperature_full_market_contributions.csv`; `scripts/experiments/review_airport_results.py` reproduces them.

## Overall errors

MAE/RMSE are EUR/MWh; ± is sample standard deviation across training seeds. Primary metrics retain every valid price target. Paired MAE additionally requires weekly persistence history.

| Variant | Validation MAE | Test MAE | Test RMSE | Test paired MAE |
|---|---:|---:|---:|---:|
| gat_airport_history_real | 21.0664 ± 0.0609 | 20.9516 ± 0.0420 | 34.7894 ± 0.2288 | 20.9551 ± 0.0433 |
| gat_airport_latest_real | 21.0613 ± 0.1373 | 20.9536 ± 0.1520 | 35.0515 ± 1.1197 | 20.9584 ± 0.1506 |
| gat_recent_real | 21.1416 ± 0.0883 | 20.9838 ± 0.0760 | 35.3302 ± 0.7389 | 20.9900 ± 0.0755 |
| mlp_airport_history_none | 21.4473 ± 0.1093 | 21.1432 ± 0.1173 | 35.4303 ± 0.3418 | 21.1466 ± 0.1179 |
| mlp_airport_latest_none | 21.4517 ± 0.0599 | 21.1216 ± 0.0656 | 35.3044 ± 0.2012 | 21.1247 ± 0.0656 |
| mlp_recent_none | 21.4859 ± 0.1120 | 21.2507 ± 0.0824 | 35.9355 ± 0.9042 | 21.2540 ± 0.0809 |
| pooled_xgb_airport_history_none | 20.8096 ± 0.0178 | 21.0841 ± 0.0155 | 37.8838 ± 0.3300 | 21.0892 ± 0.0163 |
| pooled_xgb_airport_latest_none | 20.8246 ± 0.0222 | 21.0889 ± 0.0259 | 37.7774 ± 0.4345 | 21.0927 ± 0.0269 |
| pooled_xgb_recent_none | 20.8406 ± 0.0089 | 21.4299 ± 0.0105 | 43.5951 ± 0.2861 | 21.4360 ± 0.0106 |

## Paired validation-week comparisons

Positive means the second variant improves MAE. Bootstrap resamples shared calendar weeks across countries; it conditions on the fitted models and does not include retraining uncertainty.

| Comparison | Improvement | 95% interval |
|---|---:|---:|
| pooled_xgb_recent_none_to_pooled_xgb_airport_latest_none | 0.0156 | [-0.0459, 0.0707] |
| pooled_xgb_recent_none_to_pooled_xgb_airport_history_none | 0.0304 | [-0.0296, 0.0873] |
| pooled_xgb_airport_latest_none_to_pooled_xgb_airport_history_none | 0.0147 | [-0.0076, 0.0380] |
| mlp_recent_none_to_mlp_airport_latest_none | 0.0340 | [-0.1560, 0.1675] |
| mlp_recent_none_to_mlp_airport_history_none | 0.0383 | [-0.1338, 0.1649] |
| mlp_airport_latest_none_to_mlp_airport_history_none | 0.0043 | [-0.0310, 0.0410] |
| gat_recent_real_to_gat_airport_latest_real | 0.0800 | [0.0036, 0.1535] |
| gat_recent_real_to_gat_airport_history_real | 0.0751 | [-0.0280, 0.1604] |
| gat_airport_latest_real_to_gat_airport_history_real | -0.0049 | [-0.0522, 0.0369] |
| mlp_recent_none_to_gat_recent_real | 0.3396 | [0.1717, 0.4929] |
| mlp_airport_latest_none_to_gat_airport_latest_real | 0.3856 | [0.2474, 0.5338] |
| mlp_airport_history_none_to_gat_airport_history_real | 0.3764 | [0.2432, 0.5141] |

## Country results

Each entry is test MAE averaged across seeds. Multi-zone countries pool absolute errors over their target hours; prices are never averaged into a national target.

| Country/market | Model | Price-only | + Latest | + History |
|---|---|---:|---:|---:|
| AT | pooled_xgb | 19.8071 | 19.9021 | 19.8758 |
| AT | mlp | 20.5702 | 20.4925 | 20.5206 |
| AT | gat | 19.6378 | 19.5206 | 19.4653 |
| BE | pooled_xgb | 22.4723 | 22.4615 | 22.4595 |
| BE | mlp | 23.0395 | 22.9912 | 22.9636 |
| BE | gat | 22.6190 | 22.5341 | 22.6633 |
| BG | pooled_xgb | 21.7894 | 21.7315 | 21.7437 |
| BG | mlp | 22.2616 | 22.1344 | 22.1330 |
| BG | gat | 21.3187 | 21.1032 | 21.1053 |
| CH | pooled_xgb | 14.4777 | 14.5954 | 14.5903 |
| CH | mlp | 15.2103 | 15.1116 | 15.1867 |
| CH | gat | 15.3130 | 15.4663 | 15.3794 |
| CZ | pooled_xgb | 21.6119 | 21.5743 | 21.6057 |
| CZ | mlp | 22.3972 | 22.2619 | 22.2559 |
| CZ | gat | 21.5194 | 21.5266 | 21.5347 |
| DE_LU | pooled_xgb | 24.3950 | 24.3868 | 24.3670 |
| DE_LU | mlp | 24.9204 | 24.8837 | 24.8604 |
| DE_LU | gat | 23.8700 | 23.7351 | 23.8582 |
| DK | pooled_xgb | 26.8172 | 26.7640 | 26.7664 |
| DK | mlp | 27.1018 | 27.0711 | 27.0620 |
| DK | gat | 26.2648 | 26.2560 | 26.3838 |
| EE | pooled_xgb | 39.7198 | 39.4800 | 39.4633 |
| EE | mlp | 39.7859 | 39.8195 | 39.8368 |
| EE | gat | 39.1359 | 39.3646 | 39.2227 |
| ENTSOE:10Y1001A1001A885 | pooled_xgb | 15.0188 | 14.8400 | 14.8030 |
| ENTSOE:10Y1001A1001A885 | mlp | 15.4072 | 15.2881 | 15.3753 |
| ENTSOE:10Y1001A1001A885 | gat | 15.2597 | 15.0752 | 15.2482 |
| ENTSOE:10Y1001A1001A893 | pooled_xgb | 52.4181 | 37.0944 | 37.2990 |
| ENTSOE:10Y1001A1001A893 | mlp | 24.0221 | 21.1984 | 21.3313 |
| ENTSOE:10Y1001A1001A893 | gat | 25.1589 | 23.2080 | 22.2652 |
| ENTSOE:10Y1001C--000182 | pooled_xgb | 18.5990 | 18.7240 | 18.7275 |
| ENTSOE:10Y1001C--000182 | mlp | 18.6281 | 18.6884 | 18.7555 |
| ENTSOE:10Y1001C--000182 | gat | 21.2281 | 21.0604 | 21.3152 |
| ENTSOE:10Y1001C--00100H | pooled_xgb | 28.4527 | 28.4828 | 28.4989 |
| ENTSOE:10Y1001C--00100H | mlp | 28.5163 | 28.6113 | 28.6676 |
| ENTSOE:10Y1001C--00100H | gat | 26.4781 | 26.8453 | 27.2216 |
| ENTSOE:10YAL-KESH-----5 | pooled_xgb | 28.0914 | 28.1359 | 28.1470 |
| ENTSOE:10YAL-KESH-----5 | mlp | 28.0687 | 28.1673 | 28.2161 |
| ENTSOE:10YAL-KESH-----5 | gat | 25.9493 | 26.3127 | 26.6569 |
| ENTSOE:50Y0JVU59B4JWQCU | pooled_xgb | 13.8553 | 13.8145 | 13.7967 |
| ENTSOE:50Y0JVU59B4JWQCU | mlp | 14.1641 | 14.1249 | 14.1686 |
| ENTSOE:50Y0JVU59B4JWQCU | gat | 14.3031 | 14.5518 | 14.4773 |
| ES | pooled_xgb | 17.5594 | 17.8786 | 17.9113 |
| ES | mlp | 17.4313 | 17.5584 | 17.5884 |
| ES | gat | 17.9301 | 18.2560 | 18.2334 |
| FI | pooled_xgb | 30.2099 | 29.9840 | 29.9632 |
| FI | mlp | 30.5902 | 30.4477 | 30.3970 |
| FI | gat | 29.4779 | 29.5026 | 29.3926 |
| FR | pooled_xgb | 21.5211 | 21.6991 | 21.6895 |
| FR | mlp | 21.7670 | 21.8579 | 21.8726 |
| FR | gat | 22.9379 | 23.7470 | 23.5243 |
| GR | pooled_xgb | 20.8172 | 20.9265 | 20.9545 |
| GR | mlp | 20.9872 | 20.9847 | 20.9741 |
| GR | gat | 20.5473 | 20.5755 | 20.5183 |
| HR | pooled_xgb | 20.5426 | 20.4185 | 20.4597 |
| HR | mlp | 21.2566 | 21.1371 | 21.1675 |
| HR | gat | 20.5279 | 20.3191 | 20.3505 |
| HU | pooled_xgb | 22.3924 | 22.2619 | 22.2946 |
| HU | mlp | 23.2129 | 23.0558 | 23.1059 |
| HU | gat | 22.2603 | 21.9822 | 22.0370 |
| IE_SEM | pooled_xgb | 30.9897 | 30.9676 | 30.9460 |
| IE_SEM | mlp | 30.9682 | 31.1373 | 31.0680 |
| IE_SEM | gat | 31.4767 | 31.3664 | 31.4437 |
| IT | pooled_xgb | 13.5249 | 13.5286 | 13.4581 |
| IT | mlp | 14.0330 | 13.7861 | 13.8306 |
| IT | gat | 14.4702 | 14.3589 | 14.3994 |
| LT | pooled_xgb | 37.2992 | 37.1819 | 37.1043 |
| LT | mlp | 37.4812 | 37.5141 | 37.5313 |
| LT | gat | 36.2442 | 36.1083 | 36.1380 |
| LV | pooled_xgb | 37.2938 | 37.1063 | 36.9941 |
| LV | mlp | 37.4417 | 37.4822 | 37.4541 |
| LV | gat | 36.6797 | 36.6489 | 36.6386 |
| ME | pooled_xgb | 16.6053 | 16.5420 | 16.5327 |
| ME | mlp | 16.9570 | 17.0245 | 17.0189 |
| ME | gat | 16.1688 | 16.2608 | 16.0829 |
| MK | pooled_xgb | 18.4421 | 18.5596 | 18.5663 |
| MK | mlp | 18.7218 | 18.7418 | 18.7745 |
| MK | gat | 17.3171 | 17.1960 | 17.2282 |
| NL | pooled_xgb | 22.7522 | 22.6926 | 22.6790 |
| NL | mlp | 23.3398 | 23.2680 | 23.2583 |
| NL | gat | 23.0851 | 23.1895 | 23.3071 |
| NO | pooled_xgb | 13.4297 | 13.3173 | 13.3218 |
| NO | mlp | 13.6900 | 13.5888 | 13.6336 |
| NO | gat | 14.4216 | 14.7118 | 14.6312 |
| PL | pooled_xgb | 24.6095 | 24.4776 | 24.5398 |
| PL | mlp | 25.1974 | 25.1197 | 25.1404 |
| PL | gat | 24.6020 | 24.6684 | 24.7653 |
| PT | pooled_xgb | 17.3236 | 17.5388 | 17.5346 |
| PT | mlp | 17.2778 | 17.4596 | 17.5064 |
| PT | gat | 17.4250 | 17.5940 | 17.5261 |
| RO | pooled_xgb | 22.5066 | 22.3998 | 22.4239 |
| RO | mlp | 22.9634 | 22.8751 | 22.8826 |
| RO | gat | 22.1776 | 22.0005 | 22.1166 |
| RS | pooled_xgb | 21.7545 | 21.6591 | 21.7043 |
| RS | mlp | 22.1945 | 22.1240 | 22.1235 |
| RS | gat | 20.7029 | 20.4458 | 20.4297 |
| SE | pooled_xgb | 22.0743 | 21.8998 | 21.9075 |
| SE | mlp | 22.3629 | 22.2186 | 22.2068 |
| SE | gat | 22.1117 | 21.9189 | 21.8929 |
| SI | pooled_xgb | 20.9528 | 20.8583 | 20.8605 |
| SI | mlp | 21.7587 | 21.6383 | 21.6518 |
| SI | gat | 20.7042 | 20.3876 | 20.3598 |
| SK | pooled_xgb | 22.0128 | 21.9299 | 21.9901 |
| SK | mlp | 22.7805 | 22.6259 | 22.6648 |
| SK | gat | 22.1394 | 22.0694 | 22.1005 |

## Coverage and interpretation

See `results/airport_temperature_coverage.csv` for target-hour availability by zone, month and split; `airport_temperature_source_findings.csv` for duplicates, conflicts, corrections and missing values; and `airport_temperature_geography.csv` for every mapping, including unresolved identities. Corrected reports without original versions and conflicting timestamps are quarantined. No airport was changed because of coverage or prediction scores.

A benefit supports this inexpensive proxy. A null or negative result does not rule out regional temperature, population-weighted weather or genuine archived weather forecasts. This matrix tests the existing real graph against MLP; it adds no new self-loop or shuffled-graph controls.

## Previous controlled study

These are contextual paired test MAEs, not matched weather-source ablations. The earlier rich variants included ERA5, load and JAO with unresolved publication availability. Use the new price-only controls for the matched feature comparison.

| Previous variant | Paired test MAE |
|---|---:|
| gat_recent_real | 20.9947 |
| gat_rich_real | 20.9737 |
| mlp_recent_none | 21.2540 |
| mlp_rich_none | 21.1853 |
| pooled_xgb_rich_none | 21.0584 |

Full per-zone and country metrics, seed dispersions and regime diagnostics are saved alongside the JSON summary.

## Sources

- [Eurostat annual airport passenger transport](https://ec.europa.eu/eurostat/databrowser/view/avia_paoa/default/table?lang=en). Exact 2023 query, passenger counts and station identifiers are pinned in the mapping CSV.
- [Montenegro airport operator's 2023 national totals](https://montenegroairports.com/novosti/finalne-brojke-tokom-2023-godine-opsluzeno-preko-dva-i-po-miliona-putnika/).
- [IEM observation archive](https://mesonet.agron.iastate.edu/request/download.phtml) and [API documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=).
