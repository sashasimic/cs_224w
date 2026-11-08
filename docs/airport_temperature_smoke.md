# Airport temperature experiment — smoke

Fingerprint: `8be91e76858aa68c606def353b59030b5b96c6ef94904da15287b663779e4d6b`. 9 verified fits; France supervision, short training windows.

Each country's largest airport by 2023 annual passengers is fixed before fitting. Temperature is a local proxy, not a national or bidding-zone average. Regional zones share the national airport; DE_LU and IE_SEM are explicitly combined-market proxies. DE_AT_LU and unresolved EIC identities retain missing airport inputs.

At literal D−1 noon Europe/Brussels, use observations at least two hours old and at most six hours old for latest temperature. When receipt timestamps exist they must also precede origin. The IEM export used here lacks original receipt and revision timestamps: **all weather variants remain conditional**. The same inputs are frozen across every delivery-day hour, including DST days.

Features: calendar and guarded 24h/168h prices; optionally latest temperature in Celsius plus age; optionally trailing 24h hourly-sampled mean plus coverage count (mean requires 18 hours). No load, ERA5, JAO or realized-flow predictors. XGBoost is pooled, without country IDs; MLP and GAT share the same feature groups. Target asinh(price/50), L1 objective, train-only neural preprocessing, and validation early stopping match the controlled study.

Fit: January 2016–March 2024; validation: April 2024–March 2025; test: April 2025–September 2026. October 2026 remains unassigned. Missing weather never removes a target. The test period was explored before this experiment; results are not a fresh blind evaluation.

## Overall errors

MAE/RMSE are EUR/MWh; ± is sample standard deviation across training seeds. Primary metrics retain every valid price target. Paired MAE additionally requires weekly persistence history.

| Variant | Validation MAE | Test MAE | Test RMSE | Test paired MAE |
|---|---:|---:|---:|---:|
| gat_airport_history_real | 31.9908 | 35.2994 | 45.3854 | 35.2994 |
| gat_airport_latest_real | 41.6859 | 46.8306 | 64.0100 | 46.8306 |
| gat_recent_real | 41.2305 | 44.3444 | 58.2829 | 44.3444 |
| mlp_airport_history_none | 40.8497 | 42.8946 | 56.3636 | 42.8946 |
| mlp_airport_latest_none | 35.4765 | 37.9730 | 51.2949 | 37.9730 |
| mlp_recent_none | 35.1676 | 37.4350 | 50.7346 | 37.4350 |
| pooled_xgb_airport_history_none | 25.5959 | 29.9927 | 39.9366 | 29.9927 |
| pooled_xgb_airport_latest_none | 26.6940 | 30.3097 | 40.2774 | 30.3097 |
| pooled_xgb_recent_none | 25.2037 | 29.1350 | 39.7585 | 29.1350 |

## Paired validation-week comparisons

Positive means the second variant improves MAE. Bootstrap resamples shared calendar weeks across countries; it conditions on the fitted models and does not include retraining uncertainty.

| Comparison | Improvement | 95% interval |
|---|---:|---:|
| pooled_xgb_recent_none_to_pooled_xgb_airport_latest_none | -1.4903 | [-2.1506, -0.8985] |
| pooled_xgb_recent_none_to_pooled_xgb_airport_history_none | -0.3923 | [-0.9669, 0.1392] |
| pooled_xgb_airport_latest_none_to_pooled_xgb_airport_history_none | 1.0980 | [0.4358, 1.8155] |
| mlp_recent_none_to_mlp_airport_latest_none | -0.3090 | [-1.2531, 0.6605] |
| mlp_recent_none_to_mlp_airport_history_none | -5.6822 | [-7.5839, -3.8553] |
| mlp_airport_latest_none_to_mlp_airport_history_none | -5.3732 | [-6.9154, -3.9160] |
| gat_recent_real_to_gat_airport_latest_real | -0.4554 | [-3.2260, 2.0932] |
| gat_recent_real_to_gat_airport_history_real | 9.2397 | [5.7190, 12.7385] |
| gat_airport_latest_real_to_gat_airport_history_real | 9.6951 | [7.2748, 12.1924] |
| mlp_recent_none_to_gat_recent_real | -6.0629 | [-8.6467, -3.5583] |
| mlp_airport_latest_none_to_gat_airport_latest_real | -6.2094 | [-8.5937, -4.1262] |
| mlp_airport_history_none_to_gat_airport_history_real | 8.8589 | [5.4050, 12.3995] |

## Country results

Each entry is test MAE averaged across seeds. Multi-zone countries pool absolute errors over their target hours; prices are never averaged into a national target.

| Country/market | Model | Price-only | + Latest | + History |
|---|---|---:|---:|---:|
| FR | pooled_xgb | 29.1350 | 30.3097 | 29.9927 |
| FR | mlp | 37.4350 | 37.9730 | 42.8946 |
| FR | gat | 44.3444 | 46.8306 | 35.2994 |

## Coverage and interpretation

See `results/airport_temperature_coverage.csv` for target-hour availability by zone, month and split; `airport_temperature_source_findings.csv` for duplicates, conflicts, corrections and missing values; and `airport_temperature_geography.csv` for every mapping, including unresolved identities. Corrected reports without original versions and conflicting timestamps are quarantined. No airport was changed because of coverage or prediction scores.

A benefit supports this inexpensive proxy. A null or negative result does not rule out regional temperature, population-weighted weather or genuine archived weather forecasts. This matrix tests the existing real graph against MLP; it adds no new self-loop or shuffled-graph controls.

## Sources

- [Eurostat annual airport passenger transport](https://ec.europa.eu/eurostat/databrowser/view/avia_paoa/default/table?lang=en). Exact 2023 query, passenger counts and station identifiers are pinned in the mapping CSV.
- [Montenegro airport operator's 2023 national totals](https://montenegroairports.com/novosti/finalne-brojke-tokom-2023-godine-opsluzeno-preko-dva-i-po-miliona-putnika/).
- [IEM observation archive](https://mesonet.agron.iastate.edu/request/download.phtml) and [API documentation](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=).
