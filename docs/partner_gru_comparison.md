# Residual GRU compared with previous experiments

All rows use the same canonical target cohort and chronological splits. Earlier rows are five-seed means of scores, not ensembles; the new GRU is seed 0 only. Features, architecture, training objective and checkpoint-selection objective differ, so this is context rather than a controlled architecture or weather ablation.

| Model | Seeds | Validation MAE | Test MAE | Test paired MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| partner_residual_gru_conditional | 1 | 20.3347 | 21.0677 | 20.9817 | 47.0222 |
| gat_rich_real | 5 | 20.9774 | 20.9690 | 20.9737 | 34.9183 |
| gat_weather_real | 5 | 21.0505 | 20.9010 | 20.9041 | 34.9309 |
| mlp_rich_none | 5 | 21.4420 | 21.1824 | 21.1853 | 35.9725 |
| pooled_xgb_rich_none | 5 | 20.7337 | 21.0554 | 21.0584 | 37.5695 |
| gat_airport_history_real | 5 | 21.0664 | 20.9516 | 20.9551 | 34.7894 |

The GRU reduces matched test MAE versus weekly persistence by 28.0% (20.9817 versus 29.1334 EUR/MWh). Its test MAE is close to the existing pooled XGBoost and static GAT results, while its higher RMSE shows larger squared errors. Its lower validation MAE does not establish a test improvement or a general architecture advantage.

Training stopped after 37 epochs; validation residual-plus-spread loss selected epoch 29. End-to-end fit/evaluation took 6.2 minutes. Test physical price-spread MAE is 16.1353 EUR/MWh across 92 training graph pairs. This physical metric differs from the transformed residual-contrast training loss.

The partner's published 20.40 MAE uses a different 43-zone population and pipeline; it cannot be directly ranked here. Our earlier controlled GRU+MLP score (24.50 paired test MAE) used a different recipe and fused RNN backend. Those older GRU checkpoints require their own backend checks before drawing temporal-model conclusions. Static MLP/GAT and XGBoost do not use this RNN path.

ERA5 is retrospective reanalysis, not archived weather forecasts. Load/flow vintages and anomaly climatology remain unresolved. This is a conditional experiment; it does not isolate a temperature benefit or certify deployment-time availability. Test results were previously explored, so this is not a fresh blind test.

Verification passed: 634,578 test labels, 633,885 paired labels, unchanged canonical target signatures, finite predictions, artifact hashes and recomputed zone/overall metrics. CPU float64 comparisons passed on actual validation/test batches; the full-run maximum difference was 0.00007480 EUR/MWh. Original rejected fused-run artifacts remain archived.

[Full run and learning curve](partner_gru_results.md) · [Method and backend audit](partner_gru.md). Reproduce this review with `.venv/bin/python scripts/experiments/review_partner_gru.py`. Per-zone scores are in `results/partner_gru_full_by_zone.csv`.
