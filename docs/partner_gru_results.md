# Partner-inspired residual GRU: conditional canonical-data run

Fingerprint: `9b5aaae786ce06f14abf0b82c96882638fa5a306f9a9ec064db16a2b23b235aa`. Seed 0; one full adaptation, not five-seed evidence or an exact partner reproduction.

ERA5 is reanalysis, not archived operational forecasts. Recent ERA5 observations have a known publication-latency conflict with the noon contract. Load and flow vintages and anomaly climatology remain unverified. These are retrospective conditional scores.

The initial fused ROCm GRU run was rejected after a CPU cross-check exposed a long-sequence backend discrepancy. This run uses native GPU RNN operations, verified against independent CPU float64 predictions; see results/partner_gru_backend_audit.json. Original failed checkpoints remain archived and are not scientific comparison scores.

Inputs retain original missingness and historical zone identities. The baseline used in residual reconstruction falls back from weekly to guarded daily price, then zero; targets are never filled. Training uses multivariate 168-hour histories, a 64-unit GRU, 128-unit ReLU head, day batches, and L1 residual error plus 0.5 times cross-zone residual-contrast error. This loss is not raw-price spread MAE. Early stopping uses the same residual-plus-spread validation objective.

| Split | All-target MAE | All-target RMSE | Matched MAE | Weekly persistence | rMAE |
|---|---:|---:|---:|---:|---:|
| test | 21.0677 | 47.0222 | 20.9817 | 29.1334 | 0.7202 |
| val | 20.3347 | 40.1888 | 20.3192 | 30.1822 | 0.6732 |

MAE/RMSE and separately saved physical price-spread MAE are in EUR/MWh. Target keys, values and masks match the existing 49-market evaluation. Missing weather does not remove targets. October 2026 remains unassigned.

Important adaptations relative to the partner are documented in the experiment guide: lag48 day weather, literal local noon, timestamp joins, canonical geography, conservative training-pair net imports and safe NaN masking. The partner's published 20.40 MAE uses a different 43-zone cohort and protocol; it is not directly ranked against this result. Prior controlled-study GRU scores used a different architecture, inputs and objective.

## Validation learning curve

| Epoch | Training residual + spread | Validation residual + spread | Seconds |
|---|---:|---:|---:|
| 1 | 0.21668 | 0.32074 | 10.92 |
| 2 | 0.18972 | 0.30477 | 10.05 |
| 3 | 0.18265 | 0.30161 | 10.00 |
| 4 | 0.17922 | 0.30029 | 9.90 |
| 5 | 0.17734 | 0.30057 | 9.92 |
| 6 | 0.17626 | 0.29641 | 9.96 |
| 7 | 0.17501 | 0.29312 | 9.97 |
| 8 | 0.17404 | 0.29543 | 9.88 |
| 9 | 0.17393 | 0.29358 | 9.91 |
| 10 | 0.17260 | 0.29240 | 9.81 |
| 11 | 0.17237 | 0.29129 | 9.87 |
| 12 | 0.17276 | 0.29075 | 9.89 |
| 13 | 0.17194 | 0.29025 | 9.81 |
| 14 | 0.17133 | 0.29070 | 9.89 |
| 15 | 0.17116 | 0.29099 | 9.93 |
| 16 | 0.17062 | 0.29115 | 9.91 |
| 17 | 0.17086 | 0.29025 | 9.96 |
| 18 | 0.17041 | 0.29164 | 9.94 |
| 19 | 0.16942 | 0.28986 | 9.91 |
| 20 | 0.16993 | 0.29118 | 9.95 |
| 21 | 0.16890 | 0.28925 | 9.94 |
| 22 | 0.16907 | 0.28869 | 9.92 |
| 23 | 0.16881 | 0.28965 | 9.92 |
| 24 | 0.16823 | 0.28852 | 9.95 |
| 25 | 0.16763 | 0.28928 | 9.91 |
| 26 | 0.16762 | 0.28801 | 9.87 |
| 27 | 0.16726 | 0.28850 | 9.90 |
| 28 | 0.16685 | 0.29106 | 9.97 |
| 29 | 0.16704 | 0.28612 | 9.99 |
| 30 | 0.16622 | 0.28687 | 9.93 |
| 31 | 0.16616 | 0.29023 | 9.87 |
| 32 | 0.16592 | 0.28866 | 9.90 |
| 33 | 0.16542 | 0.28848 | 9.90 |
| 34 | 0.16536 | 0.28913 | 9.91 |
| 35 | 0.16529 | 0.28777 | 9.92 |
| 36 | 0.16520 | 0.28772 | 9.89 |
| 37 | 0.16484 | 0.28723 | 9.90 |
