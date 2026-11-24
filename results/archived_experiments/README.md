# Results from experiment branches

These JSON files are byte-for-byte copies of the saved summaries. The consolidated report can be rebuilt from this checkout without the sibling worktrees.

| Source commit | Saved results | Protocol in that commit |
|---|---|---|
| `997549a` | [TimesFM full validation](timesfm_full_validation.json), [JAO ablation](timesfm_jao_validation.json), [shared week](timesfm_validation_week.json) | `docs/timesfm_pilot.md` |
| `8b6f869` | [GraphPFN shared week](graphpfn_validation_week.json) | `docs/graphpfn_pilot.md` |
| `757da79` | [Partner per-zone XGBoost](c1_xgb_zone.json), [neighbor XGBoost](c2_xgb_neighbours.json), [GRU + MLP](c25_gru_mlp.json) | Configurations and model code in the partner snapshot |

The [consolidated source manifest](../../reports/experiments/sources.json) records full commit IDs, original paths and SHA-256 hashes.
