"""Model families. Each module exposes

    run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]

returning, for every eval split, predicted prices in EUR/MWh shaped like Samples.y (n, H).
The runner (epgnn.run) builds the Samples, scores against weekly persistence and writes results/.

persistence  checkpoint 1a   price one week earlier
xgb          checkpoints 1b, 2   gradient boosting per zone, feature set from the config
gat          checkpoints 3, 3', 4a   GRU + GATv2 on the zone graph (interconnects / shuffled / + JAO summaries)
hetero       checkpoint 4b   GRU + relational GATv2 over zone and constraint nodes (HeteroConv)
encoders     the GRU history encoder shared by gat and hetero
"""
