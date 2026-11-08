"""Fixed 45-fit matrix; feature additions never selected using test scores."""

from epgnn.controlled_study.design import Job

from .data import GROUPS


def jobs(config, smoke=False):
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    return [
        Job(model, group, seed, "real" if model == "gat" else "none")
        for seed in seeds
        for group in GROUPS
        for model in ["pooled_xgb", "mlp", "gat"]
    ]
