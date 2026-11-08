"""The prespecified 100-job matrix and its fixed feature groups."""

from dataclasses import asdict, dataclass

from epgnn.experiment_suite.data import RECENT, RICH, FEATURES

FEATURE_SETS = {
    "recent": RECENT,
    "load": RECENT + ["load_fc"],
    "weather": RECENT
    + ["temp_pw_lag48", "temp_anom_lag48", "wind_cf_lag48", "solar_cf_lag48"],
    "jao": RECENT
    + ["n_presolved_lag48", "min_ram_ratio_lag48", "jao_domain_present_lag48"],
    "rich": RICH,
    "neighbours": FEATURES,
}


@dataclass(frozen=True)
class Job:
    model: str
    features: str
    seed: int
    graph: str = "none"
    rewiring_seed: int | None = None

    @property
    def variant(self):
        suffix = (
            f"_rewire{self.rewiring_seed}" if self.rewiring_seed is not None else ""
        )
        return f"{self.model}_{self.features}_{self.graph}{suffix}"

    @property
    def name(self):
        return f"{self.variant}_seed{self.seed}"

    def record(self):
        return asdict(self)


def jobs(config):
    result = []
    for seed in config["seeds"]:
        result.extend(
            Job("xgb", group, seed) for group in ["recent", "rich", "neighbours"]
        )
        result.append(Job("pooled_xgb", "rich", seed))
        for group in ["recent", "load", "weather", "jao", "rich"]:
            result.extend([Job("mlp", group, seed), Job("gat", group, seed, "real")])
        result.append(Job("gat", "rich", seed, "self"))
        result.extend(
            Job("gat", "rich", seed, "shuffled", rewire)
            for rewire in config["rewiring_seeds"]
        )
        result.extend(
            [Job("gru_mlp", "rich", seed), Job("gru_gat", "rich", seed, "real")]
        )
    if len({job.name for job in result}) != len(result):
        raise ValueError("Study contains duplicate jobs")
    return result


def feature_indices(job):
    return [FEATURES.index(name) for name in FEATURE_SETS[job.features]]
