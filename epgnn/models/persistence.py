"""Checkpoint 1a: weekly persistence. Forecast for hour h of day D = price at the same hour of D-7."""
import numpy as np

from epgnn.data import samples as S


def predict(s: S.Samples) -> np.ndarray:
    return S.day_inputs(s, ["price_lag168"])[:, :, 0]


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    return {split: predict(S.build(split, cfg.get("zones"))) for split in eval_splits}
