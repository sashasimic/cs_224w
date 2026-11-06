"""Checkpoints 1b and 2: XGBoost per zone (or pooled with the zone as a feature).

Config keys:
  feature_set   "own" (1b) or "own" + neighbours: true (2)   see epgnn.data.features
  per_zone      true = one model per zone; false = one model, zone id as a categorical feature
  params        xgboost parameters

Inputs per (zone, day, hour): the day-D feature columns of that hour, flattened; target: price (or the
asinh residual, see epgnn.data.scaling). Train on fit_splits, early-stop on "val" if present.
"""
import numpy as np


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    raise NotImplementedError("checkpoint 1b/2: see module docstring; owner A")
