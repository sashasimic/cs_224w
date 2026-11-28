"""Run the earlier baselines (xgb, gru_mlp) on cutoff-safe inputs, without changing their code.

    model: asof
    base_model: xgb            # or gru_mlp
    feature_set: own_asof      # OWN with the cutoff-safe `_asof` weather instead of `_lag24`
    ...                        # every other key is passed to the base model unchanged

The earlier models read `tables.nodes()` (zone_nodes_hourly) and `tables.edge_pairs()` directly. Two of
their inputs are wrong:
  - the `<weather>_lag24` columns of OWN are partly observed after the D-1 10:00 UTC cutoff (afternoon
    hours of day D); features/v1 has `<weather>_asof` (lag 24 where observed, else lag 48);
  - `edge_pairs()` includes 6 directed DE_LU edges (to HU, IT_NORD, SI) with no flow since the Oct 2018
    DE-AT split; `graphs.active_pairs()` drops them (xgb neighbour means, gru_mlp spread loss).

While the base model trains and predicts, this module swaps those two functions: `tables.nodes()` returns
the same table plus the four `_asof` columns from features/v1, and `tables.edge_pairs()` returns the
active pairs. Both are restored afterwards, so the scoring in epgnn.run (targets, spread metric) is the
same as for every other run.
"""
import importlib
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import pandas as pd

from epgnn.data import features as F
from epgnn.data import tables
from epgnn.pipeline import graphs
from epgnn.pipeline.tensors import FEATURES_DIR

WEATHER = ("temp_pw", "temp_anom", "wind_cf", "solar_cf")
ASOF = {f"{c}_lag24": f"{c}_asof" for c in WEATHER}
F.FEATURE_SETS.setdefault("own_asof", [ASOF.get(c, c) for c in F.OWN])
F.FEATURE_SETS.setdefault("own+jao_asof", [ASOF.get(c, c) for c in F.FEATURE_SETS["own+jao"]])


@lru_cache(maxsize=1)
def nodes_asof() -> pd.DataFrame:
    n = tables.nodes()
    cols = list(ASOF.values())
    zf = pd.read_parquet(FEATURES_DIR / "zone_features_hourly.parquet", columns=["zone", "ts_utc"] + cols)
    zf["ts_utc"] = zf.ts_utc.astype(n.ts_utc.dtype)
    out = n.merge(zf, on=["zone", "ts_utc"], how="left", validate="one_to_one")
    assert len(out) == len(n) and (out.zone.to_numpy() == n.zone.to_numpy()).all()
    for lag24, asof in ASOF.items():                       # coverage check: asof is missing only where lag24/48 are
        extra = out[asof].isna() & n[lag24].notna()
        assert extra.mean() < 0.01, f"{asof} missing on {extra.mean():.1%} of rows where {lag24} exists"
    return out


@contextmanager
def safe_inputs():
    old_nodes, old_pairs = tables.nodes, tables.edge_pairs
    n = nodes_asof()
    tables.nodes = lambda: n                               # type: ignore[assignment]
    tables.edge_pairs = graphs.active_pairs                # type: ignore[assignment]
    try:
        yield
    finally:
        tables.nodes, tables.edge_pairs = old_nodes, old_pairs


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    base = importlib.import_module(f"epgnn.models.{cfg['base_model']}")
    leaky = set(ASOF) & set(F.day_d_columns(cfg.get("feature_set", "own")))   # price_lag24 is fine
    assert not leaky, f"weather {sorted(leaky)} is partly after the cutoff; use own_asof or own+jao_asof"
    with safe_inputs():
        out = base.run(cfg, fit_splits, eval_splits)
    run.info = {**getattr(base.run, "info", {}), "base_model": cfg["base_model"],  # type: ignore[attr-defined]
                "inputs": "features/v1 _asof weather; graphs.active_pairs edges"}
    return out
