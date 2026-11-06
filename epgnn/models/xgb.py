"""Checkpoints 1b and 2: XGBoost on the gate-closure features of each hour.

One row per (zone, delivery day, hour): the day-D feature columns of that hour (epgnn.data.features), plus
the neighbours' means of the same columns for checkpoint 2. Target: asinh(price / 50) by default
("residual": asinh(price) - asinh(price one week earlier), as the torch models use). Trained on the
fit splits, early-stopped on "val" when it is among the eval splits, predictions returned in EUR/MWh.

Config keys:
  feature_set   name in epgnn.data.features.FEATURE_SETS (default "own")
  neighbours    true adds nb_<col> = mean over interconnected zones of each numeric feature (checkpoint 2)
  per_zone      true = one model per zone; false = one pooled model with the zone as a categorical feature
  target        "asinh" (default) | "residual"
  params        xgboost.XGBRegressor keyword arguments
  early_stopping_rounds   default 50 (used when "val" is an eval split)
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from epgnn.data import features as F
from epgnn.data import samples as S
from epgnn.data import scaling as SC
from epgnn.data import tables

DEFAULT_PARAMS = {"max_depth": 6, "n_estimators": 600, "learning_rate": 0.05, "subsample": 0.8,
                  "colsample_bytree": 0.8, "tree_method": "hist", "objective": "reg:absoluteerror"}


def _columns(cfg: dict) -> tuple[list[str], list[str]]:
    base = F.day_d_columns(cfg.get("feature_set", "own"))
    nb = [c for c in base if c not in F.CALENDAR] if cfg.get("neighbours") else []
    return base, nb


def _inputs(s: S.Samples, base: list[str], nb: list[str]) -> np.ndarray:
    """(n, H, n_features): base columns, then neighbour means, then lag168 for the residual target."""
    x = S.day_inputs(s, base + ["price_lag168"])
    if not nb:
        return x
    n = tables.nodes()
    means = F.neighbour_means(n, tables.edge_pairs(), nb).set_index(["zone", "ts_utc"])
    zone_rep = np.repeat(s.zone, S.H)
    ts = pd.to_datetime(s.hours.ravel(), utc=True)
    valid = s.hours.ravel() >= 0
    keys = pd.MultiIndex.from_arrays([zone_rep[valid], ts[valid]])
    flat = np.full((len(s) * S.H, len(nb)), np.nan, dtype="float32")
    flat[valid] = means.reindex(keys).to_numpy(dtype="float32")
    nbx = flat.reshape(len(s), S.H, len(nb))
    return np.concatenate([x[:, :, :-1], nbx, x[:, :, -1:]], axis=2)


def _target(y: np.ndarray, lag168: np.ndarray, kind: str) -> np.ndarray:
    return SC.residual_target(y, lag168) if kind == "residual" else SC.to_asinh(y)


def _to_price(t: np.ndarray, lag168: np.ndarray, kind: str) -> np.ndarray:
    return SC.residual_to_price(t, lag168) if kind == "residual" else SC.from_asinh(t)


def _flatten(s: S.Samples, x: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows = valid (sample, hour) cells. Returns X, target, lag168, zone-per-row."""
    lag = x[:, :, -1]
    t = _target(s.y, lag, kind)
    ok = s.mask & np.isfinite(t)
    X = x[:, :, :-1][ok]
    return X, t[ok], lag[ok], np.repeat(s.zone, S.H).reshape(len(s), S.H)[ok]


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    base, nb = _columns(cfg)
    kind = cfg.get("target", "asinh")
    params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
    zones = cfg.get("zones")

    fit = [S.build(sp, zones) for sp in fit_splits]
    fit_x = [_inputs(s, base, nb) for s in fit]
    Xtr, ytr, _, ztr = map(np.concatenate, zip(*[_flatten(s, x, kind) for s, x in zip(fit, fit_x)]))
    ev = {sp: S.build(sp, zones) for sp in eval_splits}
    ev_x = {sp: _inputs(s, base, nb) for sp, s in ev.items()}
    # early stopping on the validation split, when it is evaluated (its normal use: model selection)
    val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if "val" in ev:
        Xva, yva, _, zva = _flatten(ev["val"], ev_x["val"], kind)
        val = (Xva, yva, zva)

    def make(with_val: bool) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(**params, early_stopping_rounds=cfg.get("early_stopping_rounds", 50) if with_val else None)

    zone_list = sorted(np.unique(ztr))
    zid = {z: i for i, z in enumerate(zone_list)}

    def with_zone(X: np.ndarray, zz: np.ndarray) -> np.ndarray:
        return np.concatenate([X, np.array([zid.get(z, -1) for z in zz], dtype="float32")[:, None]], axis=1)

    models: dict[str, xgb.XGBRegressor] = {}
    if cfg.get("per_zone", True):
        for z in zone_list:
            has_val = bool(val) and bool((val[2] == z).any())      # e.g. GB has no prices after 2020
            m = make(has_val)
            tr = ztr == z
            kw: dict = {"verbose": False}
            if val and has_val:
                kw["eval_set"] = [(val[0][val[2] == z], val[1][val[2] == z])]
            m.fit(Xtr[tr], ytr[tr], **kw)
            models[z] = m
    else:
        m = make(val is not None)
        kw = {"verbose": False}
        if val:
            kw["eval_set"] = [(with_zone(val[0], val[2]), val[1])]
        m.fit(with_zone(Xtr, ztr), ytr, **kw)
        models["__pooled__"] = m

    out = {}
    for sp, s in ev.items():
        x = ev_x[sp]
        lag = x[:, :, -1]
        X = x[:, :, :-1].reshape(-1, x.shape[2] - 1)
        zz = np.repeat(s.zone, S.H)
        pred = np.full(len(X), np.nan, dtype="float32")
        if "__pooled__" in models:
            pred = models["__pooled__"].predict(with_zone(X, zz))
        else:
            for z, m in models.items():
                r = zz == z
                if r.any():
                    pred[r] = m.predict(X[r])
        out[sp] = _to_price(pred.reshape(len(s), S.H), lag, kind).astype("float32")
    return out
