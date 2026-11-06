"""Checkpoints 1b and 2: XGBoost on the gate-closure features of each hour.

What the model is
-----------------
A gradient-boosted tree regressor that predicts the price of one zone in one hour from what is known
about that zone (and, for checkpoint 2, its neighbours) at gate closure, noon on the day before.
There is no graph and no sequence model: each (zone, day, hour) is an independent row.

How a row is built
------------------
For every sample (zone, delivery day D) from epgnn.data.samples and every hour h of D (up to 25 slots,
to cover DST days) the row holds the day-D feature columns of that hour, e.g. for feature set "own":
price one day and one week earlier, the published load forecast, yesterday's weather, net import two
days earlier, and the calendar (local hour, weekday, month). With `neighbours: true` (checkpoint 2) the
same numeric features averaged over the zone's interconnected neighbours are appended as nb_<col>.
The last column is always price_lag168 (same hour, one week earlier); it is not a feature but is kept
alongside so the residual target can be formed and inverted.

Target
------
"asinh" (default): asinh(price / 50). Compresses spikes and handles negative prices; the lags give the
model last week's level as a feature, so it learns the residual implicitly.
"residual": asinh(price) - asinh(price_lag168), what the torch models use. Rows with no lag168 are
dropped from training.
Predictions are always returned in EUR/MWh, shaped like Samples.y (n_samples, 25).

Training
--------
One model per zone (`per_zone: true`) or one pooled model with the zone id as an extra feature.
Fit on the rows of `fit_splits`; when "val" is among the eval splits it is used for early stopping,
which is the normal use of a validation set (model selection); the test split is never touched during
fitting. A zone with no validation rows (GB, prices end 2020) trains for the full number of trees.
The number of trees each model actually used is printed and stored in results under info.trees_used.

Config keys
-----------
  feature_set   name in epgnn.data.features.FEATURE_SETS (default "own")
  neighbours    true adds nb_<col> = mean over interconnected zones of each numeric feature (checkpoint 2)
  per_zone      true = one model per zone; false = one pooled model with the zone as a feature
  target        "asinh" (default) | "residual"
  params        xgboost.XGBRegressor keyword arguments, merged over DEFAULT_PARAMS
  early_stopping_rounds   default 50 (used when "val" is an eval split)
  zones         optional subset of zones (useful for quick experiments)
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from epgnn.data import features as F
from epgnn.data import samples as S
from epgnn.data import scaling as SC
from epgnn.data import tables

# L1 objective (reg:absoluteerror) matches the MAE we report; "hist" is the fast tree builder.
DEFAULT_PARAMS = {"max_depth": 6, "n_estimators": 600, "learning_rate": 0.05, "subsample": 0.8,
                  "colsample_bytree": 0.8, "tree_method": "hist", "objective": "reg:absoluteerror"}


def _columns(cfg: dict) -> tuple[list[str], list[str]]:
    """The feature columns of a row: (base columns, columns to also average over neighbours)."""
    base = F.day_d_columns(cfg.get("feature_set", "own"))       # raises if a column is a day-D outcome
    # neighbours' calendar would be identical to the zone's own, so only the numeric features are averaged
    nb = [c for c in base if c not in F.CALENDAR] if cfg.get("neighbours") else []
    return base, nb


def _inputs(s: S.Samples, base: list[str], nb: list[str]) -> np.ndarray:
    """Per-hour inputs for every sample: (n_samples, H, n_features).

    Column order: base features, neighbour means (if any), and price_lag168 LAST. Everything downstream
    relies on that: `x[:, :, :-1]` are the model features, `x[:, :, -1]` is the lag used for the target.
    Slots that do not exist (hour 24 on a 24-hour day) are NaN.
    """
    x = S.day_inputs(s, base + ["price_lag168"])
    if not nb:
        return x
    # Neighbour means are computed once for the whole table (zone, hour) -> nb_<col>, then looked up for
    # every (sample, hour) cell by (zone, ts_utc). s.hours holds the UTC timestamp of each slot, -1 if none.
    n = tables.nodes()
    means = F.neighbour_means(n, tables.edge_pairs(), nb).set_index(["zone", "ts_utc"])
    zone_rep = np.repeat(s.zone, S.H)                             # zone of every (sample, slot) cell
    ts = pd.to_datetime(s.hours.ravel(), utc=True)
    valid = s.hours.ravel() >= 0
    keys = pd.MultiIndex.from_arrays([zone_rep[valid], ts[valid]])
    flat = np.full((len(s) * S.H, len(nb)), np.nan, dtype="float32")
    flat[valid] = means.reindex(keys).to_numpy(dtype="float32")   # NaN where a zone has no neighbour data
    nbx = flat.reshape(len(s), S.H, len(nb))
    return np.concatenate([x[:, :, :-1], nbx, x[:, :, -1:]], axis=2)


def _target(y: np.ndarray, lag168: np.ndarray, kind: str) -> np.ndarray:
    """Price -> the quantity the trees are fitted on."""
    return SC.residual_target(y, lag168) if kind == "residual" else SC.to_asinh(y)


def _to_price(t: np.ndarray, lag168: np.ndarray, kind: str) -> np.ndarray:
    """Inverse of _target: model output -> EUR/MWh."""
    return SC.residual_to_price(t, lag168) if kind == "residual" else SC.from_asinh(t)


def _flatten(s: S.Samples, x: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn (samples, slots) into training rows, keeping only cells with a valid price and a finite target.

    Returns X (rows, features), target (rows,), lag168 (rows,), zone of each row (rows,).
    """
    lag = x[:, :, -1]
    t = _target(s.y, lag, kind)
    ok = s.mask & np.isfinite(t)                                  # has_price and (for residual) a lag168
    X = x[:, :, :-1][ok]
    return X, t[ok], lag[ok], np.repeat(s.zone, S.H).reshape(len(s), S.H)[ok]


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    """Fit on `fit_splits`, predict every split in `eval_splits`. Returns {split: prices (n_samples, H)}."""
    base, nb = _columns(cfg)
    kind = cfg.get("target", "asinh")
    params = {**DEFAULT_PARAMS, **cfg.get("params", {})}
    zones = cfg.get("zones")

    # ---- training rows: all fit splits concatenated (e.g. pretrain + train) --------------------------
    fit = [S.build(sp, zones) for sp in fit_splits]
    fit_x = [_inputs(s, base, nb) for s in fit]
    Xtr, ytr, _, ztr = map(np.concatenate, zip(*[_flatten(s, x, kind) for s, x in zip(fit, fit_x)]))

    # ---- evaluation samples and inputs, built once and reused for prediction --------------------------
    ev = {sp: S.build(sp, zones) for sp in eval_splits}
    ev_x = {sp: _inputs(s, base, nb) for sp, s in ev.items()}

    # ---- early stopping on the validation split, when it is evaluated (model selection, not test) ----
    val: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None   # (X, target, zone per row)
    if "val" in ev:
        Xva, yva, _, zva = _flatten(ev["val"], ev_x["val"], kind)
        val = (Xva, yva, zva)

    def make(with_val: bool) -> xgb.XGBRegressor:
        # early_stopping_rounds requires an eval_set, so it is only set when one will be passed
        return xgb.XGBRegressor(**params, early_stopping_rounds=cfg.get("early_stopping_rounds", 50) if with_val else None)

    zone_list = sorted(np.unique(ztr))
    zid = {z: i for i, z in enumerate(zone_list)}                  # zone -> integer id for the pooled model

    def with_zone(X: np.ndarray, zz: np.ndarray) -> np.ndarray:
        """Append the zone id as the last feature column (pooled model only)."""
        return np.concatenate([X, np.array([zid.get(z, -1) for z in zz], dtype="float32")[:, None]], axis=1)

    # ---- fit: one model per zone, or one pooled model ------------------------------------------------
    models: dict[str, xgb.XGBRegressor] = {}
    if cfg.get("per_zone", True):
        for z in zone_list:
            has_val = val is not None and bool((val[2] == z).any())   # e.g. GB has no prices after 2020
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

    # ---- bookkeeping: trees actually used (best iteration under early stopping, else the cap) ---------
    used = {z: int(m.best_iteration) + 1 if getattr(m, "best_iteration", None) is not None else params["n_estimators"]
            for z, m in models.items()}
    vals = np.array(list(used.values()))
    print(f"xgb: trees used per model: min {vals.min()}, median {int(np.median(vals))}, max {vals.max()} "
          f"(cap {params['n_estimators']})")
    run.info = {"trees_used": used}  # type: ignore[attr-defined]   # picked up by epgnn.run into results/

    # ---- predict every eval split, all slots (masked slots come out as NaN), back to EUR/MWh ---------
    out = {}
    for sp, s in ev.items():
        x = ev_x[sp]
        lag = x[:, :, -1]
        X = x[:, :, :-1].reshape(-1, x.shape[2] - 1)               # (samples * H, features)
        zz = np.repeat(s.zone, S.H)
        pred = np.full(len(X), np.nan, dtype="float32")
        if "__pooled__" in models:
            pred = models["__pooled__"].predict(with_zone(X, zz))
        else:
            for z, m in models.items():
                r = zz == z
                if r.any():
                    pred[r] = m.predict(X[r])                      # zones never seen in training stay NaN
        out[sp] = _to_price(pred.reshape(len(s), S.H), lag, kind).astype("float32")
    return out
