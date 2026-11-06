"""Scoring. One function, used by every experiment, so numbers are comparable by construction.

score(s, yhat, yhat_persistence) -> dict of metrics for the whole split and per slice:
  mae, rmse                 EUR/MWh over valid hours
  rel_mae                   mae / mae of weekly persistence on the same hours
  spread_mae                MAE of (price_a - price_b) over interconnected zone pairs, same hour
  spread_f1                 F1 for "zones separate" (|spread| > SPREAD_EUR) actual vs predicted
Slices: pre/post 2025-10-01 (15-minute prices), summer (Jun-Aug), Iberian cap period for ES/PT.
"""
import numpy as np
import pandas as pd

from epgnn.data import tables
from epgnn.data.samples import Samples

SPREAD_EUR = 5.0


def _mae(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    return float(np.abs(a[m] - b[m]).mean()) if m.any() else float("nan")


def _spreads(s: Samples, y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Price differences over interconnected pairs for the given sample rows: (values, validity)."""
    key = {(z, d): i for i, (z, d) in enumerate(zip(s.zone, s.day))}
    pairs = tables.edge_pairs()
    pairs = pairs[pairs.from_zone < pairs.to_zone]           # each interconnector once
    sel = set(rows.tolist())
    va, vb = [], []
    for a, b in zip(pairs.from_zone, pairs.to_zone):
        for d in np.unique(s.day[rows]):
            ia, ib = key.get((a, d)), key.get((b, d))
            if ia is None or ib is None or ia not in sel or ib not in sel:
                continue
            va.append(ia)
            vb.append(ib)
    a_idx, b_idx = np.array(va, dtype=int), np.array(vb, dtype=int)
    if len(a_idx) == 0:
        return np.zeros((0, y.shape[1])), np.zeros((0, y.shape[1]), dtype=bool)
    return y[a_idx] - y[b_idx], s.mask[a_idx] & s.mask[b_idx]


def _block(s: Samples, yhat: np.ndarray, yper: np.ndarray, rows: np.ndarray) -> dict:
    m = s.mask[rows]
    y, p, q = s.y[rows], yhat[rows], yper[rows]
    out = {"n_hours": int(m.sum()), "mae": _mae(y, p, m), "rmse": float(np.sqrt(((y[m] - p[m]) ** 2).mean())) if m.any() else float("nan")}
    out["rel_mae"] = out["mae"] / _mae(y, q, m) if m.any() else float("nan")
    sy, sm = _spreads(s, s.y, rows)
    if len(sy):
        sp, _ = _spreads(s, yhat, rows)
        out["spread_mae"] = _mae(sy, sp, sm)
        act, pred = np.abs(sy[sm]) > SPREAD_EUR, np.abs(sp[sm]) > SPREAD_EUR
        tp, fp, fn = (act & pred).sum(), (~act & pred).sum(), (act & ~pred).sum()
        out["spread_f1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
    return out


def slices(s: Samples) -> dict[str, np.ndarray]:
    day = pd.to_datetime(s.day)
    out = {"all": np.ones(len(s), dtype=bool),
           "pre_2025-10": day < "2025-10-01", "post_2025-10": day >= "2025-10-01",
           "summer": day.month.isin([6, 7, 8]),
           "core_zones": np.isin(s.zone, ["AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]),
           "iberia_cap": np.isin(s.zone, ["ES", "PT"]) & (day >= "2022-06-15") & (day <= "2023-12-31")}
    return {k: np.asarray(v) for k, v in out.items()}


def score(s: Samples, yhat: np.ndarray, yhat_persistence: np.ndarray) -> dict:
    assert yhat.shape == s.y.shape == yhat_persistence.shape
    return {name: _block(s, yhat, yhat_persistence, np.flatnonzero(rows)) for name, rows in slices(s).items() if rows.any()}
