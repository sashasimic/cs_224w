"""Checkpoint 2.5: GRU + MLP per zone, no graph.

The neural counterpart of the XGBoost baseline, and the no-graph case of checkpoint 3: the same history
encoder and head as the GNN, with no message passing between zones. Comparing 3 with this isolates
what the graph adds; comparing this with 1b/2 shows what the sequence model adds over trees.

Per sample (zone, delivery day D):
  history  the 168 hours up to the cutoff (D-1 10:00 UTC) of HISTORY columns -> HistoryGRU -> h (hidden,)
  day-D    for each hour slot t of D, the gate-closure inputs x_t (feature set, e.g. "own")
  output   for each slot t: MLP([h, x_t]) -> residual target r_t = asinh(price_t) - asinh(base_t),
           base_t = price one week earlier (falls back to one day earlier, then 0, when missing)
Prices are recovered as sinh(r_t + asinh(base_t)) * 50. Loss: masked L1 + spread term (epgnn.train).

Data is prepared as day batches (B days x all zones), see epgnn.train. Inputs are z-scored with
statistics from the fit splits; NaN (missing value or non-existent slot) becomes 0 after scaling.

Config keys: feature_set, history_window (168), hidden (64), mlp_hidden (128), dropout (0.1),
fit_splits, eval_splits, epochs, lr, patience, batch_days (16), spread_weight, seed, zones.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from epgnn import train as T
from epgnn.data import features as F
from epgnn.data import samples as S
from epgnn.data import scaling as SC
from epgnn.data import tables
from epgnn.models.encoders import HistoryGRU


class GRUMLP(nn.Module):
    def __init__(self, n_hist: int, n_day: int, hidden: int = 64, mlp_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.enc = HistoryGRU(n_hist, hidden)
        self.head = nn.Sequential(nn.Linear(hidden + n_day, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def encode(self, hist: torch.Tensor) -> torch.Tensor:
        """(B, Z, W, n_hist) -> (B, Z, hidden)"""
        b, z, w, f = hist.shape
        return self.enc(hist.reshape(b * z, w, f)).reshape(b, z, -1)

    def forward(self, hist: torch.Tensor, xday: torch.Tensor, extra: dict) -> torch.Tensor:
        h = self.encode(hist)                                                   # (B, Z, hidden)
        hh = h.unsqueeze(2).expand(-1, -1, xday.shape[2], -1)                   # (B, Z, H, hidden)
        return self.head(torch.cat([hh, xday], dim=-1)).squeeze(-1)             # (B, Z, H)


# ---- data preparation ----------------------------------------------------------------------------------

def _day_major(s: S.Samples, arr: np.ndarray, zones: list[str]) -> np.ndarray:
    """Samples are zone-major (zones x days); reshape to (days, zones, ...)."""
    n_days = len(s) // len(zones)
    return arr.reshape(len(zones), n_days, *arr.shape[1:]).swapaxes(0, 1)


def _base(s: S.Samples) -> np.ndarray:
    """Reference price for the residual target: lag168, else lag24, else 0."""
    lags = S.day_inputs(s, ["price_lag168", "price_lag24"])
    base = lags[:, :, 0]
    base = np.where(np.isnan(base), lags[:, :, 1], base)
    return np.where(np.isnan(base), 0.0, base).astype("float32")


def raw(split: str, cfg: dict, zones: list[str]) -> dict:
    """Unscaled arrays of a split: samples, history, day inputs, base price, target, mask."""
    s = S.build(split, zones)
    base = _base(s)
    target = SC.residual_target(s.y, base)
    mask = s.mask & np.isfinite(target)
    return {"s": s, "base": base, "mask": mask, "target": np.where(mask, target, 0.0).astype("float32"),
            "hist": S.history(s, F.HISTORY, cfg.get("history_window", 168)),
            "xday": S.day_inputs(s, F.day_d_columns(cfg.get("feature_set", "own")))}


def fit_scalers(raws: list[dict]) -> dict:
    """Z-score statistics over all fit splits (train rows only, never val/test)."""
    return {k: SC.ZScaler().fit(np.concatenate([r[k] for r in raws])) for k in ("hist", "xday")}


def to_tensors(r: dict, scalers: dict, zones: list[str]) -> dict:
    """Scaled, day-major tensors for epgnn.train.DayBatches."""
    t = {"hist": scalers["hist"].transform(r["hist"]), "xday": scalers["xday"].transform(r["xday"]),
         "target": r["target"], "mask": r["mask"]}
    return {k: torch.as_tensor(_day_major(r["s"], v, zones)) for k, v in t.items()}


def spread_pairs(zones: list[str]) -> torch.Tensor:
    zi = {z: i for i, z in enumerate(zones)}
    p = tables.edge_pairs()
    p = p[p.from_zone.isin(zi) & p.to_zone.isin(zi) & (p.from_zone < p.to_zone)]
    return torch.as_tensor(np.array([[zi[a] for a in p.from_zone], [zi[b] for b in p.to_zone]], dtype="int64"))


# ---- entry point ---------------------------------------------------------------------------------------

def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    torch.manual_seed(cfg.get("seed", 0))
    zones = cfg.get("zones") or tables.zones()
    extra = {"spread_pairs": spread_pairs(zones)}

    # training data: fit splits concatenated along the day axis; scalers fitted on them only
    fit_raw = [raw(sp, cfg, zones) for sp in fit_splits]
    scalers = fit_scalers(fit_raw)
    parts = [to_tensors(r, scalers, zones) for r in fit_raw]
    train_t = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    ev_raw = {sp: raw(sp, cfg, zones) for sp in eval_splits}
    ev_t = {sp: to_tensors(r, scalers, zones) for sp, r in ev_raw.items()}

    model = GRUMLP(n_hist=train_t["hist"].shape[-1], n_day=train_t["xday"].shape[-1],
                   hidden=cfg.get("hidden", 64), mlp_hidden=cfg.get("mlp_hidden", 128), dropout=cfg.get("dropout", 0.1))
    bd = cfg.get("batch_days", 16)
    train_b = T.DayBatches(train_t, bd, shuffle=True, seed=cfg.get("seed", 0))
    val_b = T.DayBatches(ev_t["val"], bd, shuffle=False) if "val" in ev_t else None
    info = T.fit(model, train_b, val_b, cfg, extra, T.run_dir(cfg.get("name", "gru_mlp")))
    info["n_params"] = sum(p.numel() for p in model.parameters())
    run.info = info  # type: ignore[attr-defined]

    out = {}
    for sp, r in ev_raw.items():
        pred = T.predict(model, T.DayBatches(ev_t[sp], bd, shuffle=False), extra, cfg.get("device"))     # (days, zones, H)
        pred = pred.swapaxes(0, 1).reshape(len(r["s"]), S.H)                           # back to zone-major
        out[sp] = SC.residual_to_price(pred, r["base"]).astype("float32")
    return out
