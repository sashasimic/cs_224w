"""Checkpoint 3 attention scope: the c3 GRU + GATv2 model with a choice of neighbourhood.

Same architecture as the collaborators' c3 (`c3_gat_pe_time`, 95,425 parameters), with the edge set as the
only switch:

  graph: interconnects   attention over the 90 physical interconnectors (+ self-loop)     = c3
  graph: full            attention over every zone (+ self-loop): a Transformer over zones
  graph: self            self-loop only: the no-message-passing control

The GATv2 weights do not depend on the edges, so all three have exactly the same parameter count.

Per (zone, delivery day):
  history   168 h up to D-1 10:00 UTC of `history` columns -> GRU(64) -> h, copied to all 25 slots
  day-D     per slot: `day_inputs` columns (gate-closure only, from features/v1)
  curves    optional (`price_curves`): the zone's whole price curves of D-1, D-2, D-7 (24 local hours each,
            asinh(p/50), z-scored), the same vector for every slot of the day. Without it a slot sees only its
            own hour of D-1 and the GRU history ends at D-2 10:00 UTC (price_lag24 up to the cutoff), so the
            D-1 curve, which is public since D-2 ~13:00 CET, never reaches the model (cf. LEAR).
  position  8 Laplacian eigenvectors of the interconnector graph (sign flipped at random in training);
            `positional: false` drops them (the full arm then has no topology information at all)
  input     Linear(64 + n_day + 8 -> 64) = u0
  blocks    2 x [u += Drop(GATv2(LN(u)));  u += Drop(FFN(LN(u)))], GATv2 4 heads x 16, FFN 64-128-64 ELU
  head      LN -> 64-128 ReLU Drop 128-128 ReLU -> 1 : residual r = asinh(p/50) - asinh(base/50)
  loss      masked L1 + spread_weight x L1 on the 90 interconnected pairs (epgnn.train.masked_loss)

One graph per (day, slot); slots never exchange messages. The GATv2 layer is a dense re-implementation
of torch_geometric's GATv2Conv (share_weights=False, add_self_loops=True, concat heads, bias), verified
equal to it in tests/test_gat_scope.py; it scores only the listed edges, so memory grows with the edge
count (1,980 + 45 for full vs 180 + 45 for interconnects).

Config keys: graph, positional (true), pe_k (8), history, day_inputs (lists of columns or a named set),
hidden (64), heads (4), layers (2), ffn (128), mlp_hidden (128), dropout (0.1), epochs, lr, patience,
batch_days (16), spread_weight (0.5), weight_decay (0), seed, device, checkpoint_attention (true),
price_curves (false; true = lags [1, 2, 7] in days, or a list of day lags >= 1), strict_inputs (false: warn
about day inputs that are not known at the cutoff; true: refuse them), fit_splits, eval_splits.
rolling (off): monthly fine-tuning over the eval period, the neural counterpart of a recalibrated LEAR.
After the normal fit, the eval days (all eval splits together, in date order) are taken one calendar month
at a time: the month is predicted with the current weights, and before the next month the model is
fine-tuned on the trailing `window_days` days that end the day before that month. Every price in that
window is published before the first cutoff of the month, so nothing from the month itself is used. No early
stopping inside the roll (there is no held-out data that is not in the future); a fixed number of epochs
at a lower learning rate instead. `rolling: true` uses the defaults, or give a dict:
  window_days 730, epochs 2, lr 1e-4, weight_decay (cfg), start "continue" (keep fine-tuning the same
  weights) | "base" (restart every month from the weights of the initial fit)
The returned predictions are the rolling ones; the static ones (initial fit only) are saved as
runs/<name>/pred_<split>_static.npy and scored in info["static_metrics"].
Day sets: own_cyclic / own_cyclic_plus reproduce earlier runs and use the partly-future `_lag24` weather;
own_cyclic_asof / own_cyclic_plus_asof are the same with cutoff-safe weather (`_asof`).
"""
import copy
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as Fn
from torch.utils.checkpoint import checkpoint

from epgnn import metrics
from epgnn import train as T
from epgnn.data import samples as S
from epgnn.models import persistence
from epgnn.data import scaling as SC
from epgnn.pipeline import graphs, registry, tensors

HISTORY_SETS = {
    "base": ["price_lag24", "price_lag168", "load_fc", "net_import", "temp_pw", "temp_anom", "wind_cf", "solar_cf"],
}
DAY_SETS = {
    # the collaborators' c3 inputs with cyclic time (14 columns)
    "own_cyclic": ["price_lag24", "price_lag168", "load_fc", "temp_pw_lag24", "temp_anom_lag24", "wind_cf_lag24",
                   "solar_cf_lag24", "net_import_lag48", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                   "month_sin", "month_cos"],
}
DAY_SETS["own_cyclic_plus"] = DAY_SETS["own_cyclic"] + [
    "price_d1_mean", "price_d1_std", "price_wk_change", "europe_price_lag24", "load_fc_wk_change",
    "load_fc_day_rel", "is_weekend", "heat_stress"]
# The same sets with the weather that is actually observed at the cutoff (`<c>_asof`: lag 24 up to 10:00 UTC
# of day D, lag 48 after). The `_lag24` weather of the sets above is partly from after the cutoff; they are
# kept only to reproduce earlier runs. Use these for new runs.
_ASOF = {f"{c}_lag24": f"{c}_asof" for c in ("temp_pw", "temp_anom", "wind_cf", "solar_cf")}
DAY_SETS["own_cyclic_asof"] = [_ASOF.get(c, c) for c in DAY_SETS["own_cyclic"]]
DAY_SETS["own_cyclic_plus_asof"] = [_ASOF.get(c, c) for c in DAY_SETS["own_cyclic_plus"]]
# H2, zone-level: the D-1 flow-based domain summarised per zone (same UTC hour of D-1; NaN -> 0 outside Core)
JAO_LAG24 = ["jao_n_exposed_lag24", "jao_min_ram_exposed_lag24", "jao_tightness_lag24",
             "jao_amr_share_exposed_lag24", "jao_min_ram_grid_exposed_lag24", "jao_tightness_grid_lag24"]
DAY_SETS["own_cyclic_asof_jao"] = DAY_SETS["own_cyclic_asof"] + JAO_LAG24


def _cols(spec, sets: dict) -> list[str]:
    return list(sets[spec]) if isinstance(spec, str) else list(spec)


def check_day_inputs(cols: list[str], strict: bool = False) -> list[str]:
    """Day-D input columns that the feature registry says are not known at gate closure (timing `past` or
    `outcome`). Warns, or raises with strict=True (config `strict_inputs: true`)."""
    bad = [c for c in cols if registry.timing_of(c) in ("past", "outcome")]
    if bad:
        msg = f"day inputs not known at the cutoff (registry timing past/outcome): {bad}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + "; use the *_asof day sets for new runs", stacklevel=2)
    return bad


# ---- model ---------------------------------------------------------------------------------------------

class DenseGATv2(nn.Module):
    """GATv2Conv(dim, out, heads, concat=True, add_self_loops=True, share_weights=False) on G graphs
    that share one edge list. x: (G, Z, dim) -> (G, Z, heads*out)."""

    def __init__(self, dim: int, out: int, heads: int, edges: torch.Tensor, n_nodes: int,
                 checkpoint_scores: bool = True) -> None:
        super().__init__()
        self.h, self.c, self.z = heads, out, n_nodes
        self.lin_l = nn.Linear(dim, heads * out)
        self.lin_r = nn.Linear(dim, heads * out)
        self.att = nn.Parameter(torch.empty(1, heads, out))
        self.bias = nn.Parameter(torch.zeros(heads * out))
        loops = torch.arange(n_nodes)
        keep = edges[0] != edges[1]
        self.register_buffer("src", torch.cat([edges[0][keep], loops]), persistent=False)
        self.register_buffer("dst", torch.cat([edges[1][keep], loops]), persistent=False)
        self.checkpoint_scores = checkpoint_scores
        for lin in (self.lin_l, self.lin_r):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.att)

    def _scores(self, xl: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
        s = Fn.leaky_relu(xl[:, self.src] + xr[:, self.dst], 0.2)          # (G, E, heads, out)
        return (s * self.att).sum(-1)                                      # (G, E, heads)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        g = x.shape[0]
        xl = self.lin_l(x).view(g, self.z, self.h, self.c)
        xr = self.lin_r(x).view(g, self.z, self.h, self.c)
        if self.training and self.checkpoint_scores and torch.is_grad_enabled():
            e = checkpoint(self._scores, xl, xr, use_reentrant=False)
        else:
            e = self._scores(xl, xr)
        logits = x.new_full((g, self.z, self.z, self.h), float("-inf"))    # [graph, dst, src, head]
        logits[:, self.dst, self.src] = e
        alpha = logits.softmax(dim=2)
        out = torch.einsum("gijh,gjhc->gihc", alpha, xl).reshape(g, self.z, self.h * self.c) + self.bias
        return (out, alpha) if return_attention else out


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, dropout: float, edges: torch.Tensor, n_nodes: int,
                 checkpoint_scores: bool) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.gat = DenseGATv2(dim, dim // heads, heads, edges, n_nodes, checkpoint_scores)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ELU(), nn.Dropout(dropout), nn.Linear(ffn, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, return_attention: bool = False):
        a = self.gat(self.ln1(u), return_attention)
        a, alpha = a if return_attention else (a, None)
        u = u + self.drop(a)
        u = u + self.drop(self.ffn(self.ln2(u)))
        return (u, alpha) if return_attention else u


class ScopeGAT(nn.Module):
    def __init__(self, n_hist: int, n_day: int, edges: np.ndarray, n_zones: int, pe: np.ndarray | None,
                 hidden: int = 64, heads: int = 4, layers: int = 2, ffn: int = 128, mlp_hidden: int = 128,
                 dropout: float = 0.1, checkpoint_scores: bool = True) -> None:
        super().__init__()
        self.gru = nn.GRU(n_hist, hidden, batch_first=True)
        n_pe = 0 if pe is None else pe.shape[1]
        if pe is not None:
            self.register_buffer("pe", torch.as_tensor(pe, dtype=torch.float32))
        else:
            self.pe = None
        self.inp = nn.Linear(hidden + n_day + n_pe, hidden)
        e = torch.as_tensor(edges, dtype=torch.long)
        self.blocks = nn.ModuleList(Block(hidden, heads, ffn, dropout, e, n_zones, checkpoint_scores) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def forward(self, hist: torch.Tensor, xday: torch.Tensor, extra: dict | None = None,
                return_attention: bool = False):
        b, z, w, fh = hist.shape
        hslots = xday.shape[2]
        _, hn = self.gru(hist.reshape(b * z, w, fh))
        h = hn[-1].reshape(b, z, 1, -1).expand(-1, -1, hslots, -1)
        parts = [h, xday]
        if self.pe is not None:
            pe = self.pe
            if self.training:
                pe = pe * (torch.randint(0, 2, (1, pe.shape[1]), device=pe.device) * 2 - 1)
            parts.append(pe.view(1, z, 1, -1).expand(b, -1, hslots, -1))
        u = self.inp(torch.cat(parts, dim=-1))                             # (B, Z, H, hidden)
        u = self.node_update(u)                                            # identity here; see gat_constraints
        u = u.permute(0, 2, 1, 3).reshape(b * hslots, z, -1)               # one graph per (day, slot)
        alphas = []
        for blk in self.blocks:
            if return_attention:
                u, a = blk(u, True)
                alphas.append(a)
            else:
                u = blk(u)
        r = self.head(u).reshape(b, hslots, z).permute(0, 2, 1)            # (B, Z, H)
        return (r, alphas) if return_attention else r

    def node_update(self, u: torch.Tensor) -> torch.Tensor:
        """Hook between the input layer and the GAT blocks, u: (B, Z, H, hidden). Subclasses add messages
        from other node types here (epgnn.models.gat_constraints)."""
        return u


# ---- data ----------------------------------------------------------------------------------------------

class GatherBatches:
    """Day batches gathered on the fly from the hourly grid; yields the dict epgnn.train expects."""

    def __init__(self, data: dict, days: np.ndarray, batch_days: int, window: int, shuffle: bool, seed: int = 0) -> None:
        self.d, self.days, self.b, self.w, self.shuffle = data, days, batch_days, window, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return (len(self.days) + self.b - 1) // self.b

    def batch(self, idx: np.ndarray) -> dict:
        d = self.d
        slots = d["slots"][idx]                                            # (B, H)
        ok = slots >= 0
        s = torch.as_tensor(np.where(ok, slots, 0))
        okt = torch.as_tensor(ok)
        xday = d["xday"][s] * okt[..., None, None]                         # (B, H, Z, F)
        target = d["target"][s] * okt[..., None]
        mask = d["mask"][s] & okt[..., None]
        hidx = d["cutoff"][idx][:, None] + np.arange(-self.w + 1, 1)[None, :]
        hok = torch.as_tensor(hidx >= 0)
        hist = d["hist"][torch.as_tensor(np.clip(hidx, 0, None))] * hok[..., None, None]   # (B, W, Z, F)
        xday = xday.permute(0, 2, 1, 3)                                    # (B, Z, H, F)
        if "curves" in d:                                                  # (D, Z, Fc) per day, same for every slot
            c = d["curves"][torch.as_tensor(idx)][:, :, None, :].expand(-1, -1, xday.shape[2], -1)
            xday = torch.cat([xday, c * okt[:, None, :, None]], dim=-1)    # zero on padded slots, like xday
        return {"hist": hist.permute(0, 2, 1, 3).contiguous(), "xday": xday.contiguous(),
                "target": target.permute(0, 2, 1).contiguous(), "mask": mask.permute(0, 2, 1).contiguous()}

    def __iter__(self):
        order = self.rng.permutation(len(self.days)) if self.shuffle else np.arange(len(self.days))
        for i in range(0, len(order), self.b):
            yield self.batch(self.days[order[i:i + self.b]])


CURVE_LAGS = [1, 2, 7]


def price_curves(zt: tensors.ZoneTensors, lags: list[int]) -> tuple[np.ndarray, list[str]]:
    """Whole-day price curves of earlier delivery days, (D, Z, 24 * len(lags)) as asinh(p/50), NaN where
    missing, plus their names. Indexed by Brussels local hour, so DST days line up with normal days (the
    repeated hour of a 25-hour day keeps its second value; the skipped hour of a 23-hour day is NaN).
    Day lag k >= 1 only: the prices of D-1 are published on D-2, before the D-1 10:00 UTC cutoff."""
    assert lags and min(lags) >= 1, f"price-curve lags must be >= 1 day (day D is the target), got {lags}"
    d = pd.to_datetime(zt.days)
    assert (np.diff(d.values) == np.timedelta64(1, "D")).all(), "delivery days are not consecutive"
    local = pd.DatetimeIndex(pd.to_datetime(zt.hours, utc=True)).tz_convert("Europe/Brussels").hour.to_numpy()
    price = np.where(zt.has_price, zt.price, np.nan)
    daily = np.full((len(d), 24, len(zt.zones)), np.nan, dtype=np.float32)          # (D, local hour, Z)
    di, k = np.nonzero(zt.slots >= 0)
    t = zt.slots[di, k]
    daily[di, local[t]] = price[t]
    out = []
    for lag in lags:
        sh = np.full_like(daily, np.nan)
        sh[lag:] = daily[:-lag]
        out.append(sh)
    with np.errstate(invalid="ignore"):
        c = np.arcsinh(np.concatenate(out, axis=1) / SC.ASINH_SCALE)                # (D, 24 * L, Z)
    names = [f"price_d-{lag}_h{h:02d}" for lag in lags for h in range(24)]
    return c.transpose(0, 2, 1).astype(np.float32), names


def prepare(cfg: dict, fit_splits: list[str]) -> tuple[tensors.ZoneTensors, dict, list[str], list[str]]:
    hist_cols = _cols(cfg.get("history", "base"), HISTORY_SETS)
    day_cols = _cols(cfg.get("day_inputs", "own_cyclic"), DAY_SETS)
    check_day_inputs(day_cols, strict=cfg.get("strict_inputs", False))
    zt = tensors.load(list(dict.fromkeys(hist_cols + day_cols)))
    ci = {c: i for i, c in enumerate(zt.cols)}
    xh = zt.x[..., [ci[c] for c in hist_cols]]
    xd = zt.x[..., [ci[c] for c in day_cols]]
    fit_days = np.concatenate([zt.day_indices(sp) for sp in fit_splits])
    fit_hours = zt.hours_of(fit_days)
    sh = SC.ZScaler().fit(xh[fit_hours])
    sd = SC.ZScaler().fit(xd[fit_hours])
    with np.errstate(invalid="ignore"):
        target = SC.residual_target(zt.price, zt.base)
    mask = zt.has_price & np.isfinite(target)
    data = {"hist": torch.as_tensor(sh.transform(xh)), "xday": torch.as_tensor(sd.transform(xd)),
            "target": torch.as_tensor(np.where(mask, target, 0.0).astype(np.float32)),
            "mask": torch.as_tensor(mask), "slots": zt.slots, "cutoff": zt.cutoff}
    spec = cfg.get("price_curves", False)
    if spec:
        curves, names = price_curves(zt, CURVE_LAGS if spec is True else list(spec))
        sc = SC.ZScaler().fit(curves[fit_days])
        data["curves"] = torch.as_tensor(sc.transform(curves))
        day_cols = day_cols + names
    zt.x = np.empty(0)  # free the raw copy
    return zt, data, hist_cols, day_cols


# ---- attention diagnostic ------------------------------------------------------------------------------

def attention_report(model: ScopeGAT, batches: GatherBatches, zones: list[str], dev: torch.device,
                     max_batches: int = 12) -> dict:
    """Layer-1 attention on validation days: how much of a zone's cross-zone attention goes to its
    physical neighbours, against the share uniform attention over the same senders would give."""
    adj = torch.as_tensor(graphs.adjacency(zones) > 0)
    eye = torch.eye(len(zones), dtype=torch.bool)
    tot = {"self": torch.zeros(len(zones)), "nb": torch.zeros(len(zones)), "cross": torch.zeros(len(zones)), "n": 0}
    model.eval()
    with torch.no_grad():
        for k, b in enumerate(batches):
            if k >= max_batches:
                break
            _, alphas = model(b["hist"].to(dev), b["xday"].to(dev), None, return_attention=True)
            a = alphas[0].mean(-1).cpu()                                   # (G, dst, src), mean over heads
            tot["self"] += a[:, eye].sum(0)
            tot["nb"] += (a * adj).sum(-1).sum(0)
            tot["cross"] += (a * ~eye).sum(-1).sum(0)
            tot["n"] += a.shape[0]
    n = max(tot["n"], 1)
    senders = (~eye).sum(1).float()
    rows = []
    for i, z in enumerate(zones):
        cross = float(tot["cross"][i])
        rows.append({"zone": z, "self_weight": float(tot["self"][i]) / n,
                     "neighbour_share_of_cross": float(tot["nb"][i]) / cross if cross > 0 else float("nan"),
                     "uniform_neighbour_share": float(adj[i].sum()) / float(senders[i])})
    share = np.nanmean([r["neighbour_share_of_cross"] for r in rows])
    uni = np.mean([r["uniform_neighbour_share"] for r in rows])
    return {"graphs": n, "mean_neighbour_share": float(share), "mean_uniform_share": float(uni),
            "lift": float(share / uni), "by_zone": rows}


# ---- rolling fine-tuning -------------------------------------------------------------------------------

ROLLING_DEFAULTS = {"window_days": 730, "epochs": 2, "lr": 1e-4, "start": "continue"}


def rolling_config(cfg: dict) -> dict | None:
    spec = cfg.get("rolling", False)
    if not spec:
        return None
    r = dict(ROLLING_DEFAULTS, weight_decay=cfg.get("weight_decay", 0.0))
    if isinstance(spec, dict):
        unknown = set(spec) - set(r)
        assert not unknown, f"unknown rolling keys {sorted(unknown)}"
        r.update(spec)
    assert r["start"] in ("continue", "base"), r["start"]
    return r


def rolling_schedule(days: np.ndarray, eval_days: np.ndarray, last_fit_day: int,
                     window_days: int) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """[(month, days to predict, days to fine-tune on first or None)] over the eval days, month by month.
    Day indices are positions in the consecutive `days`. Fine-tuning happens before a month only when days
    have passed since the model last trained (so not before the first val month, which follows train
    directly), on the `window_days` days that end the day before the month starts."""
    ev = np.unique(eval_days)
    months = np.array([d[:7] for d in days[ev]])
    out, seen = [], last_fit_day
    for m in dict.fromkeys(months):
        pred = ev[months == m]
        first = int(pred[0])
        ft = None
        if first - 1 > seen:
            ft = np.arange(max(0, first - window_days), first)
            seen = first - 1
        out.append((m, pred, ft))
    return out


def predict_prices(model: nn.Module, data: dict, zt: tensors.ZoneTensors, days: np.ndarray, bd: int, window: int,
                   extra: dict, dev_name: str | None) -> np.ndarray:
    """(Z, len(days), H) prices, NaN on padded slots."""
    pred = T.predict(model, GatherBatches(data, days, bd, window, shuffle=False), extra, dev_name)
    base = zt.base[np.where(zt.slots[days] >= 0, zt.slots[days], 0)]            # (D, H, Z)
    price = SC.residual_to_price(pred, base.transpose(0, 2, 1))                  # (D, Z, H)
    price = np.where((zt.slots[days] >= 0)[:, None, :], price, np.nan)
    return price.swapaxes(0, 1)


def roll(model: nn.Module, data: dict, zt: tensors.ZoneTensors, eval_splits: list[str], last_fit_day: int,
         r: dict, cfg: dict, extra: dict, bd: int, window: int) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Monthly predict-then-fine-tune over the eval splits. Returns {split: (Z, D_split, H) prices} and a log."""
    dev = T.device(cfg.get("device"))
    ev = np.concatenate([zt.day_indices(sp) for sp in eval_splits])
    sched = rolling_schedule(zt.days, ev, last_fit_day, r["window_days"])
    base_state = copy.deepcopy(model.state_dict())
    ex = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in extra.items()}
    model.to(dev)
    opt = None
    pred = np.full((len(zt.zones), len(zt.days), tensors.H), np.nan, dtype=np.float32)
    log = []
    print(f"  rolling: {len(sched)} months, window {r['window_days']} days, {r['epochs']} epochs at lr {r['lr']}, "
          f"start={r['start']}", flush=True)
    for k, (m, days, ft) in enumerate(sched):
        t0, losses = time.time(), []
        if ft is not None:
            if r["start"] == "base":
                model.load_state_dict(base_state)
            if opt is None or r["start"] == "base":
                opt = torch.optim.Adam(model.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
            b = GatherBatches(data, ft, bd, window, shuffle=True, seed=cfg.get("seed", 0) * 1000 + k)
            for _ in range(r["epochs"]):
                losses.append(T._epoch(model, b, dev, ex, cfg, opt))
        pred[:, days] = predict_prices(model, data, zt, days, bd, window, extra, cfg.get("device"))
        log.append({"month": m, "predict_days": int(len(days)),
                    "finetune": None if ft is None else [str(zt.days[ft[0]]), str(zt.days[ft[-1]])],
                    "train_loss": losses, "seconds": round(time.time() - t0, 1)})
        print(f"  rolling {m}  finetune {log[-1]['finetune']}  loss {[round(x, 4) for x in losses]}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return {sp: pred[:, zt.day_indices(sp)] for sp in eval_splits}, log


# ---- entry point ---------------------------------------------------------------------------------------

def run(cfg: dict, fit_splits: list[str], eval_splits: list[str], ext=None) -> dict[str, np.ndarray]:
    """ext (optional, used by epgnn.models.gat_constraints): an object with
    prepare(cfg, zt, data, day_cols, fit_days) -> day_cols (may add xday channels in place) and
    build(n_hist, n_day, edges, n_zones, pe, **kw) -> model, plus an optional info() -> dict."""
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    zt, data, hist_cols, day_cols = prepare(cfg, fit_splits)
    if ext is not None:
        day_cols = ext.prepare(cfg, zt, data, day_cols,
                               np.concatenate([zt.day_indices(sp) for sp in fit_splits]))
        torch.manual_seed(seed)
    zones = zt.zones
    graph = cfg.get("graph", "interconnects")
    edges = graphs.edges_for(graph, zones)
    pe = graphs.laplacian_pe(zones, cfg.get("pe_k", 8)) if cfg.get("positional", True) else None
    model = (ext.build if ext is not None else ScopeGAT)(len(hist_cols), len(day_cols), edges, len(zones), pe, hidden=cfg.get("hidden", 64),
                     heads=cfg.get("heads", 4), layers=cfg.get("layers", 2), ffn=cfg.get("ffn", 128),
                     mlp_hidden=cfg.get("mlp_hidden", 128), dropout=cfg.get("dropout", 0.1),
                     checkpoint_scores=cfg.get("checkpoint_attention", True))
    extra = {"spread_pairs": torch.as_tensor(graphs.spread_pairs(zones))}
    window, bd = cfg.get("history_window", 168), cfg.get("batch_days", 16)
    fit_days = np.concatenate([zt.day_indices(sp) for sp in fit_splits])
    train_b = GatherBatches(data, fit_days, bd, window, shuffle=True, seed=seed)
    val_b = GatherBatches(data, zt.day_indices("val"), bd, window, shuffle=False) if "val" in eval_splits else None
    print(f"  graph={graph} edges={edges.shape[1]} (+{len(zones)} self-loops) positional={pe is not None} "
          f"history={len(hist_cols)} day_inputs={len(day_cols)} fit_days={len(fit_days)}", flush=True)
    rdir = T.run_dir(cfg.get("name", "gat_scope"))
    info = T.fit(model, train_b, val_b, cfg, extra, rdir)
    info.update({"n_params": sum(p.numel() for p in model.parameters()), "graph": graph,
                 "edges": int(edges.shape[1]), "positional": pe is not None,
                 "history_cols": hist_cols, "day_inputs": day_cols})
    if ext is not None and hasattr(ext, "info"):
        info.update(ext.info())
    dev = T.device(cfg.get("device"))
    if val_b is not None and graph != "self":
        rep = attention_report(model, val_b, zones, dev)
        (rdir / "attention.json").write_text(json.dumps(rep, indent=2))
        info["attention"] = {k: v for k, v in rep.items() if k != "by_zone"}
        print(f"  attention: neighbour share {rep['mean_neighbour_share']:.3f} vs uniform "
              f"{rep['mean_uniform_share']:.3f} (lift {rep['lift']:.2f})", flush=True)
    run.info = info  # type: ignore[attr-defined]

    def flat(p: np.ndarray) -> np.ndarray:                                            # (Z, D, H) -> zone-major rows
        return p.reshape(-1, tensors.H).astype("float32")

    out = {sp: flat(predict_prices(model, data, zt, zt.day_indices(sp), bd, window, extra, cfg.get("device")))
           for sp in eval_splits}
    r = rolling_config(cfg)
    if r is not None:
        static = out
        rolled, log = roll(model, data, zt, eval_splits, int(fit_days.max()), r, cfg, extra, bd, window)
        out = {sp: flat(p) for sp, p in rolled.items()}
        info["rolling"] = {**r, "months": log}
        info["static_metrics"] = {}
        for sp, yhat in static.items():
            np.save(rdir / f"pred_{sp}_static.npy", yhat)
            s = S.build(sp, cfg.get("zones"))
            info["static_metrics"][sp] = metrics.score(s, yhat, persistence.predict(s))
            print(f"  static {sp}: mae={info['static_metrics'][sp]['all']['mae']:.2f}", flush=True)
        torch.save(model.state_dict(), rdir / "model_rolled.pt")
    return out
