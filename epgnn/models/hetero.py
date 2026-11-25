"""Checkpoints 3, 3' and 4b: GRU + relational GATv2 over zones and, optionally, constraint nodes.

The proposal's model (Figure 1), one graph per delivery hour:
  zone nodes        h_v = GRU(168-hour history) joined with the hour's day-D inputs x_t        (checkpoint 2.5 encoder)
  constraint nodes  the tightest physical constraints (CNECs) of the hour: log RAM, log Fmax, RAM/Fmax, Core flag
  interconnects     zone <-> zone over the physical interconnector list (epgnn.data.graph.zone_graph)
  affects           zone -> constraint, the proposal's PTDF-weighted sum  c' = phi_c(c, sum_v PTDF_cv W h_v)
  rev_affects       constraint -> zone, GATv2 attention with the signed PTDF as edge feature and SUM
                    aggregation, so the number of binding constraints on a zone survives (lecture 6)
Two rounds of message passing, residual updates, then the checkpoint-2.5 head -> residual price target
(epgnn.data.scaling), masked L1 + spread loss (epgnn.train). Same data pipeline as gru_mlp, so the four
models below differ only in the graph:

  constraints: none,     graph: interconnects   checkpoint 3   GRU + GATv2 on the zone graph
  constraints: none,     graph: shuffled        checkpoint 3'  rewired zone graph, degrees kept
  constraints: real,     graph: interconnects   checkpoint 4b  the full heterogeneous model
  constraints: shuffled, graph: interconnects   4b control: constraint edges point at random domain zones
  constraints: none,     graph: none            = checkpoint 2.5 (sanity check of the pipeline)

Constraint nodes exist only inside the flow-based domain (CWE zones before June 2022, 12 Core zones after);
other zones and hours simply have no constraint edges and exchange messages over interconnectors only.

Config keys: as gru_mlp plus graph, constraints, c_hidden (32), heads (4), layers (2), max_constraints (64),
constraint_lag_hours (0 = the day-D domain, published on D-1 before the auction; 24 = the safe fallback).
"""
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from epgnn import train as T
from epgnn.data import graph as G
from epgnn.data import samples as S
from epgnn.data import scaling as SC
from epgnn.data import tables
from epgnn.models import gru_mlp as M
from epgnn.models.encoders import HistoryGRU


# ---- per-batch graph assembly ----------------------------------------------------------------------------

def batch_graph(hidx: np.ndarray, snap: dict | None, n_zones: int, shuffled: bool, seed: int = 0) -> dict:
    """Edges of the B x H hourly snapshots of a day batch, as one disjoint union.

    Zone node ids are snapshot * n_zones + zone, snapshots in (day, slot) order. Returns numpy arrays:
      x_c   (C, 4)   constraint features        c_src/c_dst (E,) zone node -> constraint   w (E,) signed PTDF
    With `shuffled`, every constraint's edges are re-pointed to a random permutation of the zones in that
    hour's domain, so the constraint data stays and only the zone attribution is destroyed.
    """
    B, H = hidx.shape
    hid = hidx.ravel()
    valid = hid >= 0
    if snap is None or not valid.any():
        return {"x_c": np.zeros((0, len(G.CONSTRAINT_FEATURES)), "float32"), "c_src": np.zeros(0, "int64"),
                "c_dst": np.zeros(0, "int64"), "w": np.zeros(0, "float32")}
    off = snap["offsets"]
    safe = np.where(valid, hid, 0)
    starts, counts = off[safe], np.where(valid, off[safe + 1] - off[safe], 0)
    total = int(counts.sum())
    row_snap = np.repeat(np.arange(B * H), counts)
    rows = np.repeat(starts, counts) + (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
    coef = snap["coef"][rows]                                   # (C, 12)
    c_i, k = np.nonzero(np.isfinite(coef))
    zone = snap["core_idx"][k]
    keep = zone >= 0
    c_i, k, zone = c_i[keep], k[keep], zone[keep]
    if shuffled:
        rng = np.random.default_rng(seed)
        snap_of_edge = row_snap[c_i]
        for sidx in np.unique(snap_of_edge):
            sel = snap_of_edge == sidx
            domain = np.unique(zone[sel])
            perm = dict(zip(domain, rng.permutation(domain)))
            zone[sel] = [perm[z] for z in zone[sel]]
    return {"x_c": snap["x"][rows], "c_src": row_snap[c_i] * n_zones + zone, "c_dst": c_i,
            "w": coef[c_i, k].astype("float32")}


# ---- model -----------------------------------------------------------------------------------------------

class RelationalLayer(nn.Module):
    """One round: zones -> constraints (PTDF-weighted sum), then {constraints, neighbours} -> zones."""

    def __init__(self, hidden: int, c_hidden: int, heads: int, dropout: float, use_zz: bool, use_c: bool) -> None:
        super().__init__()
        self.use_zz, self.use_c = use_zz, use_c
        self.zz = (GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout, add_self_loops=True)
                   if use_zz else nn.Linear(hidden, hidden))
        if use_c:
            self.zc = nn.Linear(hidden, c_hidden, bias=False)          # W_zc in the proposal's phi_c
            self.c_upd = nn.Linear(2 * c_hidden, c_hidden)
            self.cz = GATv2Conv((c_hidden, hidden), hidden // heads, heads=heads, dropout=dropout,
                                edge_dim=1, add_self_loops=False, aggr="add")
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, c: torch.Tensor | None, g: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_c and c is not None and c.shape[0] > 0:
            load = torch.zeros_like(c).index_add_(0, g["c_dst"], g["w"][:, None] * self.zc(u)[g["c_src"]])
            c = torch.nn.functional.elu(c + self.c_upd(torch.cat([c, load], dim=-1)))
        msg = self.zz(u, g["zz"]) if self.use_zz else self.zz(u)
        if self.use_c and c is not None and c.shape[0] > 0:
            msg = msg + self.cz((c, u), torch.stack([g["c_dst"], g["c_src"]]), g["w"][:, None])
        return torch.nn.functional.elu(u + self.drop(msg)), c


class HeteroGNN(nn.Module):
    def __init__(self, n_hist: int, n_day: int, n_c: int, hidden: int = 64, c_hidden: int = 32, heads: int = 4,
                 layers: int = 2, mlp_hidden: int = 128, dropout: float = 0.1, use_zz: bool = True,
                 use_c: bool = True) -> None:
        super().__init__()
        self.enc = HistoryGRU(n_hist, hidden)
        self.inp = nn.Linear(hidden + n_day, hidden)
        self.c_inp = nn.Linear(n_c, c_hidden) if use_c else None
        self.layers = nn.ModuleList(RelationalLayer(hidden, c_hidden, heads, dropout, use_zz, use_c) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))
        self.use_c = use_c

    def forward(self, hist: torch.Tensor, xday: torch.Tensor, extra: dict, batch: dict) -> torch.Tensor:
        B, Z, H, _ = xday.shape
        dev = xday.device
        h = self.enc(hist.reshape(B * Z, *hist.shape[2:])).reshape(B, Z, 1, -1).expand(-1, -1, H, -1)
        u = torch.nn.functional.elu(self.inp(torch.cat([h, xday], dim=-1)))          # (B, Z, H, hidden)
        u = u.permute(0, 2, 1, 3).reshape(B * H * Z, -1)                               # snapshot-major nodes
        g = batch_graph(batch["hidx"].cpu().numpy(), extra.get("snap") if self.use_c else None, Z,
                        extra.get("constraints") == "shuffled", int(batch["hidx"][0, 0]))
        gt = {k: torch.as_tensor(v, device=dev) for k, v in g.items()}
        zz = extra.get("zz")
        if zz is not None:
            offs = torch.arange(B * H, device=dev) * Z
            gt["zz"] = (zz[:, None, :] + offs[None, :, None]).reshape(2, -1)
        c = torch.nn.functional.elu(self.c_inp(gt["x_c"])) if self.c_inp is not None and gt["x_c"].shape[0] > 0 else None
        for layer in self.layers:
            u, c = layer(u, c, gt)
        return self.head(u).reshape(B, H, Z).permute(0, 2, 1)


# ---- entry point -----------------------------------------------------------------------------------------

def _hour_index(raws: dict[str, dict], zones: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Unique sorted delivery hours over all splits and, per split, the (days, H) index into them (-1 = no slot)."""
    per = {}
    for sp, r in raws.items():
        hrs = M._day_major(r["s"], r["s"].hours, zones)                  # (days, Z, H)
        per[sp] = hrs.max(axis=1)                                         # slots are shared across zones
    hours = np.unique(np.concatenate([v[v >= 0] for v in per.values()]))
    return hours, {sp: np.where(v >= 0, np.searchsorted(hours, np.where(v >= 0, v, hours[0])), -1) for sp, v in per.items()}


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    torch.manual_seed(cfg.get("seed", 0))
    zones = cfg.get("zones") or tables.zones()
    use_c = cfg.get("constraints", "none") != "none"
    graph = cfg.get("graph", "interconnects")

    fit_raw = {sp: M.raw(sp, cfg, zones) for sp in fit_splits}
    ev_raw = {sp: M.raw(sp, cfg, zones) for sp in eval_splits}
    scalers = M.fit_scalers(list(fit_raw.values()))
    hours, hidx = _hour_index({**fit_raw, **ev_raw}, zones)
    tens = {sp: {**M.to_tensors(r, scalers, zones), "hidx": torch.as_tensor(hidx[sp])}
            for sp, r in {**fit_raw, **ev_raw}.items()}
    train_t = {k: torch.cat([tens[sp][k] for sp in fit_splits]) for k in tens[fit_splits[0]]}

    extra: dict = {"spread_pairs": M.spread_pairs(zones), "constraints": cfg.get("constraints", "none")}
    if graph != "none":
        extra["zz"] = torch.as_tensor(G.zone_graph(zones, shuffled=graph == "shuffled", seed=cfg.get("seed", 0)))
    snap_info = {}
    if use_c:
        snap = G.constraint_snapshots(hours, zones, cfg.get("max_constraints", 64), cfg.get("constraint_lag_hours", 0))
        fit_rows = np.concatenate([np.arange(snap["offsets"][i], snap["offsets"][i + 1])
                                   for sp in fit_splits for i in hidx[sp].ravel() if i >= 0])
        snap["x"] = SC.ZScaler().fit(snap["x"][fit_rows]).transform(snap["x"])     # fit-split statistics only
        extra["snap"] = snap
        counts = np.diff(snap["offsets"])
        snap_info = {"constraint_hours": int((counts > 0).sum()), "constraints_per_hour_median": float(np.median(counts[counts > 0])),
                     "constraint_lag_hours": cfg.get("constraint_lag_hours", 0), "max_constraints": cfg.get("max_constraints", 64)}

    model = HeteroGNN(n_hist=train_t["hist"].shape[-1], n_day=train_t["xday"].shape[-1], n_c=len(G.CONSTRAINT_FEATURES),
                      hidden=cfg.get("hidden", 64), c_hidden=cfg.get("c_hidden", 32), heads=cfg.get("heads", 4),
                      layers=cfg.get("layers", 2), mlp_hidden=cfg.get("mlp_hidden", 128), dropout=cfg.get("dropout", 0.1),
                      use_zz=graph != "none", use_c=use_c)
    bd = cfg.get("batch_days", 16)
    train_b = T.DayBatches(train_t, bd, shuffle=True, seed=cfg.get("seed", 0))
    val_b = T.DayBatches(tens["val"], bd, shuffle=False) if "val" in tens else None
    info = T.fit(model, train_b, val_b, cfg, extra, T.run_dir(cfg.get("name", "hetero")))
    info.update(n_params=sum(p.numel() for p in model.parameters()), graph=graph, constraints=cfg.get("constraints", "none"), **snap_info)
    run.info = info  # type: ignore[attr-defined]

    out = {}
    for sp, r in ev_raw.items():
        pred = T.predict(model, T.DayBatches(tens[sp], bd, shuffle=False), extra, cfg.get("device"))
        pred = pred.swapaxes(0, 1).reshape(len(r["s"]), S.H)
        out[sp] = SC.residual_to_price(pred, r["base"]).astype("float32")
    return out
