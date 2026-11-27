"""Checkpoints 3 and 3': GRU + GATv2 over the zone graph (no constraint nodes; those belong to hetero.py, 4b).

One graph per delivery hour:
  zone nodes     h_v = GRU(168-hour history) joined with the hour's day-D inputs x_t (checkpoint 2.5 encoder),
                 optionally the zone's Laplacian position (zone_pe) and cyclic calendar (time_encoding)
  interconnects  zone <-> zone over the physical interconnector list (epgnn.data.graph.zone_graph), no edge features
`layers` (2) pre-norm residual blocks, each LayerNorm -> GATv2 -> skip, then LayerNorm -> feed-forward -> skip
(Block); a final LayerNorm, then the checkpoint-2.5 head -> residual price target (epgnn.data.scaling), masked L1
+ spread loss (epgnn.train). Same data pipeline as gru_mlp, so the variants differ only in the graph:

  graph: interconnects   checkpoint 3   GRU + GATv2 on the zone graph
  graph: shuffled        checkpoint 3'  rewired zone graph, out-degrees kept
  graph: none            no-message control: same blocks and size, feed-forward only

Node inputs: the GRU summary, the hour's day-D inputs (time_encoding "cyclic" turns hour/weekday/month into
sin/cos pairs), and with zone_pe = k the zone's k Laplacian eigenvectors of the interconnector graph
(epgnn.data.graph.laplacian_pe; signs flipped at random in training). Both are off by default.

Cross-hour layer (hour_mixer: bigru, off by default). Without it the 25 hours of day D never exchange
information: each hour sees the week summary and its own inputs. With it, a bidirectional GRU runs over each
zone's 25 delivery hours before the graph blocks and its output is added to every hour's state (skip), so an
hour also knows yesterday's whole price curve and tomorrow's whole load-forecast curve. It reads the week
summary and only the columns in epgnn.data.features.KNOWN_AT_CUTOFF, never the lag-24 weather, so nothing
that is observed after the cutoff can travel from one hour to another (tests/test_no_peek.py).

Config keys: as gru_mlp plus graph, heads (4), layers (2), ffn_mult (2), zone_pe (0), hour_mixer ("none").
A longer history is just history_window (168 by default): the window still ends at the cutoff and only
reaches further back.
"""
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from epgnn import train as T
from epgnn.data import features as F
from epgnn.data import graph as G
from epgnn.data import samples as S
from epgnn.data import scaling as SC
from epgnn.data import tables
from epgnn.models import gru_mlp as M
from epgnn.models.encoders import HistoryGRU


class Block(nn.Module):
    """One message-passing block, pre-norm residual (the lecture-7 layer recipe of normalisation, dropout,
    activation and skip connections, in the Transformer / GraphGPS order so the residual stream itself is
    never normalised and gradients reach the GRU unchanged):

      u <- u + Drop(GATv2(LN(u)))      messages from interconnected zones (and the zone itself, self-loop)
      u <- u + Drop(FFN(LN(u)))        per-node feed-forward, 64 -> 128 -> 64 with ELU

    LayerNorm, not BatchNorm: batches mix eras and zones, and LayerNorm keeps no running statistics that
    could differ between training and evaluation. With use_zz=False the message step is skipped.
    """

    def __init__(self, hidden: int, heads: int, dropout: float, use_zz: bool, ffn_mult: int = 2) -> None:
        super().__init__()
        self.ln_u = nn.LayerNorm(hidden)
        self.zz = GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout, add_self_loops=True) if use_zz else None
        self.ln_ffn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn_mult * hidden), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_mult * hidden, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        if self.zz is not None:
            u = u + self.drop(self.zz(self.ln_u(u), edge_index))
        return u + self.drop(self.ffn(self.ln_ffn(u)))


class ZoneGAT(nn.Module):
    def __init__(self, n_hist: int, n_day: int, hidden: int = 64, heads: int = 4, layers: int = 2,
                 mlp_hidden: int = 128, dropout: float = 0.1, use_zz: bool = True, ffn_mult: int = 2,
                 n_pe: int = 0, mix_cols: list[int] | None = None) -> None:
        super().__init__()
        self.enc = HistoryGRU(n_hist, hidden)
        self.n_pe = n_pe
        self.inp = nn.Linear(hidden + n_day + n_pe, hidden)          # embedding into the residual stream
        self.mix_cols = list(mix_cols or [])                         # day-D columns the cross-hour layer may read
        if self.mix_cols:
            self.mix_in = nn.Linear(hidden + len(self.mix_cols), hidden)
            self.mix = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(Block(hidden, heads, dropout, use_zz, ffn_mult) for _ in range(layers))
        self.ln_out = nn.LayerNorm(hidden)                           # pre-norm stacks end with a norm
        self.head = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def forward(self, hist: torch.Tensor, xday: torch.Tensor, extra: dict) -> torch.Tensor:
        B, Z, H, _ = xday.shape
        dev = xday.device
        h = self.enc(hist.reshape(B * Z, *hist.shape[2:])).reshape(B, Z, 1, -1).expand(-1, -1, H, -1)
        parts = [h, xday]
        if self.n_pe:
            pe = extra["zone_pe"]                                                       # (Z, k)
            if self.training:       # eigenvector signs are arbitrary: flip each one at random (lecture 8)
                pe = pe * (torch.randint(0, 2, (1, pe.shape[1]), device=dev) * 2 - 1)
            parts.append(pe[None, :, None, :].expand(B, -1, H, -1))
        u = self.inp(torch.cat(parts, dim=-1))                                        # (B, Z, H, hidden)
        if self.mix_cols:
            # Cross-hour layer: per zone, a GRU walks forwards and backwards over the H delivery hours. Slots
            # that do not exist (23/24-hour days) sit at the end with all-zero inputs; they only change the
            # backward start state, which then depends on the day's length (calendar, not future data).
            seq = torch.cat([h, xday[..., self.mix_cols]], dim=-1).reshape(B * Z, H, -1)
            mixed, _ = self.mix(self.mix_in(seq))                                     # (B*Z, H, hidden)
            u = u + self.drop(mixed.reshape(B, Z, H, -1))
        u = u.permute(0, 2, 1, 3).reshape(B * H * Z, -1)                               # snapshot-major nodes
        edge_index = None
        zz = extra.get("zz")
        if zz is not None:              # the zone graph copied once per (day, hour) snapshot, as one disjoint union
            offs = torch.arange(B * H, device=dev) * Z
            edge_index = (zz[:, None, :] + offs[None, :, None]).reshape(2, -1)
        for layer in self.layers:
            u = layer(u, edge_index)
        return self.head(self.ln_out(u)).reshape(B, H, Z).permute(0, 2, 1)


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    torch.manual_seed(cfg.get("seed", 0))
    zones = cfg.get("zones") or tables.zones()
    graph = cfg.get("graph", "interconnects")

    fit_raw = {sp: M.raw(sp, cfg, zones) for sp in fit_splits}
    ev_raw = {sp: M.raw(sp, cfg, zones) for sp in eval_splits}
    scalers = M.fit_scalers(list(fit_raw.values()))
    tens = {sp: M.to_tensors(r, scalers, zones) for sp, r in {**fit_raw, **ev_raw}.items()}
    train_t = {k: torch.cat([tens[sp][k] for sp in fit_splits]) for k in tens[fit_splits[0]]}

    extra: dict = {"spread_pairs": M.spread_pairs(zones)}
    if graph != "none":
        extra["zz"] = torch.as_tensor(G.zone_graph(zones, shuffled=graph == "shuffled", seed=cfg.get("seed", 0)))
    if cfg.get("zone_pe", 0):
        extra["zone_pe"] = torch.as_tensor(G.laplacian_pe(zones, cfg["zone_pe"]))

    mix_cols: list[int] = []
    if cfg.get("hour_mixer", "none") == "bigru":
        mix_cols = [i for i, c in enumerate(fit_raw[fit_splits[0]]["xday_cols"]) if c in F.KNOWN_AT_CUTOFF]
    elif cfg.get("hour_mixer", "none") != "none":
        raise ValueError(f"unknown hour_mixer {cfg['hour_mixer']!r}")

    model = ZoneGAT(n_hist=train_t["hist"].shape[-1], n_day=train_t["xday"].shape[-1], hidden=cfg.get("hidden", 64),
                    heads=cfg.get("heads", 4), layers=cfg.get("layers", 2), mlp_hidden=cfg.get("mlp_hidden", 128),
                    dropout=cfg.get("dropout", 0.1), use_zz=graph != "none", ffn_mult=cfg.get("ffn_mult", 2),
                    n_pe=extra["zone_pe"].shape[1] if "zone_pe" in extra else 0, mix_cols=mix_cols)
    bd = cfg.get("batch_days", 16)
    train_b = T.DayBatches(train_t, bd, shuffle=True, seed=cfg.get("seed", 0))
    val_b = T.DayBatches(tens["val"], bd, shuffle=False) if "val" in tens else None
    info = T.fit(model, train_b, val_b, cfg, extra, T.run_dir(cfg.get("name", "gat")))
    info.update(n_params=sum(p.numel() for p in model.parameters()), graph=graph,
                zone_pe=cfg.get("zone_pe", 0), time_encoding=cfg.get("time_encoding", "raw"),
                hour_mixer=cfg.get("hour_mixer", "none"), history_window=cfg.get("history_window", 168))
    run.info = info  # type: ignore[attr-defined]

    out = {}
    for sp, r in ev_raw.items():
        pred = T.predict(model, T.DayBatches(tens[sp], bd, shuffle=False), extra, cfg.get("device"))
        pred = pred.swapaxes(0, 1).reshape(len(r["s"]), S.H)
        out[sp] = SC.residual_to_price(pred, r["base"]).astype("float32")
    return out
