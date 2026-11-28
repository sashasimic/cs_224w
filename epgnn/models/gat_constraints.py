"""H2 arm: the gat_scope model plus Core flow-based constraints as a second node type.

Question: does the flow-based domain (which lines are tight, and how each zone loads them) help beyond the
zone graph? Control: the same config without `constraints` (c3s_gat_interconnects_curves_rolling).

Timing. The domain of day D is published on D-1 around the cutoff; the registry marks it "unverified", so
it is not used. The domain of D-1 (published on D-2) is safe: every slot of day D sees the constraints of
the same UTC hour on D-1 (24 h earlier), as the `jao_*_lag24` zone features do.

Per slot hour t of day D, the constraints c presolved at t - 24 h (about 110, at most 182) carry
  x_c   8 scale-free features: RAM/Fmax, (RAM-AMR)/Fmax, AMR/Fmax, min-RAM factor, FRM/Fmax, F0core/Fmax,
        F_LTN/Fmax (clipped to [-3, 3]), log(1 + Fmax)/8
  p_cz  the zone-to-rest PTDF towards each of the 12 Core zones (constraints.zone_ptdf, slack-invariant)
Each Core zone z attends over the constraints it is exposed to, |p_cz| >= ptdf_min (default 0.05):
  k_c   = MLP(x_c)                                                  (c_hidden)
  score = a_h . LeakyReLU(Wq u_z + Wk k_c + w_p p_cz + w_a |p_cz|)  (GATv2-style, `c_heads` heads)
  m_z   = sum_c softmax(score) (Wv k_c + v_p p_cz)                  the signed PTDF enters the message
  u_z  += Drop(Linear(m_z)) * g,  g = 1 if z has any exposed constraint, else 0
before the zone GAT blocks, so constraint information then spreads over the interconnectors. Non-Core
zones, the pretrain period (before 2022-06-09) and days without a published domain get no message.

Data path. epgnn.train passes only (hist, xday) to the model, so the constraint store (fp16, about 280 MB)
lives in the model as non-persistent buffers and every slot carries its store row in xday channel 0
(row + 1; 0 = no constraints). That channel is not scaled and is stripped before the input layer.

Config: everything gat_scope takes, plus
  constraints: {ptdf_min: 0.05, c_hidden: 16, c_heads: 4, lag_hours: 24}   (or `constraints: true`)
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as Fn
from torch.utils.checkpoint import checkpoint

from epgnn.models import gat_scope as G
from epgnn.pipeline import constraints as K
from epgnn.pipeline.tensors import FEATURES_DIR

DEFAULTS = {"ptdf_min": K.EXPOSURE_PTDF, "c_hidden": 16, "c_heads": 4, "lag_hours": 24}
C_FEATURES = ["ram_fmax", "ram_grid_fmax", "amr_fmax", "min_ram_factor", "frm_fmax", "f0core_fmax",
              "fltn_fmax", "log_fmax"]


def constraint_config(cfg: dict) -> dict:
    spec = cfg.get("constraints", True)
    c = dict(DEFAULTS)
    if isinstance(spec, dict):
        unknown = set(spec) - set(c)
        assert not unknown, f"unknown constraints keys {sorted(unknown)}"
        c.update(spec)
    assert c["lag_hours"] >= 24, "the domain of day D itself is not known at the cutoff"
    return c


def node_features(h: pd.DataFrame) -> np.ndarray:
    """(rows, 8) scale-free constraint features, NaN -> 0."""
    f = h.fmax.where(h.fmax > 0)
    cols = [h.ram_fmax, h.ram_grid_fmax, h.amr_fmax, h.min_ram_factor, h.frm / f, h.f0core / f, h.fltn / f,
            np.log1p(h.fmax.clip(lower=0)) / 8]
    x = np.stack([np.asarray(c, dtype=np.float64) for c in cols], axis=1)
    return np.nan_to_num(np.clip(x, -3, 3)).astype(np.float32)


def build_store(hourly: pd.DataFrame, hours_ns: np.ndarray, lag_hours: int):
    """Pack constraints_hourly into padded per-hour arrays and point every hour of the grid at the
    constraints of `lag_hours` earlier.
    Returns x (R, C, 8) fp16, p (R, C, 12) fp16 (zone-to-rest PTDF, 0 where padded), valid (R, C) bool,
    and row (T,) int64: store row + 1 for each grid hour, 0 where that earlier hour has no domain."""
    t = hourly.ts_utc.astype("int64").to_numpy()
    uh, inv = np.unique(t, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    cmax = int(counts.max())
    pos = np.arange(len(t)) - np.repeat(np.cumsum(counts) - counts, counts)   # position within its hour
    r_idx, c_idx = inv[order], pos
    x = np.zeros((len(uh), cmax, len(C_FEATURES)), np.float16)
    p = np.zeros((len(uh), cmax, len(K.CORE_ZONES)), np.float16)
    valid = np.zeros((len(uh), cmax), bool)
    x[r_idx, c_idx] = node_features(hourly.iloc[order])
    p[r_idx, c_idx] = np.nan_to_num(K.zone_ptdf(hourly.iloc[order], "rest")).astype(np.float16)
    valid[r_idx, c_idx] = True
    src = hours_ns - lag_hours * 3_600_000_000_000
    k = np.searchsorted(uh, src)
    hit = (k < len(uh)) & (uh[np.minimum(k, len(uh) - 1)] == src)
    row = np.where(hit, k + 1, 0).astype(np.int64)
    return x, p, valid, row


class ConstraintGAT(G.ScopeGAT):
    def __init__(self, n_hist: int, n_day: int, edges: np.ndarray, n_zones: int, pe: np.ndarray | None,
                 store: dict, core_idx: np.ndarray, ptdf_min: float, c_hidden: int, c_heads: int, **kw) -> None:
        super().__init__(n_hist, n_day - 1, edges, n_zones, pe, **kw)         # channel 0 is the store row
        hidden = self.inp.out_features
        dropout = kw.get("dropout", 0.1)
        zero = lambda shape: torch.zeros(shape)                             # noqa: E731  row 0 = "none"
        self.register_buffer("cx", torch.cat([zero((1,) + store["x"].shape[1:]).half(), torch.as_tensor(store["x"])]), persistent=False)
        self.register_buffer("cp", torch.cat([zero((1,) + store["p"].shape[1:]).half(), torch.as_tensor(store["p"])]), persistent=False)
        self.register_buffer("cv", torch.cat([torch.zeros((1,) + store["valid"].shape[1:], dtype=torch.bool),
                                              torch.as_tensor(store["valid"])]), persistent=False)
        core = torch.as_tensor(core_idx, dtype=torch.long)
        self.register_buffer("core_zone", torch.nonzero(core >= 0).flatten(), persistent=False)   # (Zc,) zone ids
        self.register_buffer("core_col", core[core >= 0], persistent=False)                       # (Zc,) PTDF cols
        self.ptdf_min, self.ch, self.heads = ptdf_min, c_hidden, c_heads
        self.c_enc = nn.Sequential(nn.Linear(len(C_FEATURES), c_hidden), nn.ELU(), nn.Linear(c_hidden, c_hidden))
        self.wq = nn.Linear(hidden, c_heads * c_hidden)
        self.wk = nn.Linear(c_hidden, c_heads * c_hidden)
        self.wp = nn.Linear(2, c_heads * c_hidden, bias=False)                # [p, |p|] -> score term
        self.att = nn.Parameter(torch.empty(1, 1, 1, c_heads, c_hidden))
        nn.init.xavier_uniform_(self.att.view(c_heads, c_hidden))
        self.wv = nn.Linear(c_hidden, c_heads * c_hidden)
        self.vp = nn.Parameter(torch.zeros(c_heads * c_hidden))               # signed-PTDF part of the message
        self.out = nn.Linear(c_heads * c_hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self._row = None
        self.last_attention = None

    def forward(self, hist: torch.Tensor, xday: torch.Tensor, extra: dict | None = None,
                return_attention: bool = False):
        self._row = xday[:, 0, :, 0].round().long()                         # (B, H), same for every zone
        try:
            return super().forward(hist, xday[..., 1:], extra, return_attention)
        finally:
            self._row = None

    def constraint_messages(self, u: torch.Tensor, row: torch.Tensor):
        """u: (B, Zc, H, hidden) Core-zone states; row: (B, H). Returns (B, Zc, H, hidden), alpha, gate."""
        b, zc, hs, _ = u.shape
        x = self.cx[row].float()                                           # (B, H, C, 8)
        p = self.cp[row].float()[..., self.core_col]                       # (B, H, C, Zc)
        ok = self.cv[row][..., None] & (p.abs() >= self.ptdf_min)           # (B, H, C, Zc)
        k = self.c_enc(x)                                                   # (B, H, C, ch)
        nh, ch = self.heads, self.ch
        q = self.wq(u).permute(0, 2, 1, 3).reshape(b, hs, 1, zc, nh, ch)    # (B, H, 1, Zc, nh, ch)
        kk = self.wk(k).view(b, hs, -1, 1, nh, ch)                           # (B, H, C, 1, nh, ch)
        pa = torch.stack([p, p.abs()], -1)                                  # (B, H, C, Zc, 2)
        if self.training and torch.is_grad_enabled():                       # the (B,H,C,Zc,nh,ch) term is
            s = checkpoint(self._scores, q, kk, pa, use_reentrant=False)    # recomputed, not stored
        else:
            s = self._scores(q, kk, pa)                                     # (B, H, C, Zc, nh)
        s = s.masked_fill(~ok[..., None], float("-inf"))
        gate = ok.any(2)                                                    # (B, H, Zc)
        alpha = torch.nan_to_num(s.softmax(dim=2), nan=0.0)                 # all-masked columns -> 0
        # message sum_c alpha (Wv k_c + vp p_cz), without forming it per (c, z)
        m = torch.einsum("bhczn,bhcnd->bhznd", alpha, self.wv(k).view(b, hs, -1, nh, ch))
        m = m + torch.einsum("bhczn,bhcz->bhzn", alpha, p)[..., None] * self.vp.view(nh, ch)
        m = m.reshape(b, hs, zc, nh * ch)                                   # (B, H, Zc, nh*ch)
        m = self.drop(self.out(m)) * gate[..., None]
        return m.permute(0, 2, 1, 3), alpha, gate

    def _scores(self, q: torch.Tensor, kk: torch.Tensor, pa: torch.Tensor) -> torch.Tensor:
        b, hs, c, zc, _ = pa.shape
        pp = self.wp(pa).view(b, hs, c, zc, self.heads, self.ch)
        return (Fn.leaky_relu(q + kk + pp, 0.2) * self.att).sum(-1)

    def node_update(self, u: torch.Tensor) -> torch.Tensor:
        if self._row is None or len(self.core_zone) == 0:
            return u
        b, z, hs, d = u.shape
        flat = self._row.reshape(-1)
        sel = torch.nonzero(flat > 0).flatten()                              # (day, slot) pairs with a domain
        if len(sel) == 0:                                                    # e.g. a batch of pretrain days
            return u
        zc = len(self.core_zone)
        uc = u[:, self.core_zone].permute(0, 2, 1, 3).reshape(b * hs, zc, d)[sel]       # (n, Zc, d)
        m, alpha, gate = self.constraint_messages(uc[:, :, None], flat[sel][:, None])   # (n, Zc, 1, d)
        if not self.training:
            self.last_attention = (alpha.detach(), gate.detach())
        full = u.new_zeros(b * hs, zc, d).index_add(0, sel, m[:, :, 0])            # sel is unique: a copy
        full = full.view(b, hs, zc, d).permute(0, 2, 1, 3)                  # (B, Zc, H, d)
        return u.index_add(1, self.core_zone, full)


class Extension:
    """The gat_scope.run extension: adds the store-row channel and builds ConstraintGAT."""

    def __init__(self, cfg: dict) -> None:
        self.c = constraint_config(cfg)
        self.stats: dict = {}

    def prepare(self, cfg, zt, data, day_cols, fit_days):
        cols = ["ts_utc"] + K.NODE_FEATURES + [f"ptdf_{z}" for z in K.CORE_ZONES]
        hourly = pd.read_parquet(FEATURES_DIR / "constraints_hourly.parquet", columns=cols)
        x, p, valid, row = build_store(hourly, np.asarray(zt.hours, dtype=np.int64), self.c["lag_hours"])
        del hourly
        self.store = {"x": x, "p": p, "valid": valid}
        self.core_idx = np.array([K.CORE_ZONES.index(z) if z in K.CORE_ZONES else -1 for z in zt.zones])
        ch = torch.as_tensor(row.astype(np.float32))[:, None, None].expand(-1, data["xday"].shape[1], 1)
        data["xday"] = torch.cat([ch, data["xday"]], dim=-1).contiguous()
        self.stats = {"store_hours": int(x.shape[0]), "max_constraints_per_hour": int(x.shape[1]),
                      "grid_hours_with_constraints": int((row > 0).sum()), "store_mb": round((x.nbytes + p.nbytes) / 2**20)}
        print(f"  constraints: {self.stats}", flush=True)
        return ["constraint_store_row"] + day_cols

    def build(self, n_hist, n_day, edges, n_zones, pe, **kw):
        return ConstraintGAT(n_hist, n_day, edges, n_zones, pe, self.store, self.core_idx, self.c["ptdf_min"],
                             self.c["c_hidden"], self.c["c_heads"], **kw)

    def info(self) -> dict:
        return {"constraints": {**self.c, **self.stats}}


def run(cfg: dict, fit_splits: list[str], eval_splits: list[str]) -> dict[str, np.ndarray]:
    out = G.run(cfg, fit_splits, eval_splits, ext=Extension(cfg))
    run.info = G.run.info                                                   # type: ignore[attr-defined]
    return out
