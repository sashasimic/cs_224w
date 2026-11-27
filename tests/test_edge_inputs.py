"""Link features: aligned with the graph, known at the cutoff, and attached as intended on the shuffled control.
Run from the repository root: python -m pytest tests/test_edge_inputs.py

  1. values equal the edge table (log1p MW), forward and reverse direction of every link
  2. on the 25-hour day the last hour's scheduled exchange does not come from that day's own schedule
  3. overwriting everything unknown at the cutoff leaves the features of that day unchanged
  4. the shuffled graph carries every real link's series exactly once
  5. in the model, the link features of one hour move only that hour's forecasts (both message types)
"""
import numpy as np
import pandas as pd
import torch

from epgnn.data import features as F
from epgnn.data import graph as G
from epgnn.data import samples as S
from epgnn.data import tables
from epgnn.models.gat import ZoneGAT

ZONES = ["BE", "DE_LU", "FR", "NL"]          # Core zones: the scheduled exchange exists from June 2022
COLS = F.EDGE_KNOWN_AT_CUTOFF
H1 = pd.Timedelta(hours=1)


def _hours(day: str) -> np.ndarray:
    n = tables.nodes()
    return pd.DatetimeIndex(n[(n.zone == "FR") & (n.delivery_date == day)].ts_utc).tz_convert("UTC").as_unit("ns").asi8


def _edges() -> pd.DataFrame:
    e = tables.edges()
    return e[e.from_zone.isin(ZONES) & e.to_zone.isin(ZONES)].copy()


def test_values_match_the_table() -> None:
    hours = _hours("2024-07-10")
    x, ei = G.edge_inputs(ZONES, hours), G.zone_graph(ZONES)
    e = _edges().set_index(["from_zone", "to_zone", "ts_utc"])
    assert x.shape == (len(hours), ei.shape[1], 2 * len(COLS))
    for k in range(ei.shape[1]):
        a, b = ZONES[ei[0, k]], ZONES[ei[1, k]]
        for t in (0, 11, 23):
            ts = pd.Timestamp(hours[t], tz="UTC")
            want = np.log1p(np.concatenate([e.loc[(a, b, ts), COLS].to_numpy("float32"), e.loc[(b, a, ts), COLS].to_numpy("float32")]))
            assert np.allclose(x[t, k], want, equal_nan=True, atol=1e-5)


def test_scheduled_exchange_never_comes_from_day_d() -> None:
    hours = _hours("2024-10-27")
    assert len(hours) == 25
    x, ei, j = G.edge_inputs(ZONES, hours), G.zone_graph(ZONES), COLS.index("flow_sched_lag24")
    e = _edges().set_index(["from_zone", "to_zone", "ts_utc"])
    last, first = pd.Timestamp(hours[24], tz="UTC"), pd.Timestamp(hours[0], tz="UTC")
    stored = np.array([e.loc[(ZONES[a], ZONES[b], last), "flow_sched_lag24"] for a, b in ei.T])
    own_day = np.array([e.loc[(ZONES[a], ZONES[b], first), "flow_sched"] for a, b in ei.T])
    assert np.allclose(stored, own_day, equal_nan=True)                 # the table's lag IS the day's own schedule
    assert np.array_equal(x[24, :, j], x[23, :, j], equal_nan=True)     # we pass on the previous hour's value
    assert not np.allclose(x[24, :, j], np.log1p(stored), equal_nan=True)


def test_destroying_the_future_changes_nothing() -> None:
    for day in ("2024-10-27", "2025-01-15", "2025-03-30"):
        hours = _hours(day)
        clean = G.edge_inputs(ZONES, hours)                              # also fills the cached link list
        small = _edges()
        t = pd.DatetimeIndex(small.ts_utc)
        cut = pd.Timestamp(day, tz="UTC") - pd.Timedelta(days=1) + pd.Timedelta(hours=S.CUTOFF_UTC)
        start = pd.Timestamp(hours[0], tz="UTC")
        small.loc[t >= cut, "flow_phys"] = 9e3                           # flows of hours not finished at the cutoff
        small.loc[t - 48 * H1 >= cut, "flow_phys_lag48"] = 9e3
        small.loc[t - 168 * H1 >= cut, "flow_phys_lag168"] = 9e3
        small.loc[t >= start, ["flow_sched", "maxbex"]] = 9e3            # schedules of day D and later: auction outcomes
        small.loc[t - 24 * H1 >= start, "flow_sched_lag24"] = 9e3
        original = tables.edges
        tables.edges = lambda: small
        try:
            bad = G.edge_inputs(ZONES, hours)
        finally:
            tables.edges = original
        assert np.isfinite(clean).any() and np.array_equal(clean, bad, equal_nan=True), day


def test_shuffled_graph_gets_every_real_series_once() -> None:
    zones, hours = tables.zones(), _hours("2024-07-10")
    real, shuf = G.edge_inputs(zones, hours), G.edge_inputs(zones, hours, shuffled=True, seed=0)
    fingerprint = lambda x: sorted(np.round(np.nan_to_num(x, nan=-1.0).sum(axis=(0, 2)), 3).tolist())
    assert real.shape == shuf.shape and fingerprint(real) == fingerprint(shuf)
    assert not np.array_equal(real, shuf, equal_nan=True)


def test_link_features_move_only_their_own_hour() -> None:
    torch.manual_seed(0)
    cols = F.FEATURE_SETS["own"]
    known = [i for i, c in enumerate(cols) if c in F.KNOWN_AT_CUTOFF]
    B, Z, H, hour = 2, 4, S.H, 7
    zz = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    hist, xday = torch.randn(B, Z, 24, len(F.HISTORY)), torch.randn(B, Z, H, len(cols))
    edge_x, hidx = torch.randn(B * H, zz.shape[1], 6), torch.arange(B * H).reshape(B, H)
    for message in ("gat", "pair"):
        model = ZoneGAT(len(F.HISTORY), len(cols), mix_cols=known, edge_dim=6, message=message).eval()
        e2 = edge_x.clone()
        e2[hidx[0, hour]] += 3.0
        with torch.no_grad():
            d = (model(hist, xday, {"zz": zz, "edge_x": e2}, {"hidx": hidx}) - model(hist, xday, {"zz": zz, "edge_x": edge_x}, {"hidx": hidx})).abs()
        assert float(d[0, :, hour].max()) > 1e-4, message
        d[0, :, hour] = 0
        assert float(d.max()) < 1e-6, message
