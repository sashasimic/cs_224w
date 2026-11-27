"""The two graph controls do what they claim. Run from the repository root: python -m pytest tests/test_graph_controls.py

  shuffled  same number of links at every zone, links two-way, no self-links or duplicates, few links kept
  self      every zone attends only to itself: changing one zone's inputs moves no other zone's forecast,
            and the model has exactly the parameters of the real-graph model
"""
import numpy as np
import torch

from epgnn.data import features as F
from epgnn.data import graph as G
from epgnn.data import samples as S
from epgnn.data import tables
from epgnn.models.gat import ZoneGAT


def test_shuffled_graph_keeps_every_zones_link_count() -> None:
    zones = tables.zones()
    real = G.zone_graph(zones)
    for seed in (0, 1, 2):
        new = G.zone_graph(zones, shuffled=True, seed=seed)
        pairs = set(map(tuple, new.T.tolist()))
        assert new.shape == real.shape and len(pairs) == new.shape[1]                 # same size, no duplicates
        assert (new[0] != new[1]).all()                                               # no self-links
        assert all((b, a) in pairs for a, b in pairs)                                 # two-way
        assert np.array_equal(np.bincount(new[0], minlength=len(zones)), np.bincount(real[0], minlength=len(zones)))
        kept = len(pairs & set(map(tuple, real.T.tolist()))) / real.shape[1]
        assert kept < 0.5, f"shuffle seed {seed} kept {kept:.0%} of the real links"
    assert not np.array_equal(G.zone_graph(zones, shuffled=True, seed=0), G.zone_graph(zones, shuffled=True, seed=1))


def test_self_loop_graph_isolates_zones() -> None:
    torch.manual_seed(0)
    cols = F.FEATURE_SETS["own"]
    known = [i for i, c in enumerate(cols) if c in F.KNOWN_AT_CUTOFF]
    B, Z, H = 2, 4, S.H
    hist, xday = torch.randn(B, Z, 24, len(F.HISTORY)), torch.randn(B, Z, H, len(cols))
    model = ZoneGAT(len(F.HISTORY), len(cols), mix_cols=known).eval()
    real = {"zz": torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])}
    alone = {"zz": torch.zeros((2, 0), dtype=torch.long)}

    def moved_elsewhere(extra: dict) -> float:
        """Largest change at any zone other than zone 1 when only zone 1's inputs change."""
        h2, x2 = hist.clone(), xday.clone()
        h2[:, 1] += 3.0
        x2[:, 1] += 3.0
        with torch.no_grad():
            d = (model(h2, x2, extra) - model(hist, xday, extra)).abs()
        assert float(d[:, 1].max()) > 1e-4
        d[:, 1] = 0
        return float(d.max())

    assert moved_elsewhere(alone) < 1e-6          # self-loops only: zones never hear each other
    assert moved_elsewhere(real) > 1e-4           # real links: they do (same model, same weights)
