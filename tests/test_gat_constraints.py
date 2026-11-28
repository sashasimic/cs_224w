"""Checks for the H2 constraint arm (epgnn.models.gat_constraints).

    python -m unittest tests/test_gat_constraints.py -v
"""
import unittest

import numpy as np
import pandas as pd
import torch

from epgnn.models import gat_constraints as GC
from epgnn.pipeline import constraints as K

HOUR = 3_600_000_000_000


def fake_hourly(hours: list[int], per_hour: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for t in hours:
        for c in range(per_hour):
            r = {"ts_utc": pd.Timestamp(t, tz="UTC"), "ram": 50.0 + c, "fmax": 100.0, "frm": 10.0, "f0core": 5.0,
                 "fltn": 3.0, "amr": 0.0, "iva": 0.0, "min_ram_factor": 0.7}
            r["ram_fmax"], r["ram_grid_fmax"], r["amr_fmax"] = r["ram"] / 100, r["ram"] / 100, 0.0
            r.update({f"ptdf_{z}": rng.normal(0, 0.1) for z in K.CORE_ZONES})
            rows.append(r)
    return pd.DataFrame(rows)


class TestStore(unittest.TestCase):
    def test_rows_point_24h_back(self):
        hours = [100 * HOUR, 101 * HOUR, 103 * HOUR]
        h = fake_hourly(hours, 3)
        h = h.iloc[:-1]                                             # last hour has only 2 constraints
        grid = np.arange(120, 130) * HOUR
        x, p, valid, row = GC.build_store(h, grid, 24)
        self.assertEqual(x.shape, (3, 3, len(GC.C_FEATURES)))
        self.assertEqual(valid.sum(1).tolist(), [3, 3, 2])
        expect = {124: 1, 125: 2, 127: 3}                            # grid hour -> store row + 1
        for i, t in enumerate(range(120, 130)):
            self.assertEqual(row[i], expect.get(t, 0), t)
        np.testing.assert_allclose(p[0, 0], K.zone_ptdf(h.iloc[[0]], "rest")[0], atol=1e-3)
        self.assertTrue(np.isfinite(x.astype(np.float32)).all())

    def test_same_day_domain_is_refused(self):
        with self.assertRaises(AssertionError):
            GC.constraint_config({"constraints": {"lag_hours": 0}})


def model_and_inputs(graph_edges: np.ndarray, zones: list[str], seed: int = 0):
    torch.manual_seed(seed)
    h = fake_hourly([0], 6, seed)
    x, p, valid, _ = GC.build_store(h, np.array([0]), 24)
    core = np.array([K.CORE_ZONES.index(z) if z in K.CORE_ZONES else -1 for z in zones])
    m = GC.ConstraintGAT(2, 4, graph_edges, len(zones), None, {"x": x, "p": p, "valid": valid}, core,
                         0.05, 8, 2, hidden=16, heads=2, layers=1, ffn=16, mlp_hidden=16, dropout=0.0)
    b, hs, w = 2, 3, 5
    hist = torch.randn(b, len(zones), w, 2)
    xday = torch.randn(b, len(zones), hs, 4)
    return m, hist, xday


class TestConstraintGAT(unittest.TestCase):
    zones = ["AT", "FR", "GB", "NO_1"]                                # 2 Core, 2 not
    self_loops = np.zeros((2, 0), dtype=np.int64)

    def run_with(self, m, hist, xday, row: int):
        x = xday.clone()
        x[..., 0] = row
        m.eval()
        with torch.no_grad():
            return m(hist, x)

    def test_messages_reach_core_zones_only(self):
        m, hist, xday = model_and_inputs(self.self_loops, self.zones)
        with torch.no_grad():
            m.out.weight.normal_()                                       # make the message non-trivial
        none, some = self.run_with(m, hist, xday, 0), self.run_with(m, hist, xday, 1)
        diff = (none - some).abs().amax(dim=(0, 2))                    # per zone
        self.assertGreater(float(diff[:2].min()), 1e-6)               # AT, FR change
        self.assertEqual(float(diff[2:].max()), 0.0)                  # GB, NO_1 do not (self graph)

    def test_unexposed_constraints_are_ignored(self):
        m, hist, xday = model_and_inputs(self.self_loops, self.zones)
        with torch.no_grad():
            m.out.weight.normal_()
            m.cp[1] = m.cp[1] * 0.01                                   # all |PTDF| now < 0.05
        none, some = self.run_with(m, hist, xday, 0), self.run_with(m, hist, xday, 1)
        torch.testing.assert_close(none, some)

    def test_backward_with_checkpoint(self):
        m, hist, xday = model_and_inputs(self.self_loops, self.zones)
        x = xday.clone()
        x[..., 0] = 1
        m.train()
        m(hist, x).sum().backward()
        self.assertIsNotNone(m.c_enc[0].weight.grad)
        self.assertGreater(float(m.c_enc[0].weight.grad.abs().sum()), 0)

    def test_store_not_in_state_dict(self):
        m, _, _ = model_and_inputs(self.self_loops, self.zones)
        self.assertFalse(any(k in m.state_dict() for k in ("cx", "cp", "cv")))


if __name__ == "__main__":
    unittest.main()
