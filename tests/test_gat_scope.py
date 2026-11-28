"""Checks for the attention-scope GAT (epgnn.models.gat_scope) and the pipeline graph helpers.

    python -m unittest tests/test_gat_scope.py -v

The PyG equivalence test runs only when torch_geometric is importable.
"""
import unittest

import numpy as np
import torch

import pandas as pd

from epgnn.models.gat_scope import DenseGATv2, GatherBatches, ScopeGAT, price_curves
from epgnn.models import gat_scope as G
from epgnn.pipeline import graphs, tensors

Z = 45


def ring(n: int) -> np.ndarray:
    a = np.arange(n)
    src = np.concatenate([a, (a + 1) % n])
    dst = np.concatenate([(a + 1) % n, a])
    return np.stack([src, dst])


def model(edges: np.ndarray, pe: bool = True, n_day: int = 14) -> ScopeGAT:
    torch.manual_seed(0)
    p = np.random.default_rng(0).normal(size=(Z, 8)).astype(np.float32) if pe else None
    return ScopeGAT(8, n_day, edges, Z, p, checkpoint_scores=False)


class TestScopeGAT(unittest.TestCase):
    def test_parameter_count_matches_c3_for_every_graph(self):
        for e in (np.zeros((2, 0), np.int64), ring(Z), graphs.full_edges(Z)):
            self.assertEqual(sum(p.numel() for p in model(e).parameters()), 95_425)
        # the collaborators' run without position and cyclic time: 11 day inputs, no PE
        self.assertEqual(sum(p.numel() for p in model(ring(Z), pe=False, n_day=11).parameters()), 94_721)

    def test_receptive_field(self):
        """Changing one far zone moves another zone's output only when attention can reach it."""
        torch.manual_seed(1)
        hist, xday = torch.randn(1, Z, 24, 8), torch.randn(1, Z, 3, 14)
        far = 20                                            # 20 hops from zone 0 on the ring
        bumped = xday.clone()
        bumped[:, far] += 5
        for edges, reach in ((np.zeros((2, 0), np.int64), False), (ring(Z), False), (graphs.full_edges(Z), True)):
            m = model(edges).eval()
            with torch.no_grad():
                d = (m(hist, bumped) - m(hist, xday))[0, 0].abs().max().item()
            self.assertEqual(d > 1e-6, reach, f"edges={edges.shape[1]} diff={d}")

    def test_attention_rows_sum_to_one_and_respect_edges(self):
        e = torch.as_tensor(ring(6))
        layer = DenseGATv2(8, 4, 2, e, 6, checkpoint_scores=False)
        _, alpha = layer(torch.randn(3, 6, 8), return_attention=True)     # (G, dst, src, heads)
        torch.testing.assert_close(alpha.sum(2), torch.ones(3, 6, 2))
        allowed = torch.eye(6, dtype=torch.bool)
        allowed[e[1], e[0]] = True
        self.assertTrue((alpha[:, ~allowed] == 0).all())

    def test_checkpointing_gives_same_gradients(self):
        e = torch.as_tensor(graphs.full_edges(7))
        torch.manual_seed(2)
        a = DenseGATv2(8, 4, 2, e, 7, checkpoint_scores=False)
        b = DenseGATv2(8, 4, 2, e, 7, checkpoint_scores=True)
        b.load_state_dict(a.state_dict())
        x = torch.randn(5, 7, 8)
        for m in (a, b):
            m.train()
            m(x).square().sum().backward()
        for pa_, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa_.grad, pb.grad)

    def test_equals_torch_geometric_gatv2conv(self):
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError:
            self.skipTest("torch_geometric not installed")
        torch.manual_seed(3)
        n, g = 9, 4
        e = torch.as_tensor(ring(n))
        ours = DenseGATv2(16, 4, 3, e, n, checkpoint_scores=False)
        ref = GATv2Conv(16, 4, heads=3, concat=True, add_self_loops=True, share_weights=False)
        with torch.no_grad():
            ref.lin_l.weight.copy_(ours.lin_l.weight)
            ref.lin_l.bias.copy_(ours.lin_l.bias)
            ref.lin_r.weight.copy_(ours.lin_r.weight)
            ref.lin_r.bias.copy_(ours.lin_r.bias)
            ref.att.copy_(ours.att)
            ref.bias.copy_(ours.bias)
        x = torch.randn(g, n, 16)
        big = torch.cat([e + k * n for k in range(g)], dim=1)               # g disjoint copies
        torch.testing.assert_close(ours(x).reshape(g * n, -1), ref(x.reshape(g * n, 16), big), rtol=1e-5, atol=1e-5)


class TestPipelineGraphs(unittest.TestCase):
    def test_full_edges(self):
        e = graphs.full_edges(4)
        self.assertEqual(e.shape, (2, 12))
        self.assertFalse((e[0] == e[1]).any())

    def test_history_window_ends_at_cutoff(self):
        t, z = 400, 3
        hist = torch.arange(t, dtype=torch.float32)[:, None, None].expand(t, z, 1).clone()
        data = {"hist": hist, "xday": torch.zeros(t, z, 2), "target": torch.zeros(t, z), "mask": torch.ones(t, z, dtype=torch.bool),
                "slots": np.array([[200 + k for k in range(24)] + [-1], [224 + k for k in range(25)]]),
                "cutoff": np.array([186, 210])}
        b = GatherBatches(data, np.array([0, 1]), 2, window=168, shuffle=False).batch(np.array([0, 1]))
        self.assertEqual(b["hist"].shape, (2, z, 168, 1))
        self.assertEqual(b["hist"][0, 0, -1, 0].item(), 186)
        self.assertEqual(b["hist"][0, 0, 0, 0].item(), 186 - 167)
        self.assertEqual(b["xday"].shape, (2, z, 25, 2))
        self.assertFalse(b["mask"][0, :, 24].any())                          # the missing 25th slot
        self.assertTrue(b["mask"][1, :, 24].all())


def fake_tensors(first_day: str, n_days: int, z: int = 2) -> tensors.ZoneTensors:
    """A ZoneTensors over n_days Brussels delivery days whose price encodes (day, local hour, zone)."""
    lo = pd.Timestamp(first_day, tz="Europe/Brussels").tz_convert("UTC")
    hi = (pd.Timestamp(first_day, tz="Europe/Brussels") + pd.Timedelta(days=n_days)).tz_convert("UTC")
    ts = pd.date_range(lo, hi, freq="h", inclusive="left")
    loc = ts.tz_convert("Europe/Brussels")
    dd = loc.strftime("%Y-%m-%d").to_numpy()
    days, first = np.unique(dd, return_index=True)
    counts = np.diff(np.append(first, len(ts)))
    slots = np.full((len(days), 25), -1, dtype=np.int64)
    for i, (f, c) in enumerate(zip(first, counts)):
        slots[i, :c] = np.arange(f, f + c)
    day_no = np.searchsorted(days, dd).astype(np.float32)
    price = (1000 * day_no[:, None] + 10 * loc.hour.to_numpy()[:, None] + np.arange(z)[None, :]).astype(np.float32)
    return tensors.ZoneTensors(hours=ts.asi8, zones=[f"Z{i}" for i in range(z)], cols=[], x=np.zeros((len(ts), z, 0)),
                               price=price, has_price=np.ones_like(price, bool), base=price, days=days,
                               slots=slots, cutoff=np.zeros(len(days), np.int64))


class TestPriceCurves(unittest.TestCase):
    def decode(self, c: np.ndarray) -> np.ndarray:
        return np.sinh(c) * 50

    def test_values_come_from_earlier_days_only(self):
        zt = fake_tensors("2024-01-08", 10)
        c, names = price_curves(zt, [1, 2, 7])
        self.assertEqual(c.shape, (10, 2, 72))
        self.assertEqual(names[0], "price_d-1_h00")
        self.assertEqual(names[24 + 5], "price_d-2_h05")
        v = self.decode(c)
        for d in (7, 9):
            for lag, blk in ((1, 0), (2, 1), (7, 2)):
                expect = 1000 * (d - lag) + 10 * np.arange(24) + 1            # zone 1
                np.testing.assert_allclose(v[d, 1, blk * 24:(blk + 1) * 24], expect, rtol=1e-4)
        self.assertTrue(np.isnan(c[0]).all())                                  # no earlier day available
        self.assertTrue(np.isnan(c[6, :, 48:]).all() and np.isfinite(c[7, :, 48:]).all())   # D-7 from day 7

    def test_rejects_same_day(self):
        with self.assertRaises(AssertionError):
            price_curves(fake_tensors("2024-01-08", 3), [0, 1])

    def test_dst_days_align_by_local_hour(self):
        zt = fake_tensors("2024-03-30", 3)                                     # 31 Mar has 23 hours
        self.assertEqual((zt.slots[1] >= 0).sum(), 23)
        v = self.decode(price_curves(zt, [1])[0])
        self.assertTrue(np.isnan(v[2, 0, 2]))                                  # 02:00 does not exist on 31 Mar
        np.testing.assert_allclose(v[2, 0, 3], 1000 + 30, rtol=1e-4)           # 03:00 is still hour 3
        np.testing.assert_allclose(v[1, 0, 2], 0 + 20, rtol=1e-4)              # 30 Mar is a normal day

    def test_batch_appends_curves_to_every_valid_slot(self):
        t, z = 400, 3
        data = {"hist": torch.zeros(t, z, 1), "xday": torch.ones(t, z, 2), "target": torch.zeros(t, z),
                "mask": torch.ones(t, z, dtype=torch.bool),
                "slots": np.array([[200 + k for k in range(24)] + [-1], [224 + k for k in range(25)]]),
                "cutoff": np.array([186, 210]), "curves": torch.arange(2 * z * 4, dtype=torch.float32).reshape(2, z, 4) + 1}
        b = GatherBatches(data, np.array([0, 1]), 2, window=24, shuffle=False).batch(np.array([1, 0]))
        self.assertEqual(b["xday"].shape, (2, z, 25, 6))
        torch.testing.assert_close(b["xday"][0, 2, 7, 2:], data["curves"][1, 2])   # batch row 0 is day 1
        torch.testing.assert_close(b["xday"][1, 0, 3, 2:], data["curves"][0, 0])
        self.assertTrue((b["xday"][1, :, 24] == 0).all())                        # padded slot stays zero


class TestRollingSchedule(unittest.TestCase):
    """Monthly fine-tuning (cfg `rolling`): never trains on the month it predicts."""

    def setUp(self):
        self.days = pd.date_range("2024-01-01", "2025-12-31", freq="D").strftime("%Y-%m-%d").to_numpy()
        self.ix = {d: i for i, d in enumerate(self.days)}

    def sched(self, first_eval: str, last_fit: str, window: int = 90):
        ev = np.arange(self.ix[first_eval], len(self.days))
        return G.rolling_schedule(self.days, ev, self.ix[last_fit], window), ev

    def test_no_future_and_full_coverage(self):
        sched, ev = self.sched("2025-01-01", "2024-12-31")
        seen = np.concatenate([p for _, p, _ in sched])
        np.testing.assert_array_equal(np.sort(seen), ev)                       # every eval day once
        for m, pred, ft in sched:
            self.assertTrue(all(self.days[i][:7] == m for i in pred))
            if ft is not None:
                self.assertEqual(ft[-1], pred[0] - 1)                          # ends the day before the month
                self.assertLessEqual(len(ft), 90)
                self.assertTrue((ft < pred.min()).all())

    def test_first_month_after_fit_is_not_finetuned(self):
        sched, _ = self.sched("2025-01-01", "2024-12-31")
        self.assertIsNone(sched[0][2])
        self.assertTrue(all(ft is not None for _, _, ft in sched[1:]))
        self.assertEqual(len(sched), 12)

    def test_gap_after_fit_is_caught_up_first(self):
        sched, _ = self.sched("2025-04-01", "2024-12-31")                    # eval test only, val skipped
        m, pred, ft = sched[0]
        self.assertEqual(self.days[ft[-1]], "2025-03-31")
        self.assertEqual(self.days[ft[0]], "2025-01-01")                      # 90-day window

    def test_config(self):
        self.assertIsNone(G.rolling_config({}))
        r = G.rolling_config({"rolling": True, "weight_decay": 0.01})
        self.assertEqual((r["window_days"], r["epochs"], r["weight_decay"]), (730, 2, 0.01))
        self.assertEqual(G.rolling_config({"rolling": {"epochs": 1, "start": "base"}})["start"], "base")
        with self.assertRaises(AssertionError):
            G.rolling_config({"rolling": {"windw_days": 30}})


class TestRollEndToEnd(unittest.TestCase):
    """`roll` on a tiny model and fake tensors: predicts every eval day, fine-tunes only on earlier days."""

    def test_roll(self):
        torch.manual_seed(0)
        zt = fake_tensors("2024-01-01", 120, z=3)
        zt.cutoff = np.clip(zt.slots[:, 0] - 14, 0, None)
        T_, Z = len(zt.hours), 3
        data = {"hist": torch.randn(T_, Z, 2), "xday": torch.randn(T_, Z, 3),
                "target": torch.randn(T_, Z) * 0.1, "mask": torch.ones(T_, Z, dtype=torch.bool),
                "slots": zt.slots, "cutoff": zt.cutoff}
        split = {"ev": np.arange(60, 120)}
        zt.day_indices = lambda sp: split[sp]
        m = ScopeGAT(2, 3, ring(Z), Z, None, hidden=8, heads=2, layers=1, ffn=16, mlp_hidden=16, dropout=0.0)
        seen = []
        orig = G.GatherBatches.__init__

        def spy(self, data, days, *a, **k):
            seen.append((np.asarray(days).copy(), k.get("shuffle", a[2] if len(a) > 2 else None)))
            orig(self, data, days, *a, **k)

        G.GatherBatches.__init__ = spy
        try:
            r = G.rolling_config({"rolling": {"window_days": 20, "epochs": 1}})
            out, log = G.roll(m, data, zt, ["ev"], 59, r, {"spread_weight": 0.0}, {"spread_pairs": None}, 8, 24)
        finally:
            G.GatherBatches.__init__ = orig
        self.assertEqual(out["ev"].shape, (Z, 60, tensors.H))
        valid = (zt.slots[60:120] >= 0)[None].repeat(Z, 0)
        self.assertTrue(np.isfinite(out["ev"][valid]).all())
        self.assertEqual([x["month"] for x in log], ["2024-03", "2024-04"])
        self.assertIsNone(log[0]["finetune"])
        # every fine-tune batch set lies before the month predicted right after it
        train_sets = [d for d, shuffle in seen if shuffle]
        self.assertEqual(len(train_sets), 1)
        self.assertLess(train_sets[0].max(), int(np.flatnonzero(np.char.startswith(zt.days.astype(str), "2024-04"))[0]))


if __name__ == "__main__":
    unittest.main()
