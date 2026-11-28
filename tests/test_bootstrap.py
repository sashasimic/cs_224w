"""Checks for the paired weekly block bootstrap (epgnn.bootstrap).

    python -m unittest tests/test_bootstrap.py -v
"""
import unittest

import numpy as np

from epgnn import bootstrap as B


class TestBlockBootstrap(unittest.TestCase):
    def test_delta_is_the_mae_difference(self):
        rng = np.random.default_rng(0)
        ea, eb = rng.gamma(2, 10, (300, 24)), rng.gamma(2, 10, (300, 24))
        ok = rng.random((300, 24)) > 0.1
        day = np.repeat(np.arange(100), 3)
        sa, sb, cnt = B.day_sums(ea, eb, ok, day, 100)
        r = B.block_bootstrap(sa, sb, cnt, n=500)
        self.assertAlmostEqual(r["delta"], ea[ok].mean() - eb[ok].mean(), places=9)
        self.assertAlmostEqual(r["mae_a"] - r["mae_b"], r["delta"], places=9)

    def test_clear_win_and_null(self):
        rng = np.random.default_rng(1)
        base = rng.gamma(2, 10, (400, 24))
        ok = np.ones_like(base, bool)
        day = np.arange(400)
        sa, sb, cnt = B.day_sums(base - 1.0, base, ok, day, 400)           # A better by exactly 1
        r = B.block_bootstrap(sa, sb, cnt, n=500)
        self.assertAlmostEqual(r["delta"], -1.0)
        self.assertLess(r["ci95"][1], 0)
        self.assertEqual(r["p_a_not_better"], 0.0)
        noise = base + rng.normal(0, 3, base.shape)                           # no systematic difference
        sa, sb, cnt = B.day_sums(np.abs(noise), np.abs(base), ok, day, 400)
        r = B.block_bootstrap(sa, sb, cnt, n=500)
        self.assertLess(r["ci95"][0], r["delta"])
        self.assertGreater(r["ci95"][1], r["delta"])

    def test_blocks_widen_ci_under_autocorrelation(self):
        rng = np.random.default_rng(2)
        weekly = np.repeat(rng.normal(0, 2, 60), 7)                           # shared weekly shocks
        sa = 20 + weekly + rng.normal(0, 0.2, 420)
        sb = np.full(420, 20.0)
        cnt = np.ones(420)
        w1 = np.diff(B.block_bootstrap(sa, sb, cnt, block=1, n=2000)["ci95"])[0]
        w7 = np.diff(B.block_bootstrap(sa, sb, cnt, block=7, n=2000)["ci95"])[0]
        self.assertGreater(w7, 1.5 * w1)


if __name__ == "__main__":
    unittest.main()
