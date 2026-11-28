"""Checks for the engineered zone features (epgnn.pipeline.zone_features): cutoff-safe weather.

    python -m unittest tests/test_zone_features.py -v

The forecast for delivery day D is made at D-1 10:00 UTC. A weather value used as a day-D input must have
been observed by then; the `_lag24` columns of the node table break that for the afternoon hours.
"""
import unittest

import numpy as np
import pandas as pd

from epgnn.models import gat_scope as G
from epgnn.pipeline import registry
from epgnn.pipeline import zone_features as ZF


def frame(first_day: str, n_days: int) -> pd.DataFrame:
    """One zone, every hour of n_days Brussels delivery days; each weather value is its own UTC hour number,
    so the value says when it was observed."""
    lo = pd.Timestamp(first_day, tz="Europe/Brussels").tz_convert("UTC")
    hi = (pd.Timestamp(first_day, tz="Europe/Brussels") + pd.Timedelta(days=n_days)).tz_convert("UTC")
    ts = pd.date_range(lo, hi, freq="h", inclusive="left")
    n = pd.DataFrame({"zone": "Z", "ts_utc": ts,
                      "delivery_date": ts.tz_convert("Europe/Brussels").strftime("%Y-%m-%d")})
    stamp = (ts.asi8 // 3_600_000_000_000).astype("float64")
    for c in ZF.WEATHER:
        n[c] = stamp
        n[f"{c}_lag24"] = n[c].shift(24)
    return n


class TestCutoffSafeWeather(unittest.TestCase):
    def test_which_hours_may_use_lag24(self):
        for day, n_ok in (("2024-01-15", 12), ("2024-07-15", 13)):        # CET: 23:00-10:00 UTC; CEST: 22:00-10:00
            n = frame(day, 1)
            ok = ZF.lag24_is_observed(n.ts_utc, n.delivery_date)
            self.assertEqual(int(ok.sum()), n_ok, day)
            self.assertTrue(ok[:n_ok].all() and not ok[n_ok:].any())    # a prefix of the day

    def test_asof_is_always_observed_at_the_cutoff(self):
        for first in ("2024-01-10", "2024-03-28", "2024-07-10", "2024-10-25"):    # incl. both DST changes
            n = frame(first, 6)
            ZF.add_weather(n)
            cutoff = (pd.to_datetime(n.delivery_date, utc=True) - pd.Timedelta(days=1)
                      + pd.Timedelta(hours=ZF.CUTOFF_UTC)).astype("int64") // 3_600_000_000_000
            src = n.temp_pw_asof
            ok = src.notna()
            self.assertTrue(ok.iloc[72:].all())
            self.assertTrue((src[ok] <= cutoff[ok]).all(), first)                   # never after the cutoff
            leak = n.temp_pw_lag24 > cutoff
            self.assertTrue(leak.iloc[72:].any(), "the old column does leak; the test would be vacuous otherwise")
            # where lag 24 is already observed it is used unchanged; otherwise lag 48
            np.testing.assert_array_equal(src[ok & ~leak], n.temp_pw_lag24[ok & ~leak])
            np.testing.assert_array_equal(src[ok & leak], n.temp_pw_lag48[ok & leak])
            np.testing.assert_array_equal(n.heat_stress[ok], n.temp_anom_asof[ok].clip(lower=0))

    def test_registry_and_day_sets(self):
        for c in ZF.WEATHER:
            self.assertEqual(registry.timing_of(f"{c}_lag24"), "past")
            self.assertEqual(registry.timing_of(f"{c}_asof"), "gate")
        self.assertEqual(G.check_day_inputs(G.DAY_SETS["own_cyclic_asof"]), [])
        self.assertEqual(G.check_day_inputs(G.DAY_SETS["own_cyclic_plus_asof"]), [])
        with self.assertWarns(UserWarning):
            G.check_day_inputs(G.DAY_SETS["own_cyclic"])
        with self.assertRaises(ValueError):
            G.check_day_inputs(G.DAY_SETS["own_cyclic"], strict=True)


if __name__ == "__main__":
    unittest.main()
