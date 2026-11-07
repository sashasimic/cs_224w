import unittest

import numpy as np
import pandas as pd

from epgnn.dataset_build.contracts import calendar
from scripts.experiments.pipeline_diagnostics import (WEATHER, prepare_features,
    profile_features, validation_block_interval)


class PipelineDiagnosticTests(unittest.TestCase):
    def test_auction_delivery_dates_and_observed_weather_require_different_masks(self):
        frame = calendar(pd.DataFrame({"zone": "FR", "ts_utc": pd.to_datetime([
            "2024-01-15T00:00Z", "2024-01-15T22:00Z", "2024-10-27T22:00Z"])}), {})
        frame["price"], frame["has_price"], frame["price_lag24"] = 10., True, 7.
        for c in WEATHER:
            frame[c + "_lag24"] = 2.
        weather = pd.DataFrame({"zone": "FR", "time_utc": frame.ts_utc - pd.Timedelta(hours=48)})
        for c in WEATHER:
            weather[c] = 1.
        weather = weather.iloc[:2]  # the missing fall source must remain missing
        out = prepare_features(frame, weather)
        self.assertEqual(out.price_lag24_guarded.iloc[:2].tolist(), [7., 7.])
        self.assertTrue(pd.isna(out.price_lag24_guarded.iloc[2]))
        self.assertEqual(out.temp_pw_lag24_past.iloc[0], 2.)
        self.assertTrue(out.temp_pw_lag24_past.iloc[1:].isna().all())
        self.assertTrue(pd.isna(out.temp_pw_lag48.iloc[2]))
        self.assertEqual(out.has_price.tolist(), [True, True, True])
        self.assertEqual(len(out), len(frame))
        past = profile_features("weather_past24")
        full = profile_features("weather_all24")
        self.assertEqual([c.removesuffix("_past") for c in past], full)

    def test_paired_week_bootstrap_keeps_identical_constant_loss_difference(self):
        before = pd.DataFrame({"zone": "FR", "ts_utc": pd.date_range("2024-04-01", periods=24*21, freq="h", tz="UTC"),
            "price": 0., "target_valid": True, "prediction": 2.})
        before["delivery_date"] = before.ts_utc.dt.strftime("%Y-%m-%d")
        after = before.assign(prediction=1.)
        out = validation_block_interval(before, after, repetitions=100)
        self.assertEqual(out["mean_mae_improvement"], 1.)
        self.assertEqual(out["ci95_lower"], 1.)
        self.assertEqual(out["ci95_upper"], 1.)
        after.loc[0, "target_valid"] = False
        with self.assertRaises(AssertionError):
            validation_block_interval(before, after, repetitions=100)


if __name__ == "__main__":
    unittest.main()
