from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from epgnn.boosting import (TEMPERATURE_CURRENT, TEMPERATURE_PERFECT,
    check_features, experiment_registry, join_temperature_history, load_inputs)
from epgnn.dataset_build.contracts import calendar, feature_registry
from epgnn.dataset_build.storage import digest, json_write


class TemperatureExperimentTests(unittest.TestCase):
    def test_oracle_requires_separate_opt_in_and_stays_excluded_from_strict(self):
        original = feature_registry()
        original_snapshot = [dict(r) for r in original]
        registry = experiment_registry(original)
        check_features(["hour_local", TEMPERATURE_CURRENT], registry, [TEMPERATURE_CURRENT])
        check_features(["hour_local", TEMPERATURE_PERFECT], registry, [], [TEMPERATURE_PERFECT])
        with self.assertRaises(ValueError):
            check_features(["hour_local", TEMPERATURE_PERFECT], registry, [TEMPERATURE_PERFECT])
        with self.assertRaises(ValueError):
            check_features(["hour_local", TEMPERATURE_PERFECT], registry)
        with self.assertRaises(ValueError):
            check_features(["hour_local", "price"], registry, [], ["price"])
        self.assertEqual(next(r["admission"] for r in registry if r["feature"] == TEMPERATURE_PERFECT), "excluded")
        self.assertEqual(original, original_snapshot)

    def test_current_temperature_is_frozen_before_local_noon_across_dst(self):
        dates = ["2024-01-15", "2024-07-15", "2024-03-31", "2024-04-01", "2024-10-27", "2024-10-28"]
        times = [pd.Timestamp(f"{day} {hour}:00", tz="Europe/Brussels").tz_convert("UTC")
                 for day in dates for hour in [0, 20]]
        frame = calendar(pd.DataFrame({"zone": "FR", "ts_utc": times}), {})
        stamps = frame.forecast_origin_utc - pd.Timedelta(hours=1)
        history = pd.DataFrame({"zone": "FR", "ts_utc": stamps.drop_duplicates()})
        history["temp_pw"] = np.arange(len(history), dtype=float) - 5.
        future = pd.DataFrame({"zone": "FR", "ts_utc": frame.ts_utc, "temp_pw": 99.})
        source = pd.concat([history, future], ignore_index=True)
        first = join_temperature_history(frame, source, [TEMPERATURE_CURRENT, TEMPERATURE_PERFECT])
        self.assertEqual(first[TEMPERATURE_CURRENT + "_source_utc"].dt.tz_convert("Europe/Brussels").dt.hour.tolist(), [11] * 12)
        self.assertTrue(first.groupby("delivery_date")[TEMPERATURE_CURRENT].nunique().eq(1).all())
        self.assertTrue(first[TEMPERATURE_PERFECT].eq(99.).all())
        source.loc[source.ts_utc.isin(frame.ts_utc), "temp_pw"] = -999.
        second = join_temperature_history(frame, source, [TEMPERATURE_CURRENT])
        pd.testing.assert_series_equal(first[TEMPERATURE_CURRENT], second[TEMPERATURE_CURRENT])
        self.assertTrue((first[TEMPERATURE_CURRENT + "_source_utc"] < first.forecast_origin_utc).all())

    def test_exact_zone_time_lookups_keep_missing_values_and_negative_temperatures(self):
        frame = calendar(pd.DataFrame({"zone": ["FR", "FR", "BE"],
            "ts_utc": pd.to_datetime(["2024-01-15T00:00Z", "2024-01-15T01:00Z", "2024-01-15T00:00Z"])}), {})
        current = frame.forecast_origin_utc.iloc[0] - pd.Timedelta(hours=1)
        source = pd.DataFrame({"zone": ["FR", "FR", "FR", "BE"],
            "ts_utc": [current, frame.ts_utc.iloc[0], current - pd.Timedelta(hours=1), frame.ts_utc.iloc[0]],
            "temp_pw": [-3., -5., 123., 20.]})
        got = join_temperature_history(frame, source, [TEMPERATURE_CURRENT, TEMPERATURE_PERFECT])
        self.assertEqual(got[TEMPERATURE_CURRENT].iloc[:2].tolist(), [-3., -3.])
        self.assertTrue(pd.isna(got[TEMPERATURE_CURRENT].iloc[2]))
        self.assertEqual(got[TEMPERATURE_PERFECT].iloc[0], -5.)
        self.assertTrue(pd.isna(got[TEMPERATURE_PERFECT].iloc[1]))
        source = source.loc[source.ts_utc != current]
        self.assertTrue(join_temperature_history(frame, source, [TEMPERATURE_CURRENT])[TEMPERATURE_CURRENT].isna().all())

    def test_temperature_history_before_output_window_comes_from_verified_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ["nodes", "strict"]:
                (root / name).mkdir()
            frame = calendar(pd.DataFrame({"zone": ["FR"], "ts_utc": pd.to_datetime(["2016-01-01T00:00Z"])}), {})
            frame["price"], frame["target_valid"] = 10., True
            frame.to_parquet(root / "strict/part.parquet", index=False)
            frame[["zone", "ts_utc"]].to_parquet(root / "nodes/part.parquet", index=False)
            source = pd.DataFrame({"zone": ["FR", "FR"], "time_utc": [
                frame.forecast_origin_utc.iloc[0] - pd.Timedelta(hours=1), frame.ts_utc.iloc[0]], "temp_pw": [1., 2.]})
            source.to_parquet(root / "weather.parquet", index=False)
            json_write(root / "source_manifest.json", {"weather_hourly": {"path": str(root / "weather.parquet"),
                "sha256": digest(root / "weather.parquet")["sha256"]}})
            cfg = {"zone": "FR", "features": ["hour_local", TEMPERATURE_CURRENT, TEMPERATURE_PERFECT],
                "conditional_features": [TEMPERATURE_CURRENT], "oracle_features": [TEMPERATURE_PERFECT]}
            got = load_inputs(root, cfg, feature_registry())
            self.assertEqual(got[TEMPERATURE_CURRENT].iloc[0], 1.)
            self.assertEqual(got[TEMPERATURE_PERFECT].iloc[0], 2.)
            with (root / "weather.parquet").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_inputs(root, cfg, feature_registry())


if __name__ == "__main__":
    unittest.main()
