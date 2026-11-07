import unittest

import numpy as np
import pandas as pd

from epgnn.dataset_build.contracts import (DE_SPLIT, NODE_KEYS, calendar,
    feature_registry, forecast_origin, lag_join, lifetime, strict_view, unique, utc, zone_map)
from epgnn.dataset_build.findings import modelling_blockers


class TimeContractTests(unittest.TestCase):
    def test_literal_local_noon_across_seasons_and_both_transitions(self):
        days = ["2024-01-15", "2024-07-15", "2024-03-31", "2024-04-01", "2024-10-27", "2024-10-28"]
        expected_hours = [11, 10, 11, 10, 10, 11]
        ts = pd.Series([pd.Timestamp(d, tz="Europe/Brussels").tz_convert("UTC") for d in days])
        origins = forecast_origin(ts)
        self.assertEqual(origins.dt.hour.tolist(), expected_hours)
        self.assertEqual(origins.dt.tz_convert("Europe/Brussels").dt.hour.tolist(), [12] * 6)
        self.assertEqual(origins.dt.tz_convert("Europe/Brussels").dt.strftime("%Y-%m-%d").tolist(),
                         ["2024-01-14", "2024-07-14", "2024-03-30", "2024-03-31", "2024-10-26", "2024-10-27"])

    def test_dst_days_have_23_and_25_hours(self):
        for day, count in [("2024-03-31", 23), ("2024-10-27", 25)]:
            a = pd.Timestamp(day, tz="Europe/Brussels")
            h = pd.date_range(a, a + pd.DateOffset(days=1), freq="h", inclusive="left")
            f = calendar(pd.DataFrame({"zone": "BE", "ts_utc": h}), {})
            self.assertEqual(len(f), count)
            self.assertFalse(f.ts_utc.duplicated().any())

    def test_timestamp_units_and_ambiguous_numeric_epochs(self):
        s = pd.Series(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
        pd.testing.assert_series_equal(utc(s.dt.as_unit("ns")), utc(s.dt.as_unit("us")))
        with self.assertRaisesRegex(ValueError, "explicit unit"):
            utc(pd.Series([1, 2]))
        self.assertEqual(utc(pd.Series(["2022-06-08T21:00:00+00:00", "2022-06-08 22:00:00+00:00"])).diff().iloc[1], pd.Timedelta(hours=1))

    def test_split_boundaries_are_local_and_october_is_unassigned(self):
        splits = {"train": ["2022-06-09", "2024-03-31"], "val": ["2024-04-01", "2025-03-31"], "test": ["2025-04-01", "2026-09-30"]}
        ts = pd.to_datetime(["2024-03-31T21:00Z", "2024-03-31T22:00Z", "2025-03-31T22:00Z", "2026-09-30T21:00Z", "2026-09-30T22:00Z"])
        f = calendar(pd.DataFrame({"ts_utc": ts}), splits)
        self.assertEqual(f.split.tolist(), ["train", "val", "test", "test", "unassigned"])


class LagAndGeographyTests(unittest.TestCase):
    def base(self):
        times = pd.to_datetime(["2024-01-08T00:00Z", "2024-01-08T22:00Z"])
        return calendar(pd.DataFrame({"zone": ["BE", "BE"], "ts_utc": times}), {})

    def test_exact_joins_preserve_missing_hours_and_zones(self):
        out = self.base()
        h = pd.DataFrame({"zone": ["BE", "FR", "BE"],
            "ts_utc": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T22:00Z", "2024-01-01T21:00Z"]),
            "price": [-20., 99., 88.]})
        got = lag_join(out, h, NODE_KEYS, "price", 168)
        self.assertEqual(got.price_lag168.iloc[0], -20)
        self.assertTrue(pd.isna(got.price_lag168.iloc[1]))
        self.assertFalse(got.price_lag168_eligible.any())

    def test_publication_and_interval_completion_both_required(self):
        out = self.base()
        h = pd.DataFrame({"zone": ["BE", "BE"], "ts_utc": out.ts_utc - pd.Timedelta(hours=24),
            "price": [1., 2.], "published": out.ts_utc - pd.Timedelta(hours=30)})
        got = lag_join(out, h, NODE_KEYS, "price", 24, availability="published")
        self.assertEqual(got.price_lag24_eligible.tolist(), [True, False])
        h["published"] = out.forecast_origin_utc + pd.Timedelta(seconds=1)
        self.assertFalse(lag_join(out, h, NODE_KEYS, "price", 24, availability="published").price_lag24_eligible.any())

    def test_future_observations_do_not_change_earlier_admitted_features(self):
        out = self.base()
        h = pd.DataFrame({"zone": "BE", "ts_utc": out.ts_utc - pd.Timedelta(hours=168),
            "price": [1., np.nan], "published": out.ts_utc - pd.Timedelta(hours=167)})
        a = lag_join(out, h, NODE_KEYS, "price", 168, availability="published")
        future = pd.DataFrame({"zone": ["BE"], "ts_utc": pd.to_datetime(["2024-01-09T00:00Z"]),
            "price": [999999.], "published": pd.to_datetime(["2024-01-09T00:00Z"])})
        b = lag_join(out, pd.concat([h, future], ignore_index=True), NODE_KEYS, "price", 168, availability="published")
        pd.testing.assert_frame_equal(a, b)

    def test_duplicates_and_conflicts_never_silently_deduplicate(self):
        h = pd.DataFrame({"zone": ["BE", "BE"], "ts_utc": pd.to_datetime(["2024-01-01T00:00Z"] * 2), "price": [1., 2.]})
        with self.assertRaisesRegex(ValueError, "conflicting=True"):
            unique(h, NODE_KEYS, "fixture")
        h["price"] = 1.
        with self.assertRaisesRegex(ValueError, "duplicate"):
            unique(h, NODE_KEYS, "fixture")

    def test_german_identities_and_lags_never_splice(self):
        names = pd.Series(["DE-AT-LU", "DE-LU", "Germany (DE)"])
        self.assertEqual(zone_map(names, pd.Series(["old", "new", "country"])).tolist(),
                         ["DE_AT_LU", "DE_LU", "ENTSOE:country"])
        out = calendar(pd.DataFrame({"zone": ["DE_LU"], "ts_utc": [DE_SPLIT]}), {})
        history = pd.DataFrame({"zone": ["DE_AT_LU"], "ts_utc": [DE_SPLIT - pd.Timedelta(hours=168)], "price": [10.]})
        self.assertTrue(lag_join(out, history, NODE_KEYS, "price", 168).price_lag168.isna().all())
        self.assertEqual(lifetime(pd.Series(["DE_AT_LU", "DE_LU", "AT"]), pd.Series([DE_SPLIT] * 3)).tolist(), [False, True, True])

    def test_registry_default_excludes_unverified_lags(self):
        r = {x["feature"]: x for x in feature_registry()}
        for c in ["price_lag48", "load_fc", "temp_pw_lag48", "n_presolved", "flow_phys_lag48", "maxbex"]:
            self.assertEqual(r[c]["admission"], "conditional")
        self.assertEqual(r["price"]["admission"], "excluded")

    def test_strict_view_masks_ineligible_rows_even_with_verified_publication_rule(self):
        out = self.base().assign(price=[3., np.nan], target_valid=[True, False], in_cwe=False, in_core=True)
        h = pd.DataFrame({"zone": ["BE", "BE"], "ts_utc": out.ts_utc - pd.Timedelta(hours=24),
            "price": [1., 2.], "published": out.ts_utc - pd.Timedelta(hours=30)})
        out = lag_join(out, h, NODE_KEYS, "price", 24, availability="published")
        # This fixture admits only the lag being tested; other required histories
        # are absent rather than fabricated for the strict projection.
        registry = [r for r in feature_registry(True) if r["feature"] not in ["price_lag48", "price_lag168"]]
        strict = strict_view(out, registry)
        self.assertEqual(strict.price_lag24.iloc[0], 1.)
        self.assertTrue(pd.isna(strict.price_lag24.iloc[1]))
        self.assertTrue(pd.isna(strict.price.iloc[1]))
        self.assertFalse(strict.target_valid.iloc[1])

    def test_partner_defects_are_reproduced(self):
        r = modelling_blockers()
        self.assertTrue(r[0]["reproduced"])
        self.assertTrue(r[1]["reproduced"])


if __name__ == "__main__":
    unittest.main()
