"""Timing, missingness and identity fixtures for the airport experiment."""

import json
import unittest

import numpy as np
import pandas as pd

from epgnn.airport_study.data import ALL_FEATURES, GROUPS, indices
from epgnn.airport_study.design import jobs
from epgnn.airport_study.features import at_origins, country_proxy, normalize
from epgnn.controlled_study.models import make_model, parameter_count
from epgnn.dataset_build.contracts import forecast_origin
from epgnn.dataset_build.storage import ROOT

CONFIG = json.loads((ROOT / "configs/experiments/airport_temperature.json").read_text())


def observations(start="2024-01-01", periods=96):
    return pd.DataFrame(
        {
            "observed_utc": pd.date_range(start, periods=periods, freq="h", tz="UTC"),
            "temp_c": np.arange(periods, dtype=float),
            "received_utc": pd.NaT,
        }
    )


class AirportStudyTests(unittest.TestCase):
    def test_frozen_matrix_and_features(self):
        self.assertEqual(len(jobs(CONFIG)), 45)
        self.assertEqual(len({job.name for job in jobs(CONFIG)}), 45)
        self.assertEqual(len(jobs(CONFIG, smoke=True)), 9)
        for group, features in GROUPS.items():
            self.assertEqual([ALL_FEATURES[j] for j in indices(group)], features)
            self.assertFalse(
                any("lag48" in name or name == "load_fc" for name in features)
            )
            size = len(features) * 2
            self.assertLess(
                abs(
                    parameter_count(make_model("mlp", size, CONFIG["neural"]))
                    / parameter_count(make_model("gat", size, CONFIG["neural"]))
                    - 1
                ),
                0.02,
            )

    def test_country_and_historical_zone_rules(self):
        self.assertEqual(country_proxy("DE_LU"), ("DE", "combined_market_proxy"))
        self.assertEqual(country_proxy("IE_SEM"), ("IE", "combined_market_proxy"))
        self.assertEqual(country_proxy("IT_SICI")[0], "IT")
        self.assertEqual(country_proxy("NO_4")[0], "NO")
        self.assertIsNone(country_proxy("DE_AT_LU")[0])
        self.assertIsNone(country_proxy("ENTSOE:unknown")[0])

    def test_winter_summer_and_both_dst_transitions(self):
        for delivery, expected in [
            ("2024-01-02", "2024-01-01 11:00Z"),
            ("2024-07-02", "2024-07-01 10:00Z"),
            ("2024-04-01", "2024-03-31 10:00Z"),
            ("2024-10-28", "2024-10-27 11:00Z"),
        ]:
            target = pd.DatetimeIndex([pd.Timestamp(delivery, tz="Europe/Brussels")])
            origin = forecast_origin(target)
            self.assertEqual(origin.iloc[0], pd.Timestamp(expected))
            source = observations(pd.Timestamp(expected) - pd.Timedelta(days=2), 96)
            for unit in ["us", "ns"]:
                source["observed_utc"] = source.observed_utc.dt.as_unit(unit)
                result = at_origins(source, origin, CONFIG)
                self.assertEqual(result[0, 1], 2)
                self.assertEqual(result[0, 3], 24)

    def test_day_features_constant_across_23_and_25_hour_days(self):
        for day, next_day, count in [
            ("2024-03-31", "2024-04-01", 23),
            ("2024-10-27", "2024-10-28", 25),
        ]:
            targets = pd.date_range(
                pd.Timestamp(day, tz="Europe/Brussels"),
                pd.Timestamp(next_day, tz="Europe/Brussels"),
                freq="h",
                inclusive="left",
            )
            self.assertEqual(len(targets), count)
            origins = forecast_origin(targets)
            self.assertEqual(origins.nunique(), 1)
            source = observations(origins.iloc[0] - pd.Timedelta(days=2), 96)
            values = at_origins(source, origins, CONFIG)
            np.testing.assert_array_equal(values, np.repeat(values[:1], count, axis=0))

    def test_cutoff_receipt_and_future_observation_poison(self):
        source = observations()
        origin = pd.DatetimeIndex([pd.Timestamp("2024-01-03 12:00Z")])
        before = at_origins(source, origin, CONFIG)
        self.assertEqual(before[0, 0], 58)
        source.loc[
            source.observed_utc.gt(origin[0] - pd.Timedelta(hours=2)), "temp_c"
        ] = -99999
        np.testing.assert_array_equal(before, at_origins(source, origin, CONFIG))
        source["received_utc"] = source.observed_utc + pd.Timedelta(minutes=5)
        source.loc[
            source.observed_utc.eq(origin[0] - pd.Timedelta(hours=2)), "received_utc"
        ] = origin[0] + pd.Timedelta(seconds=1)
        self.assertEqual(at_origins(source, origin, CONFIG)[0, 0], 57)

    def test_missing_stale_history_and_spe_ci_sampling(self):
        source = observations(periods=30)
        origin = pd.DatetimeIndex([pd.Timestamp("2024-01-02 12:00Z")])
        values = at_origins(source, origin, CONFIG)
        self.assertTrue(np.isnan(values[0, 0]))
        self.assertTrue(np.isnan(values[0, 1]))
        source = observations(periods=60)
        source = source.iloc[::2].copy()
        values = at_origins(source, origin, CONFIG)
        self.assertTrue(np.isnan(values[0, 2]))
        self.assertEqual(values[0, 3], 12)
        empty = source.iloc[:0]
        self.assertEqual(at_origins(empty, origin, CONFIG)[0, 3], 0)
        self.assertTrue(np.isnan(at_origins(empty, origin, CONFIG)[0, 0]))

    def test_hourly_mean_gives_each_hour_one_vote(self):
        source = observations(periods=48)
        source.temp_c = 10.0
        origin = pd.DatetimeIndex([pd.Timestamp("2024-01-02 12:00Z")])
        burst = pd.DataFrame(
            {
                "observed_utc": pd.date_range(
                    "2024-01-02 09:01Z", periods=50, freq="min"
                ),
                "temp_c": 34.0,
                "received_utc": pd.NaT,
            }
        )
        source = pd.concat([source, burst]).sort_values("observed_utc")
        result = at_origins(source, origin, CONFIG)
        # The 10:00 report closes the same hourly bin as the burst.
        self.assertEqual(result[0, 2], 10.0)
        self.assertEqual(result[0, 3], 24)

    def test_duplicates_conflicts_corrections_and_units(self):
        raw = pd.DataFrame(
            {
                "station": ["LFPG"] * 5,
                "valid": ["2024-01-01 00:00Z"] * 2
                + ["2024-01-01 01:00Z"] * 2
                + ["2024-01-01 02:00Z"],
                "tmpf": [32, 32, 50, 68, 86],
                "metar": ["METAR"] * 4 + ["METAR COR"],
            }
        )
        clean, audit = normalize(raw)
        self.assertEqual(clean.iloc[0].temp_c, 0)
        self.assertTrue(np.isnan(clean.iloc[1].temp_c))
        self.assertTrue(np.isnan(clean.iloc[2].temp_c))
        self.assertEqual(audit["duplicate_rows"], 2)
        self.assertEqual(audit["conflicting_timestamps"], 1)
        self.assertEqual(audit["corrected_reports"], 1)
        raw.loc[0, "station"] = "EDDF"
        with self.assertRaises(ValueError):
            normalize(raw)

    def test_unsorted_or_duplicate_observations_rejected(self):
        source = observations(periods=3)
        for invalid in [source.iloc[::-1], pd.concat([source, source.iloc[-1:]])]:
            with self.assertRaises(ValueError):
                at_origins(
                    invalid,
                    pd.DatetimeIndex([pd.Timestamp("2024-01-03 12:00Z")]),
                    CONFIG,
                )


class AirportSourceTests(unittest.TestCase):
    def test_fresh_checkout_recreates_ranking_provenance_without_changing_mapping(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from epgnn.airport_study import sources

        labels, index, values = {}, {}, {}
        for country_index, country in enumerate(
            c for c in sources.COUNTRIES if c != "ME"
        ):
            prefix = "EL" if country == "GR" else country
            for rank, ending in enumerate(["A", "B"]):
                station = (
                    "X"
                    + chr(65 + country_index // 26)
                    + chr(65 + country_index % 26)
                    + ending
                )
                code = prefix + "_" + station
                position = len(index)
                index[code] = position
                labels[code] = station
                values[str(position)] = (rank + 1) * 100
        payload = json.dumps(
            {
                "id": [
                    "freq",
                    "unit",
                    "tra_meas",
                    "rep_airp",
                    "schedule",
                    "tra_cov",
                    "time",
                ],
                "size": [1, 1, 1, len(index), 1, 1, 1],
                "dimension": {
                    "rep_airp": {"category": {"index": index, "label": labels}}
                },
                "value": values,
            }
        ).encode()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(sources, "ROOT", Path(temporary)),
            patch.object(sources, "download", return_value=payload),
        ):
            first = sources.freeze_mapping(CONFIG)
            self.assertEqual(len(first), 29)
            self.assertTrue(
                first.loc[first.country.ne("ME"), "station"].str.endswith("B").all()
            )
            ranking = (
                Path(temporary) / CONFIG["source_directory"] / "eurostat_2023.json"
            )
            ranking.unlink()
            second = sources.freeze_mapping(CONFIG)
            pd.testing.assert_frame_equal(first, second)
            self.assertTrue(ranking.exists())


if __name__ == "__main__":
    unittest.main()
