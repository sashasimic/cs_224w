"""Regression checks for frozen-score reporting, population matching and bounds."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from epgnn.reporting.investigation import CORE, Report, bounds, heatmap


class ReportingTests(unittest.TestCase):
    def test_frozen_inventory_and_headline_populations(self):
        report = Report()
        inventory = report.inventory()
        self.assertEqual(inventory.jobs.sum(), 145)
        self.assertEqual(inventory.variant.nunique(), 27)
        table = report.main_table()
        for split, hours in [("val", 428213), ("test", 633885)]:
            self.assertEqual(
                set(table.loc[table.split.eq(split), "matched_hours"]), {hours}
            )
        self.assertEqual(
            set(table.loc[table.model.eq("Weekly persistence"), "seed_SD"]), {0}
        )

    def test_reconciled_scores_reject_changed_zone_metrics(self):
        report = Report()
        report.zones["airport"].loc[0, "mae"] += 100
        with self.assertRaises(AssertionError):
            report.validate_scores()

    def test_utc_bounds_and_outcome_restrictions(self):
        a, b = bounds("2024-10-27T00:00+02:00", "2024-10-28T00:00+01:00")
        self.assertEqual(str(a.tz), "UTC")
        self.assertEqual(b - a, pd.Timedelta(hours=25))
        for start, end in [
            ("2024-07-01", "2024-07-02"),
            ("2025-04-01T00:00Z", "2025-04-02T00:00Z"),
            ("2024-07-01T00:00Z", "2024-09-01T00:00Z"),
        ]:
            with self.assertRaises(ValueError):
                bounds(start, end)

    def test_missing_predictions_are_explicit(self):
        report = Report()
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(
                report,
                "local_paths",
                return_value={"controlled": Path(temp), "airport": Path(temp)},
            ),
        ):
            frame, message = report.drilldown(
                "FR", "2024-07-08T00:00Z", "2024-07-09T00:00Z"
            )
            self.assertIsNone(frame)
            self.assertIn("unavailable", message)

    def test_cross_model_detail_target_mismatch_is_rejected(self):
        report = Report()
        with tempfile.TemporaryDirectory() as temp:
            frame = pd.DataFrame(
                {
                    "zone": ["FR"] * 2,
                    "ts_utc": pd.date_range(
                        "2024-07-08", periods=2, freq="h", tz="UTC"
                    ),
                    "price": [1.0, np.nan],
                    "target_valid": [True, False],
                    "prediction": [1.0, 2.0],
                    "price_lag168": [0.0, 0.0],
                }
            )
            for k, (_, variant, _) in enumerate(CORE):
                path = Path(temp) / (variant + "_seed224")
                path.mkdir()
                edited = frame.copy()
                if k == 1:
                    edited.loc[0, "price"] = 99
                edited.to_parquet(path / "predictions_val.parquet", index=False)
            with (
                patch.object(
                    report,
                    "local_paths",
                    return_value={"controlled": Path(temp), "airport": Path(temp)},
                ),
                self.assertRaises(AssertionError),
            ):
                report.drilldown("FR", "2024-07-08T00:00Z", "2024-07-09T00:00Z")

    def test_long_calendar_heatmap_is_bounded(self):
        figure = heatmap(pd.DataFrame(np.ones((3, 44))), "Calendar")
        self.assertLessEqual(figure.get_figwidth(), 14)
        self.assertLessEqual(len(figure.axes[0].get_xticks()), 12)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
