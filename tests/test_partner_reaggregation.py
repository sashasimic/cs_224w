import unittest

from scripts.experiments.partner_xgb_reaggregate import aggregate_countries


class PartnerCountryAggregationTests(unittest.TestCase):
    def test_country_error_pooling_weights_zone_hours_and_squared_errors(self):
        rows = [
            {"zone": "NO_1", "split": "test", "n_hours": 1, "mae": 2., "rmse": 2., "persistence_mae": 4.},
            {"zone": "NO_2", "split": "test", "n_hours": 3, "mae": 6., "rmse": 8., "persistence_mae": 8.},
        ]
        got = aggregate_countries(rows)[0]
        self.assertEqual(got["country"], "NO")
        self.assertEqual(got["n_zone_hours"], 4)
        self.assertEqual(got["mae"], 5.)
        self.assertEqual(got["rmse"], 7.)
        self.assertEqual(got["persistence_mae"], 7.)
        self.assertAlmostEqual(got["relative_mae"], 5 / 7)

    def test_combined_markets_are_not_split_and_empty_targets_stay_unscored(self):
        rows = [{"zone": zone, "split": "val", "n_hours": 0, "mae": None, "rmse": None, "persistence_mae": None}
                for zone in ["DE_LU", "IE_SEM", "GB"]]
        got = aggregate_countries(rows)
        self.assertEqual([r["country"] for r in got], ["DE_LU", "GB", "IE_SEM"])
        self.assertTrue(all(r["status"] == "no_valid_targets" and r["mae"] is None for r in got))


if __name__ == "__main__":
    unittest.main()
