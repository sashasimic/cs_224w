import unittest

import numpy as np
import pandas as pd

from epgnn.boosting_batch import aggregate, comparison_rows


class CountryComparisonTests(unittest.TestCase):
    def test_aggregate_weights_hours_and_pools_squared_errors(self):
        rows = pd.DataFrame({"split": ["test", "test"], "paired_hours": [1, 3],
            "xgb_mae": [2., 6.], "persistence_mae": [4., 8.],
            "xgb_rmse": [2., 8.], "persistence_rmse": [4., 10.]})
        got = aggregate(rows)["test"]
        self.assertEqual(got["paired_country_hours"], 4)
        self.assertEqual(got["xgb_mae"], 5.)
        self.assertEqual(got["persistence_mae"], 7.)
        self.assertEqual(got["xgb_rmse"], np.sqrt(49.))
        self.assertEqual(got["persistence_rmse"], np.sqrt(79.))
        self.assertEqual(got["countries_xgb_better"], 2)

    def test_comparison_uses_common_mask_not_full_model_score(self):
        record = {"zone": "FR", "name": "fixture", "selected_rounds": 1, "fit_rows": 100,
            "feature_policy": "conditional", "metrics": {"test": {"valid_targets": 10,
                "model": {"mae": 100.}, "weekly_persistence_conditional_common_hours": {
                    "model": {"mae": 2., "rmse": 3., "n_hours": 8},
                    "persistence": {"mae": 4., "rmse": 5., "n_hours": 8},
                    "relative_mae": 0.5, "valid_targets_without_persistence": 2}}}}
        row = comparison_rows([record]).iloc[0]
        self.assertEqual(row.xgb_mae, 2.)
        self.assertEqual(row.full_target_xgb_mae, 100.)
        self.assertEqual(row.paired_hours, 8)
        self.assertEqual(row.mae_improvement_percent, 50.)
        record["metrics"]["test"]["weekly_persistence_conditional_common_hours"]["persistence"]["n_hours"] = 9
        with self.assertRaisesRegex(ValueError, "counts differ"):
            comparison_rows([record])


if __name__ == "__main__":
    unittest.main()
