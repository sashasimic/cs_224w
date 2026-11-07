import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from epgnn.boosting import (JAO_LAG_FEATURES, check_features, errors, experiment_registry,
    extend_constraint_history, join_constraint_history, load_inputs, masks, score)
from epgnn.dataset_build.contracts import calendar, feature_registry
from epgnn.dataset_build.storage import digest, json_write


class BoostingContractTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"split": ["pretrain", "train", "val", "test", "unassigned"],
            "delivery_date": ["2020-01-01", "2023-01-01", "2024-06-01", "2025-06-01", "2026-10-01"],
            "price": [-5., 10., 20., 30., 40.], "target_valid": True,
            "hour_local": [np.nan, 1., 2., 3., 4.]})

    def cfg(self):
        return {"fit_splits": ["pretrain", "train"], "validation_split": "val", "evaluation_splits": ["val", "test"]}

    def test_conditional_outcome_and_unregistered_features_rejected(self):
        registry = feature_registry()
        check_features(["hour_local", "month", "in_core"], registry)
        for c in ["price", "price_lag168", "load_fc", "temp_pw", "n_presolved", "invented"]:
            with self.subTest(feature=c), self.assertRaisesRegex(ValueError, "non-admitted"):
                check_features(["hour_local", c], registry)

    def test_only_fit_targets_select_training_rows_even_with_missing_features(self):
        f = self.frame()
        fit, val = masks(f, self.cfg())
        self.assertEqual(fit.tolist(), [True, True, False, False, False])
        self.assertEqual(val.tolist(), [False, False, True, False, False])
        f.loc[1, ["price", "target_valid"]] = [np.nan, False]
        fit, _ = masks(f, self.cfg())
        self.assertEqual(fit.tolist(), [True, False, False, False, False])

    def test_conditional_weekly_lag_requires_explicit_limited_opt_in(self):
        registry = feature_registry()
        features = ["hour_local", "price_lag168"]
        check_features(features, registry, ["price_lag168"])
        for feature in ["price", "load_fc", "temp_pw", "price_lag48"]:
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                check_features(["hour_local", feature], registry, [feature])
        with self.assertRaises(ValueError):
            check_features(["hour_local"], registry, ["price_lag168"])
        self.assertEqual(next(r["admission"] for r in registry if r["feature"] == "price_lag168"), "conditional")

    def test_conditional_loading_keeps_missing_history_and_checks_exact_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "strict").mkdir()
            (root / "nodes").mkdir()
            f = calendar(pd.DataFrame({"zone": ["BE", "BE"], "ts_utc":
                pd.to_datetime(["2024-01-08T00:00Z", "2024-01-08T01:00Z"])}), {})
            f["price"], f["target_valid"] = [1., 2.], True
            f.to_parquet(root / "strict/part.parquet", index=False)
            h = f[["zone", "ts_utc"]].copy()
            h["price_lag168"] = [-3., np.nan]
            h["price_lag168_observation_utc"] = h.ts_utc - pd.Timedelta(hours=168)
            h["price_lag168_completion_utc"] = h.price_lag168_observation_utc + pd.Timedelta(hours=1)
            h.to_parquet(root / "nodes/part.parquet", index=False)
            cfg = {"zone": "BE", "features": ["hour_local", "price_lag168"], "conditional_features": ["price_lag168"]}
            got = load_inputs(root, cfg, feature_registry())
            self.assertEqual(len(got), 2)
            self.assertEqual(got.price_lag168.iloc[0], -3.)
            self.assertTrue(pd.isna(got.price_lag168.iloc[1]))
            h.loc[0, "price_lag168_observation_utc"] = f.forecast_origin_utc.iloc[0] + pd.Timedelta(hours=1)
            h.to_parquet(root / "nodes/part.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "exact history/forecast timing"):
                load_inputs(root, cfg, feature_registry())

    def test_test_cannot_be_fit_or_early_stop_split(self):
        for cfg in [{**self.cfg(), "fit_splits": ["train", "test"]}, {**self.cfg(), "validation_split": "test"}]:
            with self.assertRaises(ValueError):
                masks(self.frame(), cfg)

    def test_constraint_registry_is_conditional_without_changing_strict_admission(self):
        original = feature_registry()
        before = [dict(r) for r in original]
        registry = experiment_registry(original)
        self.assertEqual(original, before)
        for feature in JAO_LAG_FEATURES:
            with self.assertRaisesRegex(ValueError, "non-admitted"):
                check_features(["hour_local", feature], registry)
            check_features(["hour_local", feature], registry, [feature])
        for feature in ["n_presolved", "min_ram_ratio", "load_fc", "price", "maxbex"]:
            with self.assertRaises(ValueError):
                check_features(["hour_local", feature], registry, [feature])

    def test_constraints_keep_zeros_missing_publications_and_regional_absences(self):
        t = pd.Timestamp("2024-01-08T00:00Z")
        frame = calendar(pd.DataFrame({"zone": ["BE", "FR", "ES", "BE"],
            "ts_utc": [t, t, t, t + pd.Timedelta(hours=2)]}), {})
        source = pd.DataFrame({"zone": ["BE", "FR", "ES"], "ts_utc": [t - pd.Timedelta(hours=48)] * 3,
            "n_presolved": [0., np.nan, 50.], "min_ram_ratio": [0., np.nan, 0.9],
            "jao_domain_present": pd.Series([True, False, True], dtype="boolean"),
            "in_cwe": [False, False, False], "in_core": [True, True, False]})
        out = join_constraint_history(frame, source, list(JAO_LAG_FEATURES))
        self.assertEqual(len(out), 4)
        self.assertEqual(out.n_presolved_lag48.iloc[0], 0.)
        self.assertEqual(out.min_ram_ratio_lag48.iloc[0], 0.)
        self.assertTrue(pd.isna(out.n_presolved_lag48.iloc[1]))
        self.assertEqual(out.jao_domain_present_lag48.iloc[1], 0.)
        self.assertTrue(out.loc[[2, 3], list(JAO_LAG_FEATURES)].isna().all().all())

    def test_constraint_history_cannot_cross_historical_bidding_zone_identity(self):
        t = pd.Timestamp("2018-09-30T22:00Z")
        frame = calendar(pd.DataFrame({"zone": ["DE_LU"], "ts_utc": [t]}), {})
        source = pd.DataFrame({"zone": ["DE_AT_LU"], "ts_utc": [t - pd.Timedelta(hours=48)],
            "n_presolved": [20.], "min_ram_ratio": [0.5], "jao_domain_present": [True],
            "in_cwe": [True], "in_core": [False]})
        out = join_constraint_history(frame, source, list(JAO_LAG_FEATURES))
        self.assertTrue(out[list(JAO_LAG_FEATURES)].isna().all().all())
        source = pd.concat([source, source], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            join_constraint_history(frame, source, list(JAO_LAG_FEATURES))

    def test_constraints_read_verified_history_before_master_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            t = pd.Timestamp("2016-01-01T00:00Z")
            original = pd.DataFrame({"zone": ["BE", "BE"],
                "ts_utc": [t - pd.Timedelta(hours=48), t], "n_presolved": [0., 7.],
                "min_ram_ratio": [np.nan, 0.5], "regime": ["CWE", "CWE"]})
            original.to_parquet(root / "original.parquet", index=False)
            pd.DataFrame({"ts_utc": [t - pd.Timedelta(hours=48)], "domain_present": [True]}).to_csv(root / "coverage.csv", index=False)
            info = {name: {"path": str(root / file), "sha256": digest(root / file)["sha256"]}
                for name, file in [("jao_hourly", "original.parquet"), ("jao_metadata/coverage.csv", "coverage.csv")]}
            json_write(root / "source_manifest.json", info)
            source = original.iloc[[1]].drop(columns="regime").assign(in_cwe=True, in_core=False, jao_domain_present=True)
            extended = extend_constraint_history(root, "BE", source, t)
            self.assertEqual(len(extended), 2)
            frame = calendar(pd.DataFrame({"zone": ["BE"], "ts_utc": [t]}), {})
            got = join_constraint_history(frame, extended, list(JAO_LAG_FEATURES))
            self.assertEqual(got.n_presolved_lag48.iloc[0], 0.)
            self.assertTrue(pd.isna(got.min_ram_ratio_lag48.iloc[0]))
            self.assertEqual(got.jao_domain_present_lag48.iloc[0], 1.)
            with (root / "original.parquet").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                extend_constraint_history(root, "BE", source, t)

    def test_chronological_and_target_contracts_enforced(self):
        f = self.frame()
        f.loc[1, "delivery_date"] = "2025-01-01"
        with self.assertRaisesRegex(ValueError, "chronologically"):
            masks(f, self.cfg())
        f = self.frame()
        f.loc[1, "price"] = np.nan
        with self.assertRaisesRegex(ValueError, "validity"):
            masks(f, self.cfg())

    def test_nan_targets_and_missing_persistence_use_identical_common_mask(self):
        f = pd.DataFrame({"price": [1., np.nan, -3., 10.], "target_valid": [True, False, True, True]})
        prediction = np.array([2., 99., -1., 7.])
        persistence = np.array([0., 99., np.nan, 8.])
        r = score(f, prediction, persistence)
        self.assertEqual(r["model"]["n_hours"], 3)
        self.assertEqual(r["model"]["mae"], 2.)
        common = r["weekly_persistence_conditional_common_hours"]
        self.assertEqual(common["model"]["n_hours"], 2)
        self.assertEqual(common["persistence"]["n_hours"], 2)
        self.assertEqual(common["model"]["mae"], 2.)
        self.assertEqual(common["persistence"]["mae"], 1.5)
        self.assertEqual(common["valid_targets_without_persistence"], 1)
        self.assertIsNone(errors([np.nan], [0.], [False])["mae"])


if __name__ == "__main__":
    unittest.main()
