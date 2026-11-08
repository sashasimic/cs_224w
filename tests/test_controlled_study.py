"""Control design, time admission, graph independence and reporting invariants."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from epgnn.controlled_study.analysis import diagnostics, mean_weekly, week_interval
from epgnn.controlled_study.design import FEATURE_SETS, jobs
from epgnn.controlled_study.history import Histories, history_index
from epgnn.controlled_study.models import make_model, parameter_count
from epgnn.controlled_study.report import controlled_tex, publish
from epgnn.dataset_build.storage import ROOT
from epgnn.experiment_suite.data import FEATURES
from epgnn.experiment_suite.evaluation import evaluate

CONFIG = json.loads((ROOT / "configs/experiments/controlled_study.json").read_text())


def fixture():
    times = pd.date_range("2024-03-20", periods=24 * 15, freq="h", tz="UTC").as_unit(
        "ns"
    )
    return {
        "times": times.asi8,
        "y": np.arange(len(times) * 2, dtype=np.float32).reshape(-1, 2),
        "active": np.ones((len(times), 2), bool),
        "split": np.array([0] * (24 * 10) + [1] * (24 * 5)),
    }


class ControlledStudyTests(unittest.TestCase):
    def test_prespecified_matrix_has_100_unique_jobs_and_20_prototypes(self):
        planned = jobs(CONFIG)
        self.assertEqual(len(planned), 100)
        self.assertEqual(len({item.name for item in planned}), 100)
        self.assertEqual(sum(item.seed == 224 for item in planned), 20)
        self.assertEqual(sum(item.model.startswith("gru_") for item in planned), 10)
        self.assertEqual(
            {item.rewiring_seed for item in planned if item.graph == "shuffled"},
            {901, 902, 903},
        )

    def test_mlp_capacity_is_selected_without_looking_at_losses(self):
        for features in FEATURE_SETS.values():
            size = 2 * len(features)
            mlp = make_model("mlp", size, CONFIG["neural"])
            gat = make_model("gat", size, CONFIG["neural"])
            self.assertLess(abs(parameter_count(mlp) / parameter_count(gat) - 1), 0.02)
        self.assertEqual(make_model("mlp", 26, CONFIG["neural"]).width, 96)
        with self.assertRaises(ValueError):
            make_model("unknown", 26, CONFIG["neural"])

    def test_mlp_and_self_only_graph_do_not_mix_nodes(self):
        for kind in ["mlp", "gat", "gru_mlp"]:
            torch.manual_seed(224)
            model = make_model(kind, 26, CONFIG["neural"]).eval()
            x = torch.randn(2, 3, 26)
            empty = torch.empty((2, 0), dtype=torch.long)
            active = torch.ones(2, 3, dtype=torch.bool)
            sequence = (
                (torch.randn(2, 3, 168, 2), torch.tensor([0, 1]))
                if kind.startswith("gru")
                else None
            )
            with torch.no_grad():
                before = model(x, empty, active, sequence)
                x[:, 2] = 10000
                if sequence:
                    sequence[0][:, 2] = 10000
                edges = (
                    torch.tensor([[0, 2], [1, 1]]) if kind.endswith("mlp") else empty
                )
                after = model(x, edges, active, sequence)
            torch.testing.assert_close(before[:, :2], after[:, :2])

    def test_history_handles_dst_units_and_missing_hours(self):
        times = pd.date_range("2024-03-20", periods=24 * 15, freq="h", tz="UTC")
        for transition in ["2024-03-31", "2024-10-27"]:
            calendar = pd.date_range(
                pd.Timestamp(transition, tz="UTC") - pd.Timedelta(days=8),
                periods=24 * 10,
                freq="h",
            )
            positions, present, inverse = history_index(calendar)
            self.assertTrue((positions[present] >= 0).all())
            np.testing.assert_array_equal(
                history_index(calendar.as_unit("us"))[0],
                history_index(calendar.as_unit("ns"))[0],
            )
        gap = times.delete(170)
        _, present, _ = history_index(gap)
        self.assertTrue((~present).any())
        with self.assertRaises(ValueError):
            history_index(times[::-1])

    def test_future_prices_do_not_change_earlier_history_or_fit_statistics(self):
        arrays = fixture()
        histories = Histories(arrays).fit(np.flatnonzero(arrays["split"] == 0))
        target = 24 * 12
        before, inverse = histories.batch([target])
        arrays["y"][target - 12 :] = 999999
        later = Histories(arrays).fit(np.flatnonzero(arrays["split"] == 0))
        after, _ = later.batch([target])
        np.testing.assert_array_equal(before, after)
        self.assertEqual(histories.state(), later.state())
        self.assertFalse(np.array_equal(before[:, 0], before[:, 1]))

    def test_history_keeps_unavailable_identities_missing(self):
        arrays = fixture()
        arrays["active"][:, 1] = False
        histories = Histories(arrays)
        self.assertTrue(np.isnan(histories.values[:, 1]).all())
        self.assertTrue(np.isnan(histories.values[0]).all())

    def test_joint_week_bootstrap_and_seed_average_preserve_counts(self):
        before = pd.DataFrame({"week": ["a", "b"], "sum": [2.0, 6.0], "size": [1, 3]})
        after = before.assign(sum=[1.0, 3.0])
        result = week_interval(before, after, 100)
        self.assertEqual(result["mean_mae_improvement"], 1.0)
        self.assertEqual(result["ci95_lower"], 1.0)
        np.testing.assert_array_equal(
            mean_weekly([before, before])["size"], before["size"]
        )
        with self.assertRaises(ValueError):
            week_interval(before, after.assign(size=[2, 2]), 100)

    def test_scorer_runs_end_to_end_with_missing_target(self):
        arrays = {
            "split": np.array([1, 1]),
            "times": pd.date_range("2024-04-01", periods=2, freq="h", tz="UTC")
            .as_unit("ns")
            .asi8,
            "x": np.ones((2, 1, len(FEATURES)), np.float32),
            "y": np.array([[10.0], [np.nan]], np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            rows = evaluate(
                np.array([[11.0], [12.0]]),
                arrays,
                {"zones": ["FR"]},
                [0],
                "fixture",
                "val",
                Path(temporary),
            )
            self.assertEqual(rows[0]["hours"], 1)
            self.assertEqual(rows[0]["mae"], 1.0)

    def test_regime_thresholds_are_training_only(self):
        arrays = {
            "split": np.array([0, 0, 1, 1]),
            "y": np.array([[0.0], [100.0], [150.0], [-1.0]]),
        }
        frame = pd.DataFrame(
            {
                "zone": "FR",
                "ts_utc": pd.date_range("2024-07-01", periods=2, freq="h", tz="UTC"),
                "price": [150.0, -1.0],
                "prediction": [140.0, 0.0],
                "target_valid": True,
                "price_lag168": 0.0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.parquet"
            frame.to_parquet(path)
            rows, weekly, thresholds = diagnostics(path, arrays, {"zones": ["FR"]}, [0])
        self.assertEqual(thresholds["FR"], {"q05": 5.0, "q95": 95.0})
        self.assertEqual(
            next(row for row in rows if row["slice"] == "upper_tail")["hours"], 1
        )
        self.assertEqual(int(weekly["size"].sum()), 2)

    def test_partial_study_cannot_replace_milestone(self):
        with self.assertRaises(ValueError):
            publish({"phase": "smoke", "job_count": 20, "seeds": [224]}, CONFIG)

    @unittest.skipUnless(shutil.which("pdflatex"), "LaTeX not installed")
    def test_completed_study_template_fits_two_pages_with_fixture_values(self):
        variants = [
            "pooled_xgb_rich_none",
            "mlp_rich_none",
            "gat_rich_self",
            "gat_rich_real",
            "gru_mlp_rich_none",
            "gru_gat_rich_real",
        ]
        scores = [
            {
                "variant": variant,
                "split": split,
                "coverage": "all",
                "scope": "overall",
                "paired_mae_mean": 20.0,
                "paired_mae_std": 0.15,
            }
            for variant in variants
            for split in ["val", "test"]
        ]
        summary = {
            "scores": scores,
            "rewiring_variability": [
                {"split": split, "paired_mae_mean": 21.0, "paired_mae_training_sd": 0.2}
                for split in ["val", "test"]
            ],
            "validation_intervals": {
                "mlp_rich_none_to_gat_rich_real": {
                    "mean_mae_improvement": 0.1,
                    "ci95_lower": -0.1,
                    "ci95_upper": 0.3,
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in [
                "milestone.tex",
                "neurips_2019.sty",
                "numbers.tex",
                "suite_numbers.tex",
            ]:
                shutil.copy(ROOT / "reports/milestone" / name, directory / name)
            (directory / "controlled_study.tex").write_text(controlled_tex(summary))
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "milestone.tex",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-3000:])
            pages = subprocess.check_output(
                ["pdfinfo", str(directory / "milestone.pdf")], text=True
            )
            self.assertIn("Pages:           2", pages)


if __name__ == "__main__":
    unittest.main()
