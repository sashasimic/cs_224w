"""Forecast-time and matched input fixtures for the two graph families."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import torch

from epgnn.combined_graph_study import inputs, models, sql_features, training


class ForecastTests(unittest.TestCase):
    def test_dst_windows_use_elapsed_hours(self):
        hours = []
        for day, expected in (("2024-03-31", 23), ("2024-10-27", 25)):
            start = pd.Timestamp(day, tz="Europe/Brussels")
            end = start + pd.DateOffset(days=1)
            ts = pd.date_range(start, end, inclusive="left", freq="h")
            self.assertEqual(len(ts), expected)
            hours.extend(ts.tz_convert("UTC").as_unit("ns").asi8)
        origins, inverse, history = inputs.windows(
            np.asarray(hours), min(hours) - 216 * inputs.HOUR, 6000
        )
        self.assertTrue(np.all(inverse[:23] == 0))
        self.assertTrue(np.all(inverse[23:] == 1))
        self.assertEqual(pd.to_datetime(origins, utc=True).hour.tolist(), [11, 10])
        self.assertTrue((np.diff(history, axis=1) == 1).all())

    def test_future_publications_do_not_enter_sql_or_graph(self):
        data = np.array([0, 2, 90], np.float32).reshape(3, 1, 1)
        times = np.array([0, 1, 3])[:, None]
        history = np.array([[0, 1, 2]])
        observed = np.ones((3, 1), bool)
        before, valid = inputs.select(data, observed, times, history, np.array([2]))
        data[2] = -400
        after, new_valid = inputs.select(data, observed, times, history, np.array([2]))
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(valid, new_valid)
        with duckdb.connect() as connection:
            sql = sql_features.summaries(
                connection, after, valid, ("x",), (slice(None),), {"zone": [0]}
            )
        self.assertEqual(sql["presence_0"][0].as_py(), valid.sum())
        self.assertEqual(sql["f_x_0_latest"][0].as_py(), 2)
        self.assertEqual(sql["f_x_0_min"][0].as_py(), 0)

    def test_direction_and_missing_margin(self):
        ptdf = np.array([[0.5, 0, np.nan], [0.5, 0, np.nan]], np.float32)
        attributes = np.array([[0, 10, 0], [10, 10, 1]], np.float32)
        result = inputs.directional_margin(ptdf, attributes)
        np.testing.assert_allclose(result[0, 1], [2, 0.5, 1, 0.1])
        self.assertEqual(result[1, 0, 0], 0)
        self.assertTrue(np.isnan(result[1, 0, 2]))
        self.assertTrue(np.isnan(result[0, 2]).all())

    def test_inactive_and_missing_edges_use_local_state(self):
        torch.manual_seed(1)
        model = models.AdaptiveGNN(8, 3, "margin").eval()
        x = torch.zeros(1, 3, 10)
        active = torch.tensor([[True, False, True]])
        pair = (torch.zeros(1, 3, 3, 8), torch.zeros(1, 3, 3, dtype=torch.bool))
        physical = torch.zeros((3, 3), dtype=torch.bool)
        core = torch.arange(3)
        baseline, links = model(x, active, physical, core)
        actual, other = model(x, active, physical, core, pair)
        torch.testing.assert_close(baseline, actual)
        torch.testing.assert_close(links, other)
        self.assertEqual(links[:, 1].sum().item(), 0)

    def test_local_border_copies_block_reverse_messages(self):
        torch.manual_seed(31)
        edge = torch.tensor([[0, 1]])
        widths = {"price": 2, "weather": 8, "load": 2, "jao": 6, "flow": 2}
        first = {
            name: (
                torch.zeros(1, 6, 1 if name == "flow" else 2, width),
                torch.ones(1, 6, 1 if name == "flow" else 2, dtype=torch.bool),
            )
            for name, width in widths.items()
        }
        second = {name: (x.clone(), mask) for name, (x, mask) in first.items()}
        second["price"][0][:, :, 1] = 20
        recent = torch.zeros(1, 1, 2, 10)
        local = models.RelationalGNN(8, local=True).eval()
        crosses = models.RelationalGNN(8, local=False).eval()
        crosses.load_state_dict(local.state_dict())
        torch.testing.assert_close(
            local(first, recent, edge)[..., 0], local(second, recent, edge)[..., 0]
        )
        self.assertFalse(
            torch.equal(
                crosses(first, recent, edge)[..., 0],
                crosses(second, recent, edge)[..., 0],
            )
        )

    @unittest.skipUnless(
        torch.cuda.is_available(), "Host ROCm required for exact resume check"
    )
    def test_calibration_resume(self):
        rng = np.random.default_rng(24)
        data = SimpleNamespace(
            recent=rng.normal(size=(24, 3, 5)).astype(np.float32),
            prices=rng.normal(size=(24, 3)).astype(np.float32),
            active=np.ones((24, 3), bool),
            zones=["FR", "BE", "NL"],
            edges=np.array([[0, 1], [1, 2]]),
            core=np.arange(3),
            origins=np.zeros(1),
            day=np.zeros(24, np.int64),
            margin=rng.normal(size=(24, 3, 3, 4)).astype(np.float32),
            margin_hours=np.arange(24),
        )
        fit = np.arange(16)
        calibration = np.arange(16, 24)
        phase = training.NeuralPhase(
            data,
            "adaptive",
            fit,
            torch.device("cuda:0"),
            {"width": 8, "adaptive_hours": 8, "patience": 5, "lr": 0.001},
        )
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            phase.fit("margin", 224, fit, root / "resumed", 1, calibration)
            resumed = phase.fit("margin", 224, fit, root / "resumed", 2, calibration)[0]
            uninterrupted = phase.fit(
                "margin", 224, fit, root / "direct", 2, calibration
            )[0]
            a = phase.infer(resumed, "margin", calibration)[0]
            b = phase.infer(uninterrupted, "margin", calibration)[0]
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
