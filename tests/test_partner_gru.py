"""Partner-style learning rules with audited times, identities and missingness."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from epgnn.partner_gru.data import Days, calendar, net_imports
from epgnn.partner_gru.training import ResidualGRU, loss

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/experiments/partner_gru.json").read_text())


def fixture():
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-03-20", periods=24 * 16, freq="h", tz="UTC").as_unit(
        "ns"
    )
    return {
        "times": times.asi8,
        "y": rng.normal(50, 5, (len(times), 2)).astype(np.float32),
        "active": np.ones((len(times), 2), bool),
        "edges": np.array([[0, 1], [1, 0]]),
        "split": np.where(
            times.tz_convert("Europe/Brussels").strftime("%Y-%m-%d") < "2024-03-30",
            0,
            1,
        ).astype(np.int8),
        "hist": rng.normal(size=(len(times), 2, 8)).astype(np.float32),
        "day": rng.normal(size=(len(times), 2, 11)).astype(np.float32),
    }


class PartnerGRUTests(unittest.TestCase):
    def test_dst_units_and_completion(self):
        for date, count in [("2024-03-31", 23), ("2024-10-27", 25)]:
            times = pd.date_range(
                pd.Timestamp(date, tz="Europe/Brussels") - pd.DateOffset(days=8),
                periods=24 * 12,
                freq="h",
            )
            for unit in ["ns", "us"]:
                cal = calendar(times.as_unit(unit), np.zeros(len(times), np.int8))
                day = cal["days"].index(date)
                self.assertEqual((cal["slots"][day] >= 0).sum(), count)
                observed = times.as_unit("ns").asi8[cal["history_positions"][day]]
                present = cal["history_present"][day]
                self.assertTrue(
                    (observed[present] + 3600_000_000_000 <= cal["origins"][day]).all()
                )
        for date, utc_hour in [("2024-01-03", 11), ("2024-07-03", 10)]:
            times = pd.date_range(date, periods=24, freq="h", tz="UTC")
            cal = calendar(times, np.zeros(len(times), np.int8))
            self.assertEqual(pd.Timestamp(cal["origins"][0], tz="UTC").hour, utc_hour)

    def test_duplicate_keys_rejected(self):
        times = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        with self.assertRaises(ValueError):
            calendar(times.append(times[-1:]), np.zeros(3, np.int8))

    def test_residual_roundtrip_fallbacks_and_missing_target(self):
        a = fixture()
        a["day"][:, :, :2] = np.nan
        a["day"][:, :, 0] = 17
        a["day"][0, 0, 0] = np.nan
        a["y"][0, 0] = np.nan
        data = Days(a, [0, 1])
        raw = data.raw([0])
        self.assertEqual(raw["base"][0, 0, 0], 0)
        self.assertEqual(raw["base"][0, 1, 0], 17)
        self.assertFalse(raw["mask"][0, 0, 0])
        restored = 50 * np.sinh(raw["target"] + np.arcsinh(raw["base"] / 50))
        slots = raw["slots"][0]
        good = slots >= 0
        np.testing.assert_allclose(restored[0, :, good], a["y"][slots[good]], atol=1e-5)

    def test_future_data_cannot_change_earlier_histories_or_scalers(self):
        a = fixture()
        data = Days(a, [0, 1])
        train = np.arange(3, 7)
        earlier = data.raw([6])["hist"].copy()
        state = data.fit_scalers(train)
        cutoff = data.calendar["slots"][7, 0]
        a["hist"][cutoff:] = 9999
        a["day"][cutoff:] = 9999
        np.testing.assert_array_equal(data.raw([6])["hist"], earlier)
        self.assertEqual(state, data.fit_scalers(train))
        self.assertFalse(np.array_equal(earlier[:, 0], earlier[:, 1]))

    def test_zero_flows_vs_incomplete_physical_observations(self):
        times = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC").as_unit("ns")
        frame = pd.DataFrame(
            {
                "ts_utc": [times[0], times[0], times[1], times[2], times[2]],
                "from_zone": ["a", "b", "a", "a", "b"],
                "to_zone": ["b", "a", "b", "b", "a"],
                "flow_phys": [10.0, 3.0, 9.0, 0.0, 0.0],
            }
        )
        values, complete = net_imports(
            times.asi8,
            ["a", "b"],
            np.ones((3, 2), bool),
            np.array([[0, 1], [1, 0]]),
            [pa.RecordBatch.from_pandas(frame)],
        )
        np.testing.assert_array_equal(values[0], [-7, 7])
        self.assertTrue(np.isnan(values[1]).all())
        np.testing.assert_array_equal(values[2], [0, 0])
        self.assertFalse(complete[1].any())

    def test_unseen_training_feature_and_inactive_history_stay_missing(self):
        a = fixture()
        a["hist"][:, :, 3] = np.nan
        a["day"][:, :, 7] = np.nan
        a["active"][:, 1] = False
        data = Days(a, [0])
        scalers = data.fit_scalers(np.arange(3, 7))
        self.assertEqual(scalers["hist"]["count"][3], 0)
        self.assertTrue(np.isnan(data.raw([6])["hist"][:, 1]).all())
        a["hist"][:, :, 3] = 999999
        a["day"][:, :, 7] = 999999
        batch, _ = data.batch([7], scalers, torch.device("cpu"))
        self.assertTrue((batch["hist"][..., 3] == 0).all())
        self.assertTrue((batch["xday"][..., 7] == 0).all())

    def test_nan_safe_loss_and_residual_contrast(self):
        prediction = torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0], [999.0, 0.0]]], requires_grad=True
        )
        target = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [float("nan"), float("nan")]]])
        mask = torch.isfinite(target)
        pairs = torch.tensor([[0], [1]])
        value = loss(prediction, target, mask, pairs, 0.5)
        self.assertAlmostEqual(float(value.detach()), 1.5)
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[0, 2].abs().sum()), 0)

    def test_no_inference_message_passing(self):
        torch.manual_seed(0)
        model = ResidualGRU(CONFIG).eval()
        h = torch.randn(2, 2, 168, 8)
        x = torch.randn(2, 2, 25, 11)
        with torch.no_grad():
            before = model(h, x)
            h[:, 1] *= 99
            x[:, 1] *= 99
            after = model(h, x)
        torch.testing.assert_close(before[:, 0], after[:, 0])
