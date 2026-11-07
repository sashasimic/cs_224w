"""Audit reproductions only: partner model code is not modified."""
import importlib.util

import numpy as np
import pandas as pd

from .storage import ROOT, digest


def modelling_blockers():
    t = pd.Series(pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")).dt.as_unit("us")
    cutoff_ns = pd.Timestamp("2024-01-01T03:00Z").value
    wrong = int(np.searchsorted(t.astype("int64"), cutoff_ns, side="right"))
    correct = int(np.searchsorted(t.dt.as_unit("ns").astype("int64"), cutoff_ns, side="right"))
    pred, target, mask = np.array([2., 3.]), np.array([1., np.nan]), np.array([True, False])
    wrong_loss = float((np.abs(pred - target) * mask).sum() / max(mask.sum(), 1))
    correct_loss = float(np.abs(pred[mask] - target[mask]).mean())
    out = [
        {"id": "history_timestamp_unit", "severity": "modelling_blocker", "source": "epgnn/data/samples.py:history",
         "reproduced": wrong == 8 and correct == 4, "wrong_history_end_index": wrong,
         "correct_history_end_index": correct,
         "explanation": "astype(int64) and DatetimeIndex.asi8 each retain their own units; mismatches can empty history or admit future observations"},
        {"id": "nan_multiplication_mask", "severity": "modelling_blocker", "source": "epgnn/train.py:masked_loss",
         "reproduced": bool(np.isnan(wrong_loss) and correct_loss == 1),
         "wrong_loss": "NaN", "correct_loss": correct_loss,
         "explanation": "NaN * False remains NaN; masking after subtraction by multiplication does not protect the loss"},
        {"id": "fixed_utc_noon", "severity": "modelling_blocker", "source": "epgnn/data/samples.py:CUTOFF_UTC",
         "explanation": "10:00 UTC is Brussels noon only in summer; the required winter origin is 11:00 UTC"},
    ]
    from unittest.mock import patch
    from epgnn.data.samples import Samples, history
    sample = Samples(zone=np.array(["BE"]), day=np.array(["2024-01-02"]),
        y=np.zeros((1, 25)), mask=np.ones((1, 25), dtype=bool), hours=np.zeros((1, 25), dtype="int64"))
    observations = pd.DataFrame({"zone": "BE", "ts_utc": pd.date_range("2024-01-01", periods=16, freq="h", tz="UTC"),
                                 "value": np.arange(16, dtype=float)})
    partner_results = {}
    for unit in ["ns", "us"]:
        frame = observations.copy()
        frame["ts_utc"] = frame.ts_utc.dt.as_unit(unit)
        with patch("epgnn.data.samples.tables.nodes", return_value=frame):
            values = history(sample, ["value"], window=2).ravel()
        partner_results[unit] = [float(x) if np.isfinite(x) else None for x in values]
    out[0]["partner_history_results_by_unit"] = partner_results
    out[0]["partner_expected_history"] = [9., 10.]
    out[0]["partner_function_reproduced"] = any(v != [9., 10.] for v in partner_results.values())
    # Execute the unchanged partner implementations when torch is installed.
    if importlib.util.find_spec("torch"):
        import torch
        from epgnn.train import masked_loss
        actual = masked_loss(torch.tensor(pred), torch.tensor(target), torch.tensor(mask))
        out[1]["partner_function_reproduced"] = bool(torch.isnan(actual))
    for r in out:
        p = ROOT / r["source"].split(":")[0]
        r["code_sha256"] = digest(p)["sha256"]
    return out
