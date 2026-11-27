"""No peeking: every model input for delivery day D must be known when the forecast is made (D-1 10:00 UTC).

Run from the repository root:  python -m pytest tests/test_no_peek.py

What is checked
  1. price inputs of day D never come from day D itself (the 25-hour day of the autumn clock change is the trap)
  2. the GRU history ends at the cutoff hour, and a longer window only adds older hours
  3. the cross-hour layer carries only columns in features.KNOWN_AT_CUTOFF from one hour to another
  4. that list holds no outcome and no weather column
Not checked here, because they are documented assumptions of the dataset and not code: the lag-24 weather
columns (a stand-in for a weather forecast) and the publication time of the load forecast.
"""
import numpy as np
import pandas as pd
import torch

from epgnn.data import features as F
from epgnn.data import samples as S
from epgnn.data import tables
from epgnn.models.gat import ZoneGAT

ZONES = ["FR", "DE_LU", "PL", "ES"]
SPLIT = "val"            # Apr 2024 - Mar 2025: holds the 25-hour day 2024-10-27 and the 23-hour day 2025-03-30


def test_price_inputs_never_come_from_day_d() -> None:
    s = S.build(SPLIT, ZONES)
    lag = S.day_inputs(s, ["price_lag24"])[:, :, 0]
    start, exists = s.hours[:, :1], s.hours >= 0
    source = s.hours - 24 * S.HOUR_NS
    risky = exists & (source >= start)                       # the hour 24 h earlier lies inside day D
    assert risky.any(), "expected a 25-hour day in the split"
    rows, slots = np.nonzero(risky)
    # risky slots hold the last price of D-1 (the value of the slot before them), not the day's own first price
    assert np.array_equal(lag[rows, slots], lag[rows, slots - 1], equal_nan=True)
    first = s.y[rows, 0]
    both = np.isfinite(lag[rows, slots]) & np.isfinite(first)
    assert both.any() and not np.allclose(lag[rows, slots][both], first[both])
    # every other slot reads the price table at an hour before day D starts
    r, c = np.nonzero(exists & ~risky)
    assert (source[r, c] < start[r, 0]).all()
    price = tables.nodes().set_index(["zone", "ts_utc"]).price
    expected = price.reindex(pd.MultiIndex.from_arrays([s.zone[r], pd.to_datetime(source[r, c], utc=True)])).to_numpy(dtype="float32")
    same = np.isfinite(expected) & np.isfinite(lag[r, c])    # gap-filled lags have no table price to compare with
    assert same.mean() > 0.95 and np.allclose(lag[r, c][same], expected[same], atol=1e-3)
    # the weekly lag is a week back by construction
    assert ((s.hours - 168 * S.HOUR_NS)[exists] < np.broadcast_to(start, s.hours.shape)[exists]).all()


def test_history_ends_at_cutoff_and_longer_window_only_adds_older_hours() -> None:
    s = S.build(SPLIT, ZONES)
    h168, h336 = S.history(s, F.HISTORY, 168), S.history(s, F.HISTORY, 336)
    assert h336.shape[1] == 336
    assert np.array_equal(h336[:, -168:], h168, equal_nan=True)
    n = tables.nodes().set_index(["zone", "ts_utc"])
    cutoff = pd.DatetimeIndex(pd.to_datetime(s.day, utc=True)) - pd.Timedelta(days=1) + pd.Timedelta(hours=S.CUTOFF_UTC)
    newest = F.as_float(n.reindex(pd.MultiIndex.from_arrays([s.zone, cutoff])), F.HISTORY)
    assert np.array_equal(h168[:, -1], newest, equal_nan=True)   # the newest history row is the cutoff hour
    assert "price" not in F.HISTORY                               # prices enter the history only as 24 h / 168 h lags


def test_cross_hour_layer_carries_only_known_columns() -> None:
    torch.manual_seed(0)
    cols = F.FEATURE_SETS["own"]
    known = [i for i, c in enumerate(cols) if c in F.KNOWN_AT_CUTOFF]
    other = [i for i in range(len(cols)) if i not in known]      # the lag-24 weather columns
    assert other, "the test needs at least one column outside KNOWN_AT_CUTOFF"
    B, Z, H, slot = 2, 3, S.H, 20
    hist, xday = torch.randn(B, Z, 24, len(F.HISTORY)), torch.randn(B, Z, H, len(cols))
    extra = {"zz": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])}

    def moved_elsewhere(model: ZoneGAT, columns: list[int]) -> float:
        """Largest change at any hour other than `slot` when `columns` are changed at `slot` only."""
        x2 = xday.clone()
        x2[:, :, slot, columns] += 5.0
        with torch.no_grad():
            d = (model(hist, x2, extra) - model(hist, xday, extra)).abs()
        assert float(d[:, :, slot].max()) > 1e-4                 # the hour itself does react
        d[:, :, slot] = 0
        return float(d.max())

    mixer = ZoneGAT(len(F.HISTORY), len(cols), mix_cols=known).eval()
    assert moved_elsewhere(mixer, other) < 1e-6                  # weather of one hour never reaches another hour
    assert moved_elsewhere(mixer, known) > 1e-4                  # known columns do: the layer is connected
    plain = ZoneGAT(len(F.HISTORY), len(cols)).eval()
    assert moved_elsewhere(plain, known) < 1e-6                  # without the layer the hours are independent


def test_known_at_cutoff_list_is_clean() -> None:
    assert not [c for c in F.KNOWN_AT_CUTOFF if c in F.OUTCOMES or c in F.WEATHER_PERFECT]
    assert [c for c in F.KNOWN_AT_CUTOFF if c.endswith("_lag24")] == ["price_lag24"]
    assert set(F.KNOWN_AT_CUTOFF) <= set(F.FEATURE_SETS["own"])

