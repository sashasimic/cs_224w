"""Same-identity, timestamp-joined price histories ending at local-noon origin."""

import numpy as np
import pandas as pd

from epgnn.dataset_build.contracts import forecast_origin, utc


def history_index(times, window=168):
    times = pd.DatetimeIndex(utc(pd.Series(times)))
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("History calendar must be unique and chronological")
    origins = pd.DatetimeIndex(forecast_origin(pd.Series(times)))
    unique, inverse = np.unique(origins.asi8, return_inverse=True)
    required = unique[:, None] - np.arange(window, 0, -1)[None, :] * 3_600_000_000_000
    positions = np.searchsorted(times.asi8, required)
    safe = np.minimum(positions, len(times) - 1)
    present = (positions < len(times)) & (times.asi8[safe] == required)
    return safe, present, inverse


class Histories:
    """Store each unique forecast-origin window once; expand only a GPU batch."""

    def __init__(self, arrays, window=168):
        times = pd.to_datetime(np.asarray(arrays["times"]), unit="ns", utc=True)
        positions, present, self.day_index = history_index(times, window)
        values = np.asarray(arrays["y"][positions], dtype=np.float32)
        active = np.asarray(arrays["active"][positions])
        values = np.where(present[:, :, None] & active, values, np.nan)
        self.values = values.transpose(0, 2, 1)
        self.window = window

    def fit(self, train_indices):
        days = np.unique(self.day_index[train_indices])
        values = np.arcsinh(self.values[days] / np.float32(50))
        finite = values[np.isfinite(values)]
        self.median = float(np.median(finite)) if len(finite) else 0.0
        self.mean = float(finite.mean(dtype=np.float64)) if len(finite) else 0.0
        self.scale = (
            max(float(finite.std(dtype=np.float64)), 1e-6) if len(finite) else 1.0
        )
        return self

    def state(self):
        return {name: getattr(self, name) for name in ["median", "mean", "scale"]}

    def restore(self, state):
        for name in ["median", "mean", "scale"]:
            setattr(self, name, float(state[name]))
        return self

    def batch(self, indices):
        days, inverse = np.unique(self.day_index[indices], return_inverse=True)
        values = np.arcsinh(self.values[days] / np.float32(50))
        missing = ~np.isfinite(values)
        normalized = (np.where(missing, self.median, values) - self.mean) / self.scale
        return np.stack([normalized, missing.astype(np.float32)], axis=-1), inverse
