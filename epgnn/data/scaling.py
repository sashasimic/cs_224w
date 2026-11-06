"""Target transform and input standardisation. Statistics are fitted on training rows only."""
import numpy as np

ASINH_SCALE = 50.0   # EUR/MWh; asinh(y/50) is ~linear below 50 and log-like above, handles negatives


def to_asinh(y: np.ndarray) -> np.ndarray:
    return np.arcsinh(y / ASINH_SCALE)


def from_asinh(t: np.ndarray) -> np.ndarray:
    return np.sinh(t) * ASINH_SCALE


def residual_target(y: np.ndarray, y_lag168: np.ndarray) -> np.ndarray:
    """What the torch models predict: asinh(price) - asinh(price one week earlier)."""
    return to_asinh(y) - to_asinh(y_lag168)


def residual_to_price(r: np.ndarray, y_lag168: np.ndarray) -> np.ndarray:
    return from_asinh(r + to_asinh(y_lag168))


class ZScaler:
    """Per-feature z-score over the last axis, NaN-aware; fit on train, apply everywhere."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ZScaler":
        flat = x.reshape(-1, x.shape[-1])
        self.mean = np.nanmean(flat, axis=0)
        self.std = np.nanstd(flat, axis=0) + 1e-6
        return self

    def transform(self, x: np.ndarray, fill: float = 0.0) -> np.ndarray:
        assert self.mean is not None and self.std is not None, "fit first"
        z = (x - self.mean) / self.std
        return np.where(np.isnan(z), fill, z).astype("float32")
