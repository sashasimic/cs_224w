"""Price-only models using the same frozen UTC-hour snapshot."""

from datetime import date

import numpy as np

from .contracts import BRUSSELS, COMPONENTS, DAY, HISTORY, HOUR, THREADS, hour_grid

CURVE_DAYS = (1, 2, 3, 7)
LAGS = (24, 48, 72, 168, 336)


def local_calendar(day, grid):
    local = [t.astimezone(BRUSSELS) for t in grid]
    hour = np.asarray([t.hour for t in local])
    features = np.column_stack(
        [
            np.eye(7)[[day.weekday()] * len(grid)],
            hour,
            np.sin(2 * np.pi * day.timetuple().tm_yday / 366) * np.ones(len(grid)),
            np.cos(2 * np.pi * day.timetuple().tm_yday / 366) * np.ones(len(grid)),
            [t.utcoffset() / HOUR for t in local],
        ]
    ).astype(np.float32)
    return hour, features


def daily_inputs(day, history):
    """Return calendar, exact lag queries, daily curves, and context."""
    grid = hour_grid(day)
    n, zones = len(grid), history.shape[1]
    hour, calendar = local_calendar(day, grid)
    lag = np.full((n, zones, len(LAGS)), np.nan, np.float32)
    for j, gap in enumerate(LAGS):
        index = HISTORY + np.arange(n) - gap
        valid = (index >= 0) & (index < HISTORY)
        lag[valid, :, j] = history[index[valid]]

    curves = np.full((zones, len(CURVE_DAYS), 24), np.nan, np.float32)
    begin = grid[0] - HISTORY * HOUR
    for k, days in enumerate(CURVE_DAYS):
        for t in hour_grid(day - days * DAY):
            i = (t - begin) // HOUR
            if 0 <= i < HISTORY:
                h = t.astimezone(BRUSSELS).hour
                # During autumn, select the first available occurrence of the local hour.
                curves[:, k, h] = np.where(
                    np.isfinite(curves[:, k, h]), curves[:, k, h], history[i]
                )
    return hour, calendar, lag, curves.reshape(zones, -1), history


def weekly(inputs):
    return inputs[2][..., LAGS.index(168)]


def tree_features(inputs):
    _, calendar, lag, _, _ = inputs
    n, zones = lag.shape[:2]
    return np.concatenate(
        [
            np.broadcast_to(calendar[:, None, :], (n, zones, calendar.shape[1])),
            lag,
            np.broadcast_to(
                np.arange(zones, dtype=np.float32)[None, :, None], (n, zones, 1)
            ),
        ],
        axis=2,
    )


def filled(x, center=0):
    x = np.asarray(x)
    return np.concatenate(
        [np.where(np.isfinite(x), x, center), np.isfinite(x)], axis=-1
    )


def fit_tree(x, y, rounds):
    from xgboost import XGBRegressor

    good = np.isfinite(y)
    if good.sum() < 24:
        return None
    return XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=rounds,
        max_depth=5,
        learning_rate=0.08,
        tree_method="hist",
        n_jobs=THREADS,
        random_state=224,
    ).fit(x[good], y[good])


def predict_tree(model, x):
    if model is None:
        return np.full(x.shape[:-1], np.nan, np.float32)
    return model.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1])


def fit_lear(days, strength):
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    hours = np.concatenate([p[0] for p, _ in days])
    calendar = np.concatenate([p[1] for p, _ in days])
    prices = np.concatenate(
        [np.broadcast_to(p[3], (len(y), *p[3].shape)) for p, y in days]
    )
    targets = np.concatenate([y for _, y in days])
    fits = {}
    for z in range(targets.shape[1]):
        features = np.column_stack([calendar, filled(prices[:, z, :])])
        for h in range(24):
            valid = (hours == h) & np.isfinite(targets[:, z])
            if valid.sum() >= 30:
                # Use price-unit regularization on standardized predictors.
                fits[z, h] = make_pipeline(
                    StandardScaler(),
                    Lasso(
                        alpha=strength,
                        max_iter=4000,
                        tol=0.001,
                    ),
                ).fit(features[valid], targets[valid, z])
    return fits


def predict_lear(fits, inputs):
    hours, calendar, lag, curves, _ = inputs
    pred = np.full(lag.shape[:2], np.nan, np.float32)
    for (z, h), fit in fits.items():
        mask = hours == h
        if mask.any():
            prices = np.broadcast_to(
                filled(curves[z]), (mask.sum(), curves.shape[-1] * 2)
            )
            pred[mask, z] = fit.predict(np.column_stack([calendar[mask], prices]))
    return pred


def factor_features(inputs, center, scale, basis):
    _, calendar, lag, _, _ = inputs
    norm = (lag.transpose(0, 2, 1) - center) / scale
    observed = np.isfinite(norm)
    factors = np.where(observed, norm, 0) @ basis
    residual = norm - factors @ basis.T
    shared = np.column_stack(
        [calendar, factors.reshape(len(calendar), -1), observed.mean(axis=2)]
    )
    return shared, residual


def fit_factor(days, n_factors):
    from sklearn.linear_model import Ridge

    y = np.concatenate([y for _, y in days])
    observed = np.isfinite(y)
    counts = observed.sum(axis=0).clip(1)
    center = np.where(observed, y, 0).sum(axis=0) / counts
    norm = y - center
    scale = np.sqrt(np.where(observed, norm**2, 0).sum(axis=0) / counts).clip(1)
    norm = norm / scale
    imputed = np.where(observed, norm, 0)
    # All normalization, loadings and forecasts are fitted on this fold's training rows.
    _, eigenvectors = np.linalg.eigh(imputed.T @ imputed)
    basis = eigenvectors[:, -n_factors:]
    factors = imputed @ basis
    shared, residual_lags = zip(
        *(factor_features(p, center, scale, basis) for p, _ in days)
    )
    shared, residual_lags = np.concatenate(shared), np.concatenate(residual_lags)
    usable = observed.any(axis=1)
    common_fit = Ridge(alpha=20).fit(shared[usable], factors[usable])
    residual = norm - factors @ basis.T
    fits = {}
    for z in range(y.shape[1]):
        good = observed[:, z]
        if good.sum() >= 24:
            x = np.column_stack([shared, filled(residual_lags[:, :, z])])
            fits[z] = Ridge(alpha=20).fit(x[good], residual[good, z])
    return center, scale, basis, common_fit, fits


def predict_factor(artifact, inputs):
    center, scale, basis, common_fit, fits = artifact
    shared, residual_lags = factor_features(inputs, center, scale, basis)
    pred = common_fit.predict(shared) @ basis.T
    for z in range(pred.shape[1]):
        fit = fits.get(z)
        if fit is None:
            pred[:, z] = np.nan
        else:
            pred[:, z] += fit.predict(
                np.column_stack([shared, filled(residual_lags[:, :, z])])
            )
    return pred * scale + center


def sequence_batch(inputs):
    _, calendar, _, _, hist = inputs
    # Convert seven UTC days to 7 GRU steps with a 24-hour price and mask per step.
    recent = (hist[-168:].T.reshape(hist.shape[1], 7, 24) / 100).clip(-30, 30)
    context = filled(recent).astype(np.float32)
    cal = np.pad(
        calendar / np.array([1] * 7 + [24, 1, 1, 2], np.float32),
        ((0, 25 - len(calendar)), (0, 0)),
    )
    head = np.broadcast_to(cal.reshape(-1), (hist.shape[1], cal.size)).copy()
    return context, head


def sequence_model(zones, width):
    import torch
    from torch import nn

    class Sequence(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(zones, 8)
            self.encoder = nn.GRU(48, width, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(width + 8 + 25 * 11, width), nn.SiLU(), nn.Linear(width, 25)
            )

        def forward(self, history, calendar, ids):
            _, hidden = self.encoder(history)
            return self.head(
                torch.cat([hidden[0], calendar, self.embedding(ids)], dim=1)
            )

    return Sequence()


def fit_sequence(days, width):
    import torch

    torch.set_num_threads(THREADS)
    torch.manual_seed(224)
    zones = days[0][1].shape[1]
    model = sequence_model(zones, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    h, cal = zip(*(sequence_batch(p) for p, _ in days))
    y = np.concatenate(
        [
            np.pad(y, ((0, 25 - len(y)), (0, 0)), constant_values=np.nan).T
            for _, y in days
        ]
    )
    history, calendar = map(torch.from_numpy, (np.concatenate(h), np.concatenate(cal)))
    target = torch.from_numpy(y / 100)
    mask = torch.isfinite(target)
    clean = torch.nan_to_num(target)
    ids = torch.arange(zones).repeat(len(days))
    for _ in range(14):
        for step in torch.randperm(len(ids)).split(512):
            good = mask[step]
            if good.any():
                pred = model(history[step], calendar[step], ids[step])
                loss = (pred - clean[step]).abs()[good].mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    learned = {
        key: val.detach().cpu().numpy() for key, val in model.state_dict().items()
    }
    return width, learned, np.isfinite(y).reshape(-1, zones, 25).any(axis=(0, 2))


def predict_sequence(artifact, inputs):
    import torch

    torch.set_num_threads(THREADS)
    width, learned, supported = artifact
    context, calendar = sequence_batch(inputs)
    model = sequence_model(len(context), width)
    model.load_state_dict({key: torch.from_numpy(val) for key, val in learned.items()})
    with torch.inference_mode():
        result = model(
            torch.from_numpy(context),
            torch.from_numpy(calendar),
            torch.arange(len(context)),
        )
    pred = result.numpy().T[: len(inputs[0])] * 100
    pred[:, ~supported] = np.nan
    return pred


def timesfm_forecaster(artifact):
    import sys

    import torch

    from .contracts import file_hash

    for path, expected in artifact["hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("TimesFM asset hash mismatch")
    sys.path[:0] = artifact["python_path"]
    from timesfm3 import ModelConfig, TimesFM3Forecaster

    if artifact["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ROCm is unavailable")
    torch.set_num_threads(THREADS)
    return TimesFM3Forecaster(
        ModelConfig(
            checkpoint_path=artifact["checkpoint"],
            device=artifact["device"],
            per_core_batch_size=8,
        )
    )


def predict_timesfm(model, inputs, joint):
    hist = inputs[-1].T.copy()
    available = np.isfinite(hist).any(axis=1)
    context = hist[available]
    grid = np.arange(hist.shape[1])
    for curve in context:
        good = np.isfinite(curve)
        curve[~good] = np.interp(grid[~good], grid[good], curve[good])
    horizon = len(inputs[0])
    pred = np.full((horizon, len(available)), np.nan, np.float32)
    if available.any():
        settings = {
            "horizon": horizon,
            "use_symmetric_averaging": True,
            "make_positive": False,
        }
        if joint:
            pred[:, available] = next(
                model.predict_batch([context], **settings)
            ).forecast.T
        else:
            pred[:, available] = np.stack(
                [p.forecast for p in model.predict_batch(list(context), **settings)],
                axis=1,
            )
    return pred


def blend_predictions(weights, predictions, shape):
    stack = np.stack(
        [predictions.get(name, np.full(shape, np.nan)) for name in COMPONENTS], axis=-1
    )
    active = np.isfinite(stack) * weights
    total = active.sum(axis=-1)
    pred = np.where(np.isfinite(stack), stack, 0)
    return np.divide(
        (pred * active).sum(axis=-1), total, out=np.full(shape, np.nan), where=total > 0
    )


def predict(name, artifact, snapshot):
    inputs = daily_inputs(
        date.fromisoformat(snapshot["day"]), np.asarray(snapshot["history"], np.float32)
    )
    if name == "weekly":
        return weekly(inputs)
    if name == "pooled_xgb":
        return predict_tree(artifact, tree_features(inputs))
    if name == "zone_xgb":
        features = tree_features(inputs)
        return np.column_stack(
            [
                predict_tree(artifact[z], features[:, z, :-1])
                for z in range(features.shape[1])
            ]
        )
    if name == "lear":
        return predict_lear(artifact, inputs)
    if name == "factor":
        return predict_factor(artifact, inputs)
    if name == "sequence":
        return predict_sequence(artifact, inputs)
    if name.startswith("timesfm_"):
        return predict_timesfm(
            timesfm_forecaster(artifact), inputs, name == "timesfm_joint"
        )
    raise ValueError(f"Unknown model {name}")
