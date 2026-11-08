"""Residual GRU+MLP, safe masked residual/spread loss and resumable day batches."""

import numpy as np
import torch
from torch import nn

from epgnn.experiment_suite.neural import atomic_checkpoint, require_gpu


class ResidualGRU(nn.Module):
    def __init__(self, config, n_hist=8, n_day=11):
        super().__init__()
        self.gru = nn.GRU(n_hist, config["hidden"], batch_first=True)
        width = config["mlp_hidden"]
        self.head = nn.Sequential(
            nn.Linear(config["hidden"] + n_day, width),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, hist, xday):
        batch, zones, window, features = hist.shape
        # Keep computation on ROCm, but bypass the fused MIOpen RNN path.
        # The trained 168-step case disagreed with CPU; native GPU matched it.
        with torch.backends.cudnn.flags(enabled=False):
            _, state = self.gru(hist.reshape(batch * zones, window, features))
        encoded = state[-1].reshape(batch, zones, -1)
        encoded = encoded[:, :, None, :].expand(-1, -1, xday.shape[2], -1)
        return self.head(torch.cat([encoded, xday], dim=-1)).squeeze(-1)


def loss(prediction, target, mask, pairs, weight):
    valid = mask & torch.isfinite(target)
    if not valid.any():
        return prediction.sum() * 0
    error = prediction[valid] - target[valid]
    result = error.abs().mean()
    if pairs.numel() and weight:
        a, b = pairs
        paired = valid[:, a, :] & valid[:, b, :]
        if paired.any():
            # As in partner training: contrast of transformed residual targets,
            # NOT the raw EUR/MWh interconnector price-spread error.
            delta = (prediction[:, a, :][paired] - prediction[:, b, :][paired]) - (
                target[:, a, :][paired] - target[:, b, :][paired]
            )
            result = result + weight * delta.abs().mean()
    return result


def epoch(model, data, days, scalers, config, optimizer=None):
    device = require_gpu()
    model.train(optimizer is not None)
    pairs = torch.as_tensor(data.pairs, device=device)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(days), config["batch_days"]):
            selected = days[start : start + config["batch_days"]]
            batch, _ = data.batch(selected, scalers, device)
            prediction = model(batch["hist"], batch["xday"])
            value = loss(
                prediction,
                batch["target"],
                batch["mask"],
                pairs,
                config["spread_weight"],
            )
            if not torch.isfinite(value):
                raise ValueError("Nonfinite residual/spread loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("Nonfinite gradients")
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
            total += float(value.detach()) * len(selected)
            count += len(selected)
    return total / max(1, count)


def fit(data, config, directory, token, smoke=False):
    import time

    device = require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    rng = np.random.default_rng(config["seed"])
    train = np.flatnonzero(data.calendar["split"] == 0)
    val = np.flatnonzero(data.calendar["split"] == 1)
    if smoke:
        train, val = (
            train[-config["smoke_training_days"] :],
            val[: config["smoke_validation_days"]],
        )
    model = ResidualGRU(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scalers = data.fit_scalers(train)
    curves, best, stale, first = [], float("inf"), 0, 0
    latest = directory / "latest.pt"
    if latest.exists():
        state = torch.load(latest, weights_only=False, map_location=device)
        if state["fingerprint"] != token or state["smoke"] != smoke:
            raise ValueError("Checkpoint identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scalers, curves = state["scalers"], state["curves"]
        best, stale, first = state["best"], state["stale"], state["epoch"] + 1
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["gpu_rng"]])
    epochs = config["smoke_epochs"] if smoke else config["epochs"]
    for i in range(first, epochs if stale < config["patience"] else first):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        tr = epoch(model, data, rng.permutation(train), scalers, config, optimizer)
        va = epoch(model, data, val, scalers, config)
        improved = va < best - config["min_delta"]
        best, stale = (va, 0) if improved else (best, stale + 1)
        curve = {
            "epoch": i + 1,
            "train_residual_plus_spread": tr,
            "val_residual_plus_spread": va,
            "seconds": time.monotonic() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        curves.append(curve)
        state = {
            "fingerprint": token,
            "smoke": smoke,
            "epoch": i,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scalers": scalers,
            "curves": curves,
            "best": best,
            "stale": stale,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state_all(),
            "config": config,
        }
        if improved:
            atomic_checkpoint(directory / "best.pt", state)
        atomic_checkpoint(latest, state)
        print(curve, flush=True)
        if stale >= config["patience"]:
            break
    state = torch.load(directory / "best.pt", weights_only=False, map_location=device)
    model.load_state_dict(state["model"])
    return model, scalers, curves


def predict(model, scalers, data, config, code):
    """Undo day padding into canonical hour-major arrays, including missing labels."""
    output = np.full_like(data.arrays["y"], np.nan, dtype=np.float32)
    model.eval()
    days = np.flatnonzero(data.calendar["split"] == code)
    with torch.no_grad():
        for start in range(0, len(days), config["batch_days"]):
            selected = days[start : start + config["batch_days"]]
            batch, raw = data.batch(selected, scalers, require_gpu())
            residual = model(batch["hist"], batch["xday"]).cpu().numpy()
            price = 50 * np.sinh(residual + np.arcsinh(raw["base"] / 50))
            if not np.isfinite(price).all():
                raise ValueError("Nonfinite inverse residual transform")
            for k, positions in enumerate(raw["slots"]):
                exists = positions >= 0
                output[positions[exists]] = price[k][:, exists].T
    indices = np.flatnonzero(data.arrays["split"] == code)
    if not np.isfinite(output[indices]).all():
        raise ValueError("Unassigned canonical prediction slots")
    return output[indices][:, data.zones]


def cpu_reference_check(model, scalers, data, days):
    """Independent float64 CPU check on actual inputs, in original price units."""
    batch, raw = data.batch(days, scalers, torch.device("cpu"))
    reference = (
        ResidualGRU(
            {
                "hidden": model.gru.hidden_size,
                "mlp_hidden": model.head[0].out_features,
                "dropout": model.head[2].p,
            }
        )
        .double()
        .eval()
    )
    reference.load_state_dict(
        {k: v.detach().cpu().double() for k, v in model.state_dict().items()}
    )
    model.eval()
    with torch.no_grad():
        cpu = reference(batch["hist"].double(), batch["xday"].double()).numpy()
        gpu = (
            model(batch["hist"].to(require_gpu()), batch["xday"].to(require_gpu()))
            .cpu()
            .numpy()
        )
    base = np.arcsinh(raw["base"].astype(np.float64) / 50)
    difference = np.abs(
        50 * np.sinh(cpu + base) - 50 * np.sinh(gpu.astype(float) + base)
    )[raw["mask"]]
    if (
        not len(difference)
        or not np.isfinite(difference).all()
        or difference.max() > 0.01
        or difference.mean() > 0.0001
    ):
        raise ValueError("Actual-input native GPU/CPU GRU reference mismatch")
    return {
        "max_absolute_eur_mwh": float(difference.max()),
        "mean_absolute_eur_mwh": float(difference.mean()),
        "max_allowed_eur_mwh": 0.01,
        "mean_allowed_eur_mwh": 0.0001,
        "observations": len(difference),
        "backend": "native GPU vs independent CPU float64",
    }
