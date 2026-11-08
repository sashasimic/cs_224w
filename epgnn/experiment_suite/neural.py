"""ROCm GAT regression, train-only preprocessing, and resumable checkpoints."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from epgnn.train import masked_loss


def atomic_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def require_gpu():
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("A working ROCm GPU is required; no silent CPU fallback")
    return torch.device("cuda:0")


class Preprocessor:
    """Median imputation, standardization, and explicit missingness indicators."""

    def fit(self, values):
        self.median = np.array(
            [
                np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
                for c in values.T
            ],
            np.float32,
        )
        filled = np.where(np.isfinite(values), values, self.median)
        self.mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = filled.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        return self

    def transform(self, values):
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.median, values)
        return np.concatenate(
            [(filled - self.mean) / self.scale, missing.astype(np.float32)], axis=-1
        ).astype(np.float32)

    def state(self):
        return {k: getattr(self, k).tolist() for k in ["median", "mean", "scale"]}

    @classmethod
    def restore(cls, state):
        obj = cls()
        for k, v in state.items():
            setattr(obj, k, np.asarray(v, np.float32))
        return obj


class PriceGAT(nn.Module):
    """Two spatial layers per hourly graph; no legacy history tensors."""

    def __init__(self, features, hidden=64, heads=4, dropout=0.1):
        super().__init__()
        self.first = GATv2Conv(features, hidden // heads, heads=heads, dropout=dropout)
        self.second = GATv2Conv(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edges, active):
        batch, zones, features = x.shape
        offsets = torch.arange(batch, device=x.device)[:, None] * zones
        allowed = active[:, edges[0]] & active[:, edges[1]]
        source = (edges[0][None, :] + offsets)[allowed]
        destination = (edges[1][None, :] + offsets)[allowed]
        edge_index = torch.stack([source, destination])
        hidden = self.dropout(
            torch.nn.functional.elu(self.first(x.reshape(-1, features), edge_index))
        )
        hidden = self.dropout(torch.nn.functional.elu(self.second(hidden, edge_index)))
        return self.head(hidden).reshape(batch, zones)


def batch_tensors(arrays, indices, features, prep, device):
    raw = np.asarray(arrays["x"][indices])[:, :, features]
    active = np.asarray(arrays["active"][indices])
    x = prep.transform(raw)
    x[~active] = 0.0
    y = np.arcsinh(np.asarray(arrays["y"][indices]) / np.float32(50))
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(active.copy()).to(device),
    )


def infer(model, arrays, indices, features, prep, edges, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x, _, active = batch_tensors(
                arrays, indices[start : start + batch_size], features, prep, device
            )
            output = model(x, edges, active)
            if not torch.isfinite(output).all():
                raise RuntimeError("Nonfinite neural prediction")
            prediction = (50 * torch.sinh(output)).cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError("Inverse target transformation overflowed")
            predictions.append(prediction)
    return np.concatenate(predictions)


def train(
    arrays, features, selected_zones, directory, cfg, shuffled=False, smoke=False
):
    device = require_gpu()
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train_indices = np.flatnonzero(arrays["split"] == 0)
    val_indices = np.flatnonzero(arrays["split"] == 1)
    if smoke:
        # A contiguous recent training window, not test-driven sampling.
        train_indices = train_indices[-24 * 90 :]
        val_indices = val_indices[: 24 * 14]
    target_mask = np.zeros(arrays["y"].shape[1], bool)
    target_mask[selected_zones] = True
    prep_values = np.asarray(arrays["x"][train_indices])[:, :, features]
    prep_active = np.asarray(arrays["active"][train_indices])
    prep = Preprocessor().fit(prep_values[prep_active])
    del prep_values, prep_active
    edges = torch.tensor(
        np.asarray(arrays["shuffled_edges" if shuffled else "edges"]).copy(),
        device=device,
    )
    zone_mask = torch.from_numpy(target_mask).to(device)
    model = PriceGAT(len(features) * 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    batch_size, first_epoch, best, stale, curves = (
        cfg["batch_size"],
        0,
        float("inf"),
        0,
        [],
    )
    latest = directory / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["identity"] != cfg["identity"]:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        prep = Preprocessor.restore(state["preprocessor"])
        first_epoch, best, stale, curves = (
            state["epoch"] + 1,
            state["best"],
            state["stale"],
            state["curves"],
        )
        batch_size = state["batch_size"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["gpu_rng"]])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    epochs = 3 if smoke else cfg["epochs"]
    if stale >= cfg["patience"]:
        first_epoch = (
            epochs  # Interrupted after early stopping: do not train another epoch.
        )
    for epoch in range(first_epoch, epochs):
        started = time.monotonic()
        order = np.random.permutation(train_indices)
        model.train()
        loss_sum = 0.0
        observations = 0
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x, target, active = batch_tensors(arrays, indices, features, prep, device)
            valid = torch.isfinite(target) & zone_mask[None, :]
            if not valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, edges, active)
            loss = masked_loss(prediction, target, valid)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise RuntimeError("Nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(valid.sum())
            loss_sum += float(loss.detach()) * count
            observations += count
        if not observations:
            raise ValueError("No trainable targets")
        val_prediction = infer(
            model, arrays, val_indices, features, prep, edges, batch_size, device
        )
        actual = np.asarray(arrays["y"][val_indices])[:, selected_zones]
        predicted = val_prediction[:, selected_zones]
        valid = np.isfinite(actual)
        val_loss = float(
            np.abs(
                np.arcsinh(predicted[valid] / 50) - np.arcsinh(actual[valid] / 50)
            ).mean()
        )
        improved = val_loss < best
        if improved:
            best, stale = val_loss, 0
        else:
            stale += 1
        curve = {
            "epoch": epoch + 1,
            "train_transformed_mae": loss_sum / observations,
            "val_transformed_mae": val_loss,
            "seconds": time.monotonic() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "batch_size": batch_size,
        }
        curves.append(curve)
        state = {
            "identity": cfg["identity"],
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "preprocessor": prep.state(),
            "best": best,
            "stale": stale,
            "curves": curves,
            "batch_size": batch_size,
            "torch_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        atomic_checkpoint(latest, state)
        if improved:
            atomic_checkpoint(directory / "best.pt", state)
        print(f"GAT shuffled={shuffled}: {curve}", flush=True)
        if stale >= cfg["patience"]:
            break
    state = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model, prep, edges, curves, batch_size
