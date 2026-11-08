"""One FP32 training/inference path for every classical and graph control."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from epgnn.experiment_suite import neural
from epgnn.experiment_suite.data import shuffle_edges
from epgnn.train import masked_loss
from .models import make_model, parameter_count


def graph_for(job, arrays):
    if job.graph in {"none", "self"}:
        return np.empty((2, 0), np.int64)
    edges = np.asarray(arrays["edges"]).copy()
    return shuffle_edges(edges, job.rewiring_seed) if job.graph == "shuffled" else edges


def tensors(arrays, indices, columns, preprocessing, history, device):
    x, target, active = neural.batch_tensors(
        arrays, indices, columns, preprocessing, device
    )
    sequence = None
    if history is not None:
        values, inverse = history.batch(indices)
        sequence = (
            torch.from_numpy(values).to(device),
            torch.from_numpy(inverse).to(device),
        )
    return x, target, active, sequence


@dataclass
class Fitted:
    model: torch.nn.Module
    preprocessing: neural.Preprocessor
    history: object
    edges: torch.Tensor
    curves: list
    batch_size: int


def predict(fitted, arrays, indices, columns, device):
    fitted.model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), fitted.batch_size):
            x, _, active, sequence = tensors(
                arrays,
                indices[start : start + fitted.batch_size],
                columns,
                fitted.preprocessing,
                fitted.history,
                device,
            )
            output = fitted.model(x, fitted.edges, active, sequence)
            prediction = (50 * torch.sinh(output)).cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError(
                    "Nonfinite neural prediction or inverse-transform overflow"
                )
            predictions.append(prediction)
    return np.concatenate(predictions)


def fit(
    job,
    arrays,
    columns,
    zones,
    directory,
    config,
    token,
    batch_size,
    histories=None,
    smoke=False,
):
    device = neural.require_gpu()
    torch.set_num_threads(4)
    torch.manual_seed(job.seed)
    torch.cuda.manual_seed_all(job.seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = config["neural"]
    train = np.flatnonzero(arrays["split"] == 0)
    validation = np.flatnonzero(arrays["split"] == 1)
    epochs = settings["epochs"]
    if smoke:
        train = train[-24 * config["smoke"]["training_days"] :]
        validation = validation[: 24 * config["smoke"]["validation_days"]]
        epochs = config["smoke"]["epochs"]
    raw = np.asarray(arrays["x"][train])[:, :, columns]
    preprocessing = neural.Preprocessor().fit(raw[np.asarray(arrays["active"][train])])
    del raw
    history = histories if job.model.startswith("gru_") else None
    if history is not None:
        history.fit(train)
    edges = torch.tensor(graph_for(job, arrays), device=device)
    model = make_model(job.model, len(columns) * 2, settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    zone_mask = torch.zeros(arrays["y"].shape[1], dtype=torch.bool, device=device)
    zone_mask[zones] = True
    first_epoch, best, stale, curves = 0, float("inf"), 0, []
    latest = directory / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state["identity"] != token or state["job"] != job.record():
            raise ValueError("Checkpoint architecture/configuration identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        preprocessing = neural.Preprocessor.restore(state["preprocessing"])
        if history is not None:
            history.restore(state["history"])
        first_epoch, best, stale, curves = (
            state["epoch"] + 1,
            state["best"],
            state["stale"],
            state["curves"],
        )
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["gpu_rng"]])
        if state["batch_size"] != batch_size:
            raise ValueError("Cannot resume with a changed batch schedule")
    fitted = Fitted(model, preprocessing, history, edges, curves, batch_size)
    if stale >= settings["patience"]:
        first_epoch = epochs
    for epoch in range(first_epoch, epochs):
        started = time.monotonic()
        # Identical order across architectures, reproducible at an epoch boundary.
        order = np.random.default_rng(job.seed + epoch).permutation(train)
        model.train()
        total, count = 0.0, 0
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(order), batch_size):
            x, target, active, sequence = tensors(
                arrays,
                order[start : start + batch_size],
                columns,
                preprocessing,
                history,
                device,
            )
            valid = torch.isfinite(target) & zone_mask[None, :]
            if not valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x, edges, active, sequence), target, valid)
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
            n = int(valid.sum())
            total += float(loss.detach()) * n
            count += n
        if not count:
            raise ValueError("No valid training labels")
        prediction = predict(fitted, arrays, validation, columns, device)[:, zones]
        actual = np.asarray(arrays["y"][validation])[:, zones]
        valid = np.isfinite(actual)
        val_loss = float(
            np.abs(
                np.arcsinh(prediction[valid] / 50) - np.arcsinh(actual[valid] / 50)
            ).mean()
        )
        if not np.isfinite(val_loss):
            raise RuntimeError("Invalid validation loss")
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        curve = {
            "epoch": epoch + 1,
            "train_transformed_mae": total / count,
            "val_transformed_mae": val_loss,
            "seconds": time.monotonic() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        curves.append(curve)
        state = {
            "identity": token,
            "job": job.record(),
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "preprocessing": preprocessing.state(),
            "history": history.state() if history else None,
            "best": best,
            "stale": stale,
            "curves": curves,
            "batch_size": batch_size,
            "torch_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state_all(),
            "parameters": parameter_count(model),
            "architecture": {
                "kind": job.model,
                "input_features": len(columns) * 2,
                "head_width": model.width,
                "history_hidden": settings["history_hidden"],
            },
        }
        if improved:
            neural.atomic_checkpoint(directory / "best.pt", state)
        # Publish the selected weights before the resumable epoch marker. A crash
        # between the two writes then repeats an epoch instead of losing a best model.
        neural.atomic_checkpoint(latest, state)
        print(f"{job.name}: {curve}", flush=True)
        if stale >= settings["patience"]:
            break
    best_state = torch.load(
        directory / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_state["model"])
    return fitted


def reload_fitted(job, columns, arrays, directory, config, histories=None):
    device = neural.require_gpu()
    state = torch.load(
        Path(directory) / "best.pt", map_location=device, weights_only=False
    )
    if state["job"] != job.record():
        raise ValueError("Checkpoint belongs to a different model")
    model = make_model(job.model, len(columns) * 2, config["neural"]).to(device)
    model.load_state_dict(state["model"])
    preprocessing = neural.Preprocessor.restore(state["preprocessing"])
    history = histories if job.model.startswith("gru_") else None
    if history is not None:
        history.restore(state["history"])
    return Fitted(
        model,
        preprocessing,
        history,
        torch.tensor(graph_for(job, arrays), device=device),
        state["curves"],
        state["batch_size"],
    )
