"""Train, resume and score the validation-only CORE graph experiment."""

import itertools
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epgnn.boosting import errors
from epgnn.dataset_build.storage import digest, json_write
from epgnn.experiment_suite import data, neural
from epgnn.train import masked_loss

from . import catalog, graph

VARIANTS = ("interconnector", "flat_jao", "bipartite", "shuffled")
CONTRASTS = (
    ("bipartite", "flat_jao"),
    ("bipartite", "shuffled"),
    ("flat_jao", "interconnector"),
)


def study_inputs(settings, directory, output):
    marker = Path(settings["benchmark_arrays"]) / "complete.json"
    for name, expected in catalog.read_json(marker).items():
        actual = digest(marker.parent / name)["sha256"]
        if actual != expected:
            raise ValueError(f"Benchmark cache checksum mismatch: {name}")
    meta, arrays = data.load(marker.parent)
    if meta["dataset_fingerprint"] != settings["dataset_fingerprint"]:
        raise ValueError("Benchmark arrays belong to a different canonical dataset")
    zones = meta["zones"]
    # Only the pre-existing 49-market roster and train-derived topology are reused.
    # No test price values reach the training process or scoring tables.
    days = pd.to_datetime(arrays["times"], unit="ns", utc=True).tz_convert(
        "Europe/Brussels"
    )
    fit = (arrays["split"] == 0) & (days >= pd.Timestamp("2022-06-09", tz=days.tz))
    val = arrays["split"] == 1
    selected = fit | val
    times = arrays["times"][selected]
    raw = np.asarray(arrays["x"][selected, :, : len(data.RECENT)])
    prices = arrays["y"][selected]
    active = arrays["active"][selected]
    cache_file = output / "graphs.npz"
    if not cache_file.exists():
        print("Preparing timestamp-aligned CORE graphs", flush=True)
        cache = graph.observations(directory / "forecast.duckdb", times, zones)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            staged = Path(temporary) / cache_file.name
            np.savez(staged, **cache)
            staged.replace(cache_file)
    with np.load(cache_file) as stored:
        cache = dict(stored)
    train_size = int(fit.sum())
    return meta, times, raw, prices, active, arrays["edges"], train_size, cache


def fit_job(
    variant,
    seed,
    settings,
    directory,
    raw,
    prices,
    active,
    edges,
    train,
    validation,
    snapshots,
    roster,
    device,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    preprocessing = neural.Preprocessor().fit(raw[train][active[train]])
    inputs = preprocessing.transform(raw)
    inputs[~active] = 0
    transformed = np.arcsinh(prices / np.float32(50))
    is_bipartite = variant in ("bipartite", "shuffled")
    features = inputs.shape[-1]
    model = (
        graph.ConstraintGAT(features) if is_bipartite else neural.PriceGAT(features)
    ).to(device)
    edges = torch.tensor(edges, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    zone_mask = np.zeros(raw.shape[1], dtype=bool)
    zone_mask[roster] = True
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pt"
    best, stale, start, curves = np.inf, 0, 0, []
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["gpu_rng"]])
        best, stale, start, curves = (
            state["best"],
            state["stale"],
            state["epoch"],
            state["curves"],
        )

    def predict(indices):
        x = torch.from_numpy(inputs[indices]).to(device)
        current_active = active[indices]
        if is_bipartite:
            return model(
                x,
                snapshots.batch(indices, current_active, device, variant == "shuffled"),
            )
        return model(x, edges, torch.from_numpy(current_active).to(device))

    def batches(indices):
        size = settings["batch_size"]
        for start in range(0, len(indices), size):
            yield indices[start : start + size]

    @torch.no_grad()
    def infer():
        model.eval()
        prediction = np.concatenate(
            [predict(batch).cpu().numpy() for batch in batches(validation)]
        )
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite model predictions")
        return prediction

    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        began = time.monotonic()
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        train_order = np.random.default_rng(seed + epoch).permutation(train)
        for batch in batches(train_order):
            label = transformed[batch]
            valid = np.isfinite(label) & zone_mask[None, :]
            if not valid.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(
                predict(batch),
                torch.from_numpy(label).to(device),
                torch.from_numpy(valid).to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
        estimate, actual = infer()[:, roster], transformed[validation][:, roster]
        mask = np.isfinite(actual)
        value = float(np.abs(estimate[mask] - actual[mask]).mean())
        if not np.isfinite(value):
            raise ValueError(
                "Cannot select a checkpoint without finite validation loss"
            )
        improved = value < best
        best, stale = (value, 0) if improved else (best, stale + 1)
        curves.append(
            {
                "epoch": epoch + 1,
                "val_transformed_mae": value,
                "seconds": time.monotonic() - began,
                "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )
        if improved:
            neural.atomic_checkpoint(directory / "best.pt", model.state_dict())
        neural.atomic_checkpoint(
            latest,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "gpu_rng": torch.cuda.get_rng_state_all(),
                "best": best,
                "stale": stale,
                "epoch": epoch + 1,
                "curves": curves,
            },
        )
        print(f"{variant} seed {seed} {curves[-1]}", flush=True)
    model.load_state_dict(
        torch.load(directory / "best.pt", map_location=device, weights_only=True)
    )
    predictions = 50 * np.sinh(infer()[:, roster])
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite forecast after reversing the target transform")
    return predictions, curves, sum(p.numel() for p in model.parameters())


def scores(frame):
    y = frame.price.to_numpy()
    prediction = frame.prediction.to_numpy()
    weekly = frame.weekly.to_numpy()
    valid = frame.target_valid.to_numpy()
    paired = valid & np.isfinite(weekly)
    model = errors(y, prediction, valid)
    common = errors(y, prediction, paired)
    baseline = errors(y, weekly, paired)
    return {
        "n": model["n_hours"],
        "mae": model["mae"],
        "rmse": model["rmse"],
        "paired_n": common["n_hours"],
        "paired_mae": common["mae"],
        "persistence_mae": baseline["mae"],
        "persistence_skill": 1 - common["mae"] / baseline["mae"]
        if baseline["mae"]
        else None,
    }


def paired_intervals(common, fitted, variants, repetitions):
    results = []
    for scope, keep in (
        ("all", common.target_valid),
        ("core_covered", common.target_valid & common.core_covered),
    ):
        if not keep.any():
            continue
        days = common.ts_utc[keep].dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
        weeks = days.dt.to_period("W-SUN")
        y = common.price.to_numpy()[keep]
        for model, control in CONTRASTS:
            if model not in variants or control not in variants:
                continue
            model_losses = np.mean([np.abs(p[keep] - y) for p in fitted[model]], axis=0)
            control_losses = np.mean(
                [np.abs(p[keep] - y) for p in fitted[control]], axis=0
            )
            delta = control_losses - model_losses
            grouped = pd.Series(delta).groupby(weeks.to_numpy()).agg(["sum", "size"])
            rng = np.random.default_rng(224)
            sample = rng.integers(len(grouped), size=(repetitions, len(grouped)))
            draws = grouped["sum"].to_numpy()[sample].sum(axis=1) / grouped[
                "size"
            ].to_numpy()[sample].sum(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            results.append(
                {
                    "scope": scope,
                    "model": model,
                    "control": control,
                    "mae_improvement": float(delta.mean()),
                    "ci95": [float(low), float(high)],
                    "weeks": len(grouped),
                    "n": int(keep.sum()),
                }
            )
    return results


def run(settings, smoke):
    device = neural.require_gpu()
    torch.set_num_threads(4)
    directory = catalog.build(settings)
    config = settings | ({"epochs": settings["smoke"]["epochs"]} if smoke else {})
    manifest = {
        "settings": config,
        "catalog": directory.name,
        "smoke": smoke,
        "benchmark": digest(Path(settings["benchmark_arrays"]) / "complete.json")[
            "sha256"
        ],
        "code": {
            name: digest(Path(__file__).with_name(name))["sha256"]
            for name in ("graph.py", "experiment.py")
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
    }
    output = (
        Path(settings["runs"])
        / ("smoke" if smoke else "validation")
        / catalog.identity(manifest)
    )
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "manifest.json", manifest)
    print(f"Writing experiment to {output}", flush=True)
    meta, times, recent, prices, active, edges, train_size, cache = study_inputs(
        settings, directory, output
    )
    roster = np.asarray(meta["eligible"])
    train, validation = np.arange(train_size), np.arange(train_size, len(times))
    if smoke:
        train = train[-24 * config["smoke"]["training_days"] :]
        validation = validation[: 24 * config["smoke"]["validation_days"]]
    snapshots = graph.Snapshots(
        cache, active, meta["zones"], train, config["shuffle_seed"]
    )
    full_features = np.concatenate((recent, cache["summary"]), axis=-1)
    del cache
    target = prices[np.ix_(validation, roster)].ravel()
    weekly = recent[
        np.ix_(validation, roster, [data.RECENT.index("price_lag168")])
    ].ravel()
    common = pd.DataFrame(
        {
            "ts_utc": np.repeat(
                pd.to_datetime(times[validation], unit="ns", utc=True), len(roster)
            ),
            "zone": np.tile(np.asarray(meta["zones"])[roster], len(validation)),
            "price": target,
            "target_valid": np.isfinite(target),
            "weekly": weekly,
            "core_covered": (snapshots.connected & active)[
                np.ix_(validation, roster)
            ].ravel(),
        }
    )
    fitted = {variant: [] for variant in VARIANTS}
    rows, job_results = [], []
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    for variant, seed in itertools.product(VARIANTS, seeds):
        job = output / f"{variant}_{seed}"
        features = recent if variant == "interconnector" else full_features
        prediction, curve, parameters = fit_job(
            variant,
            seed,
            config,
            job,
            features,
            prices,
            active,
            edges,
            train,
            validation,
            snapshots,
            roster,
            device,
        )
        frame = common.assign(prediction=prediction.ravel())
        frame.to_parquet(job / "predictions.parquet", index=False)
        fitted[variant].append(frame.prediction.to_numpy())
        covered = frame.core_covered
        strata = [
            ("all", frame),
            ("core_covered", frame[covered]),
            ("other", frame[~covered]),
        ]
        strata.extend((zone, group) for zone, group in frame.groupby("zone", sort=True))
        rows.extend(
            dict(variant=variant, seed=seed, scope=scope, **scores(group))
            for scope, group in strata
        )
        job_results.append(
            {
                "variant": variant,
                "seed": seed,
                "epochs": curve,
                "parameters": parameters,
            }
        )
        json_write(
            job / "result.json", job_results[-1] | {"scores": rows[-len(strata) :]}
        )
        print(
            f"{variant} seed {seed} France {scores(frame[frame.zone.eq('FR')])}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    intervals = paired_intervals(
        common, fitted, VARIANTS, config["validation_bootstraps"]
    )
    json_write(
        output / "results.json",
        {
            "admission": "conditional",
            "metrics": rows,
            "contrasts": intervals,
            "jobs": job_results,
        },
    )
    print(f"Completed {output}", flush=True)
