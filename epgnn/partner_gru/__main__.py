"""python -m epgnn.partner_gru {audit,gpu-smoke,smoke,full,status}."""

import argparse
import fcntl
import json
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epgnn.controlled_study.runner import check_complete, signatures
from epgnn.dataset_build.storage import ROOT
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.evaluation import evaluate, grouped_scores
from epgnn.experiment_suite.neural import require_gpu

from . import training
from .data import Days, prepare


def gpu_smoke(config, output):
    """Actual architecture, 16 days x 62 nodes, residual loss and masked gradients."""
    torch.set_num_threads(4)
    device = require_gpu()
    torch.manual_seed(config["seed"])
    model = training.ResidualGRU(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    hist = torch.randn(config["batch_days"], 62, 168, 8, device=device)
    day = torch.randn(config["batch_days"], 62, 25, 11, device=device)
    target = torch.randn(config["batch_days"], 62, 25, device=device)
    target[:, -1] = float("nan")
    mask = torch.isfinite(target)
    pairs = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(hist, day)
        loss = training.loss(prediction, target, mask, pairs, config["spread_weight"])
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise ValueError("GPU smoke has invalid gradients")
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.monotonic() - started)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.pt"
        torch.save(model.state_dict(), path)
        model.eval()
        before = model(hist, day).detach()
        restored = training.ResidualGRU(config).to(device).eval()
        restored.load_state_dict(
            torch.load(path, weights_only=True, map_location=device)
        )
        torch.testing.assert_close(before, restored(hist, day), rtol=1e-5, atol=1e-5)
        cpu = training.ResidualGRU(config).double().eval()
        cpu.load_state_dict(
            {k: v.detach().cpu().double() for k, v in model.state_dict().items()}
        )
        with torch.no_grad():
            independent = cpu(hist.cpu().double(), day.cpu().double()).numpy()
        np.testing.assert_allclose(
            before.cpu().numpy(), independent, rtol=1e-4, atol=1e-5
        )
    peak = torch.cuda.max_memory_allocated() / 2**30
    if peak > 12:
        raise ValueError("Preflight exceeds the 12 GiB tensor budget")
    # Exercise the real fit/resume path, not merely state_dict serialization.
    rng = np.random.default_rng(0)
    times_axis = pd.date_range(
        "2024-01-01", periods=24 * 12, freq="h", tz="UTC"
    ).as_unit("ns")
    arrays = {
        "times": times_axis.asi8,
        "split": np.where(
            times_axis.tz_convert("Europe/Brussels").strftime("%Y-%m-%d")
            < "2024-01-09",
            0,
            1,
        ).astype(np.int8),
        "y": rng.normal(50, 5, (len(times_axis), 2)).astype(np.float32),
        "hist": rng.normal(size=(len(times_axis), 2, 8)).astype(np.float32),
        "day": rng.normal(size=(len(times_axis), 2, 11)).astype(np.float32),
        "active": np.ones((len(times_axis), 2), bool),
        "edges": np.array([[0, 1], [1, 0]]),
    }
    arrays["y"][0, 0] = np.nan
    fixture = Days(arrays, [0, 1])
    local = dict(config, epochs=1)
    with tempfile.TemporaryDirectory() as tmp:
        resumed, continuous = Path(tmp) / "resumed", Path(tmp) / "continuous"
        resumed.mkdir()
        continuous.mkdir()
        training.fit(fixture, local, resumed, "resume_fixture")
        local["epochs"] = 2
        training.fit(fixture, local, resumed, "resume_fixture")
        training.fit(fixture, local, continuous, "resume_fixture")
        first = torch.load(
            resumed / "latest.pt", weights_only=False, map_location="cpu"
        )
        second = torch.load(
            continuous / "latest.pt", weights_only=False, map_location="cpu"
        )
        if [r["epoch"] for r in first["curves"]] != [1, 2]:
            raise ValueError("Resume repeated/skipped epoch")
        for name in first["model"]:
            torch.testing.assert_close(
                first["model"][name], second["model"][name], rtol=1e-4, atol=1e-5
            )
    evidence = {
        "passed": True,
        "gpu": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_vram_gib": peak,
        "warm_batch_seconds": float(np.mean(times[1:])),
        "config": config,
        "checks": [
            "forward/backward",
            "NaN-masked residual and spread loss",
            "Adam",
            "checkpoint reload",
            "epoch-boundary resume against continuous fit",
            "native GPU forward agrees with independent CPU float64",
        ],
    }
    write_json(output, evidence)
    print(json.dumps(evidence, indent=2), flush=True)
    return evidence


def run(config, smoke):
    cache, token, metadata, arrays = prepare(config)
    root = ROOT / "runs/partner_gru" / token
    directory = root / ("smoke" if smoke else "full")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Phase already running") from error
        if (directory / "complete.json").exists():
            check_complete(directory)
            print("Already complete:", directory)
            return
        try:
            if smoke:
                gpu_smoke(config, root / "gpu_preflight.json")
            else:
                checked = check_complete(root / "smoke")
                if checked["fingerprint"] != token or not checked["verified"]:
                    raise ValueError("Matching verified smoke required")
            zones = [metadata["zones"].index("FR")] if smoke else metadata["eligible"]
            data = Days(arrays, zones, config["history_hours"])
            write_json(
                directory / "provenance.json",
                {
                    "fingerprint": token,
                    "cache": str(cache),
                    "config": config,
                    "zones": [metadata["zones"][j] for j in zones],
                    "differences_from_partner": metadata["differences_from_partner"],
                    "availability": metadata["availability"],
                    "scale_fit": "fit days, scored identities only; padded xday slots excluded",
                },
            )
            write_json(
                directory / "status.json", {"state": "training", "fingerprint": token}
            )
            started = time.monotonic()
            model, scalers, curves = training.fit(data, config, directory, token, smoke)
            write_json(directory / "scalers.json", scalers)
            rows, spread_metrics, cpu_checks = [], {}, {}
            for split, code in [("val", 1), ("test", 2)]:
                prediction = training.predict(model, scalers, data, config, code)
                days = np.flatnonzero(data.calendar["split"] == code)
                cpu_checks[split] = training.cpu_reference_check(
                    model, scalers, data, days[: config["batch_days"]]
                )
                # Independent checkpoint round trip on actual data.
                restored = training.ResidualGRU(config).to(require_gpu())
                state = torch.load(
                    directory / "best.pt",
                    weights_only=False,
                    map_location=require_gpu(),
                )
                restored.load_state_dict(state["model"])
                actual = training.predict(
                    restored, state["scalers"], data, config, code
                )
                np.testing.assert_allclose(prediction, actual, rtol=1e-5, atol=1e-4)
                label = "partner_residual_gru_conditional"
                rows.extend(
                    r
                    for r in evaluate(
                        prediction, arrays, metadata, zones, label, split, directory
                    )
                    if r["coverage"] == "all"
                )
                frame = pd.read_parquet(directory / f"predictions_{split}.parquet")
                valid = frame.target_valid.to_numpy(bool)
                price = frame.price.to_numpy(float)
                pred = frame.prediction.to_numpy(float)
                paired = valid & np.isfinite(frame.price_lag168.to_numpy(float))
                if (
                    frame.duplicated(["zone", "ts_utc"]).any()
                    or not np.isfinite(pred).all()
                ):
                    raise ValueError("Invalid saved prediction keys")
                expected = np.asarray(arrays["y"][arrays["split"] == code])[
                    :, zones
                ].T.reshape(-1)
                np.testing.assert_array_equal(price, expected)
                selected = np.flatnonzero(arrays["split"] == code)
                y = np.asarray(arrays["y"][selected])
                p = np.full_like(y, np.nan)
                p[:, zones] = prediction
                a, b = data.pairs
                mask = np.isfinite(y[:, a]) & np.isfinite(y[:, b])
                delta = (p[:, a] - p[:, b]) - (y[:, a] - y[:, b])
                spread_metrics[split] = {
                    "pairs": len(a),
                    "paired_edge_hours": int(mask.sum()),
                    "price_spread_mae_eur_mwh": float(np.abs(delta[mask]).mean())
                    if mask.any()
                    else None,
                }
                # Recompute both evaluation masks from saved original-unit predictions.
                original = next(
                    r
                    for r in grouped_scores(rows)
                    if r["scope"] == "overall" and r["split"] == split
                )
                np.testing.assert_allclose(
                    [
                        np.abs(pred[valid] - price[valid]).mean(),
                        np.abs(pred[paired] - price[paired]).mean(),
                    ],
                    [original["mae"], original["paired_mae"]],
                    rtol=0,
                    atol=1e-9,
                )
            record = {
                "fingerprint": token,
                "verified": True,
                "phase": "smoke" if smoke else "full",
                "seed": config["seed"],
                "seconds": time.monotonic() - started,
                "curves": curves,
                "parameters": sum(p.numel() for p in model.parameters()),
                "rows": rows,
                "aggregate": grouped_scores(rows),
                "spread_metrics": spread_metrics,
                "cpu_reference_checks": cpu_checks,
                "target_signatures": signatures(directory),
                "availability": metadata["availability"],
                "limitations": [
                    "Single-seed adaptation, not an exact partner reproduction",
                    "Retrospective ERA5 is not available weather forecasts",
                    *metadata["differences_from_partner"],
                ],
            }
            if not smoke:
                reference = json.loads(
                    (
                        ROOT / "results/airport_temperature_full_verification.json"
                    ).read_text()
                )
                if record["target_signatures"] != reference["target_signatures"]:
                    raise ValueError("Canonical comparison target keys/masks changed")
            write_json(directory / "result.json", record)
            from epgnn.dataset_build.storage import digest

            write_json(
                directory / "complete.json",
                {
                    p.name: digest(p)["sha256"]
                    for p in directory.iterdir()
                    if p.is_file()
                    and p.name not in {"complete.json", "status.json", "run.lock"}
                },
            )
            prefix = ROOT / (
                "results/partner_gru_smoke" if smoke else "results/partner_gru_full"
            )
            write_json(str(prefix) + ".json", record)
            pd.DataFrame(rows).to_csv(str(prefix) + "_by_zone.csv", index=False)
            if not smoke:
                publish_report(record)
            write_json(
                directory / "status.json", {"state": "complete", "fingerprint": token}
            )
            print("COMPLETE:", directory, flush=True)
        except BaseException:
            write_json(
                directory / "status.json",
                {
                    "state": "failed",
                    "traceback": traceback.format_exc(),
                    "fingerprint": token,
                },
            )
            raise


def publish_report(record):
    lines = [
        "# Partner-inspired residual GRU: conditional canonical-data run",
        "",
        f"Fingerprint: `{record['fingerprint']}`. Seed {record['seed']}; one full adaptation, not five-seed evidence or an exact partner reproduction.",
        "",
        "ERA5 is reanalysis, not archived operational forecasts. Recent ERA5 observations have a known publication-latency conflict with the noon contract. Load and flow vintages and anomaly climatology remain unverified. These are retrospective conditional scores.",
        "",
        "The initial fused ROCm GRU run was rejected after a CPU cross-check exposed a long-sequence backend discrepancy. This run uses native GPU RNN operations, verified against independent CPU float64 predictions; see results/partner_gru_backend_audit.json. Original failed checkpoints remain archived and are not scientific comparison scores.",
        "",
        "Inputs retain original missingness and historical zone identities. The baseline used in residual reconstruction falls back from weekly to guarded daily price, then zero; targets are never filled. Training uses multivariate 168-hour histories, a 64-unit GRU, 128-unit ReLU head, day batches, and L1 residual error plus 0.5 times cross-zone residual-contrast error. This loss is not raw-price spread MAE. Early stopping uses the same residual-plus-spread validation objective.",
        "",
        "| Split | All-target MAE | All-target RMSE | Matched MAE | Weekly persistence | rMAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in record["aggregate"]:
        if row["scope"] == "overall":
            ratio = row["paired_mae"] / row["persistence_mae"]
            lines.append(
                f"| {row['split']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['paired_mae']:.4f} | {row['persistence_mae']:.4f} | {ratio:.4f} |"
            )
    lines += [
        "",
        "MAE/RMSE and separately saved physical price-spread MAE are in EUR/MWh. Target keys, values and masks match the existing 49-market evaluation. Missing weather does not remove targets. October 2026 remains unassigned.",
        "",
        "Important adaptations relative to the partner are documented in the experiment guide: lag48 day weather, literal local noon, timestamp joins, canonical geography, conservative training-pair net imports and safe NaN masking. The partner's published 20.40 MAE uses a different 43-zone cohort and protocol; it is not directly ranked against this result. Prior controlled-study GRU scores used a different architecture, inputs and objective.",
        "",
        "## Validation learning curve",
        "",
        "| Epoch | Training residual + spread | Validation residual + spread | Seconds |",
        "|---|---:|---:|---:|",
    ]
    for curve in record["curves"]:
        lines.append(
            f"| {curve['epoch']} | {curve['train_residual_plus_spread']:.5f} | {curve['val_residual_plus_spread']:.5f} | {curve['seconds']:.2f} |"
        )
    (ROOT / "docs/partner_gru_results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["audit", "gpu-smoke", "smoke", "full", "status"]
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/partner_gru.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "gpu-smoke":
        gpu_smoke(config, ROOT / "runs/partner_gru_gpu_preflight.json")
    elif args.command in {"audit", "status"}:
        cache, token, metadata, _ = prepare(config)
        info = {
            "fingerprint": token,
            "cache": str(cache),
            "metadata": metadata if args.command == "audit" else None,
        }
        if args.command == "status":
            for phase in ["smoke", "full"]:
                path = ROOT / "runs/partner_gru" / token / phase / "status.json"
                info[phase] = (
                    json.loads(path.read_text())
                    if path.exists()
                    else {"state": "not_started"}
                )
        print(json.dumps(info, indent=2))
    else:
        run(config, smoke=args.command == "smoke")


if __name__ == "__main__":
    main()
