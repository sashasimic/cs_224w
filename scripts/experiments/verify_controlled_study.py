"""Verify a completed study phase from artifacts, without training or using GPU."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.controlled_study.design import jobs
from epgnn.controlled_study.preflight import DEFAULT_CONFIG, checked
from epgnn.controlled_study.runner import check_complete, study_identity
from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.artifacts import write_json


def verify(config, phase, save=False):
    token, _ = study_identity(config)
    root = ROOT / "runs/controlled_study" / token
    directory = root / phase
    summary = json.loads((directory / "summary.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    planned = [
        job for job in jobs(config) if phase == "full" or job.seed == config["seeds"][0]
    ]
    if summary["fingerprint"] != token or status["state"] != "complete":
        raise ValueError("Study phase has not completed for this source identity")
    if sorted(item["name"] for item in summary["jobs"]) != sorted(
        job.name for job in planned
    ):
        raise ValueError("Completed jobs differ from the predefined matrix")
    reference = {}
    for job in planned:
        result = check_complete(directory / job.name)
        for split in ["val", "test"]:
            frame = pd.read_parquet(
                directory / job.name / f"predictions_{split}.parquet"
            )
            keys = frame[["zone", "ts_utc", "price", "target_valid"]]
            if frame.duplicated(["zone", "ts_utc"]).any():
                raise ValueError("Duplicate prediction keys")
            if split in reference:
                pd.testing.assert_frame_equal(reference[split], keys)
            else:
                reference[split] = keys
            prediction = frame.prediction.to_numpy(float)
            price = frame.price.to_numpy(float)
            valid = frame.target_valid.to_numpy(bool)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite saved predictions")
            residual = prediction[valid] - price[valid]
            paired = valid & np.isfinite(frame.price_lag168.to_numpy(float))
            record = next(
                row
                for row in result["aggregate"]
                if row["split"] == split
                and row["scope"] == "overall"
                and row["coverage"] == "all"
            )
            actual = [
                np.abs(residual).mean(),
                np.sqrt(np.square(residual).mean()),
                np.abs(prediction[paired] - price[paired]).mean(),
            ]
            expected = [record["mae"], record["rmse"], record["paired_mae"]]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        print(
            f"PASS {job.name}: hashes, unique/aligned targets, finite predictions and recomputed scores"
        )
    gpu = checked(config)
    evidence = {
        "passed": True,
        "phase": phase,
        "fingerprint": token,
        "job_count": len(planned),
        "summary_sha256": digest(directory / "summary.json")["sha256"],
        "checks": [
            "artifact hashes",
            "unique identical keys/targets/masks",
            "finite predictions",
            "recomputed MAE/RMSE/paired MAE",
            "matching current GPU preflight",
        ],
    }
    if save:
        write_json(
            ROOT / f"results/controlled_study_{phase}_verification.json", evidence
        )
        write_json(ROOT / "results/controlled_study_gpu_preflight.json", gpu)
        estimate = json.loads((root / "estimate.json").read_text())
        write_json(
            ROOT / "results/controlled_study_estimate.json",
            {"fingerprint": token, **estimate},
        )
    print(f"Verified {len(planned)} jobs; fingerprint {token}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    verify(json.loads(args.config.read_text()), args.phase, args.save)


if __name__ == "__main__":
    main()
