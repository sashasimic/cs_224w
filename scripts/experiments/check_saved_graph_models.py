"""Check original graph checkpoints against saved predictions after code maintenance.

This reads a trusted local suite result and its artifacts. It requires ROCm,
does not train models, and does not modify any saved results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epgnn.dataset_build.storage import ROOT
from epgnn.experiment_suite import data, neural


def check(result_path):
    summary = json.loads(Path(result_path).read_text())
    directory = ROOT / "runs/vibing_suite" / summary["fingerprint"] / "full"
    provenance = json.loads((directory / "provenance.json").read_text())
    metadata, arrays = data.load(provenance["cache"])
    zones = metadata["eligible"]
    device = neural.require_gpu()

    for name in ["real_gat", "shuffled_gat"]:
        experiment = next(
            item for item in summary["experiments"] if item["experiment"] == name
        )
        features = [
            metadata["features"].index(column) for column in experiment["features"]
        ]
        checkpoint = torch.load(
            directory / name / "best.pt", map_location=device, weights_only=False
        )
        model = neural.PriceGAT(len(features) * 2).to(device)
        model.load_state_dict(checkpoint["model"])
        preprocessing = neural.Preprocessor.restore(checkpoint["preprocessor"])
        edge_key = "shuffled_edges" if name == "shuffled_gat" else "edges"
        edges = torch.tensor(np.asarray(arrays[edge_key]).copy(), device=device)
        batch_size = checkpoint["batch_size"]
        indices = np.flatnonzero(arrays["split"] == 2)[:batch_size]
        times = pd.to_datetime(np.asarray(arrays["times"][indices]), utc=True)

        predictions = neural.infer(
            model, arrays, indices, features, preprocessing, edges, batch_size, device
        )[:, zones]
        saved = pd.read_parquet(
            directory / name / "predictions_test.parquet",
            filters=[("ts_utc", ">=", times[0]), ("ts_utc", "<=", times[-1])],
        )
        expected = (
            saved.pivot(index="ts_utc", columns="zone", values="prediction")
            .reindex(index=times, columns=[metadata["zones"][index] for index in zones])
            .to_numpy()
        )
        np.testing.assert_allclose(predictions, expected, rtol=1e-5, atol=1e-4)
        difference = float(np.abs(predictions - expected).max())
        print(
            f"PASS {name}: historical checkpoint and predictions match; max difference {difference:.8f} EUR/MWh"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", type=Path, default=ROOT / "results/vibing_suite_full.json"
    )
    args = parser.parse_args()
    check(args.result)


if __name__ == "__main__":
    main()
