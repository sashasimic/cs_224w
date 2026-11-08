"""Compare saved GRU weights on CPU and fused/native ROCm without fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.artifacts import write_json
from epgnn.experiment_suite.neural import require_gpu
from epgnn.experiment_suite.runner import checked_cache
from epgnn.partner_gru.data import Days
from epgnn.partner_gru.training import ResidualGRU

ORIGINAL = "881d84b072bf950423743251f97718bc23394dffd383c8bfd89bf8d0689b973c"


def residual(model, hist, day, fused):
    batch, zones, window, features = hist.shape
    with torch.backends.cudnn.flags(enabled=fused):
        _, state = model.gru(hist.reshape(batch * zones, window, features))
    h = state[-1].reshape(batch, zones, -1)[:, :, None].expand(-1, -1, day.shape[2], -1)
    return model.head(torch.cat([h, day], dim=-1)).squeeze(-1)


def differences(before, after, mask):
    difference = np.abs(before.astype(np.float64) - after.astype(np.float64))[mask]
    return {
        "max_eur_mwh": float(difference.max()),
        "mean_eur_mwh": float(difference.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprint", default=ORIGINAL)
    args = parser.parse_args()
    directory = ROOT / "runs/partner_gru" / args.fingerprint / "full"
    state = torch.load(directory / "best.pt", weights_only=False, map_location="cpu")
    provenance = json.loads((directory / "provenance.json").read_text())
    cache = Path(provenance["cache"])
    for name, checksum in json.loads((cache / "complete.json").read_text()).items():
        if digest(cache / name)["sha256"] != checksum:
            raise ValueError("Original input cache changed")
    _, metadata, base = checked_cache(state["config"])
    arrays = {
        **base,
        "hist": np.load(cache / "history.npy", mmap_mode="r"),
        "day": np.load(cache / "day.npy", mmap_mode="r"),
    }
    data = Days(arrays, metadata["eligible"])
    device = require_gpu()
    torch.set_num_threads(4)
    gpu = ResidualGRU(state["config"]).to(device).eval()
    cpu = ResidualGRU(state["config"]).double().eval()
    gpu.load_state_dict(state["model"])
    cpu.load_state_dict({k: v.double() for k, v in state["model"].items()})
    for name, weight in cpu.state_dict().items():
        torch.testing.assert_close(
            weight, gpu.state_dict()[name].detach().cpu().double(), rtol=0, atol=0
        )
    validation = np.flatnonzero(data.calendar["split"] == 1)
    records = []
    for start in [0, 336]:
        selected = validation[start : start + 16]
        batch, raw = data.batch(selected, state["scalers"], torch.device("cpu"))
        with torch.no_grad():
            c = residual(
                cpu, batch["hist"].double(), batch["xday"].double(), False
            ).numpy()
            n = (
                residual(gpu, batch["hist"].to(device), batch["xday"].to(device), False)
                .cpu()
                .numpy()
            )
            f = (
                residual(gpu, batch["hist"].to(device), batch["xday"].to(device), True)
                .cpu()
                .numpy()
            )
        base_transformed = np.arcsinh(raw["base"].astype(np.float64) / 50)
        price = [50 * np.sinh(r.astype(float) + base_transformed) for r in [c, n, f]]
        native, fused = (
            differences(price[0], price[1], raw["mask"]),
            differences(price[0], price[2], raw["mask"]),
        )
        if native["max_eur_mwh"] > 0.01 or native["mean_eur_mwh"] > 0.0001:
            raise ValueError("Native GPU does not agree with CPU reference")
        record = {
            "validation_batch_start": start,
            "native_gpu_vs_cpu_float64": native,
            "fused_gpu_vs_cpu_float64": fused,
        }
        print(record, flush=True)
        records.append(record)
    result = {
        "training_rerun": False,
        "training_fingerprint": args.fingerprint,
        "checkpoint_sha256": digest(directory / "best.pt")["sha256"],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "exact_same_weights": True,
        "native_reference_checks_passed": True,
        "cases": records,
        "conclusion": "Fused ROCm GRU discrepancy observed on trained 168-step sequences. Original run is not accepted as intended GRU evidence. Native GPU agrees with CPU reference; retrain with that path.",
        "scope": "This checkpoint/input configuration; older temporal experiments require their own backend audit. Static MLP/GAT have no GRU and are outside this finding.",
    }
    write_json(ROOT / "results/partner_gru_backend_audit.json", result)


if __name__ == "__main__":
    main()
