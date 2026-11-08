# Setup

Run commands from the repository root. Existing local `.venv`, datasets, staging
and model artifacts can be reused; installing the package does not fetch data.

## Python environment

For a new general model/inspection environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[model,inspection]'
```

For this Linux AMD machine, use the pinned ROCm environment instead:

```bash
# Create .venv only if it does not exist.
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-rocm.txt
.venv/bin/python scripts/check_rocm.py
```

When converting an existing NVIDIA environment, remove `triton` before installing
the ROCm requirements because its files overlap with `triton-rocm`:

```bash
uv pip uninstall --python .venv/bin/python triton
uv pip install --python .venv/bin/python -r requirements-rocm.txt
```

The local pinned build is PyTorch 2.14.1+rocm7.2 with its matching Triton dependency.
ROCm uses PyTorch's `cuda` API; `torch.version.hip` distinguishes the AMD build.
The check runs GPU matrix multiplication, GRU/GATv2, masked loss, backward and
Adam. The suite's stronger `gpu-smoke` command also checks checkpoint resume.
Run these in a host terminal with access to `/dev/kfd` and `/dev/dri/renderD*`.
XGBoost uses CPU; the suite's graph models require ROCm and fail rather than
silently falling back to CPU.

## Data

This shared checkout already contains the audited generation under
`staging/forecast_dataset_v1.0/current`, original sources under
`staging/forecast_sources/`, and the original comparison tables under `datasets/`.
Construction paths and checksums are recorded in
[`configs/dataset_construction.json`](../configs/dataset_construction.json).
See [the construction guide](forecast_dataset.md) for audit/build/validate.

The original downloader uses GCS credentials. With access configured, these
commands inspect or retrieve explicitly selected releases:

```bash
scripts/datasets/datasets.py status
scripts/datasets/datasets.py pull entsoe_prices@v1.2
scripts/datasets/datasets.py pull entsoe_load_forecast@v1.2
scripts/datasets/datasets.py pull entsoe_physical_flows@v1.2
```

Keep the construction source pins separate from the partner model-table manifest
in `configs/datasets.json`. The supplied weather archive's requested v1.1 release
identity remains unverified; current data are reanalysis, not forecast vintages.
This cleanup does not download, publish, repin or remove datasets.

## Run and test

Follow [the suite guide](vibing_suite.md) for GPU/France gates and a full tmux run.
The [results report](vibing_suite_results.md) describes the completed evaluation.

```bash
.venv/bin/python -m epgnn.experiment_suite status --saved
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests/dataset_build -p 'test_*.py'
```
