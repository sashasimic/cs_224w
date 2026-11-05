"""Resolve dataset file paths from the manifest, so code never hardcodes a version.

    from epgnn.paths import dataset_path
    prices = pd.read_parquet(dataset_path("entsoe_prices", "entsoe_prices_hourly.parquet"))
    # -> datasets/entsoe_prices_hourly_v1.1.parquet

Filenames in the manifest are version-free; on disk the version is appended to the stem.
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "configs" / "datasets.json"
DATASETS = ROOT / "datasets"


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def dataset_version(name: str) -> str:
    return manifest()["datasets"][name]["version"]


def dataset_path(name: str, filename: str) -> Path:
    """Local path of one file of a dataset, at the manifest's version."""
    ds = manifest()["datasets"][name]
    if filename not in ds["files"]:
        raise KeyError(f"{filename!r} is not a file of dataset {name!r} {ds['version']}; "
                       f"known: {sorted(ds['files'])}")
    f = Path(filename)
    p = DATASETS / f"{f.stem}_{ds['version']}{f.suffix}"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/datasets.py pull {name}")
    return p
