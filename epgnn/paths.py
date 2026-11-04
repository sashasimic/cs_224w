"""Resolve dataset file paths from the manifest, so code never hardcodes a version.

    from epgnn.paths import dataset_path
    prices = pd.read_parquet(dataset_path("entsoe_prices", "electricity_prices_hourly.parquet"))
    weather_dir = dataset_path("weather")          # datasets/weather/v1.0
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


def dataset_path(name: str, filename: str | None = None) -> Path:
    """Local path of a dataset directory, or of one file in it, at the manifest's version."""
    ds = manifest()["datasets"][name]
    d = DATASETS / name / ds["version"]
    if filename is None:
        return d
    if filename not in ds["files"]:
        raise KeyError(f"{filename!r} is not a file of dataset {name!r} {ds['version']}; "
                       f"known: {sorted(ds['files'])}")
    p = d / filename
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/datasets.py pull {name}")
    return p
