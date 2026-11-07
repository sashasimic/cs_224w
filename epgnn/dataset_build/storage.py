"""Content identities, bounded source partitioning, and atomic local publication."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .contracts import utc

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "entsoe_prices": ("DeliveryHourUTC", ["AreaCode", "DeliveryHourUTC"], "IsValidTarget"),
    "entsoe_load_forecast": ("DeliveryHourUTC", ["AreaCode", "DeliveryHourUTC"], "IsValidValue"),
    "entsoe_physical_flows": ("DeliveryHourUTC", ["OutAreaCode", "InAreaCode", "DeliveryHourUTC"], "IsValidValue"),
    "weather_hourly": ("time_utc", ["zone", "time_utc"], None),
    "jao_hourly": ("ts_utc", ["zone", "ts_utc"], None),
    "jao_border_hourly": ("ts_utc", ["from_zone", "to_zone", "ts_utc"], None),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def digest(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return dict(sha256=sha.hexdigest(), md5=base64.b64encode(md5.digest()).decode(), size=Path(path).stat().st_size)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def config(path):
    p = Path(path).resolve()
    cfg = json.loads(p.read_text())
    # Paths are repository-relative unless root is explicitly supplied (fixtures).
    cfg["root"] = str(Path(cfg.get("root", ROOT)).resolve())
    split_file = resolve(cfg, cfg.get("splits_file", "configs/splits.json"))
    cfg["splits"] = json.loads(split_file.read_text())["splits"]
    spans = sorted((a, b, k) for k, (a, b) in cfg["splits"].items())
    for a, b, k in spans:
        if a > b:
            raise ValueError(f"Invalid split: {k}")
    if any(spans[i][1] >= spans[i + 1][0] for i in range(len(spans) - 1)):
        raise ValueError("Split dates overlap")
    pub = cfg.get("price_publication", {})
    if bool(pub.get("column")) != bool(pub.get("evidence")):
        raise ValueError("Publication admission requires both a column and supporting evidence")
    return cfg


def resolve(cfg, value):
    p = Path(value)
    return p if p.is_absolute() else Path(cfg["root"]) / p


def source_paths(cfg):
    paths = {k: resolve(cfg, v["path"]) for k, v in cfg["sources"].items()}
    jao = resolve(cfg, cfg["jao_directory"]).resolve()
    paths.update({k: jao / f"{k}.parquet" for k in ["jao_hourly", "jao_border_hourly", "jao_cnec_hourly", "cnec_provenance"]})
    for name in cfg.get("jao_metadata", ["build.json", "verification.json", "coverage.csv", "zone_history.csv", "timestamp_mapping.json", "sources.json", "cwe_border_sources.jsonl", "cwe_border_verification.json"]):
        paths[f"jao_metadata/{name}"] = jao / name
    for name, path in cfg.get("references", {}).items():
        paths[f"reference/{name}"] = resolve(cfg, path)
    for name, path in cfg.get("archives", {}).items():
        paths[f"archive/{name}"] = resolve(cfg, path)
    return paths


def identities(cfg):
    out = {}
    for name, path in source_paths(cfg).items():
        out[name] = {"path": str(path.resolve()), "exists": path.exists()}
        if path.exists():
            out[name].update(digest(path))
        if name in cfg["sources"]:
            out[name].update({k: v for k, v in cfg["sources"][name].items() if k not in {"path", "md5"}})
            expected = cfg["sources"][name].get("md5")
            lock = resolve(cfg, cfg["output_directory"]) / "downloads" / f"{name}.json"
            if not expected and lock.exists():
                downloaded = json.loads(lock.read_text())
                if downloaded["version"] == cfg["sources"][name]["version"]:
                    expected = downloaded["files"][f"{name}.parquet"]["md5"]
            out[name]["checksum_verified"] = bool(expected and out[name].get("md5") == expected)
            out[name]["expected_md5"] = expected
    return out


def digest_md5(path):
    h = hashlib.md5()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return base64.b64encode(h.digest()).decode()


def fingerprint(cfg, manifest):
    code = {p.name: digest(p)["sha256"] for p in Path(__file__).parent.glob("*.py")}
    code["zones.py"] = digest(ROOT / "epgnn/zones.py")["sha256"]
    code["dataset_guide"] = digest(ROOT / "docs/forecast_dataset.md")["sha256"]
    versions = {"pandas": pd.__version__, "pyarrow": pa.__version__, "python": sys.version.split()[0]}
    return hashlib.sha256(canonical(dict(config=cfg, sources=manifest, code=code, runtime=versions))).hexdigest()


def partition_sources(cfg, manifest):
    """Single pass per source into calendar-month buckets, then bounded joins.

    Every key includes its timestamp, so duplicates cannot cross month buckets.
    Source row counts and fragment checksums are checked on reuse.
    """
    cache_root = resolve(cfg, cfg["output_directory"]) / "source_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    folders = {}
    for name, (ts, _, _) in SPECS.items():
        info = manifest[name]
        if not info["exists"]:
            continue
        token = hashlib.sha256(canonical([info["sha256"], "month-cache-v2", pa.__version__])).hexdigest()[:20]
        dest = cache_root / f"{name}-{token}"
        marker = dest / "complete.json"
        if marker.exists():
            meta = json.loads(marker.read_text())
            actual = {str(p.relative_to(dest)) for p in dest.glob("*/*.parquet")}
            if actual == set(meta["fragments"]) and all(p.exists() and digest(p)["sha256"] == h for rel, h in meta["fragments"].items() for p in [dest / rel]):
                folders[name] = dest
                continue
            raise ValueError(f"Corrupt source cache: {dest}")
        work = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=cache_root))
        fragments = {}
        try:
            q = pq.ParquetFile(info["path"])
            total = 0
            writers = {}
            schema = pa.schema([pa.field(f.name, pa.timestamp("ns", tz="UTC") if f.name == ts else f.type)
                                for f in q.schema_arrow])
            for i, batch in enumerate(q.iter_batches(batch_size=cfg.get("batch_size", 65536))):
                frame = batch.to_pandas()
                frame[ts] = utc(frame[ts])
                month = frame[ts].dt.strftime("%Y-%m")
                for m in month.unique():
                    sub = frame.loc[month == m]
                    d = work / m
                    d.mkdir(exist_ok=True)
                    p = d / "part.parquet"
                    if m not in writers:
                        writers[m] = pq.ParquetWriter(p, schema, compression="zstd")
                    writers[m].write_table(pa.Table.from_pandas(sub, schema=schema, preserve_index=False))
                    total += len(sub)
            for writer in writers.values():
                writer.close()
            fragments = {str(p.relative_to(work)): digest(p)["sha256"] for p in work.glob("*/*.parquet")}
            if total != q.metadata.num_rows:
                raise ValueError(f"Cache row reconciliation failed: {name}")
            json_write(work / "complete.json", {"rows": total, "fragments": fragments})
            os.rename(work, dest)
            folders[name] = dest
        finally:
            if work.exists():
                shutil.rmtree(work)
    return folders


def read_month(folder, month, columns=None):
    p = Path(folder) / month
    if not p.exists():
        # Return typed empty source schema from any fragment, not all-null columns.
        files = sorted(Path(folder).glob("*/*.parquet"))
        if not files:
            raise ValueError(f"Empty source cache: {folder}")
        schema = pq.read_schema(files[0])
        if columns:
            schema = pa.schema([schema.field(c) for c in columns])
        return pa.Table.from_batches([], schema=schema).to_pandas()
    return ds.dataset(p, format="parquet").to_table(columns=columns).to_pandas()


def read_window(folder, ts_col, start, end):
    """Read historical observations even when they precede the output window."""
    months = pd.period_range(start.tz_localize(None).to_period("M"), end.tz_localize(None).to_period("M"), freq="M")
    frames = [read_month(folder, str(m)) for m in months]
    f = pd.concat(frames, ignore_index=True)
    return f.loc[(f[ts_col] >= start) & (f[ts_col] < end)].reset_index(drop=True)


def download_missing(config_path, cfg):
    """Reuse the existing downloader with exact versions and verify=True.

    Child processes bound credential/network retries. No pin, prune, push or latest.
    Metadata is saved independently of the partners' manifest.
    """
    reports = []
    for name, spec in cfg["sources"].items():
        if resolve(cfg, spec["path"]).exists():
            continue
        if not spec["version"].startswith("v"):
            raise ValueError("An explicit dataset version is required")
        command = [sys.executable, "-m", "epgnn.dataset_build.download", str(config_path), name]
        try:
            p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=cfg.get("download_timeout_seconds", 40))
            reports.append({"source": name, "returncode": p.returncode, "output": (p.stdout + p.stderr)[-3000:]})
        except subprocess.TimeoutExpired:
            reports.append({"source": name, "returncode": "timeout", "output": "GCS credentials/network did not respond within timeout"})
    return reports
