#!/usr/bin/env python3
"""Pull and push versioned datasets from the team GCS bucket.

Layout in the bucket:   gs://<bucket>/<dataset>/<version>/<file>
Layout on disk:         datasets/<dataset>/<version>/<file>   (mirrors the bucket)
Manifest (tracked):     configs/datasets.json                 (which version of each dataset the code expects)

Code resolves paths with `epgnn.paths.dataset_path(name, filename)`, never by hand.

Usage:
    scripts/datasets.py pull                       # download everything in the manifest
    scripts/datasets.py pull weather               # one dataset
    scripts/datasets.py push weather v1.1 datasets/weather_hourly.parquet datasets/weather_daily.parquet
    scripts/datasets.py status                     # what's local vs manifest vs bucket

Auth: `gcloud auth application-default login` once per machine.
"""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "configs" / "datasets.json"
LOCAL = ROOT / "datasets"


def load_manifest():
    return json.loads(MANIFEST.read_text())


def save_manifest(m):
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")


def md5_b64(path: Path) -> str:
    """MD5 in the base64 form GCS stores in blob.md5_hash."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def bucket(m):
    return storage.Client(project=m["project"]).bucket(m["bucket"])


def blob_path(name, version, filename):
    return f"{name}/{version}/{filename}"


def local_path(name, version, filename):
    return LOCAL / name / version / filename


def cmd_pull(args):
    m = load_manifest()
    b = bucket(m)
    names = [args.dataset] if args.dataset else list(m["datasets"])
    for name in names:
        ds = m["datasets"][name]
        for filename, md5 in ds["files"].items():
            dest = local_path(name, ds["version"], filename)
            if dest.exists() and md5_b64(dest) == md5:
                print(f"  ok      {name}/{ds['version']}/{filename}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob = b.blob(blob_path(name, ds["version"], filename))
            print(f"  fetch   {name}/{ds['version']}/{filename} ...", end="", flush=True)
            blob.download_to_filename(dest)
            got = md5_b64(dest)
            if got != md5:
                dest.unlink()
                sys.exit(f"\nchecksum mismatch for {filename}: manifest {md5}, downloaded {got}")
            print(" done")


def cmd_push(args):
    m = load_manifest()
    b = bucket(m)
    files = {}
    for p in args.files:
        p = Path(p)
        if not p.exists():
            sys.exit(f"missing: {p}")
        blob = b.blob(blob_path(args.dataset, args.version, p.name))
        if blob.exists() and not args.force:
            sys.exit(f"already in bucket: {blob.name} (use --force to overwrite)")
        print(f"  upload  {blob.name} ...", end="", flush=True)
        blob.upload_from_filename(str(p))
        blob.reload()
        files[p.name] = blob.md5_hash
        print(" done")
    m["datasets"][args.dataset] = {"version": args.version, "files": files}
    save_manifest(m)
    print(f"manifest updated: {args.dataset} -> {args.version}")


def cmd_status(args):
    m = load_manifest()
    b = bucket(m)
    for name, ds in m["datasets"].items():
        for filename, md5 in ds["files"].items():
            dest = local_path(name, ds["version"], filename)
            local = "missing" if not dest.exists() else ("ok" if md5_b64(dest) == md5 else "STALE")
            remote = "ok" if b.blob(blob_path(name, ds["version"], filename)).exists() else "MISSING"
            print(f"  {name:28s} {ds['version']:6s} {filename:50s} local={local:8s} bucket={remote}")
    print("\nversions on disk (datasets/):")
    for p in sorted(LOCAL.glob("*/v*")):
        if p.is_dir():
            cur = m["datasets"].get(p.parent.name, {}).get("version")
            tag = "  <- manifest" if cur == p.name else "  (other version)"
            print(f"  {p.parent.name}/{p.name}{tag}")
    print("\nversions in bucket:")
    seen = set()
    for blob in b.list_blobs():
        key = "/".join(blob.name.split("/")[:2])
        if key not in seen:
            seen.add(key)
            print(f"  {key}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull"); p.add_argument("dataset", nargs="?"); p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("push"); p.add_argument("dataset"); p.add_argument("version"); p.add_argument("files", nargs="+")
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_push)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
