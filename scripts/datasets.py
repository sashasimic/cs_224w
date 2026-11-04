#!/usr/bin/env python3
"""Pull and push versioned datasets from the team GCS bucket.

Layout in the bucket:   gs://<bucket>/<dataset>/<version>/<file>
Layout on disk:         datasets/<stem>_<version>.<ext>       (flat, e.g. weather_hourly_v1.0.parquet)
Manifest (tracked):     configs/datasets.json                 (which version of each dataset the code expects)

Code resolves paths with `epgnn.paths.dataset_path(name, filename)`, never by hand.

Usage:
    scripts/datasets.py pull                     # every dataset at the version pinned in the manifest
    scripts/datasets.py pull weather             # one dataset, manifest version
    scripts/datasets.py pull weather@v1.0        # one dataset, a specific version (manifest untouched)
    scripts/datasets.py pull --latest            # newest version in the bucket for every dataset
    scripts/datasets.py pull --latest --pin      # ...and write those versions into the manifest
    scripts/datasets.py pull --prune             # also delete local copies of other versions
    scripts/datasets.py pull --verify            # md5-check local files instead of trusting size
    scripts/datasets.py push weather v1.1 path/to/weather_hourly.parquet path/to/weather_daily.parquet
    scripts/datasets.py status                   # local vs manifest vs bucket

A file already on disk with the expected size is not downloaded again.

Auth: `gcloud auth application-default login` once per machine.
"""
import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "configs" / "datasets.json"
LOCAL = ROOT / "datasets"


# ---------------------------------------------------------------- helpers

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


def local_path(version, filename):
    p = Path(filename)
    return LOCAL / f"{p.stem}_{version}{p.suffix}"


def version_key(v: str):
    """'v1.10' sorts after 'v1.9'."""
    return tuple(int(x) for x in re.findall(r"\d+", v))


def bucket_versions(b, name):
    """Versions of a dataset present in the bucket, oldest first."""
    seen = set()
    for blob in b.list_blobs(prefix=f"{name}/"):
        parts = blob.name.split("/")
        if len(parts) == 3:
            seen.add(parts[1])
    return sorted(seen, key=version_key)


def bucket_files(b, name, version):
    """{filename: {"md5":..., "size":...}} for one dataset version, read from blob metadata."""
    files = {}
    for blob in b.list_blobs(prefix=f"{name}/{version}/"):
        files[blob.name.split("/")[-1]] = {"md5": blob.md5_hash, "size": blob.size}
    if not files:
        sys.exit(f"not in bucket: {name}/{version}")
    return files


def fetch(b, name, version, files, verify):
    """Download the files of one dataset version that are missing or wrong on disk."""
    LOCAL.mkdir(exist_ok=True)
    for filename, meta in files.items():
        dest = local_path(version, filename)
        if dest.exists() and dest.stat().st_size == meta["size"] and (not verify or md5_b64(dest) == meta["md5"]):
            print(f"  ok      {name}/{version}/{filename}")
            continue
        print(f"  fetch   {name}/{version}/{filename} ...", end="", flush=True)
        b.blob(blob_path(name, version, filename)).download_to_filename(dest)
        got = md5_b64(dest)
        if got != meta["md5"]:
            dest.unlink()
            sys.exit(f"\nchecksum mismatch for {filename}: bucket {meta['md5']}, downloaded {got}")
        print(" done")


def prune(name, keep_version, files):
    """Delete local copies of this dataset's files at versions other than keep_version."""
    for filename in files:
        p = Path(filename)
        for other in LOCAL.glob(f"{p.stem}_v*{p.suffix}"):
            if other != local_path(keep_version, filename):
                other.unlink()
                print(f"  pruned  {other.name}")


# ---------------------------------------------------------------- commands

def cmd_pull(args):
    m = load_manifest()
    b = bucket(m)

    # which datasets, and which version of each
    if args.dataset:
        name, _, version = args.dataset.partition("@")
        if name not in m["datasets"]:
            sys.exit(f"unknown dataset {name!r}; known: {sorted(m['datasets'])}")
        targets = {name: version or None}
    else:
        targets = {name: None for name in m["datasets"]}

    for name, version in targets.items():
        pinned = m["datasets"][name]["version"]
        if args.latest:
            version = bucket_versions(b, name)[-1]
        elif version is None:
            version = pinned

        if version == pinned and not args.latest:
            files = m["datasets"][name]["files"]
        else:
            files = bucket_files(b, name, version)

        fetch(b, name, version, files, args.verify)
        if args.prune:
            prune(name, version, files)

        if version != pinned:
            if args.pin:
                m["datasets"][name] = {"version": version, "files": files}
                print(f"  pinned  {name} -> {version} (manifest updated)")
            else:
                print(f"  note    {name} {version} pulled; manifest still pins {pinned} (use --pin to switch)")

    if args.pin:
        save_manifest(m)


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
        files[p.name] = {"md5": blob.md5_hash, "size": blob.size}
        print(" done")
    m["datasets"][args.dataset] = {"version": args.version, "files": files}
    save_manifest(m)
    print(f"manifest updated: {args.dataset} -> {args.version}")


def cmd_status():
    m = load_manifest()
    b = bucket(m)
    print("manifest:")
    for name, ds in m["datasets"].items():
        for filename, meta in ds["files"].items():
            dest = local_path(ds["version"], filename)
            if not dest.exists():
                local = "missing"
            elif dest.stat().st_size != meta["size"]:
                local = "WRONG SIZE"
            else:
                local = "ok"
            remote = "ok" if b.blob(blob_path(name, ds["version"], filename)).exists() else "MISSING"
            print(f"  {name:24s} {ds['version']:6s} {filename:46s} local={local:11s} bucket={remote}")

    print("\non disk (datasets/):")
    current = {local_path(ds["version"], f).name for ds in m["datasets"].values() for f in ds["files"]}
    for p in sorted(LOCAL.glob("*.parquet")) if LOCAL.exists() else []:
        tag = "<- manifest" if p.name in current else "(other version; pull --prune removes)"
        print(f"  {p.name:52s} {tag}")

    print("\nin bucket:")
    for name in m["datasets"]:
        versions = bucket_versions(b, name)
        pinned = m["datasets"][name]["version"]
        tags = [v + (" *" if v == pinned else "") for v in versions]
        print(f"  {name:24s} {', '.join(tags)}    (* = pinned)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="download datasets")
    p.add_argument("dataset", nargs="?", help="dataset name, optionally name@version")
    p.add_argument("--latest", action="store_true", help="newest version in the bucket instead of the manifest's")
    p.add_argument("--pin", action="store_true", help="write the pulled version into the manifest")
    p.add_argument("--prune", action="store_true", help="delete local copies of other versions")
    p.add_argument("--verify", action="store_true", help="md5-check files already on disk")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("push", help="upload a new dataset version")
    p.add_argument("dataset"); p.add_argument("version"); p.add_argument("files", nargs="+")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("status", help="local vs manifest vs bucket")
    p.set_defaults(fn=lambda _: cmd_status())

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
