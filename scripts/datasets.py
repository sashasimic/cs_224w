#!/usr/bin/env python3
"""Pull and push versioned datasets from the team GCS bucket.

A dataset version is its data file(s) plus its doc, <dataset>.md, stored and pulled together.

Layout in the bucket:   gs://<bucket>/<dataset>/<version>/<file>      data files and <dataset>.md
Layout on disk:         datasets/<stem>_<version>.<ext>              e.g. weather_hourly_v1.0.parquet, weather_v1.0.md
Staging for push:       staging/<dataset>/<file>                     data files and <dataset>.md, unversioned names
Manifest (tracked):     configs/datasets.json                        which version of each dataset the code expects

Code resolves paths with `epgnn.paths.dataset_path(name, filename)`, never by hand.

Usage:
    scripts/datasets.py pull                     # every dataset at the version pinned in the manifest
    scripts/datasets.py pull weather             # one dataset, manifest version
    scripts/datasets.py pull weather@v1.0        # one dataset, a specific version (manifest untouched)
    scripts/datasets.py pull --latest            # newest version in the bucket for every dataset
    scripts/datasets.py pull --latest --pin      # ...and write those versions into the manifest
    scripts/datasets.py pull --prune             # also delete local copies of other versions
    scripts/datasets.py pull --verify            # md5-check local files instead of trusting size
    scripts/datasets.py push weather v1.1        # upload everything in staging/weather/ as weather/v1.1
    scripts/datasets.py clean                    # delete local files not at a manifest-pinned version
    scripts/datasets.py clean -n                 # ...just show what would be deleted
    scripts/datasets.py status                   # local vs manifest vs bucket

A file already on disk with the expected size is not downloaded again.

`push` refuses unless staging/<dataset>/<dataset>.md exists and has a version-history row
for the new version (the human-written "what changed"). It rewrites the mechanical header
lines (version, bucket path, local filenames), uploads, updates the manifest, and copies the
staged files into datasets/ under their versioned names.

Auth: `gcloud auth application-default login` once per machine.
"""
import argparse
import base64
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

from google.cloud import storage

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "configs" / "datasets.json"
LOCAL = ROOT / "datasets"
STAGING = ROOT / "staging"


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


DOC_TEMPLATE = """# <title>

| | |
|---|---|
| Dataset id | `{name}` |
| Version | **{version}** |
| Bucket | `gs://<bucket>/{name}/{version}/` |
| Local file(s) | `datasets/<file>_{version}.parquet`, `datasets/{name}_{version}.md` |
| Source | <where the data comes from, period covered> |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("{name}", "<file>.parquet"))
```

Version history:

| Version | What |
|---|---|
| {version} | <what changed> |

---

<schema, quality findings, caveats>
"""


def staged_files(name, version):
    """Files in staging/<name>/: the doc must be present; returns {filename: path}."""
    d = STAGING / name
    files = {p.name: p for p in sorted(d.iterdir()) if p.is_file() and not p.name.startswith(".")} if d.exists() else {}
    doc_name = f"{name}.md"
    if doc_name not in files:
        sys.exit(f"staging/{name}/{doc_name} is missing. Put the data file(s) and the doc there; doc template:\n\n"
                 + DOC_TEMPLATE.format(name=name, version=version))
    if len(files) == 1:
        sys.exit(f"staging/{name}/ has only the doc; add the data file(s).")
    return files


def check_doc(doc, version):
    """Refuse the push unless the doc has a history row for this version."""
    if not re.search(rf"^\|\s*{re.escape(version)}\s*\|", doc.read_text(), re.M):
        sys.exit(f"{doc.relative_to(ROOT)} has no version-history row for {version}.\n"
                 f"Add a line like   | {version} | <what changed> |   under 'Version history', then push again.")


def update_doc_header(doc, name, version, files, bucket_name):
    """Rewrite the mechanical header lines so they match what is being pushed."""
    text = doc.read_text()
    local = ", ".join(f"`datasets/{Path(f).stem}_{version}{Path(f).suffix}`" for f in files)
    subs = {
        r"^\| (?:Pinned v|V)ersion \|.*$": f"| Version | **{version}** |",
        r"^\| Bucket \|.*$":              f"| Bucket | `gs://{bucket_name}/{name}/{version}/` |",
        r"^\| Local file\(s\) \|.*$":     f"| Local file(s) | {local} |",
    }
    for pattern, line in subs.items():
        text, n = re.subn(pattern, line, text, count=1, flags=re.M)
        if n == 0:
            print(f"  warn    {doc.name}: header line not found for {pattern!r}; add it by hand")
    doc.write_text(text)


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
    name, version = args.dataset, args.version
    m = load_manifest()
    staged = staged_files(name, version)
    doc = staged[f"{name}.md"]
    check_doc(doc, version)
    b = bucket(m)
    if not args.force:
        for filename in staged:
            if b.blob(blob_path(name, version, filename)).exists():
                sys.exit(f"already in bucket: {name}/{version}/{filename} (use --force to overwrite)")
    update_doc_header(doc, name, version, list(staged), m["bucket"])

    files = {}
    for filename, p in staged.items():
        blob = b.blob(blob_path(name, version, filename))
        print(f"  upload  {blob.name} ...", end="", flush=True)
        blob.upload_from_filename(str(p))
        blob.reload()
        files[filename] = {"md5": blob.md5_hash, "size": blob.size}
        print(" done")
    m["datasets"][name] = {"version": version, "files": files}
    save_manifest(m)

    LOCAL.mkdir(exist_ok=True)
    for filename, p in staged.items():
        dest = local_path(version, filename)
        shutil.copyfile(p, dest)
        print(f"  local   {dest.relative_to(ROOT)}")
    print(f"manifest updated: {name} -> {version}")
    doc_local = local_path(version, f"{name}.md").relative_to(ROOT)
    print(f"next: git add configs/datasets.json {doc_local} && git commit    (staging/{name}/ can be deleted)")


def pinned_local_names(m):
    return {local_path(ds["version"], f).name for ds in m["datasets"].values() for f in ds["files"]}


def cmd_clean(args):
    """Delete files in datasets/ that are not at the version the manifest pins."""
    m = load_manifest()
    keep = pinned_local_names(m)
    extra = [p for p in sorted(LOCAL.glob("*_v*.*")) if LOCAL.exists() and p.is_file() and p.name not in keep]
    if not extra:
        print("nothing to clean: datasets/ holds only manifest-pinned versions")
        return
    for p in extra:
        print(f"  {'would delete' if args.dry_run else 'deleted'}  {p.name}")
        if not args.dry_run:
            p.unlink()
    if args.dry_run:
        print("run without -n to delete")
    elif any(p.suffix == ".md" for p in extra):
        print("note: deleted .md docs are tracked; `git status` will show them as deletions")


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
    current = pinned_local_names(m)
    for p in sorted(LOCAL.glob("*_v*.*")) if LOCAL.exists() else []:
        tag = "<- manifest" if p.name in current else "(other version; `clean` removes)"
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

    p = sub.add_parser("push", help="upload staging/<dataset>/ as a new dataset version")
    p.add_argument("dataset"); p.add_argument("version")
    p.add_argument("--force", action="store_true", help="overwrite a version that already exists in the bucket")
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("clean", help="delete local files not at a manifest-pinned version")
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would be deleted")
    p.set_defaults(fn=cmd_clean)

    p = sub.add_parser("status", help="local vs manifest vs bucket")
    p.set_defaults(fn=lambda _: cmd_status())

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
