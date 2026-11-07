"""Bounded GCS worker, using the team's existing downloader functions."""
import importlib.util
from pathlib import Path
import sys

from .storage import ROOT, config, json_write, resolve


def main():
    cfg = config(sys.argv[1])
    name = sys.argv[2]
    spec = cfg["sources"][name]
    module_spec = importlib.util.spec_from_file_location("team_downloader", ROOT / "scripts/datasets/datasets.py")
    downloader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(downloader)
    b = downloader.bucket(downloader.load_manifest())
    files = downloader.bucket_files(b, name, spec["version"])
    downloader.fetch(b, name, spec["version"], files, verify=True)
    # Preserve the partner manifest. Store independent resolved metadata.
    lock = resolve(cfg, cfg["output_directory"]) / "downloads" / f"{name}.json"
    json_write(lock, {"version": spec["version"], "files": files})
    target = resolve(cfg, spec["path"])
    source = downloader.local_path(spec["version"], f"{name}.parquet")
    if target.resolve() != source.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(source, target)


if __name__ == "__main__":
    main()
