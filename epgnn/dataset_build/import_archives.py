"""Extract only pinned input members from user-provided ZIPs, preserving originals.

python -m epgnn.dataset_build.import_archives --config configs/dataset_construction.json
"""
import argparse
import os
import tempfile
import zipfile

from .storage import config, digest_md5, resolve


def import_archives(config_path):
    cfg = config(config_path)
    for name, spec in cfg["sources"].items():
        target = resolve(cfg, spec["path"])
        if target.exists():
            if digest_md5(target) != spec["md5"]:
                raise ValueError(f"Existing source checksum mismatch: {name}; not overwriting")
            continue
        archive = resolve(cfg, cfg["archives"][spec["archive"]])
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            member = z.getinfo(spec["archive_member"])
            if f"{member.CRC:08x}" != spec["archive_member_crc32"]:
                raise ValueError(f"Archive CRC metadata changed: {name}")
            # Fixed configured target; no archive-provided paths are extracted.
            fd, temp = tempfile.mkstemp(prefix=".extract-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as dst, z.open(member) as src:
                    import shutil
                    shutil.copyfileobj(src, dst, 8 << 20)
                if digest_md5(temp) != spec["md5"]:
                    raise ValueError(f"Archive member checksum mismatch: {name}")
                os.replace(temp, target)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    import_archives(p.parse_args().config)
