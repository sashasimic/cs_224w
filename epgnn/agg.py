"""Helpers shared by the scripts/build_agg_*.py scripts: provenance table and doc writing."""
from pathlib import Path

from epgnn.paths import manifest


def sources_table(names: list[str]) -> str:
    """Markdown table of the pinned version and checksum of each source dataset, from the manifest."""
    rows = ["| Source dataset | Version | File | md5 |", "|---|---|---|---|"]
    for n in names:
        ds = manifest()["datasets"][n]
        for f, meta in ds["files"].items():
            if not f.endswith(".md"):
                rows.append(f"| `{n}` | {ds['version']} | `{f}` | `{meta['md5']}` |")
    return "\n".join(rows)


def write_doc(out_dir: Path, name: str, text: str) -> Path:
    p = out_dir / f"{name}.md"
    p.write_text(text)
    return p
