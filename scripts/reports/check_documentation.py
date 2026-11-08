"""Check local documentation links and the consolidated report's source hashes."""

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.identifiers = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "href" in attrs:
            self.links.append(attrs["href"])
        if "id" in attrs:
            self.identifiers.add(attrs["id"])


def main():
    paths = [
        ROOT / "README.md",
        ROOT / "configs/experiments/README.md",
        ROOT / "results/README.md",
    ]
    paths += sorted((ROOT / "docs").glob("*.md"))
    paths += sorted((ROOT / "reports").glob("*/README.md"))
    paths.append(ROOT / "reports/experiments/experiments.html")
    checked, failures = 0, []
    for path in paths:
        text = path.read_text()
        identifiers = None
        if path.suffix == ".html":
            parser = Links()
            parser.feed(text)
            links, identifiers = parser.links, parser.identifiers
        else:
            links = re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text)
        for target in links:
            target = target.strip().split(' "')[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            if parsed.path:
                checked += 1
                if not (path.parent / unquote(parsed.path)).exists():
                    failures.append(f"{path.relative_to(ROOT)} -> {target}")
            elif parsed.fragment and identifiers is not None:
                checked += 1
                if unquote(parsed.fragment) not in identifiers:
                    failures.append(f"{path.relative_to(ROOT)} -> {target}")
    for location in [
        "reports/experiments/sources.json",
        "reports/milestone_v3/sources.json",
    ]:
        path = ROOT / location
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        for group in ["sources", "artifacts", "preserved_milestones"]:
            for name, expected in manifest.get(group, {}).items():
                actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                if actual != expected:
                    failures.append("Changed report input/output: " + name)
    if failures:
        raise ValueError("\n".join(failures))
    print(
        f"PASS {checked} local links across {len(paths)} documents; report sources/output hashes match"
    )


if __name__ == "__main__":
    main()
