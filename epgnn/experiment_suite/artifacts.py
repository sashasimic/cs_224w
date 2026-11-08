"""Small helpers for atomic experiment status and result files."""

import json
import os
from pathlib import Path


def write_json(path, value):
    """Publish status/results atomically; readers never see truncated JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)
