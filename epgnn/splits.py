"""The chronological split, read from configs/splits.json. Loaders filter by delivery day (Brussels time).

Tables are kept whole; the `split` column in zone_nodes_hourly / zone_edges_hourly is the same labelling
written at build time, but the config file is the source of truth so the split can change without a rebuild.
"""
import json
from functools import lru_cache

import pandas as pd

from epgnn.paths import ROOT

CONFIG = ROOT / "configs" / "splits.json"


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return json.loads(CONFIG.read_text())


def splits() -> dict[str, tuple[str, str]]:
    """{name: (first_day, last_day)} inclusive, as YYYY-MM-DD."""
    return {k: (v[0], v[1]) for k, v in _cfg()["splits"].items()}


def train_all() -> tuple[str, str]:
    a, b = _cfg()["train_all"]
    return a, b


def split_of(delivery_day: pd.Series) -> pd.Series:
    """Label each delivery day (YYYY-MM-DD string or date) with its split; NaN outside all splits."""
    d = pd.to_datetime(delivery_day)
    out = pd.Series(pd.NA, index=d.index, dtype="string")
    for name, (a, b) in splits().items():
        out = out.mask((d >= a) & (d <= b), name)
    return out
