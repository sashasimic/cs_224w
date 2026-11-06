"""The chronological split, defined once. Loaders filter by delivery day (Brussels time) from here.

Tables are kept whole; the `split` column in zone_nodes_hourly / zone_edges_hourly is the same labelling
written at build time, but this module is the source of truth so the split can change without a rebuild.

    pretrain  2016-01-01 .. 2022-06-08   ENTSO-E + weather only (CWE era; Core constraints not yet published)
    train     2022-06-09 .. 2024-03-31   Core constraints available
    val       2024-04-01 .. 2025-03-31
    test      2025-04-01 .. 2026-09-30   two summers; prices move to 15-minute MTUs on 2025-10-01
"""
import pandas as pd

SPLITS: dict[str, tuple[str, str]] = {
    "pretrain": ("2016-01-01", "2022-06-08"),
    "train": ("2022-06-09", "2024-03-31"),
    "val": ("2024-04-01", "2025-03-31"),
    "test": ("2025-04-01", "2026-09-30"),
}

# Models that use the whole history as one training set (e.g. with CWE constraints from jao_cnec_hourly)
TRAIN_ALL = ("2016-01-01", "2024-03-31")


def split_of(delivery_day: pd.Series) -> pd.Series:
    """Label each delivery day (YYYY-MM-DD string or date) with its split; NaN outside all splits."""
    d = pd.to_datetime(delivery_day)
    out = pd.Series(pd.NA, index=d.index, dtype="string")
    for name, (a, b) in SPLITS.items():
        out = out.mask((d >= a) & (d <= b), name)
    return out
