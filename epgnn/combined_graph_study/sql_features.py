"""DuckDB aggregation over the same admitted observations supplied to GraphSAGE."""

import os
import tempfile
from contextlib import closing

import duckdb
import numpy as np
import pyarrow as pa

from .inputs import FIELDS

PRICE_FIELDS = ("price",)
PRICE_SLICES = (slice(0, 96), slice(96, 144), slice(144, 168))
OTHER_SLICES = (slice(0, 24), slice(24, 48))


def observations_to_arrow(values, mask, fields, slices):
    """One list per source field, entity and age bucket, in timestamp order."""
    columns = {}
    for bucket, window in enumerate(slices):
        valid = mask[:, window].transpose(0, 2, 1)
        width = valid.shape[-1]
        columns[f"presence_{bucket}"] = pa.array(valid.sum(axis=-1).ravel())
        for f, name in enumerate(fields):
            flat = values[:, window, :, f].transpose(0, 2, 1).ravel()
            array = pa.array(flat, mask=~np.isfinite(flat) | ~valid.ravel())
            columns[f"f_{name}_{bucket}"] = pa.FixedSizeListArray.from_arrays(
                array, width
            )
    return pa.table(columns)


def summaries(db, values, mask, fields, slices, extra):
    table = observations_to_arrow(values, mask, fields, slices)
    for name, value in extra.items():
        table = table.append_column(name, pa.array(value))
    db.register("observations", table)
    fields = [name for name in table.column_names if name.startswith("f_")]
    functions = [
        f"list_{fun}({name}) AS {name}_{fun}"
        for name in fields
        for fun in ("count", "avg", "min", "max")
    ]
    functions.extend(
        f"list_last(list_filter({name}, x -> x IS NOT NULL)) AS {name}_latest"
        for name in fields
    )
    keys = list(extra) + [
        name for name in table.column_names if name.startswith("presence_")
    ]
    result = db.execute(
        f"SELECT {', '.join(keys + functions)} FROM observations ORDER BY {', '.join(extra)}"
    ).to_arrow_table()
    db.unregister("observations")
    return result


def numeric(table, size):
    fields = [
        column.to_numpy(zero_copy_only=False).astype(np.float32)
        for column in table.columns
    ]
    return np.column_stack(fields).reshape(size[0], size[1], -1)


def build_features(inputs, rich):
    result = []
    zones = len(inputs.zones)
    with closing(duckdb.connect(config={"threads": 4})) as db:
        for start in range(0, len(inputs.origins), 32):
            days = np.arange(start, min(start + 32, len(inputs.origins)))
            loc = {
                "day": np.repeat(days, zones),
                "zone": np.tile(np.arange(zones), len(days)),
            }
            size = (len(days), zones)
            price = summaries(
                db, *inputs.history("price", days), PRICE_FIELDS, PRICE_SLICES, loc
            )
            local = [numeric(price.drop(["day", "zone"]), size)]
            if rich:
                for name in ("weather", "load", "jao"):
                    stats = summaries(
                        db, *inputs.history(name, days), FIELDS[name], OTHER_SLICES, loc
                    )
                    local.append(numeric(stats.drop(["day", "zone"]), size))
                source, dest = inputs.edges.T
                flow_keys = {
                    "day": np.repeat(days, len(source)),
                    "src": np.tile(source, len(days)),
                    "dst": np.tile(dest, len(days)),
                }
                flow = summaries(
                    db,
                    *inputs.history("flow", days),
                    FIELDS["flow"],
                    OTHER_SLICES,
                    flow_keys,
                )
                db.register("price_stats", price)
                db.register("flow_stats", flow)
                flow_names = [f.name for f in flow.schema if f.name not in flow_keys]
                price_names = [
                    f"f_price_{bucket}_{agg}"
                    for bucket in range(3)
                    for agg in ("avg", "latest")
                ]
                for own, neighbor in (("src", "dst"), ("dst", "src")):
                    aggregates = [f"avg(b.{name}) AS {name}" for name in flow_names]
                    aggregates.extend(
                        f"avg(p.{name}) AS {name}" for name in price_names
                    )
                    result_table = db.execute(f"""
                        SELECT {", ".join(aggregates)}, count(b.day) AS border_count
                        FROM price_stats seeds LEFT JOIN flow_stats b
                         ON b.day = seeds.day AND b.{own} = seeds.zone
                         AND (b.presence_0 + b.presence_1) > 0
                        LEFT JOIN price_stats p ON p.day = b.day AND p.zone = b.{neighbor}
                        GROUP BY seeds.day, seeds.zone ORDER BY seeds.day, seeds.zone
                    """).to_arrow_table()
                    local.append(numeric(result_table, size))
                db.unregister("price_stats")
                db.unregister("flow_stats")
            result.append(np.concatenate(local, axis=-1))
    return np.concatenate(result)


def features(inputs, rich, cache):
    path = cache / ("sql_full.npy" if rich else "sql_price.npy")
    if not path.exists():
        print(f"Building {'full' if rich else 'price'} SQL aggregates", flush=True)
        aggregate = build_features(inputs, rich)
        # SQL caches live in the run directory, separate from immutable source panels.
        fd, tmp = tempfile.mkstemp(suffix=".npy", dir=path.parent)
        os.close(fd)
        try:
            np.save(tmp, aggregate)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return np.load(path, mmap_mode="r")
