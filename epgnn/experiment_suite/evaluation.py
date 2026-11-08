"""Aligned prediction tables and weighted zone/country error summaries."""

import numpy as np
import pandas as pd

from epgnn.boosting import score
from . import data


def evaluate(predictions, arrays, meta, zones, experiment, split, directory):
    indices = np.flatnonzero(arrays["split"] == {"val": 1, "test": 2}[split])
    if (
        predictions.shape != (len(indices), len(zones))
        or not np.isfinite(predictions).all()
    ):
        raise ValueError(
            "Evaluation requires finite predictions for every selected zone-hour"
        )
    times = pd.to_datetime(np.asarray(arrays["times"][indices]), utc=True)
    price_column = data.FEATURES.index("price_lag168")
    jao_column = data.FEATURES.index("jao_domain_present_lag48")
    records, keyed = [], []
    for k, j in enumerate(zones):
        y = np.asarray(arrays["y"][indices, j])
        p = np.asarray(arrays["x"][indices, j, price_column])
        domain = np.asarray(arrays["x"][indices, j, jao_column])
        f = pd.DataFrame(
            {
                "zone": meta["zones"][j],
                "ts_utc": times,
                "price": y,
                "target_valid": np.isfinite(y),
                "prediction": predictions[:, k],
                "price_lag168": p,
                "jao_domain_present_lag48": domain,
            }
        )
        result = score(f, predictions[:, k], p)
        for coverage, selection in [
            ("all", np.ones(len(f), bool)),
            ("jao_published", domain == 1),
        ]:
            if coverage != "all":
                result = score(
                    f.loc[selection], predictions[selection, k], p[selection]
                )
            common = result["weekly_persistence_conditional_common_hours"]
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "zone": meta["zones"][j],
                    "coverage": coverage,
                    "hours": result["model"]["n_hours"],
                    "mae": result["model"]["mae"],
                    "rmse": result["model"]["rmse"],
                    "paired_hours": common["model"]["n_hours"],
                    "paired_mae": common["model"]["mae"],
                    "persistence_mae": common["persistence"]["mae"],
                    "persistence_rmse": common["persistence"]["rmse"],
                    "relative_mae": common["relative_mae"],
                }
            )
        keyed.append(f)
    frame = pd.concat(keyed, ignore_index=True)
    if frame.duplicated(["zone", "ts_utc"]).any():
        raise ValueError("Duplicate prediction keys")
    frame.to_parquet(directory / f"predictions_{split}.parquet", index=False)
    return records


def grouped_scores(rows):
    frame = pd.DataFrame(rows)
    # DE_LU/IE_SEM stay combined markets; regional targets are pooled, not averaged prices.
    frame["country"] = frame.zone.map(
        lambda z: z.split("_")[0] if z.startswith(("IT_", "NO_", "SE_", "DK_")) else z
    )
    result = []
    for keys in [
        ["experiment", "split", "coverage"],
        ["experiment", "split", "coverage", "country"],
    ]:
        for group, f in frame.groupby(keys):
            n, paired = int(f.hours.sum()), int(f.paired_hours.sum())

            def weighted(column, weights):
                ok = f[column].notna() & (f[weights] > 0)
                return (
                    float(np.average(f.loc[ok, column], weights=f.loc[ok, weights]))
                    if ok.any()
                    else None
                )

            # Pool squared errors rather than averaging RMSEs.
            squared = f.rmse.fillna(0).to_numpy(float) ** 2 * f.hours
            mae, pm = weighted("paired_mae", "paired_hours"), weighted(
                "persistence_mae", "paired_hours"
            )
            result.append(
                {
                    **dict(zip(keys, group)),
                    "scope": "overall" if len(keys) == 3 else "country",
                    "hours": n,
                    "paired_hours": paired,
                    "mae": weighted("mae", "hours"),
                    "rmse": float(np.sqrt(squared.sum() / n)) if n else None,
                    "paired_mae": mae,
                    "persistence_mae": pm,
                    "relative_mae": mae / pm if pm else None,
                }
            )
    return result
