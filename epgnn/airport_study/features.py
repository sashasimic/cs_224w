"""Explicit observation joins, conservative availability, and no filling."""

import numpy as np
import pandas as pd

from epgnn.dataset_build.contracts import utc

TEMPERATURE_FEATURES = [
    "airport_temp_c",
    "airport_temp_age_h",
    "airport_temp_mean24_c",
    "airport_temp_hours24",
]
HOUR = pd.Timedelta(hours=1).value


def country_proxy(zone):
    if zone == "DE_LU":
        return "DE", "combined_market_proxy"
    if zone == "IE_SEM":
        return "IE", "combined_market_proxy"
    if zone == "DE_AT_LU" or zone.startswith("ENTSOE:"):
        return None, "unresolved_geography"
    if zone.startswith(("IT_", "NO_", "SE_", "DK_")):
        return zone.split("_")[0], "national_airport_for_regional_zone"
    return zone, "national_airport_proxy"


def normalize(frame):
    """Collapse identical records; quarantine conflicting/corrected timestamps.

    Optional received_utc supports availability fixtures and future richer feeds.
    IEM's historical CSV itself does not supply first-receipt timestamps.
    """
    frame = frame.copy()
    frame["observed_utc"] = utc(frame["valid"]).to_numpy()
    frame["temp_c"] = (pd.to_numeric(frame.tmpf, errors="coerce") - 32) * 5 / 9
    corrected = frame.metar.fillna("").str.contains(r"\bCOR\b", regex=True)
    implausible = ~frame.temp_c.between(-90, 60) & frame.temp_c.notna()
    frame.loc[corrected | implausible, "temp_c"] = np.nan
    if "received_utc" not in frame:
        frame["received_utc"] = pd.NaT
    frame["received_utc"] = pd.to_datetime(frame.received_utc, utc=True).dt.as_unit(
        "ns"
    )
    if frame.station.nunique() > 1:
        raise ValueError("Normalize one station at a time")
    groups = frame.groupby("observed_utc", sort=True)
    conflict = groups.temp_c.nunique(dropna=False).gt(1)
    result = groups.agg(
        temp_c=("temp_c", "first"), received_utc=("received_utc", "max")
    )
    ambiguous = corrected.groupby(frame.observed_utc).any() | conflict
    result.loc[ambiguous, "temp_c"] = np.nan
    result = result.reset_index()
    return result, {
        "input_rows": len(frame),
        "unique_timestamps": len(result),
        "duplicate_rows": len(frame) - len(result),
        "conflicting_timestamps": int(conflict.sum()),
        "corrected_reports": int(corrected.sum()),
        "implausible_temperatures": int(implausible.sum()),
        "missing_temperatures": int(result.temp_c.isna().sum()),
        "receipt_times_present": int(result.received_utc.notna().sum()),
    }


def at_origins(observations, origins, config):
    """Latest report and hourly-sampled history available at each origin."""
    origin = pd.DatetimeIndex(utc(origins))
    observed = pd.DatetimeIndex(utc(observations.observed_utc)).asi8
    if len(observed) > 1 and np.any(np.diff(observed) <= 0):
        raise ValueError("Observations must have unique sorted timestamps")
    temperature = observations.temp_c.to_numpy(float)
    receipt = (
        pd.DatetimeIndex(pd.to_datetime(observations.received_utc, utc=True))
        .as_unit("ns")
        .asi8
    )
    has_receipt = observations.received_utc.notna().to_numpy()
    output = np.full((len(origin), 4), np.nan, dtype=np.float32)
    window = config["mean_window_hours"] * HOUR
    for i, now in enumerate(origin.asi8):
        cutoff = now - config["reporting_buffer_hours"] * HOUR
        lower = np.searchsorted(observed, cutoff - window, side="right")
        upper = np.searchsorted(observed, cutoff, side="right")
        positions = np.arange(lower, upper)
        positions = positions[~has_receipt[positions] | (receipt[positions] <= now)]
        if len(positions):
            last = positions[-1]
            age = (now - observed[last]) / HOUR
            if age <= config["max_observation_age_hours"] and np.isfinite(
                temperature[last]
            ):
                output[i, :2] = temperature[last], age
            # Sample each completed hourly bin once; SPECI bursts get no extra
            # weight. The exact history window is (cutoff-24h, cutoff].
            bins = (observed[positions] - 1) // HOUR
            _, reversed_first = np.unique(bins[::-1], return_index=True)
            sampled = temperature[positions[::-1][reversed_first]]
            finite = sampled[np.isfinite(sampled)]
            output[i, 3] = len(finite)
            if len(finite) >= config["mean_min_hours"]:
                output[i, 2] = finite.mean()
        else:
            output[i, 3] = 0
    return output
