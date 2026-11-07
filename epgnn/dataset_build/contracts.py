"""Time, geography, uniqueness, and feature admission contracts."""
from datetime import timedelta

import numpy as np
import pandas as pd

from epgnn.zones import ENTSOE_NAME_TO_ZONE

TZ = "Europe/Brussels"
DE_SPLIT = pd.Timestamp("2018-09-30T22:00:00Z")
CORE_START = pd.Timestamp("2022-06-08T22:00:00Z")
WEATHER = ["temp_pw", "temp_anom", "wind_cf", "low_wind", "solar_cf"]
NODE_KEYS = ["zone", "ts_utc"]
EDGE_KEYS = ["from_zone", "to_zone", "ts_utc"]
CALENDAR = ["hour_local", "dow", "month", "utc_offset_hours"]


def utc(values):
    """Normalize datetime values without interpreting microseconds as nanoseconds.

    Integer epochs are deliberately rejected: an explicit source unit is required.
    """
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s.dtype):
        raise ValueError("Numeric timestamps require an explicit unit before ingestion")
    # Original combined JAO CSV mixes ISO date/time separators (T and space).
    kwargs = {"format": "ISO8601"} if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object else {}
    out = pd.to_datetime(s, utc=True, errors="raise", **kwargs).dt.as_unit("ns")
    if out.isna().any():
        raise ValueError("Null timestamps are not valid keys")
    return out


def forecast_origin(ts):
    """Previous *calendar* date at literal Brussels noon, including DST weekends."""
    t = utc(ts)
    days = t.dt.tz_convert(TZ).dt.date
    unique = {d: pd.Timestamp.combine(d - timedelta(days=1),
              pd.Timestamp("12:00").time()).tz_localize(TZ).tz_convert("UTC")
              for d in days.unique()}
    return pd.to_datetime(days.map(unique), utc=True).dt.as_unit("ns")


def calendar(frame, splits):
    out = frame.copy()
    out["ts_utc"] = utc(out.ts_utc)
    local = out.ts_utc.dt.tz_convert(TZ)
    out["delivery_date"] = local.dt.strftime("%Y-%m-%d").astype("string")
    out["forecast_origin_utc"] = forecast_origin(out.ts_utc)
    out["interval_end_utc"] = out.ts_utc + pd.Timedelta(hours=1)
    out["hour_local"] = local.dt.hour.astype("int16")
    out["dow"] = local.dt.day_of_week.astype("int16")
    out["month"] = local.dt.month.astype("int16")
    # Vectorized, preserving the repeated autumn hour's two distinct offsets.
    out["utc_offset_hours"] = ((local.dt.tz_localize(None) -
                                out.ts_utc.dt.tz_localize(None)).dt.total_seconds() / 3600)
    out["split"] = pd.Series("unassigned", index=out.index, dtype="string")
    for name, (a, b) in splits.items():
        out.loc[out.delivery_date.between(a, b), "split"] = name
    out["regime"] = pd.Series(np.where(out.ts_utc < CORE_START, "CWE", "CORE"), dtype="string")
    return out


def zone_map(names, codes=None):
    """Identity mapping; never splice DE_AT_LU onto DE_LU.

    Unresolved ENTSO-E areas remain separate, inspectable identities in the grid.
    Equal spelling alone is not evidence that country load is bidding-zone load.
    """
    names = names.astype("string")
    out = names.map({**ENTSOE_NAME_TO_ZONE, "DE-AT-LU": "DE_AT_LU"}).astype("string")
    if codes is not None:
        out = out.fillna("ENTSOE:" + codes.astype("string"))
    return out


def lifetime(zones, ts):
    active = pd.Series(True, index=zones.index)
    active &= ~((zones == "DE_AT_LU") & (ts >= DE_SPLIT))
    active &= ~((zones.isin(["DE_LU", "AT"])) & (ts < DE_SPLIT))
    return active


def unique(frame, keys, label):
    if frame[keys].isna().any().any():
        raise ValueError(f"{label}: null keys")
    bad = frame.duplicated(keys, keep=False)
    if bad.any():
        conflicting = len(frame.loc[bad].drop_duplicates()) > len(frame.loc[bad, keys].drop_duplicates())
        raise ValueError(f"{label}: duplicate keys; conflicting={conflicting}; "
                         f"sample={frame.loc[bad, keys].head(3).to_dict('records')}")


def lag_join(output, history, keys, value, hours, *, availability=None):
    """Exact elapsed-hour lookup within the same identities; no row offsets/fills.

    Availability requires an independently supported per-record publication time.
    A completed interval alone does not establish publication.
    """
    unique(history, keys, f"{value} lag history")
    col = f"{value}_lag{hours}"
    h = history[keys + [value]].copy()
    h[f"{col}_observation_utc"] = h.ts_utc
    h[f"{col}_completion_utc"] = h.ts_utc + pd.Timedelta(hours=1)
    if availability and availability in history:
        h[f"{col}_publication_utc"] = history[availability]
    else:
        h[f"{col}_publication_utc"] = pd.Series(pd.NaT, index=h.index, dtype="datetime64[ns, UTC]")
    h["ts_utc"] += pd.Timedelta(hours=hours)
    h = h.rename(columns={value: col})
    out = output.merge(h, on=keys, how="left", validate="one_to_one")
    out[f"{col}_eligible"] = (
        out[col].notna() & (out[f"{col}_observation_utc"] < out.forecast_origin_utc)
        & (out[f"{col}_completion_utc"] <= out.forecast_origin_utc)
        & (out[f"{col}_publication_utc"] <= out.forecast_origin_utc))
    return out


def feature_registry(price_publication_verified=False):
    """Registry is explicit: lag names are never evidence of availability."""
    rows = []

    def add(feature, table, source, status, timing, transformation, units, geography="exact canonical identity"):
        rows.append(dict(feature=feature, table=table, source=source, admission=status,
                         timing_rule=timing, transformation=transformation, units=units,
                         geography=geography, normalization="unfitted; physical units; preserve NaNs"))

    for c in CALENDAR:
        add(c, "node", "delivery calendar", "admitted", "deterministic at origin", "Brussels calendar from UTC", "calendar")
    add("price", "node", "entsoe_prices", "excluded", "same-delivery-day outcome", "valid unfilled PriceEUR only", "EUR/MWh")
    for h in [24, 48, 168]:
        add(f"price_lag{h}", "node", "entsoe_prices",
            "admitted" if price_publication_verified else "conditional",
            "observation < origin; interval end <= origin; verified publication <= origin; otherwise NaN in strict view",
            f"timestamp join at t-{h}h on zone; never filled", "EUR/MWh")
    add("load_fc", "node", "entsoe_load_forecast", "conditional", "forecast vintage/publication unverified", "unfilled LoadForecastMW", "MW")
    for c in WEATHER:
        for h in [0, 48]:
            add(c if not h else f"{c}_lag{h}", "node", "weather_hourly", "conditional",
                "reanalysis and historical publication unverified; lag does not verify availability",
                "unfilled raw value" if not h else "exact timestamp join at t-48h", "degC" if c.startswith("temp") else "dimensionless",
                "weather DE_LU before 2018-10-01 unresolved; never attach to DE_AT_LU")
    for c in ["n_presolved", "min_ram_ratio", "jao_domain_present", "jao_maxbex_present", "jao_exchange_present"]:
        add(c, "node", "original combined JAO", "conditional", "historical domain/publication vintage unverified",
            "source-preserving left join; missing distinct from zero", "count" if c == "n_presolved" else "dimensionless")
    for c in ["in_cwe", "in_core"]:
        add(c, "node", "dated market geography", "admitted", "dated regional membership known in advance",
            "historical member set; independent of publication presence", "boolean")
    add("flow_phys", "edge", "entsoe_physical_flows", "excluded", "same-day realised outcome", "raw directional FlowMW", "MW")
    for h in [48, 168]:
        add(f"flow_phys_lag{h}", "edge", "entsoe_physical_flows", "conditional",
            "completion necessary; historical publication unverified", f"exact directed-pair join at t-{h}h", "MW")
    for c in ["flow_sched", "maxbex"]:
        add(c, "edge", "original combined JAO border", "conditional", "publication vintage unverified",
            "CWE BEC / CORE scheduled exchange" if c == "flow_sched" else "single-border MaxBex, not a physical or jointly feasible capacity", "MW")
    return rows


def strict_view(nodes, registry):
    cols = NODE_KEYS + ["delivery_date", "forecast_origin_utc", "split", "price", "target_valid"]
    admitted = [r["feature"] for r in registry if r["table"] == "node" and r["admission"] == "admitted"]
    out = nodes[cols + admitted].copy()
    for c in admitted:
        if f"{c}_eligible" in nodes:
            out[c] = out[c].where(nodes[f"{c}_eligible"])
            out[f"{c}_eligible"] = nodes[f"{c}_eligible"]
    return out
