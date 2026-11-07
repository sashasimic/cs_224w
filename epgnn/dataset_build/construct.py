"""Source-preserving adapters and monthly master construction."""
import numpy as np
import pandas as pd

from .contracts import (CALENDAR, CORE_START, DE_SPLIT, EDGE_KEYS, NODE_KEYS,
                        WEATHER, calendar, lag_join, lifetime, unique, utc, zone_map)
from .storage import SPECS, read_month, read_window


def adapt(raw, name):
    """Retain original attributes, including upstream fill diagnostics, under prefixes.

    *_input fields survive only as source diagnostics. Derived features exclusively
    use original values and their source validity flags.
    """
    ts = SPECS[name][0]
    f = raw.reset_index(drop=True).copy()
    # Stable nullable types across months, including completely absent sources.
    for c in f:
        dtype = f[c].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            f[c] = pd.to_datetime(f[c], utc=True).dt.as_unit("ns")
        elif pd.api.types.is_bool_dtype(dtype):
            f[c] = f[c].astype("boolean")
        elif pd.api.types.is_integer_dtype(dtype):
            f[c] = f[c].astype("float64")
        elif pd.api.types.is_float_dtype(dtype):
            f[c] = f[c].astype("float64")
        else:
            f[c] = f[c].astype("string")
    f[ts] = utc(f[ts]) if len(f) else pd.Series([], dtype="datetime64[ns, UTC]")
    if name.startswith("entsoe_"):
        if name == "entsoe_physical_flows":
            keys = EDGE_KEYS
            f["from_zone"] = zone_map(f.OutAreaName, f.OutAreaCode)
            f["to_zone"] = zone_map(f.InAreaName, f.InAreaCode)
            prefix, value, valid = "physical", "FlowMW", "IsValidValue"
            target = "flow_phys"
        else:
            keys = NODE_KEYS
            f["zone"] = zone_map(f.AreaName, f.AreaCode)
            prefix, value, valid, target = (("price", "PriceEUR", "IsValidTarget", "price")
                if name == "entsoe_prices" else ("load", "LoadForecastMW", "IsValidValue", "load_fc"))
        f = f.rename(columns={ts: "ts_utc"})
        unique(f, keys, name + " canonical keys")
        out = f[keys].copy()
        for c in f.columns.difference(keys, sort=False):
            out[f"{prefix}_source_{c}"] = f[c]
        out[f"{prefix}_row_present"] = True
        out[f"{prefix}_source_valid"] = f[valid].fillna(False).astype(bool)
        out[f"{prefix}_source_filled"] = f.get("is_filled", pd.Series(False, index=f.index)).fillna(False).astype(bool)
        out[target] = pd.to_numeric(f[value], errors="raise").astype("float64").where(
            out[f"{prefix}_source_valid"] & np.isfinite(f[value]))
        if name == "entsoe_prices":
            # Filled flags refer to *_input. The raw target must additionally be unfilled.
            out[target] = out[target].where(~out.price_source_filled & lifetime(out.zone, out.ts_utc))
            out["target_valid"] = out[target].notna()
        if name == "entsoe_load_forecast":
            out["load_geography_status"] = pd.Series(np.where(out.zone.str.startswith("ENTSOE:"), "unresolved_area", "exact_name_identity"), dtype="string")
        return out
    f = f.rename(columns={ts: "ts_utc"})
    if name == "weather_hourly":
        unique(f, NODE_KEYS, name)
        out = f[NODE_KEYS + WEATHER].copy()
        for c in WEATHER:
            out[c] = out[c].astype("float64")
        out["weather_row_present"] = True
        out["weather_source_zone"] = f.zone.astype("string")
        out["weather_geography_status"] = pd.Series(np.where(f.zone == "DE_LU", "germany_only_proxy_unresolved", "source_zone_identity"), dtype="string")
        return out
    if name == "jao_hourly":
        unique(f, NODE_KEYS, name)
        out = f.rename(columns={c: f"jao_source_{c}" for c in f if c not in NODE_KEYS + ["n_presolved", "min_ram_ratio"]})
        out["jao_row_present"] = True
        out["n_presolved"] = out.n_presolved.astype("float64")
        out["min_ram_ratio"] = out.min_ram_ratio.astype("float64")
        return out
    unique(f, EDGE_KEYS, name)
    out = f.rename(columns={"flow": "flow_sched", **{c: f"border_source_{c}" for c in f if c not in EDGE_KEYS + ["flow", "maxbex"]}})
    out["border_row_present"] = True
    out["exchange_definition"] = pd.Series(np.where(f.regime == "CWE", "allocated_BEC", "scheduled_exchange"), dtype="string")
    out["maxbex_value_present"] = out.maxbex.notna()
    out["exchange_value_present"] = out.flow_sched.notna()
    return out


def regional_membership(nodes):
    z, t = nodes.zone, nodes.ts_utc
    cwe = t < CORE_START
    nodes["in_cwe"] = cwe & (z.isin(["BE", "FR", "NL"]) |
        ((z == "DE_AT_LU") & (t < DE_SPLIT)) |
        (z.isin(["AT", "DE_LU"]) & (t >= DE_SPLIT)))
    nodes["in_core"] = ~cwe & z.isin(["AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"])
    return nodes


def coverage_join(out, coverage):
    cols = {"domain_present": "jao_domain_present", "maxbex_present": "jao_maxbex_present", "flow_present": "jao_exchange_present"}
    c = coverage[["ts_utc"] + list(cols)].rename(columns=cols).copy()
    unique(c, ["ts_utc"], "JAO publication coverage")
    c["jao_coverage_row_present"] = True
    out = out.merge(c, on="ts_utc", how="left", validate="many_to_one")
    # No coverage record is not a published zero. Keep nullable publication flags.
    out["jao_coverage_row_present"] = out.jao_coverage_row_present.fillna(False).astype(bool)
    for col in cols.values():
        out[col] = out[col].astype("boolean")
    return out


def node_month(cfg, folders, zones, hours, coverage):
    start, end = hours.min(), hours.max() + pd.Timedelta(hours=1)
    grid = pd.MultiIndex.from_product([zones, hours], names=NODE_KEYS).to_frame(index=False)
    out = calendar(grid, cfg["splits"])
    reconciliation = {}
    for name in ["entsoe_prices", "entsoe_load_forecast", "weather_hourly", "jao_hourly"]:
        raw = read_window(folders[name], SPECS[name][0], start, end)
        src = adapt(raw, name)
        before = len(out)
        out = out.merge(src, on=NODE_KEYS, how="left", validate="one_to_one")
        assert len(out) == before
        joined = len(src.merge(grid[NODE_KEYS], on=NODE_KEYS, how="inner", validate="one_to_one"))
        reconciliation[name] = {"window_source_rows": len(src), "joined_rows": joined, "unmatched_rows": len(src) - joined}
    for c in ["price_row_present", "load_row_present", "weather_row_present", "jao_row_present", "target_valid"]:
        out[c] = out[c].fillna(False).astype(bool)
    out["price_structurally_available"] = lifetime(out.zone, out.ts_utc)
    # Source-defined structural exclusions are retained rather than inferred from gaps.
    if "price_source_row_status" in out:
        out["price_structurally_available"] &= ~out.price_source_row_status.isin(["OUTSIDE_ZONE_LIFETIME", "STRUCTURAL_ABSENCE", "NOT_APPLICABLE"])
    out["price_absence_class"] = pd.Series(np.select([
        ~out.price_structurally_available, out.target_valid,
        ~out.price_row_present], ["structural", "observed_valid", "missing_source_row"],
        default="invalid_or_missing_value"), dtype="string")
    out["geography_status"] = pd.Series(np.where(out.zone.str.startswith("ENTSOE:"), "unresolved_area", "canonical_identity"), dtype="string")
    out = regional_membership(out)
    out = coverage_join(out, coverage)
    member = out.in_cwe | out.in_core
    out["jao_absence_class"] = pd.Series(np.select([
        ~member, ~out.jao_domain_present.fillna(False), ~out.jao_row_present],
        ["outside_region", "missing_domain_publication", "missing_hub_row"], default="observed"), dtype="string")
    pub = cfg.get("price_publication", {}).get("column")
    for name, values, lags in [("entsoe_prices", ["price"], [24, 48, 168]),
                               ("weather_hourly", WEATHER, [48])]:
        history = adapt(read_window(folders[name], SPECS[name][0], start - pd.Timedelta(hours=max(lags)), end), name)
        for val in values:
            for h in lags:
                out = lag_join(out, history, NODE_KEYS, val, h,
                               availability=f"price_source_{pub}" if pub and val == "price" else None)
    # These are explicit publication timestamps only. Update timestamps are kept as
    # source attributes and are never relabelled as historical forecast vintages.
    out["price_publication_status"] = "verified_column" if pub else "unverified_historical_vintage"
    unique(out, NODE_KEYS, "node master")
    out = out.sort_values(NODE_KEYS).reset_index(drop=True)
    out.attrs["reconciliation"] = reconciliation
    return out


def edge_month(cfg, folders, pairs, hours, coverage, node_zones):
    start, end = hours.min(), hours.max() + pd.Timedelta(hours=1)
    grid = pairs.merge(pd.DataFrame({"ts_utc": hours}), how="cross")
    out = calendar(grid, cfg["splits"])
    reconciliation = {}
    for name in ["entsoe_physical_flows", "jao_border_hourly"]:
        source = adapt(read_window(folders[name], SPECS[name][0], start, end), name)
        out = out.merge(source, on=EDGE_KEYS, how="left", validate="one_to_one")
        reconciliation[name] = {"window_source_rows": len(source), "joined_rows": len(source), "unmatched_rows": 0}
    for c in ["physical_row_present", "border_row_present", "maxbex_value_present", "exchange_value_present"]:
        out[c] = out[c].fillna(False).astype(bool)
    out["edge_structurally_available"] = lifetime(out.from_zone, out.ts_utc) & lifetime(out.to_zone, out.ts_utc)
    out["endpoints_in_node_grid"] = out.from_zone.isin(node_zones) & out.to_zone.isin(node_zones)
    out = coverage_join(out, coverage)
    history = adapt(read_window(folders["entsoe_physical_flows"], "DeliveryHourUTC", start - pd.Timedelta(hours=168), end), "entsoe_physical_flows")
    for h in [48, 168]:
        out = lag_join(out, history, EDGE_KEYS, "flow_phys", h)
    # Do not sum MaxBex or infer any jointly feasible import limit.
    unique(out, EDGE_KEYS, "edge master")
    out = out.sort_values(EDGE_KEYS).reset_index(drop=True)
    out.attrs["reconciliation"] = reconciliation
    return out


def coverage_report(frame, table):
    """Structural and missingness reports; deliberately no target distributions."""
    f = frame.assign(calendar_month=frame.ts_utc.dt.strftime("%Y-%m"))
    ids = ["zone"] if table == "node" else ["from_zone", "to_zone"]
    group = ids + ["calendar_month", "split"]
    features = (["price", "load_fc"] + WEATHER + ["n_presolved", "min_ram_ratio"]
                if table == "node" else ["flow_phys", "flow_sched", "maxbex"])
    result = []
    for c in features:
        r = f.assign(missing=f[c].isna()).groupby(group, dropna=False).agg(rows=(c, "size"), missing=("missing", "sum")).reset_index()
        r["feature"] = c
        r["table"] = table
        result.append(r)
    if table == "node":
        for c in ["price_absence_class", "jao_absence_class", "geography_status", "weather_geography_status", "load_geography_status"]:
            r = f.groupby(group + [c], dropna=False).size().rename("rows").reset_index().rename(columns={c: "classification"})
            r["feature"] = c
            r["table"] = table
            result.append(r)
    return pd.concat(result, ignore_index=True)
