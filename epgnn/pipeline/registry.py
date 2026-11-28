"""The feature dictionary: one entry per column the pipeline writes.

`timing` says when the value is known relative to the forecast for delivery day D (made at D-1 10:00 UTC):
  gate      known at gate closure for every hour of D: a legal day-D input
  past      a realised value; legal only inside the history window (hours up to the cutoff)
  outcome   an outcome of day D itself: never an input (target or diagnostics only)
  unverified JAO domain values for day D: published around gate closure, timing not verified;
            use the `_lag24` version (day D-1's domain) for a strictly safe setting
  static    does not change over time
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    timing: str
    source: str
    meaning: str


Z = "zone_nodes_hourly"
E = "zone_edges_hourly"
J = "jao_cnec_hourly + cnec_provenance"
P = "pipeline"

ZONE: list[Feature] = [
    # ---- target and masks
    Feature("price", "target", "outcome", Z, "day-ahead price, EUR/MWh; the target"),
    Feature("has_price", "target", "outcome", Z, "price exists and is valid: the loss mask"),
    Feature("price_filled", "target", "past", Z, "the lag inputs of this hour came from a short gap fill"),
    # ---- calendar
    Feature("hour_local", "calendar", "gate", Z, "Brussels local hour 0-23"),
    Feature("dow", "calendar", "gate", Z, "day of week, 0 = Monday"),
    Feature("month", "calendar", "gate", Z, "month 1-12"),
    Feature("hour_sin", "calendar", "gate", P, "sin(2 pi hour/24)"),
    Feature("hour_cos", "calendar", "gate", P, "cos(2 pi hour/24)"),
    Feature("dow_sin", "calendar", "gate", P, "sin(2 pi dow/7)"),
    Feature("dow_cos", "calendar", "gate", P, "cos(2 pi dow/7)"),
    Feature("month_sin", "calendar", "gate", P, "sin(2 pi (month-1)/12)"),
    Feature("month_cos", "calendar", "gate", P, "cos(2 pi (month-1)/12)"),
    Feature("is_weekend", "calendar", "gate", P, "Saturday or Sunday"),
    # ---- prices known at gate closure
    Feature("price_lag24", "price", "gate", Z, "price of the same hour on D-1 (D-1 prices are published on D-2)"),
    Feature("price_lag168", "price", "gate", Z, "price of the same hour one week earlier; the persistence forecast"),
    Feature("price_wk_change", "price", "gate", P, "price_lag24 - price_lag168"),
    Feature("price_d1_mean", "price", "gate", P, "mean of the zone's D-1 prices (mean of price_lag24 over D)"),
    Feature("price_d1_min", "price", "gate", P, "min of the zone's D-1 prices"),
    Feature("price_d1_max", "price", "gate", P, "max of the zone's D-1 prices"),
    Feature("price_d1_std", "price", "gate", P, "std of the zone's D-1 prices (intra-day shape / volatility)"),
    Feature("europe_price_lag24", "price", "gate", P, "mean price_lag24 over all priced zones that hour"),
    Feature("price_lag24_vs_europe", "price", "gate", P, "price_lag24 - europe_price_lag24"),
    Feature("nb_price_lag24", "price", "gate", P, "mean price_lag24 over interconnected neighbours (non-graph baselines)"),
    # ---- load
    Feature("load_fc", "load", "gate", Z, "ENTSO-E day-ahead load forecast, MW (vintage caveat in the dataset doc)"),
    Feature("load_fc_lag168", "load", "gate", P, "load forecast of the same hour one week earlier"),
    Feature("load_fc_wk_change", "load", "gate", P, "load_fc / load_fc_lag168 - 1"),
    Feature("load_fc_ramp", "load", "gate", P, "load_fc minus the previous hour's load_fc"),
    Feature("load_fc_day_rel", "load", "gate", P, "load_fc / mean load_fc of delivery day D (daily shape)"),
    # ---- weather
    Feature("temp_pw", "weather", "outcome", Z, "population-weighted temperature, same hour (perfect-forecast ablation only)"),
    Feature("temp_anom", "weather", "outcome", Z, "temperature anomaly vs climatology, same hour (ablation only)"),
    Feature("wind_cf", "weather", "outcome", Z, "wind capacity factor, same hour (ablation only)"),
    Feature("low_wind", "weather", "outcome", Z, "low-wind flag, same hour (ablation only)"),
    Feature("solar_cf", "weather", "outcome", Z, "solar capacity factor, same hour (ablation only)"),
    Feature("temp_pw_lag24", "weather", "past", Z, "temperature same hour D-1; NOT a legal day-D input: hours after 10:00 UTC are D-1 after the cutoff (use temp_pw_asof)"),
    Feature("temp_anom_lag24", "weather", "past", Z, "temperature anomaly same hour D-1; not a legal day-D input (use temp_anom_asof)"),
    Feature("wind_cf_lag24", "weather", "past", Z, "wind capacity factor same hour D-1; not a legal day-D input (use wind_cf_asof)"),
    Feature("solar_cf_lag24", "weather", "past", Z, "solar capacity factor same hour D-1; not a legal day-D input (use solar_cf_asof)"),
    Feature("temp_pw_lag48", "weather", "gate", P, "temperature same hour D-2: strictly observed at gate closure"),
    Feature("temp_anom_lag48", "weather", "gate", P, "temperature anomaly same hour D-2"),
    Feature("wind_cf_lag48", "weather", "gate", P, "wind capacity factor same hour D-2"),
    Feature("solar_cf_lag48", "weather", "gate", P, "solar capacity factor same hour D-2"),
    Feature("temp_pw_asof", "weather", "gate", P, "temperature of the same clock hour, latest observed at the cutoff: lag 24 up to 10:00 UTC, lag 48 after"),
    Feature("temp_anom_asof", "weather", "gate", P, "temperature anomaly, as temp_pw_asof"),
    Feature("wind_cf_asof", "weather", "gate", P, "wind capacity factor, as temp_pw_asof"),
    Feature("solar_cf_asof", "weather", "gate", P, "solar capacity factor, as temp_pw_asof"),
    Feature("heat_stress", "weather", "gate", P, "max(temp_anom_asof, 0): positive anomaly only (H3)"),
    # ---- flows
    Feature("net_import", "flows", "past", Z, "physical imports - exports with modelled zones, MW (outcome for D; fine in history)"),
    Feature("net_import_lag48", "flows", "gate", Z, "net_import two days earlier"),
    # ---- JAO Core domain
    Feature("in_core", "jao", "static", Z, "zone belongs to the Core flow-based region"),
    Feature("n_presolved", "jao", "unverified", Z, "Core-wide count of presolved constraints (same for every Core zone)"),
    Feature("min_ram_ratio", "jao", "unverified", Z, "Core-wide minimum RAM/Fmax (same for every Core zone)"),
    Feature("jao_n_exposed", "jao", "unverified", J, "number of presolved constraints with |zone-to-rest PTDF| >= 0.05 (PTDF_z - mean of the other Core hubs; slack-invariant)"),
    Feature("jao_min_ram_exposed", "jao", "unverified", J, "smallest RAM/Fmax among those constraints: how tight the zone's binding lines are"),
    Feature("jao_tightness", "jao", "unverified", J, "PTDF-weighted mean scarcity: sum |zone-to-rest PTDF| (1 - min(RAM/Fmax, 1)) / sum |zone-to-rest PTDF|, in [0, 1]; not a count"),
    Feature("jao_amr_share_exposed", "jao", "unverified", J, "share of those constraints whose RAM was lifted by AMR (minimum-RAM policy)"),
    Feature("jao_min_ram_grid_exposed", "jao", "unverified", J, "smallest (RAM - AMR)/Fmax among them: the tightest margin before the policy uplift"),
    Feature("jao_tightness_grid", "jao", "unverified", J, "jao_tightness computed on (RAM - AMR)/Fmax: weighted mean scarcity without the uplift, in [0, 1]"),
    Feature("jao_n_exposed_lag24", "jao", "gate", J, "jao_n_exposed of the same hour on D-1 (day D-1's domain: safe)"),
    Feature("jao_min_ram_exposed_lag24", "jao", "gate", J, "jao_min_ram_exposed on D-1"),
    Feature("jao_tightness_lag24", "jao", "gate", J, "jao_tightness on D-1"),
    Feature("jao_amr_share_exposed_lag24", "jao", "gate", J, "jao_amr_share_exposed on D-1"),
    Feature("jao_min_ram_grid_exposed_lag24", "jao", "gate", J, "jao_min_ram_grid_exposed on D-1"),
    Feature("jao_tightness_grid_lag24", "jao", "gate", J, "jao_tightness_grid on D-1"),
]

EDGE: list[Feature] = [
    Feature("flow_phys", "flows", "outcome", E, "ENTSO-E physical flow from_zone -> to_zone, MW"),
    Feature("flow_phys_lag48", "flows", "gate", E, "physical flow two days earlier"),
    Feature("flow_phys_lag168", "flows", "gate", E, "physical flow one week earlier"),
    Feature("flow_sched", "flows", "outcome", E, "JAO day-ahead scheduled exchange (Core borders)"),
    Feature("flow_sched_lag24", "flows", "gate", E, "scheduled exchange on D-1"),
    Feature("maxbex", "capacity", "unverified", E, "JAO maximum bilateral exchange, MW (Core borders from 9 Jun 2022)"),
    Feature("maxbex_lag24", "capacity", "gate", P, "maxbex on D-1 (safe)"),
    Feature("net_flow_lag48", "flows", "gate", P, "flow_phys_lag48(a->b) - flow_phys_lag48(b->a): net direction"),
    Feature("util_lag48", "flows", "gate", P, "flow_phys_lag48 / maxbex_lag24 where both exist: how congested the border was"),
    Feature("is_core_border", "capacity", "static", P, "both ends are Core zones"),
]

CONSTRAINT: list[Feature] = [
    Feature("constraint_id", "id", "static", J, "stable id: sha1(CneEic | Direction | ContName)[:12]; one line under one contingency"),
    Feature("ram", "domain", "unverified", J, "remaining available margin, MW"),
    Feature("fmax", "domain", "unverified", J, "maximum admissible flow, MW"),
    Feature("ram_fmax", "domain", "unverified", J, "RAM / Fmax: 0 = no headroom, 1 = full line available; > 1 from counter-flows (fltn < 0), not policy"),
    Feature("frm", "domain", "unverified", J, "flow reliability margin, MW"),
    Feature("f0core", "domain", "unverified", J, "flow with zero Core net positions, MW (loop / transit flow)"),
    Feature("fltn", "domain", "unverified", J, "flow including long-term nominations (FtotalLtn = f0core + LTN flow), MW; RAM = fmax - frm - fltn + amr - iva"),
    Feature("amr", "domain", "unverified", J, "adjustment for minimum RAM, MW: margin added by the 70 % (or derogation) policy"),
    Feature("iva", "domain", "unverified", J, "individual validation adjustment by the TSO, MW (reduces RAM)"),
    Feature("min_ram_factor", "domain", "unverified", J, "minimum-RAM target as a fraction of Fmax (0.7 under the CEP; lower with derogations)"),
    Feature("ram_grid_fmax", "domain", "unverified", P, "(ram - amr) / fmax: margin the grid model leaves before the policy uplift (can be < 0)"),
    Feature("amr_fmax", "domain", "unverified", P, "amr / fmax: size of the policy uplift"),
    Feature("ptdf_<zone>", "domain", "unverified", J, "published zone-to-slack PTDF of the 12 Core zones (DE -> DE_LU); only differences are physical: use constraints.rest_ptdf / zone_ptdf"),
]


def timing_of(name: str) -> str | None:
    """The registered timing of a zone, edge or constraint column (None if it is not registered)."""
    return next((f.timing for f in all_features() if f.name == name), None)


def all_features() -> list[Feature]:
    return ZONE + EDGE + CONSTRAINT


def names(group_list: list[Feature], timing: tuple[str, ...] | None = None) -> list[str]:
    return [f.name for f in group_list if timing is None or f.timing in timing]
