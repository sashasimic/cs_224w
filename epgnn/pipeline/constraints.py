"""JAO Core constraints as graph nodes (checkpoint 4b), and zone-level summaries of them.

Source: the JAO Core views (`jao_core_hourly_views/`): `jao_cnec_hourly.parquet` holds one row per
presolved constraint observation and hour (RAM/Fmax, PTDF per hub); `cnec_provenance.parquet` holds the
original fields of the same rows (line EIC, name, direction, contingency, RAM, Fmax, FRM, F0core).

`cnec_key` identifies one published observation, not a line, so it cannot be a node id. A constraint
node here is one monitored element under one contingency in one direction:

    constraint_id = sha1(CneEic | Direction | ContName)[:12]     (CneName where CneEic is "NA" or blank)

RAM and the minimum-RAM policy. On every physical Core row the published margin decomposes exactly as

    RAM = Fmax - FRM - F_LTN + AMR - IVA        (F_LTN = FtotalLtn = F0core + flow of long-term nominations)

AMR is the adjustment that lifts RAM to the minimum-RAM target (70 % of Fmax under the Clean Energy Package,
lower for TSOs with derogations: `min_ram_factor`). It is positive on 18.8 % of rows (median 8 % of Fmax), so
for those rows RAM/Fmax measures policy, not the grid. `ram_grid_fmax = (RAM - AMR) / Fmax` is the margin the
grid model leaves before the uplift (1.2 % of rows are negative), `amr_fmax = AMR / Fmax` the uplift itself.
RAM/Fmax > 1 (17 % of rows) is not policy: all of them have F_LTN < 0, a counter-flow that frees margin.

Synthetic constraints (equality / external / default import-export limits, `is_synthetic`) are dropped in
`load` before ids are assigned.

On the Core data (Jun 2022 - Oct 2026) that gives 6,319 constraints, about 230 per delivery day and
120 per hour; the median constraint appears on 7 days, a few on all 1,599 domain days. So the constraint
set changes from day to day: build one constraint graph per delivery day (`constraint_graph`).

PTDF hubs: the 12 Core zones; JAO's DE is DE_LU. ALBE/ALDE (ALEGrO virtual hubs) are dropped.

PTDF reference. Published hub PTDFs are zone-to-slack: flow = F0 + sum_z PTDF_z * NP_z with sum_z NP_z = 0,
so adding one constant to all hub PTDFs of a constraint changes no flow. |PTDF_z| on its own is therefore
not physical, and a threshold on it depends on the slack. Everything zone-level here uses the
**zone-to-rest PTDF** instead (`rest_ptdf`):

    PTDF_z,rest = PTDF_z - mean_{n != z} PTDF_n

the flow change per MW exported from z and taken up evenly by the other Core hubs. It is invariant to the
constant, keeps the sign (positive: exports from z load the constraint) and equals 12/11 x the PTDF
de-meaned over the hubs. constraints_hourly keeps the published (raw) PTDFs; the transform is applied when
exposure features and constraint graphs are built (`reference="raw"` reproduces the old behaviour).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from epgnn.jao import SYNTHETIC
from epgnn.paths import ROOT

JAO_DIR = ROOT.parent / "data" / "jao_core_hourly_views"
HUBS = ["AT", "BE", "CZ", "DE", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]
ZONE_OF_HUB = {h: ("DE_LU" if h == "DE" else h) for h in HUBS}
CORE_ZONES = [ZONE_OF_HUB[h] for h in HUBS]
# published margin and its parts (MW unless noted); see "RAM and the minimum-RAM policy" in the module doc
NODE_FEATURES = ["ram_fmax", "ram", "fmax", "frm", "f0core", "fltn", "amr", "iva", "min_ram_factor",
                 "ram_grid_fmax", "amr_fmax"]
POLICY_FEATURES = ["jao_amr_share_exposed", "jao_min_ram_grid_exposed", "jao_tightness_grid"]
EXPOSURE_FEATURES = ["jao_n_exposed", "jao_min_ram_exposed", "jao_tightness"] + POLICY_FEATURES
EXPOSURE_PTDF = 0.05


NUMERIC = {"Ram": "ram", "Fmax": "fmax", "Frm": "frm", "F0core": "f0core", "FtotalLtn": "fltn", "Amr": "amr",
           "Iva": "iva", "MinRamFactor": "min_ram_factor"}
MISSING_EIC = {"", "NA"}   # the JAO export writes "NA" for elements without an EIC; fall back to the name


def is_synthetic(names: pd.Series) -> np.ndarray:
    """True for constraints that are not a physical element: equality constraints of the ALEGrO hubs,
    external import/export limits (ALEGrO, the advanced-hybrid-coupling DC links, NL) and the
    Default_<zone>_Import/Export rows (epgnn.jao.SYNTHETIC). On Core Jun 2022 - Oct 2026 these are 58 names,
    8.0 % of the presolved rows, all with CneEic "NA" and no ElementType; their Fmax is a placeholder."""
    s = names.astype("category") if not isinstance(names.dtype, pd.CategoricalDtype) else names
    hit = s.cat.categories.astype(str).str.upper().str.contains(SYNTHETIC)
    return s.isin(s.cat.categories[hit]).to_numpy()


def add_policy_columns(df: pd.DataFrame) -> None:
    """In place: min_ram_factor as a fraction, ram_grid_fmax = (ram - amr) / fmax, amr_fmax = amr / fmax
    (NaN where fmax <= 0 or missing). Expects ram, fmax, amr and min_ram_factor (percent, as published)."""
    fmax = df.fmax.where(df.fmax > 0)
    df["min_ram_factor"] = (df.min_ram_factor / 100).astype("float32")
    df["ram_grid_fmax"] = ((df.ram - df.amr.fillna(0)) / fmax).astype("float32")
    df["amr_fmax"] = (df.amr / fmax).astype("float32")


def _sid(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _to_float(col: pa.ChunkedArray) -> np.ndarray:
    """Provenance numbers are text ("240.0000000000"); blanks and junk become NaN."""
    try:
        col = pc.if_else(pc.equal(col, ""), pa.scalar(None, col.type), col)
        return pc.cast(col, pa.float32()).to_numpy(zero_copy_only=False).astype(np.float32)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return pd.to_numeric(col.to_pandas(), errors="coerce").to_numpy(dtype=np.float32)


def load(jao_dir: Path = JAO_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(hourly, catalog). hourly: one row per (ts_utc, constraint_id); catalog: one row per constraint."""
    cnec, prov = jao_dir / "jao_cnec_hourly.parquet", jao_dir / "cnec_provenance.parquet"
    # the two tables hold the same observations; align provenance rows to the CNEC rows by cnec_key
    ck = pq.read_table(cnec, columns=["cnec_key"]).column(0)
    pk = pq.read_table(prov, columns=["cnec_key"]).column(0).cast(ck.type)
    order = None
    if not (len(ck) == len(pk) and pc.all(pc.equal(ck, pk)).as_py()):
        order = pd.Index(pk.to_pandas()).get_indexer(ck.to_pandas())
        assert (order >= 0).all(), "provenance is missing CNEC observations"
    del ck, pk
    df = pd.read_parquet(cnec, columns=["ts_utc", "ram_fmax"] + [f"ptdf_{h}" for h in HUBS])
    text = ["CneEic", "CneName", "Direction", "ContName", "Tso", "ElementType"]
    pt = pq.read_table(prov, columns=text + list(NUMERIC), read_dictionary=text)
    if order is not None:
        pt = pt.take(pa.array(order))
    for col in text:
        df[col] = pt.column(col).to_pandas().astype("category")
    for src, dst in NUMERIC.items():
        df[dst] = _to_float(pt.column(src))
    del pt
    add_policy_columns(df)
    # synthetic constraints are not grid elements: drop them before they get ids (they would otherwise all
    # share CneEic "NA" and collapse into one node through the variant dedupe below)
    syn = is_synthetic(df.CneName)
    load.synthetic_dropped = int(syn.sum())  # type: ignore[attr-defined]
    df = df[~syn].reset_index(drop=True)
    for col in text:
        df[col] = df[col].cat.remove_unused_categories()
    # constraint id per distinct (element, direction, contingency), computed once per combination
    ident = ["CneEic", "CneName", "Direction", "ContName"]
    grp = df.groupby(ident, observed=True, dropna=False, sort=False).ngroup().to_numpy()
    first = np.unique(grp, return_index=True)[1]          # first row of group 0, 1, 2, ...
    combos = df.iloc[first][ident].astype(object).where(lambda f: f.notna() & (f != ""), None)
    keys = [f"{(e if e not in MISSING_EIC else None) or n or ''}|{d or ''}|{c or 'BASECASE'}"
            for e, n, d, c in combos.itertuples(index=False)]
    sid = np.array([_sid(k) for k in keys])
    df["constraint_id"] = pd.Categorical(sid[grp])
    df["ram_fmax"] = df.ram_fmax.astype("float32")
    df = df.rename(columns={f"ptdf_{h}": f"ptdf_{ZONE_OF_HUB[h]}" for h in HUBS})
    ptdf = [f"ptdf_{z}" for z in CORE_ZONES]
    df[ptdf] = df[ptdf].astype("float32")
    df["ts_utc"] = df.ts_utc.astype("datetime64[ns, UTC]")

    # the same constraint can be published twice in an hour (variants); keep the tightest
    df = df.sort_values(["ts_utc", "constraint_id", "ram_fmax"]).reset_index(drop=True)
    dup = df.duplicated(["ts_utc", "constraint_id"])
    load.duplicates_dropped = int(dup.sum())  # type: ignore[attr-defined]
    df = df[~dup]

    day = df.ts_utc.dt.tz_convert("Europe/Brussels").dt.date
    cat = df.assign(day=day).groupby("constraint_id", observed=True).agg(
        cne_eic=("CneEic", "first"), cne_name=("CneName", "first"), direction=("Direction", "first"),
        contingency=("ContName", "first"), tso=("Tso", "first"), element_type=("ElementType", "first"),
        first_ts=("ts_utc", "min"), last_ts=("ts_utc", "max"), n_hours=("ts_utc", "size"),
        n_days=("day", "nunique"), mean_ram_fmax=("ram_fmax", "mean"), mean_fmax=("fmax", "mean"),
    ).reset_index()
    hourly = df[["ts_utc", "constraint_id"] + NODE_FEATURES + ptdf].reset_index(drop=True)
    hourly["constraint_id"] = hourly.constraint_id.astype(str)
    return hourly, with_ptdf_summary(cat, hourly)


def with_ptdf_summary(cat: pd.DataFrame, hourly: pd.DataFrame, reference: str = "rest") -> pd.DataFrame:
    """Set the catalog's main_zone (the Core zone with the largest |PTDF|, most often over the constraint's
    hours) and max_abs_ptdf, in the given PTDF reference. Replaces any existing columns of those names."""
    a = np.abs(np.nan_to_num(zone_ptdf(hourly, reference)))
    rows = pd.DataFrame({"constraint_id": hourly.constraint_id.astype(str).to_numpy(),
                         "main_zone": np.array(CORE_ZONES)[a.argmax(1)], "max_abs_ptdf": a.max(1).astype(np.float32)})
    g = rows.groupby("constraint_id", sort=False)
    summ = pd.DataFrame({"main_zone": g.main_zone.agg(lambda s: s.mode().iat[0]), "max_abs_ptdf": g.max_abs_ptdf.max()})
    out = cat.drop(columns=["main_zone", "max_abs_ptdf"], errors="ignore")
    key = out.constraint_id.astype(str)
    out["main_zone"] = key.map(summ.main_zone).to_numpy()
    out["max_abs_ptdf"] = key.map(summ.max_abs_ptdf).to_numpy(dtype=np.float32)
    out["ptdf_reference"] = reference
    return out


REFERENCES = ("rest", "raw")


def rest_ptdf(p: np.ndarray) -> np.ndarray:
    """Zone-to-rest PTDFs from hub PTDFs, (..., Zc) -> (..., Zc): PTDF_z - mean of the other hubs' PTDFs.
    NaN-aware: a missing hub is left out of the means and stays NaN itself."""
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    n = ok.sum(-1, keepdims=True)
    s = np.where(ok, p, 0.0).sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rest_mean = np.where(n > 1, (s - np.where(ok, p, 0.0)) / (n - 1), np.nan)
    return np.where(ok, p - rest_mean, np.nan)


def zone_ptdf(hourly: pd.DataFrame, reference: str = "rest") -> np.ndarray:
    """(rows, 12) PTDFs of the CORE_ZONES for every row of constraints_hourly, in the chosen reference."""
    assert reference in REFERENCES, f"reference must be one of {REFERENCES}, got {reference!r}"
    p = hourly[[f"ptdf_{z}" for z in CORE_ZONES]].to_numpy(dtype=np.float64)
    return rest_ptdf(p) if reference == "rest" else p


def zone_exposure(hourly: pd.DataFrame, threshold: float = EXPOSURE_PTDF, reference: str = "rest") -> pd.DataFrame:
    """Zone-specific JAO summaries per (ts_utc, zone) for the 12 Core zones, with P = the zone-to-rest PTDF
    (see the module doc; `reference="raw"` uses the published zone-to-slack PTDF, the pre-fix behaviour).

    jao_n_exposed        constraints with |P_zone| >= threshold
    jao_min_ram_exposed  min RAM/Fmax among them
    jao_tightness        PTDF-weighted mean scarcity, sum |P_zone| s / sum |P_zone| with s = 1 - min(RAM/Fmax, 1),
                         over the constraints with a valid RAM/Fmax: in [0, 1], independent of how many
                         constraints were presolved (that count is jao_n_exposed); NaN if sum |P_zone| = 0
    policy-adjusted (NaN if the table has no ram_grid_fmax / amr, i.e. was built before they existed):
    jao_amr_share_exposed     share of the exposed constraints whose RAM was lifted by AMR (> 0)
    jao_min_ram_grid_exposed  min (RAM - AMR)/Fmax among the exposed: the tightest margin before the uplift
    jao_tightness_grid        jao_tightness with s = 1 - clip((RAM - AMR)/Fmax, 0, 1): scarcity without the uplift
    (The node table's n_presolved / min_ram_ratio use PTDF != 0, which makes them nearly identical for
    every Core zone; the threshold is what makes these zone-specific.)
    """
    out = []
    scarcity = 1 - hourly.ram_fmax.clip(upper=1)                       # NaN where RAM/Fmax is missing
    policy = "ram_grid_fmax" in hourly and "amr" in hourly
    if policy:
        grid_scarcity = 1 - hourly.ram_grid_fmax.clip(0, 1)
        lifted = (hourly.amr > 0).astype("float32").where(hourly.amr.notna())
    P = np.abs(zone_ptdf(hourly, reference)).astype(np.float32)
    for i, z in enumerate(CORE_ZONES):
        a = pd.Series(P[:, i], index=hourly.index)
        exposed = a >= threshold
        w = a.fillna(0)
        f = pd.DataFrame({"ts_utc": hourly.ts_utc, "exp": exposed.astype("int32"), "ram": hourly.ram_fmax.where(exposed),
                          "tight": (w * scarcity).fillna(0), "w": w.where(scarcity.notna(), 0)})
        agg = {"jao_n_exposed": ("exp", "sum"), "jao_min_ram_exposed": ("ram", "min"),
               "_tight": ("tight", "sum"), "_w": ("w", "sum")}
        if policy:
            f["lifted"] = lifted.where(exposed)
            f["ram_grid"] = hourly.ram_grid_fmax.where(exposed)
            f["tight_grid"] = (w * grid_scarcity).fillna(0)
            f["w_grid"] = w.where(grid_scarcity.notna(), 0)
            agg.update(jao_amr_share_exposed=("lifted", "mean"), jao_min_ram_grid_exposed=("ram_grid", "min"),
                       _tight_grid=("tight_grid", "sum"), _w_grid=("w_grid", "sum"))
        g = f.groupby("ts_utc").agg(**agg).reset_index()
        g["jao_tightness"] = g._tight / g._w.where(g._w > 0)
        if policy:
            g["jao_tightness_grid"] = g._tight_grid / g._w_grid.where(g._w_grid > 0)
        g = g.drop(columns=[c for c in g.columns if c.startswith("_")])
        g["zone"] = z
        out.append(g)
    e = pd.concat(out, ignore_index=True)
    e["ts_utc"] = e.ts_utc.astype("datetime64[ns, UTC]")
    for c in EXPOSURE_FEATURES:
        e[c] = e[c].astype("float32") if c in e else np.float32(np.nan)
    return e


def constraint_graph(hourly: pd.DataFrame, day_hours: np.ndarray, zone_list: list[str],
                     ptdf_min: float = EXPOSURE_PTDF, reference: str = "rest") -> dict:
    """The constraint part of the hetero graph for one delivery day.

    hourly     the constraints_hourly table (pass a slice for speed; index not required)
    day_hours  (H,) int64 ns ts_utc of the day's slots, -1 where the slot does not exist (25 slots)
    reference  "rest" (default): zone-to-rest PTDFs, invariant to the slack; "raw": the published hub PTDFs
    Returns
      ids        (C,) constraint ids active in any hour of the day
      x          (C, H, len(NODE_FEATURES)) hourly node features, NaN where the constraint is not presolved
      present    (C, H) bool
      ptdf       (C, H, Zc) hourly PTDF towards the Core zones in `reference` (NaN where absent); Zc = len(CORE_ZONES)
      edges      (2, E) [zone index in zone_list, constraint index] where |mean PTDF over the day| >= ptdf_min
      weight     (E,) the signed mean PTDF of each edge
      reference  the reference used
    """
    h = len(day_hours)
    ts = pd.to_datetime(day_hours[day_hours >= 0], utc=True)
    sub = hourly[hourly.ts_utc.isin(ts)]
    ids = np.array(sorted(sub.constraint_id.unique()))
    ci = {c: i for i, c in enumerate(ids)}
    slot = {t: k for k, t in enumerate(day_hours) if t >= 0}
    rows = sub.constraint_id.map(ci).to_numpy()
    cols = np.array([slot[t] for t in sub.ts_utc.astype("int64")])
    x = np.full((len(ids), h, len(NODE_FEATURES)), np.nan, dtype=np.float32)
    x[rows, cols] = sub[NODE_FEATURES].to_numpy(dtype=np.float32)
    ptdf = np.full((len(ids), h, len(CORE_ZONES)), np.nan, dtype=np.float32)
    ptdf[rows, cols] = zone_ptdf(sub, reference).astype(np.float32)
    present = np.zeros((len(ids), h), dtype=bool)
    present[rows, cols] = True
    with np.errstate(invalid="ignore"):
        mean_ptdf = np.nanmean(ptdf, axis=1) if len(ids) else np.zeros((0, len(CORE_ZONES)))
    zi = {z: i for i, z in enumerate(zone_list)}
    c_idx, z_idx = np.nonzero(np.abs(np.nan_to_num(mean_ptdf)) >= ptdf_min)
    edges = np.stack([[zi[CORE_ZONES[k]] for k in z_idx], c_idx]).astype(np.int64) if len(c_idx) else np.zeros((2, 0), np.int64)
    return {"ids": ids, "x": x, "present": present, "ptdf": ptdf, "edges": edges,
            "weight": mean_ptdf[c_idx, z_idx].astype(np.float32), "reference": reference}
