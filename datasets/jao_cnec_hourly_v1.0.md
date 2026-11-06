# JAO flow-based constraints, one row per physical constraint and hour

| | |
|---|---|
| Dataset id | `jao_cnec_hourly` |
| Version | **v1.0** |
| Bucket | `gs://cs224w-epgnn-datasets/jao_cnec_hourly/v1.0/` |
| Local file(s) | `datasets/jao_cnec_hourly_v1.0.md`, `datasets/jao_cnec_hourly_v1.0.parquet` |
| Source | JAO publication tool, CWE (2015-05-20 – 2022-06-08) and Core (2022-06-09 – 2026-10-29) final flow-based domains, via Arsen's `jao_cwe_core_hourly_views` bundle |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("jao_cnec_hourly", "jao_cnec_hourly.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | first build from the JAO bundle: physical constraints only, RAM/Fmax in MW, PTDF per zone with weather zone codes, `market_day_cet` (rules below) |

Built by [scripts/build_jao_cnec_hourly.py](../scripts/build_jao_cnec_hourly.py); the shared rules (synthetic-name pattern, zone codes, neighbours) are in [epgnn/jao.py](../epgnn/jao.py).

---

## What it is

This is the **constraint-node table** for the proposal's graph (Figure 1 of the proposal): one row per **physical transmission constraint (CNEC) per hour** in the day-ahead flow-based domain. Each row gives the line's remaining margin and capacity, and its sensitivity (PTDF) to injections in each of the 12 flow-based zones.

For one hour, the hetero graph is built directly from the rows of that hour:

```python
h = df[df.ts_utc == t]                                   # the constraint nodes of hour t
x_c = h[["ram", "fmax", "ram_fmax"]]                     # constraint-node features
ptdf = h[[f"ptdf_{z}" for z in ZONES]]                   # zone -> constraint edge weights (NaN = zone not in domain)
```

`jao_zone_hourly` is the per-zone summary of this table for models without constraint nodes.

**Alignment with the other datasets (verified):**

- `ts_utc` is the delivery hour in UTC, the same grid as `DeliveryHourUTC` in the ENTSO-E tables and `time_utc` in `weather_hourly`. Every JAO hour from 2016 on is in the ENTSO-E grid, and `market_day_cet` equals ENTSO-E's `MarketDayCET` for every hour.
- Zone codes in the `ptdf_*` columns (`AT, BE, CZ, DE_LU, FR, HR, HU, NL, PL, RO, SI, SK`) are the `weather_hourly` codes; the milestone plan's `zones.csv` maps them to ENTSO-E area codes.
- Coverage: 100,064 hours, 2015-05-19 22:00 → 2026-10-29 20:00 UTC. 279 ENTSO-E hours since 2016 have no physical constraint row (120 of them in 2026, the rest single days); the domain for those hours is simply empty and the model falls back to interconnector edges only.

**Timing.** The day-ahead domain for day D is published on D-1 before the 12:00 CET auction, which is before the project's D-1 10:00 UTC cutoff. The proposal's open question (whether the CNEC-level detail is only released later) is not settled by this dataset; if it turns out to be post-cutoff, use day D-1's rows as the input for day D.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `ts_utc` | timestamp (UTC) | delivery hour |
| `market_day_cet` | string | calendar day of the hour in Europe/Brussels, `YYYY-MM-DD`; equals ENTSO-E `MarketDayCET` |
| `regime` | string | `CWE` (to 2022-06-08 21:00 UTC) or `Core` (from 2022-06-08 22:00 UTC) |
| `cnec_key` | string | observation id from the bundle, `<source sha256>:<row>`; joins back to the raw tables. An observation id, not a stable asset id |
| `cne_name` | string | source constraint name, e.g. `Ensdorf - Vigy VIGY2 S`, `[BE-FR] ACHENE - LONNY 380.19 [OPP] [BE]` (CWE names carry border, direction and contingency tags) |
| `tso` | string | owning TSO (Core only; empty for CWE rows) |
| `ram` | float, MW | remaining available margin for the market |
| `fmax` | float, MW | maximum admissible flow on the line |
| `ram_fmax` | float | `ram / fmax`, share of capacity left; 14% of rows exceed 1 (margin above nominal), 1% are below 0.1 |
| `ptdf_AT` … `ptdf_SK` | float | zone-to-slack PTDF: MW loading of the line per MW injected in the zone; **null when the zone is not in that hour's domain** (CZ, HR, HU, PL, RO, SI, SK before June 2022; AT before Oct 2018) |

6,641,751 rows; no duplicate (`ts_utc`, `cnec_key`). Constraints per hour: CWE median 21 (3 – 341), Core median 109 (40 – 310).

## Rules used to build it

1. **Physical constraints only.** A source row is synthetic if its name (`CriticalBranchName` in CWE, `CneName` in Core) matches `LTA | IMPORT | EXPORT | EC_ | EXTERNAL | CONSTRAINT | NET POSITION | NETPOSITION | MAXBEX`: long-term-allocation corners, external/net-position limits, equality constraints. They shape the market domain but their `Fmax` is a placeholder (in CWE 2021–22 they are 94% of all rows, median ratio ≈ 4), so they cannot be constraint nodes with meaningful margin features. Dropped: 11.0M of 17.7M source rows. They remain in the raw bundle.
2. **Valid ratio.** Rows with null `ram_fmax` are dropped (RAM or Fmax missing or Fmax ≤ 0). `ram` and `fmax` are taken from the source fields (`RemainingAvailableMargin`/`Ram`, `Fmax`); `ram_fmax` equals their quotient exactly.
3. **Germany before the 2018 split.** Before delivery day 2018-10-01 the source has one combined zone `DE_AT_LU`; its PTDF is stored in `ptdf_DE_LU` and `ptdf_AT` is null. Use it for the DE-LU node; the proposal treats Germany pre-split as DE-LU anyway.
4. **Virtual hubs dropped.** `ALBE` and `ALDE` (the ALEGrO cable ends) are not bidding zones and have no price; their PTDF columns are not carried.
5. **`market_day_cet`** is the hour converted to Europe/Brussels and truncated to the date, the same definition as the ENTSO-E v1.x tables.
6. **Nothing else is changed.** All physical rows are kept, in every hour, with their published values; ratios above 1 are kept (clip in the model if needed), and `presolved` is dropped because it is `True` on every source row.

## How to use it in the graph

- **Constraint nodes:** one per row of the hour. Features `ram`, `fmax`, `ram_fmax` (z-score `ram` and `fmax` on train, clip `ram_fmax` to [0, 1] or use `1 − ram_fmax` as tightness). There is no shadow price; JAO does not publish it for these domains.
- **Zone → constraint and constraint → zone edges:** for each row and each zone with a non-null PTDF. Edge weight: the PTDF itself (signed), or, slack-free, the zone-to-zone spread `max_n |ptdf_zone − ptdf_n|` over the zone's neighbours (see `jao_zone_hourly` rule 3 and `epgnn.jao.NEIGHBOURS`). The published PTDFs are zone-to-slack and Germany's is ≈ 0 by construction, so a threshold on `|ptdf_zone|` alone under-connects DE-LU; prefer the spread or no threshold.
- **Zones outside the domain** (the other 26 project zones, and the Core-only zones before June 2022) have no edges to constraint nodes; they exchange messages over interconnectors only, as the proposal's fallback describes.
- **Regime:** feed `regime` (or the date) as an input; the constraint set and PTDF conventions differ between CWE and Core, and `jao_zone_hourly` documents how the summary features compare across the switch.

## Caveats

- **CWE hours are derived**, not published: CWE files give delivery day + period, mapped to UTC via Europe/Brussels. Core hours are published UTC. The two are contiguous at the switch. A handful of market days have 17–22 hours because the source is missing periods.
- **Minimum-RAM floor from 2019.** Checked on the raw data: both RAM and Fmax are present on 100% of physical rows in both regimes (CWE field `RemainingAvailableMargin`, Core field `Ram`; `Fmax` in both), on the same scale (median Fmax ≈ 1,400–1,600 MW). But until 2018 RAM could be as low as 4–5% of Fmax (1st percentile of `ram_fmax` 0.04–0.05), while from 2019 onward the 1st percentile is exactly 0.20 in every year, CWE and Core alike: the EU minimum-RAM (20%) rule applied in CWE from 2019 and inherited by Core. `ram_fmax` therefore has a floor at 0.2 from 2019 but not before; give the model the year (or a `minram_floor` flag) so 2015–2018 is not read as structurally tighter.
- **Extreme ratios:** `ram_fmax` goes up to 299 where `fmax` is a tiny value (1 MW); 0.01% of rows. Clip.
- **No stable line identity across regimes or even across days:** `cnec_key` is per observation, and the same line can appear under different names in CWE and Core. `cne_name` is the best handle for tracking a line over time within a regime.
- **PTDF convention differs between regimes** (different slack / domain construction, see the bundle's `DATASET_GUIDE.md`); zone-to-zone differences are comparable, absolute values less so.
- **Synthetic constraints are excluded**, so the table does not reproduce the full market domain, only its physical part.
- Source-level verification is in the raw bundle (`build.json`, `verification.json`, `DATASET_GUIDE.md`).
