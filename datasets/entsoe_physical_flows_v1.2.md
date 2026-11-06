# ENTSO-E physical cross-border flows

| | |
|---|---|
| Dataset id | `entsoe_physical_flows` |
| Version | **v1.2** |
| Bucket | `gs://cs224w-epgnn-datasets/entsoe_physical_flows/v1.2/` |
| Local file(s) | `datasets/entsoe_physical_flows_v1.2.md`, `datasets/entsoe_physical_flows_v1.2.parquet` |
| Source | ENTSO-E Transparency Platform, `PhysicalFlows`, Jan 2016 – Oct 2026 |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("entsoe_physical_flows", "entsoe_physical_flows.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | first clean build: hourly averaging, both directions kept, zeros kept |
| v1.1 | short gaps patched for inputs, `FlowMW_input` and `row_status`/`fix_*` columns |
| v1.2 | range extended to Oct 2026 (13 months), same schema and rules; overlap identical to v1.1 except ENTSO-E-revised Serbian/Croatian border hours; NL→BE conflicts resolved (see *What changed in v1.2*) |

---

Source: ENTSO-E Transparency Platform, `PhysicalFlows`. Archive: January 2016 – October 2026 (v1.0/v1.1: to September 2025).

## Raw table description

The **PhysicalFlows** table records **measured electricity flow from one reporting area to another during a specified interval**, expressed as average power in **MW**.

For example, **1,000 MW from France to Belgium between 14:00 and 15:00** represents an average flow of 1,000 MW in that reported direction — equivalent to 1,000 MWh over the hour.

The sending and receiving areas describe the border crossing. They do not identify the original generator, final consumer, or a particular commercial transaction. ENTSO-E describes these measurements as average netted physical flows, although the archive contains direction-reporting ambiguities discussed below.

The archive covers **January 2016 – September 2025** and contains:

| Measure | Count |
|---|---|
| Raw records analyzed | 39,652,996 |
| Monthly files | 117 |
| Endpoint area codes | 89 |
| Directed area pairs | 426 |
| Unordered endpoint pairs | 213 |

These **213 endpoint pairs are not necessarily 213 distinct physical interconnectors**: country, control-area and bidding-zone definitions can overlap.

| # | Raw column | Meaning |
|---|---|---|
| 1 | `DateTime(UTC)` | Start of the measurement interval. |
| 2 | `ResolutionCode` | Interval duration. This archive contains hourly and quarter-hour records. |
| 3 | `OutAreaCode` (+ name/type/map fields) | Reporting area electricity flows out of. |
| 4 | `InAreaCode` (+ name/type/map fields) | Reporting area electricity flows into. |
| 5 | `Flow[MW]` | Average measured flow in the reported direction. |
| 6 | `UpdateTime(UTC)` | Recorded update timestamp, not necessarily original publication time. |

## Raw-data findings

### A. Basic data quality and duplicates

All timestamps, durations and flow values were readable. There were:

- No negative or non-finite flow values.
- No duplicated directed-pair/start-time/duration combinations.
- No conflicting or overlapping intervals within a directed pair.
- No self-links or misaligned interval starts.

Missing endpoint metadata is addressed separately below.

**Transformation used for V1.0:** Standardized timestamps and numerical types while preserving original endpoint identifiers. No duplicate removal or selection between conflicting records was necessary.

### B. Mixed time resolutions

| Resolution | Raw records |
|---|---|
| One hour | 24,140,970 |
| 15 minutes | 15,512,026 |

Four consecutive quarter-hour observations provide one covered hour.

Because flow is measured in **MW**, hourly conversion requires averaging. Four quarter-hour observations of 1,000 MW produce an hourly flow of **1,000 MW**, not 4,000 MW.

**Transformation used for V1.0:** Calculated duration-weighted hourly averages, producing **3,877,726 complete hours from quarter-hour records**. Recorded ordinary averaging in `TransformationNote`. Required all 60 minutes for a usable hourly value; partial hours remain blank.

### C. Missing history and incomplete hours

The output contains one row for every observed directed pair across the full study period: **426 directed pairs × 85,464 hours = 36,407,664 rows.**

| Coverage category | Directed-pair hours |
|---|---|
| Complete source coverage | 28,018,696 |
| Entirely missing | 8,388,377 |
| Partial coverage | 591 |

Of the entirely missing hours:

- **256,571** are internal gaps between a series' first and last observations.
- **8,131,806** precede or follow its observed history.

**101 directed pairs** have complete hourly coverage throughout the entire period.

Missing history does not establish whether a physical connection existed, was inactive, or simply lacked reporting.

**Transformation used for V1.0:** Created explicit missing rows and recorded continuous gap ranges in the issue log. Missing and partial-hour values remain blank. No interpolation, forward filling or conversion of missing observations into zero flow was applied.

### D. Opposite-direction reporting needs interpretation

Every directed pair has a reverse-direction series somewhere in the archive. However, their coverage does not necessarily match at every interval.

For **152 unordered endpoint pairs**, both directions have positive values over overlapping reported intervals at least once. For example, one Albania–Greece hour reports **2 MW one way and 21 MW the other**.

This could reflect aggregation, within-interval behavior or reporting conventions. It does not establish that either record is wrong. Also, hourly averages can legitimately be positive in both directions when flow reverses within the hour.

**Transformation used for V1.0:** Preserved both directed series separately and flagged affected pairs for interpretation. Did not subtract the directions, negate one series, or assume a missing reverse observation means zero. Reverse-direction diagnostics are included in `Pair_summary.csv`.

### E. Geographic overlap and missing endpoint metadata

**124 directed pairs** contain at least one endpoint without a bidding-zone designation, including the unnamed Ukrainian endpoint.

The archive includes overlapping country, control-area and bidding-zone series. Some different identifiers produce identical hourly series — for example, Denmark's country and control-area representations for flows to the Netherlands. Adding such series together could double-count the same physical flow.

The endpoint **`10YUA-WEPS-----0`** has no name, area type or map label in this archive. It appears in six directed pairs involving Hungary, Romania and Slovakia.

**Transformation used for V1.0:** Preserved all endpoint codes and available names/types. Recorded geographic and missing-metadata concerns in the issue log. No geographic merging, inferred mapping or automatic graph-node selection was performed.

### F. Zero values, magnitude checks and constant runs

The raw archive contains **17,835,297 zero-valued records**. After hourly conversion, **12,838,065 complete hours report zero flow**.

**Zero physical flow can be valid**, so zeros were not treated as missing or automatically rejected.

Four directed series are zero throughout their complete observed hours:

- IT-South → IT-Brindisi.
- IT-South → IT-Foggia.
- IT-South → IT-Rossano.
- Poland → UA-DobTPP.

The largest raw value is **9,841.5 MW**, reported for France → Italy and France → IT-North.

The audit also identified **65 positive constant-flow runs lasting at least one week**. The longest is **0.02 MW from Finland to Russia for 3,708 hours**. Such a small constant value could reflect measurement behavior or reporting conventions; it is not proof of stale data.

**Transformation used for V1.0:** Preserved zeros and recorded warnings for all-zero series and extended positive constant runs. Applied a broad magnitude screen above the greater of **10,000 MW or ten times the directed pair's positive historical median**. No records triggered it. This screen is not an engineering capacity check and cannot establish that every value is correct. No smoothing, clipping or guessed corrections were applied. Consequently, all **28,018,696 complete hours** pass the numerical checks, while interpretation warnings remain separate.

### G. Actual flows require careful timing in our model

These are **actual measurements**, so tomorrow's physical flows cannot be used to predict tomorrow's day-ahead prices.

ENTSO-E specifies publication as close to real time as possible, and no later than one hour after the operating period (source: eepublicdownloads.entsoe.eu). However, this archive does not establish original publication vintages.

Every recorded update timestamp is after its measurement interval. Update timestamps range from **August 29, 2025, to October 5, 2026**. This does not prove every value was revised, but it prevents us from establishing original availability from `UpdateTime` alone.

**Transformation used for V1.0:** Retained source update timestamps and set **`IsAvailabilityVerified = false` throughout**. No model-specific lag was applied. Historical features must eventually be selected relative to our prediction cutoff, with publication delay allowed for.

**A fixed 24-hour shift is not sufficient for every next-day delivery hour.** For example, today's 23:00 flow is unavailable when predicting tomorrow's prices at noon today.

### H. Flow does not establish capacity or congestion

The table does not contain:

- Available transmission capacity.
- A physical line's engineering limit.
- A congestion indicator.
- Individual circuit identifiers.
- The commercial transaction associated with the flow.

A large observed flow therefore does not, by itself, establish that a connection was congested.

Historical changes in area definitions and reporting coverage also mean that observing a pair later in the archive does not establish its suitability as an edge throughout the earlier years.

**Transformation used for V1.0:** Kept the reported flow measurements without manufacturing capacity, utilization or congestion variables. Preserved separate area histories. No graph topology was backcast or joined to the other datasets.

### I. Validation

The final core contains **36,407,664 unique directed-pair/hour rows and 16 columns**.

Every complete hourly average was independently reconciled against the contributing raw values and durations. The maximum numerical difference was approximately **0.00000000000091 MW**. Raw-record accounting, hourly-key uniqueness and issue-reference ranges also passed validation.

**Transformation used for V1.0:** Produced one standalone Parquet and a verified ZIP package. No price/load join, currency conversion, feature scaling, model fitting or training/validation/test split was performed.

## Transformations implemented

| # | Transformation | Implementation | Related |
|---|---|---|---|
| 1 | Hourly resolution | Duration-weighted average MW. | B |
| 2 | Complete time grid | One row per observed directed pair × UTC hour. | C |
| 3 | Missing and partial hours | Leave flow blank and reference the issue log. | B, C |
| 4 | Direction preservation | Keep both directions separately; no automatic netting. | D |
| 5 | Geographic identity | Preserve endpoint codes and types; flag concerns. | E |
| 6 | Zero handling | Retain explicit zero flow. | F |
| 7 | Constant-run review | Warnings without automatic value removal. | F |
| 8 | Availability tracking | Preserve update provenance; mark availability unverified. | G |
| 9 | Transformation notes | Record ordinary hourly averaging separately from issues. | B |
| 10 | Validation | Reconcile hourly values, source counts and issue references. | I |

## Schema of the transformed core table

| # | Column | Meaning |
|---|---|---|
| 1 | `OutAreaCode` | Original sending-area identifier. |
| 2 | `OutAreaName` | Sending-area name, where available. |
| 3 | `OutAreaTypeCode` | Sending-area geographic role. |
| 4 | `InAreaCode` | Original receiving-area identifier. |
| 5 | `InAreaName` | Receiving-area name, where available. |
| 6 | `InAreaTypeCode` | Receiving-area geographic role. |
| 7 | `DeliveryHourUTC` | Start of the measurement hour in UTC. |
| 8 | `FlowMW` | Hourly average reported directional flow; blank for incomplete hours. |
| 9 | `CoverageMinutes` | Minutes covered by exactly one source interval. |
| 10 | `SourceRecordCount` | Number of contributing source intervals. |
| 11 | `IsValidValue` | Complete hour passing numerical checks; does not certify geographic suitability or direction conventions. |
| 12 | `IsAvailabilityVerified` | Whether original historical availability is established. Currently false throughout. |
| 13 | `SourceUpdateTimeMinUTC` | Earliest contributing recorded update timestamp. |
| 14 | `SourceUpdateTimeMaxUTC` | Latest contributing recorded update timestamp. |
| 15 | `IssueIDs` | References to relevant issue-log entries. |
| 16 | `TransformationNote` | Normal conversion notes, such as quarter-hour averaging. |

## Files in the delivered package

Folder: `Electricity_physical_flows_V1_0_data`

| # | File | Purpose |
|---|---|---|
| 1 | `Core_physical_flows_hourly.parquet` | Standalone hourly physical-flow data. |
| 2 | `Issue_log.csv` | Coverage, direction, geography and availability concerns. |
| 3 | `Pair_summary.csv` | Coverage and diagnostic statistics for each directed pair. |
| 4 | `README.txt` | Definitions, transformation rules, limitations and validation results. |

No flagged-source-record file is included because there were no conflicting records or values triggering the sign/magnitude screen. Interpretation warnings remain in the issue log.

## Critical gaps and issues remaining in V1.0

| # | Remaining issue | Action available to our project |
|---|---|---|
| 1 | **Actual flows are unavailable before their delivery period.** | Use historical flows selected relative to the prediction cutoff, allowing publication delay. Do not use same-delivery-day measurements or assume a 24-hour shift always prevents leakage. |
| 2 | **Original publication vintages are unverified.** | Obtain timestamped historical publications if strict availability verification is required. Otherwise document the possibility of revised historical inputs. |
| 3 | **Switzerland–IT-North and Slovenia–IT-North have substantial early gaps.** | **January 2018 onward has complete coverage in both directions for these pairs.** This is a concrete option for avoiding their earlier gaps. |
| 4 | **IT-Sardinia–IT-SACODC has a 13,105-hour gap from July 2019 to January 2021, plus later gaps.** | Verify this special area's graph role first. If retained, 2023 onward improves coverage substantially, but still requires handling missing and partial hours. |
| 5 | **Russia/Belarus–UA-IPS series have long 2022–2023 gaps.** | Reacquire observations or exclude affected edge-periods. Do not replace the gap with zero based on assumptions about operations. |
| 6 | **Positive values occur in both directions for 152 endpoint pairs.** | Verify aggregation/reporting conventions before deriving net flow. Require appropriate coverage in both directions. |
| 7 | **Geographic levels overlap.** | Choose verified endpoint definitions for the graph. Avoid summing national totals and their component control-area or bidding-zone flows. |
| 8 | **One Ukrainian endpoint lacks metadata.** | Verify `10YUA-WEPS-----0` before mapping its six directed series into the graph. |
| 9 | **All-zero series and long constant runs need interpretation.** | Review their recorded issue ranges against operator information. Do not automatically delete them or infer that the physical connection does not exist. |
| 10 | **Capacity and congestion are absent.** | Use the appropriate capacity/constraint datasets before constructing utilization ratios or congestion labels. |

## What changed in v1.1

1. **Two end hours removed** (30 Sep 2025, 22:00–23:00 UTC), as for prices and load.
2. **Nothing hidden.** Zero flows are normal (power usually goes one way on a link at a time). The week-long flat flows are real cable outages or fixed-setting operation, which the model should see. No conflicts existed.
3. **Short gaps patched for model inputs:** up to 48 hours, from the same hour on an earlier day (29,866 hours). Longer gaps stay empty.
4. **Same labelling as the other tables:** `row_status`, `fix_reason`. All v1.0 values and columns are unchanged.

### What's ready to use

Links between the candidate zones (the 33 core zones plus DE-LU, AT, PL, HR, BG, RS) were checked. Of these, **83 connected pairs** have flows reported in both directions, which is what is needed for net-flow edges.

**Ready now: about 50 links, essentially complete 2016 – Sep 2025.** These cover the whole Nordic and Baltic system, France–Spain–Portugal, France–Belgium–Netherlands–Switzerland, Austria and its neighbours, and Central/South-east Europe (CZ, SK, PL, HU, HR, SI, RO, BG, RS, GR).

**Usable with a small tweak**

- **Germany:** links to DE-AT-LU exist until Sep 2018, and to DE-LU and AT after. Join them into one "Germany" edge, the same tweak as for prices and load. That covers Germany's links to FR, NL, BE, CH, CZ, PL, DK1, DK2 and SE4.
- **Inside Italy:** the links between Italian zones start in 2018, so they're missing the first 2 training years. France, Switzerland and Slovenia → IT-North are 71–85% complete in training.
- **Cables built during the period:** these are real changes, not data gaps. Switch the edge on from its start date:
  - Norway–Germany (NordLink, from 2020–21)
  - Denmark–Netherlands (COBRA, from 2019)
  - Greece–Italy (training years partly missing)
  - Hungary–Slovenia (from 2022)

**Gaps to know about**

- **Sicily and Sardinia have few or no links in this set.** Sicily's link goes through IT-Calabria, which isn't in the zone list, so Sicily would be an isolated node. Add Calabria as a pass-through node, or drop Sicily.
- **Duplicate country and control-area pairs** (e.g. Sweden→Germany vs SE4→DE-LU): use only bidding-zone pairs.
- **Timing:** at noon on D-1, flows are only known up to about 11:00 that morning. Build flow history with that cutoff.

**Bottom line across all three tables (prices, load, flows):** the 33 core zones, plus DE-LU and AT with the Germany tweak, have prices, load and links ready for the whole period. The only exceptions are Ireland (load missing for validation and test) and Sicily (no link unless Calabria is added).

## What changed in v1.2

Same format and rules as v1.1, with the date range extended to **1 Jan 2016 – 31 Oct 2026** (Brussels-time market days; UTC hours 2016-01-01 00:00 to 2026-10-31 22:00). The data is ENTSO-E PhysicalFlows (12.1.G) for 426 directed area pairs, built from a fresh raw download (130 monthly files, 2016_01 … 2026_10). It matches `entsoe_prices` v1.2 and `entsoe_load_forecast` v1.2.

### Same as v1.1

- **Schema:** identical: the same 24 columns in the same order with the same types, and still one Parquet row group per directed pair. The issue log and pair summary keep their v1.1 columns.
- **Values:** for every hour v1.1 covers, v1.2 equals v1.1, except where ENTSO-E has since revised the source (see below). Issue IDs are also renumbered.
- **Rules:** before applying them, the v1.0 rules were re-run on the new raw files over the v1.0 period. They reproduced v1.0 exactly, apart from those revisions.
- **Extreme-value screen:** uses **frozen thresholds** (`ExtremeThresholdMW` from v1.1). Still no hour is screened out.
- **Float noise:** where the recomputed hourly average differs from v1.1 only by floating-point noise (≤ 1e-9 relative, 42,902 values), the v1.1 value is kept.
- **Still no masking for flows:** zeros, week-long constant runs and all-zero pairs stay valid, as in v1.1.

### What's different from v1.1

| # | Change | Rows |
|---|---|---|
| 1 | **13 more months:** Oct 2025 – Oct 2026 | +4,049,130 rows (426 pairs × 9,505 h) |
| 2 | **ENTSO-E revisions in the v1.1 period:** Serbia's borders (AL, BA, BG, ME, MK, RO; Jul 2017 – Jan 2024) have revised values, and 128 hours that were missing are now published and valid. Croatia ↔ Bosnia / Slovenia / Serbia have small revisions (Apr – Aug 2025). | 875 values |
| 3 | **UA-WEPS (Burshtyn island)** is now published with a display name (`UA-BEI`). Its type label is still blank, so it keeps its `MissingAreaMetadata` issue. | 6 pairs (names only) |
| 4 | **IT-Centre-North ↔ IT-SACODC, 15–30 Sep 2025:** these 373 h each way were the end of the data in v1.1. The pair continues, so they now fall inside its lifetime and change from `OUTSIDE_ZONE_LIFETIME` to `MASKED_UNFILLED` (`GAP_GT_48H_NOT_FILLED`). | 746 h |
| 5 | **New: NL → BE conflicts (30 Jun – 7 Aug 2026).** ENTSO-E published two versions of these hours: hourly values in July, then 15-minute values in August. Under the v1.0 rules the hour becomes blank. As with prices, v1.1 rules resolve it with the **latest-updated version** (the 15-minute data). These hours get `row_status` = `VALID_CORRECTED`, `fix_reason` = `CONFLICT_RESOLVED_LATEST_UPDATE`, and both versions in `fix_detail`. The versions differ by 0.16 MW (median) and at most 11.6 MW. Every hour is listed in `conflict_resolutions_v1.2.csv`. | 841 h |
| 6 | **New: agreeing duplicates at two resolutions.** For BE → NL (924 h) and NL → BE (83 h), the hourly record equals the mean of the four 15-minute records. It is collapsed like an identical duplicate (v1.0 rule), counted in `AgreeingDuplicatesRemoved`, and noted in `TransformationNote`. The hour stays `VALID`. | 1,007 h |
| 7 | **Issue IDs renumbered.** New issues are inserted per pair, so `FLOW-xxxxxx` numbers differ from v1.1. The version entries are `FLOW-V12-001` (edge hour dropped) and `FLOW-V12-002` (partial first market day). | — |

What is new in the 13 months, kept as in v1.1:
- 2,360 internal gaps (24,703 h)
- 1,041 partial hours, mostly on the Italian virtual-area links and NL ↔ GB/UK
- 2 new week-long constant runs, NL → GB and NL → UK, 6–15 Sep 2026
- 6 more `BothDirectionsPositive` pairs: SE ↔ DE(TenneT), SE ↔ DE and SE4 ↔ DE-LU
- **Latvia ↔ Russia** stops reporting on 30 Oct 2025; the hours after that are `OUTSIDE_ZONE_LIFETIME`

### Verification against v1.1 (done before pushing)

- Schema identical to v1.1 (24 columns, same order and Arrow types).
- Full grid: 426 directed pairs × 94,967 hours = 40,455,942 rows, same 426 pairs as v1.1, no duplicate (pair, hour) keys, no hour gaps; every pair has 24 hours on the first and last market day; DST days are 23 h (2026-03-29) and 25 h (2025-10-26, 2026-10-25).
- Overlap (2016-01-01 → 2025-09-30, 36,406,812 rows): every v1.1 key present exactly once. `FlowMW` differs on 875 rows, all on Serbia's borders (AL, BA, BG, ME, MK, RO, HR; Jul 2017 – Jan 2024, Aug 2025) and HR ↔ BA / SI (Apr – Jun 2025): 747 revised values (median |Δ| 2.7 MW, max 358 MW) and 128 newly published hours (`MASKED_FILLED` → `VALID`). `SourceUpdateTime*` changed on 1,728 rows on the same borders (republished, values mostly unchanged). 746 IT-Centre-North ↔ IT-SACODC hours (15–30 Sep 2025) moved `OUTSIDE_ZONE_LIFETIME` → `MASKED_UNFILLED`, still masked. The UA-WEPS endpoint gained the display name `UA-BEI` on 6 pairs. Everything else identical apart from renumbered `IssueIDs`.
- Extension (Oct 2025 – Oct 2026): `IsValidValue` consistent with `row_status`; no valid row with a blank value; `FlowMW_input` equals `FlowMW` wherever valid; the 841 `VALID_CORRECTED` hours are all NL → BE, 30 Jun – 7 Aug 2026, as declared. Of the 60 bidding-zone pairs live in the extension, all but the IT-SACODC/SACOAC cable pairs are ≥ 98.5% valid. Per-direction mean flows move a lot year-on-year on Nordic/DE-DK borders (hydro year; e.g. NO2 → NO1 ×4), but 99th-percentile flows stay within interconnector capacity, so these are real dispatch changes, not scaling errors. No constant non-zero runs ≥ 7 days on bidding-zone pairs. All 360 pairs that were live on 30 Sep 2025 continue into the extension; Latvia ↔ Russia ends 30 Oct 2025, and 8 DK/GB-related pairs are missing only the final hour (2026-10-31 22:00 UTC).

### Counts (v1.2)

| row_status | rows |
|---|---|
| VALID | 31,414,052 |
| VALID_CORRECTED | 841 |
| MASKED_FILLED | 37,954 |
| MASKED_UNFILLED | 245,535 |
| OUTSIDE_ZONE_LIFETIME | 8,757,560 |
| **Total** | **40,455,942** (426 pairs × 94,967 hours) |

| fix_reason code | hours |
|---|---|
| `CONFLICT_RESOLVED_LATEST_UPDATE` | 841 |
| `INPUT_FILLED_RULE_B` | 37,954 |
| `GAP_GT_48H_NOT_FILLED` | 243,198 |
| `NO_SOURCE_WITHIN_7D_NOT_FILLED` | 2,337 |

### Companion files (not in the bucket)

The build also produced `issue_log_v1.2.csv` (16,716 issues; new type `ConflictingRecords`), `pair_summary_v1.2.csv` (`ExtremeThresholdMW` = the frozen threshold), `conflict_resolutions_v1.2.csv`, `build_rules.md` and the build scripts (`merge_raw_flows.py`, `f10.py`, `fconf.py`, `fb11.py`, `fverify12.py`, `fcmp12.py`, `fcmp10.py`). Only the parquet and this doc were delivered for v1.2; ask the builder for the rest if needed. There is no CSV copy of the table (≈ 6 GB).

### Open items (unchanged)

- **Timing / leakage:** actual flows for day D cannot be inputs for day D's prices. Build features relative to the D-1 noon cutoff. Original publication vintages are not available.
- **Both directions positive** on 310 directed pairs: decide the netting convention before building edge features.
- **Overlapping geographies:** countries and control areas sit alongside bidding zones; use bidding-zone pairs for graph edges.
- **Deferred decisions:**
  - pair lifetimes
  - the DE-AT-LU split
  - the Italian virtual areas
  - the Ukrainian, Russian and Belarusian endpoints
