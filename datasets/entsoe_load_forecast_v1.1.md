# ENTSO-E day-ahead load forecast

| | |
|---|---|
| Dataset id | `entsoe_load_forecast` |
| Version | **v1.1** |
| Bucket | `gs://cs224w-epgnn-datasets/entsoe_load_forecast/v1.1/` |
| Local file(s) | `datasets/electricity_load_forecasts_hourly_v1.1.parquet`, `datasets/entsoe_load_forecast_v1.1.md` |
| Source | ENTSO-E Transparency Platform, `DayAheadTotalLoadForecast`, Jan 2016 – Sep 2025 |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("entsoe_load_forecast", "electricity_load_forecasts_hourly.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | first clean build: hourly averaging, zero/extreme screening, availability unverified |
| v1.1 | 404 flat-run hours masked, short gaps patched for inputs, `LoadForecastMW_input` and `row_status`/`fix_*` columns |

---

Source: ENTSO-E Transparency Platform, `DayAheadTotalLoadForecast`. Archive: January 2016 – September 2025.

## Raw table description

The **DayAheadTotalLoadForecast** table records the system operator's **forecast of electricity demand in a specified area during a delivery interval**, measured in **MW**.

For example, **10,000 MW for France on January 1, from 14:00 to 15:00**, means expected average demand of 10,000 MW during that hour — equivalent to **10,000 MWh of expected consumption over the hour**.

This is forecast demand. It does not report actual consumption, auction trading volume, or how much electricity will come from domestic generation versus imports.

The areas include **bidding zones, control areas and countries**. These can overlap geographically and should not automatically become separate nodes in our graph.

The uploaded archive covers **January 2016 – September 2025**, with **9,028,424 records, 117 monthly files and 68 area codes**. All records were analyzed.

| # | Raw column | Meaning |
|---|---|---|
| 1 | `DateTime(UTC)` | Start of the delivery interval, not forecast publication time. |
| 2 | `ResolutionCode` | Interval duration: 15 minutes, 30 minutes or one hour. |
| 3 | `AreaCode` | Unique area identifier. |
| 4 | `AreaDisplayName` | Area name. |
| 5 | `AreaTypeCode` | Bidding zone (`BZN`), control area (`CTA`), country (`CTY`), or a combination. |
| 6 | `AreaMapCode` | Short area label. |
| 7 | `TotalLoad[MW]` | Forecast average demand during the interval. |
| 8 | `UpdateTime(UTC)` | Recorded update timestamp; original publication time is not established. |

## Raw-data findings

### A. Basic data quality and duplicates

All timestamps, durations and numerical values were readable. There were no missing raw fields, negative or non-finite load values, duplicated area/start-time/duration combinations, or overlapping intervals within an area.

Area names and area types remained consistent within each area code. The UK country series changed its shorthand map label between `GB` and `UK`.

**Transformation used for V1.0:** Preserved original area codes, names and area types. Recorded the UK map-label change in the issue log. No duplicate removal or conflicting-record selection was necessary.

### B. Mixed time resolutions

| Resolution | Raw records |
|---|---|
| 15 minutes | 4,853,818 |
| 30 minutes | 578,878 |
| One hour | 3,595,728 |

Coverage was measured using each record's duration. Four consecutive quarter-hour records provide one covered hour.

Because load is measured in **MW**, the values must be averaged when converting to hourly resolution. Four quarter-hour forecasts of 1,000 MW produce an hourly forecast of **1,000 MW**, not 4,000 MW.

**Transformation used for V1.0:** Calculated duration-weighted hourly averages:

- **1,213,454 complete hours** were constructed from quarter-hour records.
- **289,439 complete hours** were constructed from half-hour records.

These normal conversions are documented in `TransformationNote`.

Complete hourly values require all 60 minutes. Two hours were incomplete: **DE-AT-LU and Luxembourg, both on September 15, 2018, at 22:00 UTC**. Their hourly forecasts remain blank.

### C. Missing history and internal gaps

The complete hourly grid contains **5,811,552 rows**: 68 areas × 85,464 hours.

| Coverage category | Hours |
|---|---|
| Complete source coverage | 5,098,621 |
| Entirely missing | 712,929 |
| Partial coverage | 2 |

Of the entirely missing hours, **242,675 fall between an area's first and last observations**. The remainder occur before or after its observed history.

These counts span overlapping geographic areas; they are not independent measures of missing demand across Europe.

**Transformation used for V1.0:** Created one row per area and UTC hour across the full period. Missing and partial-hour forecasts remain blank. Recorded continuous missing ranges in `Issue_log.csv`. No interpolation, forward filling or backcasting was applied.

### D. Geographic overlap and compatibility with prices

The load table contains **68 area codes**, including **14 without a bidding-zone designation**.

Some series describe overlapping geographic levels:

- Germany's four control-area forecasts sum to its national forecast within numerical precision.
- Sweden's bidding-zone forecasts sum exactly to its national forecast in all comparable hours.

Adding national totals to their component areas would double-count demand.

Of the **61 area codes in the separate price dataset**, **49 have an exact match** in this load dataset. The other 12 are special Italian areas and `NO2NSL`.

**Transformation used for V1.0:** Retained all 68 load areas and their `AreaTypeCode`. Geographic concerns are recorded in the issue log. **The delivered load dataset remains separate from prices.** No price columns, geographic substitutions or reconstructed zone forecasts were added. A future join can use `AreaCode` and `DeliveryHourUTC` after the appropriate areas are selected.

### E. Zero and extreme forecast values

There are **3,606 zero-valued source intervals**, affecting **3,360 hourly rows**. Some occur in long runs — for example, **640 consecutive zero hours for Moldova**.

The most extreme value is **14,501,460 MW for Georgia on August 21, 2020, at 21:00 UTC**. The archive does not establish the intended correction.

A broad diagnostic screen also identified **1,322 hours** containing a positive source value below **0.1×**, or above **5×**, that area's positive historical median. These thresholds identify values for investigation; they do not prove that every flagged value is incorrect.

**Transformation used for V1.0:** Preserved complete-hour numerical values, but set `IsValidValue = false` for hours containing zero or screened extreme intervals. Saved the original suspicious records in `Flagged_source_records.csv`, including their source filenames and row numbers. No clipping, rescaling or guessed corrections were applied.

Overall, **4,681 complete hours fail the value checks**, leaving **5,093,940 hours passing them**.

The median screen uses the full history for retrospective auditing. Any thresholds learned for model preprocessing must instead be fitted using training data only.

### F. Unusually flat forecasts and abrupt changes

The audit examined constant forecasts and changes between adjacent hours.

There are **70 constant-value runs lasting at least 24 hours**, including zero runs. One positive example is a Czech forecast of **8,715 MW for 83 consecutive hours**, beginning February 27, 2021, at 12:00 UTC.

A flat forecast may be legitimate or stale. A sharp change alone also does not prove a data error.

**Transformation used for V1.0:** Recorded positive constant runs lasting at least 24 hours as issue-log warnings. Their values remain present and are not automatically invalidated. No smoothing or blanket removal of demand peaks was performed. Additional diagnostic exports were omitted from the simplified package.

### G. Original forecast availability is unverified

**All 9,028,424 recorded update timestamps are later than their delivery timestamps.** They range from **September 26, 2025, to September 30, 2026**.

Consequently, `UpdateTime(UTC)` cannot establish when the original forecast was available. This does not prove every value was revised after delivery; the timestamps could also reflect later processing or republication.

The unresolved question for our project is whether the archived value matches the forecast available before our prediction cutoff.

**Transformation used for V1.0:** Retained the earliest and latest update timestamps contributing to each hour. Set **`IsAvailabilityVerified = false` throughout**, with corresponding issue-log entries.

Numerical validity and historical availability are therefore separate: a forecast can pass value checks without being verified as an eligible pre-auction feature.

### H. Historical boundaries and uneven endings

Several series have important historical limits:

- **DE-AT-LU:** observed load coverage ends at September 30, 2018, 00:00 UTC.
- **DE-LU:** begins October 1, 2018, 00:00 UTC, followed by substantial late-2018 gaps.
- **Ukrainian load series:** stop at February 26, 2022, 22:00 UTC.
- **GB:** stops in June 2021. The UK country series uses a different area code and has substantial gaps and suspect low values.

These boundaries do not always align with the corresponding price histories.

**Transformation used for V1.0:** Preserved separate area histories and explicit missing periods. Did not splice DE-AT-LU into DE-LU, substitute UK for GB, or extend Ukrainian forecasts.

### I. Validation

The transformed table contains **5,811,552 unique area-hour rows and 13 columns**.

Every hourly average was independently reconciled against the original values and interval durations. The maximum floating-point difference was below **0.000000001 MW**. Source-record counts and issue-reference ranges also passed validation.

**Transformation used for V1.0:** Produced one standalone core Parquet. No price join, currency conversion, scaling, graph-node selection, model fitting or training/validation/test split is included.

## Transformations implemented

| # | Transformation | Implementation | Related |
|---|---|---|---|
| 1 | Hourly resolution | Duration-weighted average MW; require 60 minutes of coverage. | B |
| 2 | Complete hourly grid | One row per source area × UTC hour across the full period. | C |
| 3 | Missing and partial hours | Leave forecast blank and reference the issue log. | B, C |
| 4 | Zero/extreme screening | Preserve complete-hour values but mark them invalid pending review. | E |
| 5 | Flat-forecast warnings | Flag extended constant positive forecasts without automatically removing them. | F |
| 6 | Geographic identity | Preserve original codes and area types; log concerns. | D, H |
| 7 | Forecast availability | Preserve update provenance and mark availability unverified. | G |
| 8 | Transformation notes | Document ordinary averaging separately from issues. | B |
| 9 | Validation | Reconcile hourly values, record counts and issue references. | I |

## Schema of the transformed core table

| # | Column | Meaning |
|---|---|---|
| 1 | `AreaCode` | Original area identifier. |
| 2 | `AreaName` | Original area name. |
| 3 | `AreaTypeCode` | Bidding zone, control area, country, or combination. |
| 4 | `DeliveryHourUTC` | Start of the delivery hour in UTC. |
| 5 | `LoadForecastMW` | Hourly average forecast; blank for missing/partial hours. Suspicious complete values remain present. |
| 6 | `CoverageMinutes` | Minutes covered by exactly one source interval, from 0 to 60. |
| 7 | `SourceRecordCount` | Number of contributing raw records. |
| 8 | `IsValidValue` | Complete hour passing zero/extreme screening. Does not certify geography or historical availability. |
| 9 | `IsAvailabilityVerified` | Whether availability before the prediction cutoff is established. Currently false throughout. |
| 10 | `SourceUpdateTimeMinUTC` | Earliest update timestamp among contributing records. |
| 11 | `SourceUpdateTimeMaxUTC` | Latest update timestamp among contributing records. |
| 12 | `IssueIDs` | References to relevant issue-log entries. |
| 13 | `TransformationNote` | Normal transformations, such as averaging quarter-hour forecasts. |

## Files in the delivered package

Folder: `Electricity_load_forecasts_V1_0_data`

| # | File | Purpose |
|---|---|---|
| 1 | `Core_load_forecast_hourly.parquet` | Standalone transformed hourly forecasts. |
| 2 | `Issue_log.csv` | Missing periods, suspicious patterns, geographic concerns and availability limitations. |
| 3 | `Area_summary.csv` | Coverage and quality counts for each source area. |
| 4 | `Flagged_source_records.csv` | Original zero/extreme records for investigation. |
| 5 | `README.txt` | Column definitions, transformation rules and limitations. |

## Critical gaps and issues remaining in V1.0

| # | Remaining issue | Action available to our project |
|---|---|---|
| 1 | **Original forecast availability is unverified for every area.** | Obtain original published forecasts or a timestamped archive. Otherwise, describe experiments as using potentially revised historical forecasts. A later starting date does not resolve this. |
| 2 | **DE-LU has 840 missing hours in October–December 2018.** | Consider **January 2019** as a cleaner starting point. There are still **50 later missing hours**: 48 in 2022 and one each in 2023 and 2024. |
| 3 | **IE(SEM) becomes extremely sparse after 2020.** | Reacquire all-island forecasts or exclude affected periods/features. Later starts do not help: only **96, 264, 336 and 192 hours** are present in 2022, 2023, 2024 and January–September 2025, respectively. Do not substitute Republic of Ireland demand alone. |
| 4 | **Albania has extensive historical gaps.** | **2024 onward** has complete interval coverage, although 13 hours in 2024 are flagged as extreme. Earlier use requires reacquisition or missing-feature handling. |
| 5 | **Ukrainian load histories stop in February 2022.** | Obtain later forecasts or exclude affected node-periods. Do not carry the final value forward. |
| 6 | **Twelve price-area codes have no exact load-area match.** | Resolve their special/interconnector roles before a future join. Define appropriate features or exclude those nodes rather than copying national demand into them. |
| 7 | **Other areas have smaller internal gaps.** | Use `Issue_log.csv` for exact ranges and `Area_summary.csv` for coverage totals. Reacquire records, mask missing features or select usable periods. |
| 8 | **Zeros, extremes and flat runs remain unresolved.** | Review original flagged records. Apply `IsValidValue` as a provisional quality filter and assess flat-run warnings separately. Correct values only with supporting evidence. |
| 9 | **Countries, control areas and bidding zones overlap.** | Select appropriate geographic identities for graph nodes. Avoid counting national totals and component areas together. |
| 10 | **Prediction cutoff and daylight-saving handling remain undecided.** | Keep UTC hourly data, then define the local prediction cutoff and treatment of 23/25-hour delivery days during model preparation. |

## What changed in v1.1

1. **Two end hours removed** (30 Sep 2025, 22:00–23:00 UTC), as for prices.
2. **404 hours of flat placeholder forecasts hidden:** runs where the forecast sat at one exact value for 24+ hours (Albania, Bosnia, Czechia, Austria, Georgia). Real load forecasts change every hour. The values are kept but marked not usable.
3. **Zero and extreme values stay hidden**, as v1.0 already marked them. Nothing new there.
4. **Short gaps patched for model inputs:** up to 48 hours, from the same hour on an earlier day (17,035 hours). Longer gaps stay empty.
5. **Everything is labelled the same way as prices:** `row_status`, `fix_reason`, `fix_detail`. Old columns are untouched.

One difference from prices: a hidden value stays in `LoadForecastMW`, so **always filter with `IsValidValue`, or use `LoadForecastMW_input`**.

### What's ready to use

Train 2016–21, validation 2022, test 2023–Sep 2025.

**Ready now: the same 33 zones that are ready for prices.** All have at least 98% of hours in every period:

- **Western/Central:** FR, BE, NL, CH, CZ, SK, HU, SI
- **Iberia:** ES, PT
- **Nordic:** NO1–NO5, SE1–SE4, DK1, DK2, FI
- **Baltic:** EE, LV, LT
- **South-east:** GR, RO
- **Italy:** North, Centre-North, Centre-South, South, Sicily, Sardinia

So prices and load line up for the core set.

**Better than prices here**

- **Austria, Poland, Croatia, Bulgaria, Serbia:** load is complete from 2016, even where prices start later or have gaps. Load won't be the limiting factor for these.

**Usable with a small tweak**

- **Germany:** DE-LU load is complete from Oct 2018. Before that, use the combined DE-AT-LU load (about 99% complete), matching the price tweak.

**Problem: Ireland**

- IE(SEM) load is only about 1% present in 2022 and 3% in 2023–25, effectively missing for validation and test. The "Ireland (IE)" country series doesn't fix it: it excludes Northern Ireland and is only 58% complete in 2022. **Drop Ireland as a node,** or find another source.

**Not suitable**

- **GB and the UK:** end in 2021.
- **Ukrainian areas:** end in Feb 2022.
- **Cyprus, Moldova, Georgia:** outside the market area, and very incomplete.
- **Albania, Kosovo, North Macedonia, Montenegro:** load exists, but their prices only start in 2023–24.

**Don't double-count**

This table also has whole countries and control areas next to bidding zones. Examples:

- **Germany:** the country, plus its four grid-operator areas, overlap DE-LU.
- **Norway, Sweden, Denmark and Italy:** the country totals overlap their zones.

For graph nodes, use only the bidding-zone rows that match the price zones.

So the **33 core zones, plus DE-LU and AT**, have both prices and load ready for the full period. Ireland is the one zone that drops out because of load.
