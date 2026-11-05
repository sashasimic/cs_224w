# ENTSO-E day-ahead prices

| | |
|---|---|
| Dataset id | `entsoe_prices` |
| Version | **v1.1** |
| Bucket | `gs://cs224w-epgnn-datasets/entsoe_prices/v1.1/` |
| Local file(s) | `datasets/entsoe_prices_v1.1.parquet`, `datasets/entsoe_prices_v1.1.md` |
| Source | ENTSO-E Transparency Platform, day-ahead prices, Jan 2016 – Sep 2025 |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("entsoe_prices", "entsoe_prices.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | first clean build: auction selection, currency conversion, hourly grid, gaps flagged (sections 1–5 below) |
| v1.1 | 74 conflicts resolved, 4,115 placeholder prices masked, short gaps patched for inputs, `row_status`/`fix_*` columns (see *What changed in v1.1*) |

---

Source: ENTSO-E Transparency Platform, day-ahead prices. Archive: January 2016 – September 2025.

## 1. What the raw `energyprices` table describes

The table records **the electricity auction price for delivery in a particular bidding zone during a specified interval**.

For example, a day-ahead row showing **€100/MWh for France on January 1, from 14:00 to 15:00**, means that the previous day's auction established a wholesale price of €100 per megawatt-hour for electricity delivered in the French bidding zone during that hour.

**"France" identifies the delivery zone.** It does not mean the electricity was necessarily generated in France or that the companies buying and selling it are French. Supply can come from local generation or connected zones, subject to transmission limits.

Buyers can include electricity suppliers serving households and businesses, large industrial consumers, traders and storage operators. Sellers can include generators, traders and storage operators. The price is the zone's **auction clearing price for accepted purchases and sales**, rather than the price of an identified shipment.

The row does **not** tell us:

- How much electricity was bought or sold.
- Which buyers, sellers or generators participated.
- How much electricity came from local generation versus imports.
- How much local generation was consumed within the zone versus exported.
- Who ultimately consumed the electricity or whether it was subsequently traded onward.

For our project, the table provides **a bidding zone, a delivery interval and an auction price**. Generation, traded quantities and cross-border flows require other datasets.

### Raw columns

| # | Field | Meaning |
|---|---|---|
| 1 | `DateTime(UTC)` | Start of the electricity delivery interval. |
| 2 | `ResolutionCode` | Interval duration: 15, 30 or 60 minutes. |
| 3 | `AreaCode` | Identifier of the area to which the price applies. |
| 4 | `AreaDisplayName` / `AreaMapCode` | Readable area name and short label. |
| 5 | `AreaTypeCode` | Area classification, generally bidding zone (`BZN`). |
| 6 | `ContractType` | Market category, such as day-ahead or intraday. |
| 7 | `Sequence` | Distinguishes auction series where supplied. |
| 8 | `Price[Currency/MWh]` | Auction price per megawatt-hour. |
| 9 | `Currency` | Currency in which that price is reported. |
| 10 | `UpdateTime(UTC)` | When the stored record was updated — not necessarily when the auction result was first published. |
| 11 | `InstanceCode` | Source-system record identifier. |

## 2. Raw-data findings

### A. Archive contents and consistency

| # | Check | Finding |
|---|---|---|
| 1 | Monthly files | All **117 months** present |
| 2 | Day-ahead records | **4,864,725** |
| 3 | Intraday records | **6,256**, all belonging to Spain |
| 4 | Distinct area codes | **61**, including historical and special areas |
| 5 | Time intervals | 15, 30 and 60 minutes |
| 6 | Currencies | EUR, GBP, PLN, RON, BGN, UAH |
| 7 | Invalid timestamps or nonnumeric prices | None found |
| 8 | Blank raw price cells | None found |
| 9 | Exact duplicate full rows | None found |

The Spanish intraday rows describe separate trading after the day-ahead auction.

**Transformation used for V1.0 — A:** Parsed timestamps and prices, normalized auction-sequence whitespace, and retained only day-ahead records. Preserved original area identifiers and names. The original archive, checksum and builder support reproducibility; conflicting records also retain source-file and row references. Traded quantities were not added because they are absent from the source.

### B. All months are present, but some zones have missing data

Missing prices often mean **entire rows are absent**, rather than price fields being blank.

Coverage was measured using actual interval durations. One 60-minute record or four consecutive 15-minute records both provide one complete hour.

| # | Zone | Missing or limited raw coverage |
|---|---|---|
| 1 | Poland | **2,755 missing hours**: 2,469 in 2016 and 286 in 2017. Continuous from 2018 onward. |
| 2 | Bulgaria | Begins October 31, 2016 UTC; also has **360 missing hours** within its 2016 coverage. |
| 3 | Croatia | Begins October 17, 2017 UTC. |
| 4 | Serbia | Begins December 22, 2016 UTC. |
| 5 | Ireland / SEM | **197 missing hours**, including entire 25-hour autumn clock-change days in 2019–2023. |
| 6 | Romania | **48 missing hours**, across two days in 2017–2018. |
| 7 | Slovakia | **72 missing hours**, across three days in 2016. |
| 8 | Great Britain | Ends December 2020; no prices for the proposed validation or test periods. |
| 9 | Albania, Montenegro, North Macedonia | Begin in April–May 2023. |
| 10 | Kosovo | Begins January 31, 2024 UTC. |
| 11 | Ukraine IPS | Begins September 2020; **7,512 missing hours** within its coverage, mostly in 2022. |

After selecting the main auction, the 12 Core-market zones checked — **AT, BE, HR, CZ, FR, DE-LU, HU, NL, PL, RO, SK and SI** — have uninterrupted, unambiguous price coverage from the October 2018 split through September 2025.

**Transformation used for V1.0 — B:** Created a complete UTC hourly grid for every area code. Missing hours remain as rows with blank prices and `IsValidTarget = false`. Distinguished internal gaps, periods before or after observed history, and Austrian periods excluded by auction selection. No missing electricity prices were interpolated or forward-filled. Observed history boundaries were not treated as proof of a zone's legal lifetime.

### C. Austria and Germany contain two different day-ahead auctions

Both auctions establish prices for tomorrow's delivery, but they occur at different times and can produce different prices. For the Austrian and Germany–Luxembourg series:

| Sequence | Meaning |
|---|---|
| **1** | Main European coupled day-ahead auction. |
| **2** | Separate, earlier EXAA auction, around 10:15. |

**The numbers identify auctions — not their chronological order or revision versions.**

For example, the earlier auction might establish €95/MWh, while the main auction establishes €100/MWh for the same delivery hour. Both can be correct.

Before October 2018, Austrian records contain only sequence 2. The earlier joint-zone prices appear under Germany–Austria–Luxembourg. Most other zones have a blank `Sequence`.

**Transformation used for V1.0 — C:** Selected **sequence 1 for Austria and Germany–Luxembourg**, retaining blank-sequence day-ahead records elsewhere. Excluded **578,884 sequence-2 records** from the core table. The two auctions were never averaged, and sequence 2 was not used to fill missing main-auction prices. The original archive retains the excluded records.

### D. Bidding-zone definitions change during the study

Before October 2018, Germany, Austria and Luxembourg shared a bidding zone. Afterward, Austria and Germany–Luxembourg have separate prices. The archive reflects this:

- `DE_AT_LU` contains the earlier joint-zone prices.
- `DE_LU` and Austrian sequence-1 records begin at the split.

Modern Austria and Germany–Luxembourg cannot therefore be treated as independently priced zones throughout 2016–2025. Several Italian codes also start or disappear, while special border or interconnector areas require separate interpretation.

**Transformation used for V1.0 — D:** Retained **all 61 original area codes separately**. No historical prices were assigned to modern replacement zones, and no zones were merged. Recorded concerns about historical splits, special Italian areas, `NO2NSL` and missing Ukrainian metadata. Authoritative zone lifetimes and final graph-node eligibility remain unresolved.

### E. Some repeated records disagree on the price

After distinguishing zone, delivery time, interval length, currency and auction sequence:

- **456 raw day-ahead interval groups** contain repeated records.
- **382 groups** agree on price.
- **74 groups** disagree.

| Area | Conflicting hours |
|---|---|
| North Macedonia | 72 |
| Albania | 1 |
| Kosovo | 1 |

For example, Albania and Kosovo each contain both **€3.40/MWh and €13.40/MWh for April 1, 2024, at 13:00 UTC**.

Later update timestamps suggest corrections, but the export does not explicitly establish which record supersedes another.

**Transformation used for V1.0 — E:** After auction selection, removed **190 redundant records** whose interval, currency, sequence and price agreed. The other 192 agreeing duplicate groups belonged to the already-excluded sequence-2 series. All **74 conflicting hours** retain blank prices, `IsValidTarget = false` and issue references. No latest-update preference or averaging of disputed values was applied.

### F. Some zones change reporting currency

| Zone | Observed currencies |
|---|---|
| Bulgaria | BGN, then EUR from January 2022 |
| Romania | RON, then EUR from June 2021 |
| Poland | Several EUR/PLN switches, then EUR from November 2019 |
| Great Britain | GBP |
| Ukrainian series | UAH |

Poland has **nine consecutive currency runs**, so one currency assumption for its entire history would be wrong. Ignoring currency would create artificial price jumps and meaningless cross-border differences.

**Transformation used for V1.0 — F:** Converted **209,752 non-euro hourly prices** to EUR/MWh using each record's reported currency. Preserved `PriceOriginal`, `Currency` and `FXRateDate`; EUR prices remain unchanged.

For local delivery date *D*, the conversion uses the latest FX rate dated **before *D−1***, the assumed day-ahead auction date. This is the documented preprocessing convention.

ECB rates supply GBP, PLN, RON and BGN conversions. NBU official rates retrieved through PrivatBank supply UAH conversions. All required conversions have rates. Cross-source comparison discrepancies were retained as approximate notes rather than silently corrected.

### G. Interval lengths and delivery calendars need consistent treatment

Some Italian, Norwegian, Serbian, Belgian and French records use 15-minute intervals before October 2025. Ireland also has half-hourly records.

Many earlier quarter-hour records repeat the same hourly price. The final two UTC hours of the archive correspond to October 1 in continental European local time, when quarter-hour prices can vary within an hour.

Local delivery days also have **23 or 25 hours** at daylight-saving changes. A model producing "24 prices for tomorrow" therefore needs an explicit policy for those days.

**Transformation used for V1.0 — G:** Produced strictly hourly rows using complete, non-overlapping intervals from the selected auction. Combined **92,059 hours from quarter-hour records** and **23,950 hours from half-hour records**. For four complete quarter-hours:

$$p_{\text{hour}} = \frac{p_1 + p_2 + p_3 + p_4}{4}$$

`TransformationNote` identifies these aggregations. `CoverageMinutes` records usable interval coverage. UTC timestamps preserve distinct hours through daylight-saving changes. Local forecast-day grouping and the model's 23/25-hour output policy have **not** been implemented.

### H. Some price patterns need verification

The raw day-ahead records contain **50,207 negative-price rows** and **12,333 zero-price rows**, across different auctions and interval lengths.

Negative and zero prices can be valid. However, these patterns deserve checking:

- **Ireland:** prolonged zero-price runs around the 2018 market transition and in 2019.
- **North Macedonia:** several full zero-price days, in addition to disputed records.
- **`IT-SACODC`:** long plateaus, including 657 hours at €3,000/MWh and 378 hours at €1,750/MWh.
- **Unnamed Ukrainian area:** 48 zero-price hours immediately before its history ends.

Ireland's earlier price definition also requires review against the later I-SEM day-ahead market.

**Transformation used for V1.0 — H:** Preserved negative prices, zeros and spikes without clipping. V1.0 contains **31,424 negative-price hours** and **10,209 zero-price hours** among its populated prices. Flagged selected long zero runs, special-area high plateaus and the legacy Irish market. These warnings do not automatically invalidate a technically complete price; their underlying causes have not yet been independently resolved.

### I. Original publication times are not preserved

`DateTime(UTC)` identifies **when electricity is delivered**. `UpdateTime(UTC)` identifies **when the stored record was updated**. Updates range from October 2024 through October 2026, including updates to electricity prices from 2016.

The export cannot establish exactly which historical price version was available at each prediction cutoff.

**Transformation used for V1.0 — I:** Used delivery timestamps to construct hourly rows. Update timestamps were not interpreted as original publication times or used to resolve conflicting prices. The original archive preserves that metadata, while the compact core table omits it. V1.0 uses the downloaded historical price versions and documents this limitation; original publication histories were not reconstructed.

## 3. Transformations implemented in V1.0

| # | Step | V1.0 implementation or remaining work | Related |
|---|---|---|---|
| 1 | Parse consistently | Parsed the archive; preserved reproducibility through the original source, checksum and builder. | A |
| 2 | Retain area codes | Included all 61 codes separately. Final node eligibility and historical mappings remain open. | B, D |
| 3 | Select the auction | Day-ahead only; sequence 1 for AT/DE-LU and blank sequences elsewhere. | A, C |
| 4 | Resolve repeated records | Removed agreeing duplicates; left all 74 disputed hours unavailable. | E |
| 5 | Standardize currencies | Converted non-EUR prices and recorded the historical FX date used. | F |
| 6 | Produce hourly prices | Aggregated complete shorter intervals and added `TransformationNote`. | G |
| 7 | Represent gaps | Added explicit blank hourly rows with blocking issue references. No electricity-price imputation. | B, D |
| 8 | Flag suspicious patterns | Preserved values and recorded review warnings; external verification remains open. | H |
| 9 | Align delivery days | UTC hourly alignment completed. Local forecast-day grouping and DST output policy remain open. | G |
| 10 | Track availability limitations | Documented the historical-version limitation. Original publication history remains unavailable. | I |
| 11 | Build historical windows | **Not implemented.** Usable seven-day histories and causal handling of missing inputs remain necessary. | B, I |
| 12 | Construct spread targets | **Not implemented.** Requires graph connections and period-appropriate zone definitions. | C, D, F |
| 13 | Prepare evaluation | Chronological splits, training-only scaling and comparable model samples remain to be implemented. | — |

## 4. Schema of the transformed core table

**File:** `Core_prices_hourly.csv`

**One row represents one original area code × one UTC delivery hour.** The unique key is `AreaCode` + `DeliveryHourUTC`.

| # | Column | Description |
|---|---|---|
| 1 | `AreaCode` | Original ENTSO-E area identifier. Historical and special areas retain their own codes. |
| 2 | `AreaName` | Original readable area name. Blank where the source does not provide one. |
| 3 | `DeliveryHourUTC` | Start of the one-hour delivery interval in UTC. E.g. `2020-01-01T14:00:00Z` represents 14:00–15:00 UTC. |
| 4 | `PriceOriginal` | Hourly price in the original currency per MWh. For shorter intervals, the duration-weighted hourly average. Blank for missing, incomplete or conflicting hours. |
| 5 | `Currency` | Currency of `PriceOriginal`. Blank when no source record exists; not inferred across gaps. |
| 6 | `FXRateDate` | Historical exchange-rate date used to calculate `PriceEUR`. Blank for prices already in EUR and for hours without a usable price. |
| 7 | `PriceEUR` | Hourly price standardized to EUR/MWh. **The common price measure for comparisons and model targets.** |
| 8 | `CoverageMinutes` | Number of usable, non-overlapping source minutes supporting the hour. **60 means complete coverage.** Disputed intervals do not contribute usable minutes. |
| 9 | `IsValidTarget` | `true` when the hourly price is complete, unambiguous and convertible; otherwise `false`. Does not independently approve the area's geographic or historical suitability. |
| 10 | `IssueIDs` | References to concerns affecting the row, separated by semicolons; blank when none apply. |
| 11 | `TransformationNote` | Normal processing applied, e.g. `Averaged four 15-minute prices`, `Averaged two 30-minute prices`, `Collapsed agreeing duplicate intervals`. Blank when none applies. |

**Missing hours remain in the table.** Their price fields are blank, and `IsValidTarget` is false. Zero is preserved as a legitimate numeric price and is never used to represent missing data.

## 5. Critical gaps and issues remaining in V1.0

V1.0 has **1,227,854 hours without selected price records**, plus **74 conflicting hours**. The first count includes periods outside observed zone histories and auction exclusions — not just missing observations within active markets.

The remaining issues should guide **which zones and dates we use**:

| # | Issue | Affected zones / dates | Action or decision for our project |
|---|---|---|---|
| 1 | Early coverage gaps and the German–Austrian split | Poland has gaps in 2016–2017; Bulgaria, Croatia and Serbia begin during 2016–2017. Separate main-auction prices for AT and DE-LU begin October 2018. | **Consider starting the primary experiment on October 1, 2018.** Starting January 2018 resolves several early coverage problems but still requires handling the joint DE-AT-LU zone until October. |
| 2 | A continuous core group is available | AT, BE, HR, CZ, FR, DE-LU, HU, NL, PL, RO, SK and SI have uninterrupted, unambiguous price coverage after the October 2018 split. | **Use these 12 zones as the initial candidate panel**, then expand after checking weather, forecasts and transmission data. This establishes price coverage only. |
| 3 | Great Britain has no post-2020 prices | GB ends December 2020, before the proposed validation and test periods. | Obtain another price source or **exclude GB from the main experiment**. |
| 4 | Some zones appear only in the test period | Albania and Montenegro begin April 2023; North Macedonia begins May 2023; Kosovo begins early 2024. | **Exclude them from a fixed-node experiment requiring training history**, obtain earlier data where the market existed, or evaluate them separately as previously unseen nodes. |
| 5 | Ukraine has a major validation-period gap | Ukraine IPS has 7,512 internal missing hours, mostly in 2022. The unnamed Ukrainian series ends February 2022 and lacks area metadata. | **Exclude Ukrainian areas from the initial continuous panel**, or obtain replacement data and verify identities before inclusion. |
| 6 | Smaller internal gaps remain | Ireland has 197 missing hours; Romania and Slovakia have earlier gaps; other series also have isolated gaps. | **Mask missing target hours.** If a gap falls inside a 168-hour input window, exclude that example or use documented causal imputation with missingness indicators. |
| 7 | 74 prices remain disputed | North Macedonia: 72 hours; Albania and Kosovo: one hour each. | **Keep these targets excluded**, as already marked. Verify against the originating exchange if retaining these zones makes recovery worthwhile. |
| 8 | Some areas may not be suitable graph nodes | Joint DE-AT-LU, several Italian areas, `NO2NSL`, and the unnamed Ukrainian code. | **Choose ordinary geographic bidding zones for the initial graph.** Review special codes individually before adding them; do not automatically connect or merge them based on their names. |
| 9 | Some price patterns and definitions remain uncertain | Legacy Irish prices, selected full-day zero runs, and high plateaus in `IT-SACODC`. | An October 2018 start avoids the legacy Irish period. **Exclude `IT-SACODC` initially unless its role is established.** Verify flagged zero runs in retained zones or compare results with and without them. Preserve ordinary negative and zero prices. |
| 10 | Currency conversion introduces uncertainty | Historical GBP, PLN, RON, BGN and UAH prices; larger discrepancies occur for some UAH dates. | **Use V1.0's conversion convention consistently.** If retaining UAH, test sensitivity to the FX source. An October 2018 start does not eliminate all currency conversions. |
| 11 | Daylight-saving days do not always have 24 hours | Local delivery days with 23 or 25 hours. | **Choose a local-day output policy:** variable-length targets, padded outputs with masks, or exclusion of those days with reported counts. Keep the UTC timestamps. |
| 12 | Original publication versions are unavailable | Historical price and FX records may include later revisions. | **Document use of the downloaded historical versions.** Obtain archived releases if strict historical availability is required. Check forecast-feature publication timing separately. |
| 13 | The model needs seven days of input history | First examples after the chosen start and windows crossing gaps. | For an October 2018 start, **begin forecasting after 168 complete hours of usable history**, or obtain justified earlier history. Apply identical window rules to every model. |

**Important interpretation of `IsValidTarget`:** it establishes price completeness and internal consistency, not geographic suitability, consistent historical auction meaning or original publication availability. **596,611 populated hours carry warnings**, so filtering only on `IsValidTarget = true` is insufficient to finalize the modeling dataset.

**Suggested starting configuration:** use **October 2018 onward and the 12 continuously covered Core-market zones** for the first experiment, retaining the proposed **2022 validation** and **2023–September 2025 test** periods. Expand the zone set once the other required datasets have been checked.

This changes the proposed training period from **2016–2021 to October 2018–2021**, with the earliest forecast examples requiring sufficient input history. It is a recommendation for the experiment — not a change to V1.0, which still preserves all 61 area codes and the full historical period.

## What changed in v1.1

1. **Two end hours removed.** The last two hours (30 Sep 2025, 22:00–23:00 UTC) belong to 1 Oct, outside the project period.
2. **74 conflicting hours fixed** (North Macedonia June 2023, Albania and Kosovo one hour each). Where ENTSO-E had two versions, the latest correction was taken, checked against neighbouring prices.
3. **4,115 fake-looking prices hidden:** full days of exactly 0 €/MWh (Ireland, North Macedonia, Ukraine) and the flat placeholder values in `IT-SACODC`. They're marked not usable; the original value is kept in a note column.
4. **Short gaps patched for model inputs only.** For gaps up to 48 hours, a stand-in from the same hour on an earlier day is used as history input (1,273 hours). Real prices to predict are never patched. Poland's gaps were deliberately left unpatched.
5. **Everything is labelled.** Every changed row says what happened and why (`row_status`, `fix_reason`, `fix_detail`). Old column names still work; the new columns are added at the end.

### What's ready to use

Share of hours with a valid price; train 2016–21, validation 2022, test 2023–Sep 2025.

**Ready now: 33 zones, essentially complete in all three periods**

- **Western/Central:** FR, BE, NL, CH, CZ, SK, HU, SI
- **Iberia:** ES, PT
- **Nordic:** NO1–NO5, SE1–SE4, DK1, DK2, FI
- **Baltic:** EE, LV, LT
- **South-east:** GR, RO
- **Italy:** North, Centre-North, Centre-South, South, Sicily, Sardinia

That's more than the proposal's 25, so zones can be chosen freely.

**Usable with a small tweak**

- **Germany and Austria:** fully covered from Oct 2018, the 2018 zone split. Either use the old combined DE-AT-LU zone as "Germany" before then, or start training in Oct 2018.
- **Poland:** 95% in training (gaps in 2016–17), 100% after. Mask the gaps, or recover them from TGE.
- **Croatia, Bulgaria, Serbia:** complete once their markets start (Oct 2017, Nov 2016, Dec 2016). Mask the hours before.
- **Ireland:** 99.6%, but before Oct 2018 it was a different market design. Use it from Oct 2018, or treat the earlier years separately.

**Not suitable as main graph nodes**

- **Albania, Kosovo, Montenegro, North Macedonia:** only exist from 2023–24, so only in the test period.
- **GB:** ends in 2020, after Brexit.
- **Old Italian virtual zones:** end between 2017 and 2020.
- **Ukrainian zones:** mostly missing.
- **`NO2NSL`, `IT-SACOAC`, `IT-SACODC`:** cable links, not regions. `IT-SACODC` also had the placeholder values.
- **IT-Calabria:** only from 2021.

So the **33 ready zones** cover the full 2016 – Sep 2025 range with no extra work. Add **DE-LU and AT** with one of the two tweaks above, since Germany is central to the graph.
