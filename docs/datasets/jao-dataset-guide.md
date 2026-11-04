# JAO Dataset Guide

Flow-based constraint data from the Joint Allocation Office (JAO), CWE and CORE regimes.

## One-page overview

**What we have.** Daily JAO exports of final, unfiltered flow-based constraint records for European day-ahead capacity calculation, from **20 May 2015 through 31 December 2024**. The intended scope contains **3,514 delivery days: 2,577 CWE days through 8 June 2022, then 937 CORE days from 9 June 2022**. Acquisition markers cover every expected day. A marker establishes a completed download, not complete hours, correct units or a validated network identity. The whole-archive conversion and column profile supply the separate record-level evidence; their live status must show complete before interpreting observed ranges as full-history measurements.

**What a row means.** Treat a row as one published constraint observation for a delivery interval and calculation domain. It often describes a monitored network element under a contingency, with a remaining available margin (RAM) and sensitivities to bidding-zone positions (PTDFs). The same branch appears across hours, contingencies, directions and variants. Some rows instead represent external or equality constraints. This is therefore a **time-varying domain dataset, not a static list of transmission lines**.

| Aspect | CWE | CORE |
|---|---|---|
| Local delivery coverage | 2015-05-20 – 2022-06-08 | 2022-06-09 – 2024-12-31 |
| Time representation | `DeliveryDate` + `Period` | `DateTimeUtc` |
| Margin field | `RemainingAvailableMargin` | `Ram` |
| Sensitivities | Repeated zone-label / `Factor` pairs | `Ptdf_TOKEN` columns |
| Contingency representation | `OutageName` / `OutageEIC` | `Cont*` fields, including component slots |
| Physical identity | Not established | Not established |

**What the main values mean.** `Fmax` is the allowable power flow used in the calculation; RAM is the margin offered for market exchanges after the relevant background loading, allowances and adjustments; PTDFs describe how a proposed change in trading positions changes a monitored flow. Documented RAM and principal flow terms use MW, whereas current and voltage use A and kV. Numeric strings, flags and source tokens remain preserved.

**How to use it.** Read this guide for concepts and interpretation, then use the column reference or CSV dictionary (separate artifacts) for exact spellings, measured ranges, examples and availability. Daily profiles show when a field is absent or empty. First/last populated dates alone do not prove uninterrupted availability.

**Main cautions.**

- Empty PTDFs are not zeros.
- Presolved is not proof of binding at market clearing.
- Source IDs are not persistent physical IDs.
- CWE periods need explicit UTC anchoring, especially at DST.
- Delivery time and download time do not establish historical availability before a forecast cutoff.
- The two regimes must retain separate conventions until their differences are resolved.

## Measured coverage snapshot

**Complete:** 3514/3514 days profiled; CORE: **390,509,064 rows**, CWE: **424,279,680 rows**.

| Regime | Field | Observed finite range | Empty / present records |
|---|---|---|---|
| CORE | `Presolved` | categorical | 0 / 390,509,064 |
| CORE | `Ram` | 0 … 2,463,376 | 0 / 390,509,064 |
| CORE | `Fmax` | 0 … 2,462,335 | 0 / 390,509,064 |
| CORE | `Fref` | −6,219 … 6,219 | 0 / 390,509,064 |
| CWE | `Presolved` | categorical | 0 / 424,279,680 |
| CWE | `RemainingAvailableMargin` | 0 … 69,268,512 | 0 / 424,279,680 |
| CWE | `Fmax` | 0 … 69,268,073 | 70,744 / 424,279,680 |
| CWE | `Fref` | −4,467 … 4,269 | 70,744 / 424,279,680 |

Selected source-zone availability (first/last populated dates are not continuity guarantees):

| Regime | Token | Nonempty days | First / last populated |
|---|---|---|---|
| CORE | ALBE | 937 | 2022-06-09 / 2024-12-31 |
| CORE | ALDE | 937 | 2022-06-09 / 2024-12-31 |
| CORE | AT | 937 | 2022-06-09 / 2024-12-31 |
| CORE | CH | 0 | — |
| CWE | ALBE | 582 | 2020-11-04 / 2022-06-08 |
| CWE | ALDE | 582 | 2020-11-04 / 2022-06-08 |
| CWE | AT | 1347 | 2018-10-01 / 2022-06-08 |

Counts and ranges are observations. The full source-zone availability table, yearly zone measurements, all fields, availability dates and extreme-row locators are in the separate column reference.

## Main values in physical terms

### The real-world situation behind a row

Imagine a transmission line connecting two substations. Electricity already flows through it because of generation and consumption across the interconnected grid. Additional international electricity trading changes that flow, even when the trade is between countries far from the line. The operator must keep loading within acceptable limits, including under specified failure scenarios.

A JAO constraint describes part of that calculation: **which element is monitored, under which scenario, how much room is offered to the market, and how each zone's trading position affects the element.** These are mostly calculated parameters for a future delivery interval. They are not a time series of sensor measurements of the actual electricity that eventually flowed.

### Physical objects, scenarios and units

| Concept / fields | Physical or operational meaning |
|---|---|
| **Monitored element** — `CneName`, `CriticalBranchName` | Usually a line or transformer whose loading is checked. A transformer connects voltage levels or helps control flows; a substation is a location where network equipment connects. Special domain constraints also occur, so a row name does not always identify a physical line. |
| **Contingency** — `Cont*`, `OutageName` | An assumed event such as another line becoming unavailable. The calculation asks whether the monitored element would remain within limits after the resulting redistribution of electricity. A labelled `BASECASE` is the reference case without an additional contingency outage. Blank fields need separate interpretation. |
| **Direction** — `Direction`, or `DIR`/`OPP` in a CWE name | The orientation in which a flow is counted as positive. Reversing the monitored direction reverses flow signs; it does not guarantee identical margins. `NA` leaves the published orientation unresolved. |
| **Power: MW** | A rate of energy transfer. A constant 100 MW over one hour transfers 100 MWh. RAM and power-flow terms use MW, whereas electricity prices commonly use €/MWh. |
| **Current: A; voltage: kV** — `Imax`, `U` | Current is the movement of electric charge, which heats conductors; voltage is the electrical potential difference. `Imax` is an admissible current and `U` a reference voltage. Neither is itself an amount of traded electricity. |

Contingency analysis supports **N−1 security**: checking operation after losing one component. An event can involve multiple components, and the export also contains basecase observations. A contingency label does not establish that a failure occurred, that it was forecast to occur, or how likely it was. (See ENTSO-E: calculation inputs and scenarios.)

### The three main quantities: Fmax, RAM and PTDF

**Fmax — the power-flow limit.** The maximum allowable active-power flow used for the monitored element/scenario, in MW. Physically, limits can reflect equipment heating and operating conditions. A seasonal limit changes with seasonal assumptions; dynamic and fixed limits use different determination methods. Fmax is an input to the calculation, rather than a measurement of present loading. Exceptional published values and special constraints need their own interpretation.

**RAM / RemainingAvailableMargin — the market's remaining margin.** Some of the element's allowable loading is already accounted for by the calculation's background/reference flows and reliability allowances. RAM is the resulting margin offered for the trading positions represented by the domain, after the applicable adjustments. Thus a 1,000 MW line limit does not automatically allow 1,000 MW of additional cross-border trade. Policy and validation adjustments also mean RAM is not simply a physical "unused capacity" reading from a meter.

**PTDF / Factor — how trading changes loading.** A Power Transfer Distribution Factor is a model sensitivity: the change in the monitored flow per MW change in a specified zone position, under the given scenario and reference convention. A coefficient of 0.3 means a 1 MW injection increase in that zone, balanced at the model's reference/slack, changes monitored flow by approximately +0.3 MW. A negative coefficient changes it in the opposite monitored direction. This is a marginal relationship around a modelled system state.

A zone's **net position** is its net commercial export or import; in the usual net-export convention, positive means export and negative means import. A **bidding zone** is a market area with a common day-ahead price, not necessarily a country. Net positions are quantities to which these sensitivities refer; clearing net-position values are not acquired in this constraint export. For an export increase in zone A balanced by an import increase in zone B, the sensitivity is the **difference** `PTDF_A − PTDF_B`. The reference convention matters when interpreting individual coefficients. (See JAO: domain parameters and units.)

### A worked example: from a trade to a line constraint

The following numbers are illustrative, not an extracted JAO record. Assume one monitored direction and contingency:

| Quantity | Illustrative value | Meaning |
|---|---|---|
| Allowable flow | 1,000 MW | The element's limit in this scenario |
| Background loading | 600 MW | Loading already accounted for before the proposed additional exchange |
| Reliability allowance | 100 MW | Room reserved for uncertainty |
| Remaining trading margin | 300 MW | Headroom in this simplified example, with no other adjustments |
| PTDF for A / B | 0.3 / −0.1 | A-to-B exchange sensitivity is `0.3 − (−0.1) = 0.4` |

An additional **100 MW export from A to B** therefore adds approximately **40 MW** to this monitored flow, consuming 40 MW of the 300 MW margin. On this constraint alone, the illustrative exchange limit is `300 / 0.4 = 750 MW`. Other constraints can impose a smaller limit. Electricity follows the network's electrical properties; the commercial trade does not choose a dedicated route through a single line.

The subtraction above explains the idea of headroom. It is **not a universal formula for reconstructing the source RAM**: the published calculation uses regime-specific references, nominations and adjustments. Use the published margin unless a reconstruction has been validated for that process and era.

### What the other main values contribute

| Fields | Real-world interpretation |
|---|---|
| **Fref; FrefInit** | Calculated reference loading of the monitored element; FrefInit belongs to the initial computation. The reference system includes assumptions about generation, consumption and exchanges. These are not realised measured flows. |
| **F0core; F0all** | Counterfactual flow baselines with different sets of commercial exchanges removed. They describe loading that remains under those modelling assumptions. Keep the CORE-only and wider exchange-removal cases separate. |
| **FRM / Frm** | Flow reliability margin: an allowance for uncertainty between the capacity-calculation model and operation. Reserving this margin protects against prediction/model error. |
| **Fnrao** | Expected flow change associated with non-costly remedial-action optimisation. Such actions can change network configuration or controllable equipment settings to redistribute loading. |
| **AMR / Amr; minimum-RAM factors/targets** | An adjustment and targets intended to ensure minimum capacity for market exchanges. This is a capacity-calculation/policy mechanism; it does not build a new line or raise its physical rating. Export target scales still require confirmation. |
| **Cva; Iva; FAV** | Coordinated validation, individual validation, and final adjustment terms respectively. They modify the calculated domain in light of the relevant process; preserve their source signs rather than assuming one common subtraction rule. |
| **Fltn; FtotalLtn; LtaMargin** | Long-term nominations and allocation rights affect how capacity is represented. Fltn is interpreted as a nomination-related flow component; FtotalLtn includes its effect. LtaMargin serves allocation inclusion. These are different concepts, and the Fltn export mapping remains inferred. |
| **Presolved** | Whether the observation survives the relevant domain-redundancy treatment. A retained constraint can bound some possible trading combinations without being reached by the market's chosen combination. Therefore Presolved does not identify which line actually constrained clearing. |

For the adjustment terminology see JAO's CORE parameter breakdown; the calculation and operational context is described by ENTSO-E. This table explains roles, not an arithmetic identity connecting all exported columns.

The physical reading of a row is therefore: **"For this interval and assumed network scenario, market positions affect this monitored direction according to these sensitivities, and their modelled contribution must respect this margin."** All rows together describe the flow-based portion of the permissible trading region. They do not by themselves identify the final market outcome or every actual physical flow.

## What exists and what it covers

The local acquisition has two roots, `jao_exports` and `jao_core_exports`, inside `jao_pull_bundle`. Some completed CORE days exist in both roots. Identical source hashes allow the inventory to select one copy; conflicting hashes are quarantined rather than resolved by guessing which version is preferable. The project cutoff remains **2024-12-31** even while later acquisition continues. This is a historical snapshot, not a claim that JAO has no later data.

Raw storage preserves the downloaded application-level ZIP for CORE and gzip-compressed original CSV responses for CWE. Each completed attempt has metadata and a completion marker pointing to its file. Incomplete attempts and partial downloads are outside this guide's measured population. They remain on disk and should not be counted as additional delivery-day observations.

The source-preserving layer parses those completed files into daily Parquet. Its original fields remain strings, including empty strings and decimal representations. Added member/row provenance allows an analyst to locate a record. Separate manifests record source hash, parser identity, sizes, row counts and diagnostics. A source hash recorded in a download marker is distinct from rehashing the raw file during analysis; the manifests identify whether optional verification occurred.

Do not confuse the present scope with every publication exposed by JAO. **Prices, shadow prices, clearing net positions, generation, demand, actual physical flows and separate allocation-constraint publications have not been acquired as part of this dataset.** An endpoint may publish them, but their existence elsewhere does not make them columns available here. The CORE handbook explicitly separates final computation from post-coupling shadow-price publications (CORE handbook, §§5.13 and 5.23).

## Conceptual data model

The model below explains relationships in the source observations. It is not an implemented relational database, and it does not assert that the physical entities already have validated primary keys.

```
Completed daily source artifact
├── Source hash and member provenance
└── Published interval observations
    ├── Monitored element or special constraint
    ├── Contingency and component slots
    ├── Direction and source variants
    ├── RAM and flow components
    ├── Source-zone PTDF coefficients
    │   └── Bidding-zone or virtual-hub token
    └── Added analysis row locator
```

A day's domain consists of many inequalities, with coefficients and margins varying by interval. A schematic constraint is `sum(PTDF_z × position_z) <= RAM`. It is a constraint on a vector of market positions, not an independent capacity attached to every zone pair. This describes the flow-based portion only; CORE feasibility also uses extended-LTA logic and separate bilateral restrictions. The final-domain export is not the entire market-clearing input set (CORE handbook v2.2, §5.1). Two rows with different contingencies can describe the same monitored element under different assumed system states. Their margins should not be added as if they were separate assets. The CORE market-check description describes this domain interpretation.

Use three different notions of identity:

1. **Artifact identity** identifies the exact saved source, using its hash.
2. **Observation locator** identifies a row within that artifact, using `_analysis_row` and optionally `_analysis_member`.
3. **Physical constraint identity** would match the underlying monitored element, direction, contingency and relevant variants over time. The third is not established merely because the first two are available.

The 18-day identity audit illustrates the distinction: 4,989,319 rows contained no exact source duplicates, but 40,375 candidate-key groups had different RAM/PTDF source values. Those counts belong to the audited sample, not the whole historical archive. CORE examples have `Direction=NA` and opposite flow/PTDF signs under otherwise matching keys. CWE examples retain different parameters even under the same labelled branch and outage. These findings argue for preserving observations, not deleting collisions. The July 2024 CORE handbook, §§5.10–5.12 explicitly includes filtered-out monitored elements, non-CORE-exchange calculation elements with `Imax=9999`, and valid same-name variants reflecting remedial actions. These categories must not be equated with corrupt or duplicate data. They do not establish the classification of a particular row without its own evidence.

## What RAM values usually look like

A histogram (figure in the original guide; regenerate with `jao_ram_histogram.py`) covers **every published constraint row through 2024**, with CWE and CORE shown separately. Each row has equal weight, so dates with more published constraints contribute more observations. It is not a distribution of independent physical lines or a probability forecast of congestion. **99.27% of CWE rows and 99.90% of CORE rows have RAM below 5,000 MW** in this archive.

| Regime | N | RAM ≥ 5,000 MW | RAM = 0 | Shape of central range (0–5,000 MW) |
|---|---|---|---|---|
| CWE | 424,279,680 | 0.734% | 0.001% | Small peak near ~400–500 MW, main mode ~1,500–1,700 MW, tail to ~3,500 MW. Full range has isolated clusters up to ~10⁷ MW. |
| CORE | 390,509,064 | 0.099% | 0.025% | Small peak near ~400–500 MW, main mode ~1,300–1,500 MW, tail to ~4,000 MW. Very few values above 10⁴ MW. |

The central-range panels use 100 MW bins over 0 ≤ RAM < 5,000 MW. The full-range panels use logarithmically spaced bins and a logarithmic frequency axis so the small tail remains visible. Zero margins are counted separately. No values have been clipped or removed. The 5,000 MW boundary is a viewing choice, not a validity threshold.

**Data-quality note:** some published RAM and Fmax values have unusually large magnitudes. These could reflect source errors, reporting conventions or special calculation records; their interpretation is unresolved. Preserve and flag them, and check sensitivity to their inclusion in analysis. Detailed examples belong in the quality evidence, rather than defining the main explanation of RAM.

Zero RAM also needs context: it occurs in equality constraints and some line records. Neither zero nor Presolved alone establishes an erroneous record or a constraint binding at market clearing.

The bin counts and coverage summary make the figure reproducible. Run `jao_ram_histogram.py` to regenerate it from existing Parquet; an SVG version is available for presentations.

## Reading the variable families

### Time and delivery intervals

**CORE** provides a UTC timestamp field. The analysis compares its interval timestamps with the UTC instants obtained from successive Europe/Brussels calendar midnights. An ordinary delivery day has 24 hourly instants; a spring clock-change day has 23, and an autumn change day has 25. UTC avoids the duplicated local-clock label in autumn. The explicit 2024-03-31 and 2024-10-27 probes matched these expectations without missing or unexpected timestamps.

**CWE** provides a source business date and a period label. The 2021 DST probes have period sets 1–23 and 1–25. That supports ordinal periods on those days; it does not independently prove the exact UTC instant for every label. Preserve both fields and retain the unresolved mapping status until documented operational anchors establish it. Neither a filename nor a source midnight string should silently substitute for that evidence.

### Network, contingency and orientation

Names are human-readable descriptors. EICs are identifiers, while hub/substation fields describe endpoints. None of these by itself establishes a complete observation key. Names can include direction text and operator tags; literal whitespace is part of the preserved string representation. Normalising whitespace can be a later comparison aid, but it must not overwrite the source.

A contingency is the assumed event associated with the monitored observation. `BASECASE` and blank contingency fields need explicit treatment rather than being interpreted as the same missing category. CORE's numbered `Cont*` fields are component slots in the flat export. A suffix is not a count of independent scenarios. Slot attributes should remain grouped, and the first slot can be empty without implying that all later slots are empty. The CWE handbook also records historical Elia naming limitations, including only `DIR` labels and potentially inconsistent hub labels before tool adaptation. This provides a documented reason to distrust a name-only orientation key in that era, without assigning an inferred direction to any individual row (CWE handbook, pp. 21–22).

### Export-specific margin conventions

The physical roles above do not establish exact API mappings. Exported `MinRamFactor`/`MinRamTarget` still need confirmation against handbook targets and scales: 0.7 must not silently become either 0.7% or 70%. Likewise, validate any RAM reconstruction against the specific regime, era and process stage before using it in analysis.

### PTDFs and source zones

A PTDF is a sensitivity coefficient, so its numerical unit differs from the MW margin.

**CWE** repeats a zone-token field followed by its `Factor`. The parser names repeated columns with `.1`, `.2`, and subsequent suffixes; **these suffixes are not zone identities**. The zone identity comes from the paired source string. Always interpret a coefficient together with that token.

**CORE** uses wide fields such as `Ptdf_FR`. The suffix identifies the source token directly. Keep virtual and external hub labels separate from countries. **ALBE and ALDE** represent the Belgian and German virtual connection hubs for ALEGrO, rather than ordinary country nodes (JAO methodology annex).

A column that is empty throughout a period contributes no measured coefficient during that period; filling it with zero would turn "unpublished" into "measured no sensitivity." Reference/slack assumptions must also be established before comparing vectors longitudinally. The sampled extrema lying in [−1, 1] never established a universal valid range.

### Flags, classifications and explanations

Presolved concerns the shape/redundancy of the domain. Binding at a realised clearing point is a different concept, requiring the market outcome or associated post-coupling evidence. The CWE handbook also describes a virgin final-computation view where all flags are `FALSE` because presolving was not performed. Thus `FALSE` cannot universally mean that an algorithm proved redundancy until the publication view is established. A status label, asset classification, limit method or free-text justification should similarly be interpreted according to its own field, rather than collapsed into a generic quality flag.

Free-text explanations can clarify exceptional records but are not guaranteed to have stable wording. Their bounded examples in the reference are illustrations. They do not enumerate the complete vocabulary or prove the absence of other explanations. `CneStatus` and `Cnec` have deliberately conservative definitions where a precise export-specific interpretation has not been established.

## Availability, ranges and evidence

Every numerical range in the reference is measured, not an engineering acceptance threshold. The profile saves finite minima and maxima together with their original strings, delivery dates and row locators. It separately counts empty strings, nulls, invalid numeric text, parsed overflow/nonfinite values, zeros and negative values. Use the distribution above to understand typical values; observed extrema are not physical acceptance limits. Detailed source checks are retained in the quality evidence.

Availability has multiple denominators. "Days present" counts days where a column exists in the schema. "Days populated" counts days with at least one nonempty cell. Row-level missingness uses records where the column exists, while absent-record counts measure structural absence. Both denominators matter when a schema evolves. For CWE `Factor` fields, source-zone availability must additionally be read from the paired token and its coefficient diagnostics.

The daily field table provides the evidence between first and last populated dates. Use those rows to identify interruptions or era changes. The schema table records original parsed column lists by day. CWE has an additional `AMR_Exclusion` flag and explanation only from **2019-10-02 through 2019-12-11 (71 days)**. Their absence elsewhere is not a `FALSE` flag. The older handbook uses exclusion terminology differently; the observed boolean/text split is kept explicit. The source converter's PTDF diagnostics separately aggregate coefficients by their actual source-zone tokens rather than treating slot positions as permanent zones.

The whole-history profile does not repeat the expensive sampled identity audit over every historical candidate key. Its purpose is column coverage and value characterisation. No full-history physical-identity certificate or duplicate-free guarantee follows from field profiling. High-cardinality names and source IDs have representative examples rather than an enumerated global vocabulary; exact frequencies are retained for declared categorical fields.

## Disambiguations

| # | Similar-looking concepts | Distinction | Consequence |
|---|---|---|---|
| 1 | Source record vs physical element | A record belongs to an artifact and interval | Preserve source provenance in joins |
| 2 | Missing vs zero | Absence carries no measured numerical value | Do not zero-fill PTDFs |
| 3 | Presolved vs market binding | Domain property versus realised outcome | Do not label bottlenecks from the flag alone |
| 4 | CORE vs CWE RAM | Similar role, different process conventions | Keep regime and era metadata |
| 5 | Fmax vs RAM | Limit component versus remaining margin | Do not substitute one for the other |
| 6 | Bidding zone vs country | Labels can describe market areas or virtual hubs | Do not map tokens by country name alone |
| 7 | Factor suffix vs zone token | Parser slot position versus source identity | Pair label and coefficient before alignment |
| 8 | Delivery time vs publication vintage | Target interval versus information availability | Prevent look-ahead in forecasting |
| 9 | Acquired day vs validated data | Container completion versus row/hour evidence | Inspect both coverage layers |

## Using the data in the project

A practical analysis starts with provenance and timing, then derives scoped features. Select a delivery interval using the regime's established time representation; preserve the source artifact and calculation-domain context. Decide explicitly whether an analysis uses all rows or a labelled filtered view. Summaries such as minimum margin, number of constraints or zone sensitivities depend on that choice and on which zones were actually published.

For cross-zone features, interpret the PTDF vector jointly with its reference assumptions and the margin. A **bipartite zone/constraint representation** can retain the distinction between a physical network graph and a published constraint domain. It should not present endpoint hub labels as complete bus topology, or count each contingency observation as a separate physical line. Source identity ambiguity remains relevant even if a model can consume anonymous row features.

Temporal comparison requires attention to changing coverage. An increase in row count can result from publication changes, added contingency variants or different domain construction; it does not by itself measure worsening congestion. Likewise, a newly populated zone coefficient can mark a changed market representation rather than a sudden appearance of a physical country in the grid.

For next-day price prediction, first establish what version of each feature was available at the intended cutoff. The downloader's acquisition timestamp is in 2026 and cannot certify availability in 2018 or 2023. The CWE handbook distinguishes pre-coupling products from the detailed final-domain section under ex-post publication. **The handbook gives 10:30 on delivery day D for that detailed domain, rather than D−1.** Endpoint names alone do not resolve which historical vintage the saved response represents. Treat this as an information-set audit before claiming an operational forecasting improvement.

## Unresolved questions

The main interpretation questions are the operational CWE period-to-UTC mapping, exact CORE target-field scales, PTDF reference changes, and a physical-identity policy for missing directions and multiple source variants. Outlier treatment is a separate data-quality decision, as noted with the histogram.

Publication vintage is an additional modelling prerequisite. Evidence should connect the acquired endpoint and historical record version to a known publication stage and release time. An official schedule is useful context but does not prove that every stored historical value is the value visible at that time. Where evidence is missing, preserve an explicit unknown status and continue descriptive analysis with appropriately limited claims.

## References and reproduction

Consult the CWE handbook v1.9 (23 November 2021), the CORE handbook v1.3 (18 November 2021), and ENTSO-E's CORE process explanation. Also consult CORE handbook v2.2 (July 2024), especially its version history and clarification of domain scope. These are process references, not proof that one handbook version governs every historical day. ENTSO-E distinguishes the 8 June 2022 go-live from first delivery on 9 June, explaining the local regime boundary.

The reproducible build is `jao_dataset_build.py`; it converts completed sources and profiles their columns using eight low-priority local workers. It makes no acquisition requests. `jao_dictionary.py` renders the reference and CSV from the saved field profiles. Existing sample Parquet is reused when source/parser identity, size, schema and footer checks match. Per-day profile artifacts allow failed or interrupted work to resume without rereading every successful day.

The authoritative completion evidence is profile status together with conversion status and explicit failure records. Until both complete successfully, treat generated ranges as provisional. Original acquisitions are retained, no clipping or deduplication is applied, and no data is uploaded by this workflow.
