# JAO flow-based constraints, aggregated per zone and hour

| | |
|---|---|
| Dataset id | `jao_zone_hourly` |
| Version | **v1.0** |
| Bucket | `gs://cs224w-epgnn-datasets/jao_zone_hourly/v1.0/` |
| Local file(s) | `datasets/jao_zone_hourly_v1.0.md`, `datasets/jao_zone_hourly_v1.0.parquet` |
| Source | JAO publication tool, CWE (2015-05-20 – 2022-06-08) and Core (2022-06-09 – 2026-10-29) final flow-based domains, via Arsen's `jao_cwe_core_hourly_views` bundle |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("jao_zone_hourly", "jao_zone_hourly.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | first build from the JAO bundle: physical constraints only, neighbour-spread relevance ≥ 0.10, four features + availability flags per zone-hour (rules below) |

Built by [scripts/build_jao_zone_hourly.py](../scripts/build_jao_zone_hourly.py), which is the executable form of the rules in this doc.

---

## What it is

One row per **(zone, hour)** for the 12 bidding zones that take part in the Core flow-based market coupling: AT, BE, CZ, DE_LU, FR, HR, HU, NL, PL, RO, SI, SK. Each row summarises how tight the transmission constraints that matter for that zone were in that hour's day-ahead flow-based domain.

The source is the hour × constraint table (`jao_cnec_hourly`, 17.7M rows): for each hour and each critical network element under a contingency (CNEC), JAO publishes the remaining available margin as a share of the line's capacity (`ram_fmax`) and the sensitivity of the line to injections in each zone (the PTDFs). This table collapses that to a handful of zone-level numbers so the model never has to touch the 870 MB source.

**Timing.** JAO publishes the final day-ahead domain for day D on D-1 in the morning, before the 12:00 CET auction, so these features are legitimately known at the project's D-1 10:00 UTC cutoff and can be used as inputs for day D.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `ts_utc` | timestamp (UTC) | delivery hour |
| `zone` | string | `AT`, `BE`, `CZ`, `DE_LU`, `FR`, `HR`, `HU`, `NL`, `PL`, `RO`, `SI`, `SK` |
| `regime` | string | `CWE` or `Core`: which capacity calculation produced that hour; null when the hour is missing from JAO |
| `jao_hour` | bool | the hour exists in the JAO data (false for 120 hours in 2026) |
| `in_domain` | bool | the zone had PTDF coefficients in that hour's domain (i.e. was part of the flow-based region) |
| `cnec_avail` | bool | at least one relevant physical constraint for the zone that hour; **the mask for the four feature columns** |
| `n_phys_hour` | int | physical constraints in the whole domain that hour (context; 0 when `jao_hour` is false) |
| `n_rel` | int | relevant physical constraints for the zone (rule 3); 0 when `cnec_avail` is false |
| `min_ratio` | float | smallest `ram_fmax` among the relevant constraints: the tightest line the zone can load. Not clipped, so values > 1 occur (margin above nominal capacity) |
| `wmean_tight` | float | spread-weighted mean of tightness `1 − clip(ram_fmax, 0, 1)` over the relevant constraints |
| `top3` | float | sum of the three largest `spread × tightness` products: the pressure from the zone's three most binding lines |

`min_ratio`, `wmean_tight`, `top3` are null wherever `cnec_avail` is false. **Model code should fill them with 0 and feed `cnec_avail` as an input**, so a zero with the flag off means "no information", not "no congestion".

Grid: 100,343 hours (2015-05-19 22:00 → 2026-10-29 20:00 UTC) × 12 zones = 1,204,116 rows, hourly, no gaps in the grid itself.

## Rules used to build it

1. **Physical constraints only.** A source row is *synthetic* if its constraint name (`CriticalBranchName` in CWE, `CneName` in Core) matches `LTA | IMPORT | EXPORT | EC_ | EXTERNAL | CONSTRAINT | NET POSITION | NETPOSITION | MAXBEX`, i.e. long-term-allocation corners, external/net-position limits and equality constraints. Everything else is a physical line, transformer or phase shifter. Synthetic rows are real parts of the market domain but their `Fmax` is a placeholder, so their `ram_fmax` is meaningless as a congestion measure (in CWE 2021–22 they are 94% of all rows and have a median ratio of about 4). They are excluded from every feature. All rows in the source are `presolved = True`.
2. **Valid ratio.** Rows with null `ram_fmax` are dropped. Tightness is `1 − clip(ram_fmax, 0, 1)`; `min_ratio` keeps the unclipped value.
3. **Relevance to a zone: zone-to-zone PTDF with a neighbour ≥ 0.10.** For each zone and constraint, `spread = max over neighbours n of |ptdf_zone − ptdf_n|`, where the neighbours are the zone's AC-interconnected partners inside the domain (table below). The constraint is relevant to the zone if `spread ≥ 0.10`, i.e. shifting 1 MW of exchange between the zone and some neighbour changes the line's loading by at least 0.1 MW.
   *Why not `|ptdf_zone| ≥ 0.10`:* JAO publishes zone-to-slack PTDFs, whose absolute size depends on the slack convention. Germany's PTDF is ≈ 0 by construction in every year since 2018 (median exactly 0.000), so an absolute threshold keeps almost no constraints for DE_LU (1–10% of hours in 2020–21). Differences between zones are independent of the slack, and the zone-to-zone PTDF is also what drives price spreads.
4. **Weight.** Each relevant constraint is weighted by its `spread`.
5. **Features per zone-hour** over the relevant constraints: `n_rel` (count), `min_ratio` (min of `ram_fmax`), `wmean_tight` (Σ spread × tightness / Σ spread), `top3` (sum of the 3 largest spread × tightness).
6. **Germany before the 2018 split.** Before delivery day 2018-10-01 the source has one combined zone `DE_AT_LU`. Its features are reported under `DE_LU`; `AT` is not in the domain then (`in_domain = false`). From 2018-10-01, `DE_LU` and `AT` are separate in the source.
7. **Nothing is invented.** The output is the full hourly grid × 12 zones. Hours missing from JAO have `jao_hour = false` and null `regime`; zones outside the domain have `in_domain = false`; zone-hours with no relevant constraint have `cnec_avail = false`. All three cases have `n_rel = 0` and null features.

Neighbours used in rule 3:

| Zone | Neighbours in the domain |
|---|---|
| AT | DE_LU (DE_AT_LU), CZ, HU, SI |
| BE | FR, NL, DE_LU (DE_AT_LU) |
| CZ | DE_LU, AT, PL, SK |
| DE_LU | AT, BE, CZ, FR, NL, PL |
| DE_AT_LU (pre-2018) | BE, FR, NL, CZ |
| FR | BE, DE_LU (DE_AT_LU) |
| HR | HU, SI |
| HU | AT, HR, RO, SI, SK |
| NL | BE, DE_LU (DE_AT_LU) |
| PL | DE_LU, CZ, SK |
| RO | HU |
| SI | AT, HR, HU |
| SK | CZ, HU, PL |

Neighbours that are not in the domain in a given hour (e.g. CZ before June 2022) contribute nothing; the spread is taken over the neighbours that are present.

## Why these features, and how they behave across the CWE → Core switch

The capacity calculation changed on 2022-06-09 (CWE: 5 zones, Core: 12 zones, different domain construction). A feature is only useful across the whole period if it means the same thing on both sides. Checked on Mar–Sep 2022 (3 months either side of the switch) for the four zones present in both regimes:

| | BE CWE → Core | FR | NL | DE_LU |
|---|---|---|---|---|
| `wmean_tight` median | 0.21 → 0.22 | 0.21 → 0.22 | 0.21 → 0.23 | 0.22 → 0.25 |
| `min_ratio` median | 0.43 → 0.36 | 0.48 → 0.48 | 0.46 → 0.37 | 0.40 → 0.24 |
| `top3` median | 0.37 → 0.38 | 0.30 → 0.28 | 0.34 → 0.37 | 0.30 → 0.41 |
| `n_rel` median | 21 → 25 | 18 → 23 | 19 → 24 | 22 → 44 |

`wmean_tight` and `top3` carry over with no adjustment; `min_ratio` shifts somewhat for DE_LU; `n_rel` roughly doubles for DE_LU because Core adds the Polish and Czech borders. Recommendation: use `wmean_tight`, `top3`, `min_ratio` as they are, standardise `n_rel` per regime (or drop it), and give the model `regime` as an input.

Counts, stats and coverage (physical constraints per hour: CWE 16–25 typical, Core 108–121):

| Zone | Hours with `cnec_avail` | From | `wmean_tight` median | `min_ratio` median |
|---|---|---|---|---|
| BE, DE_LU, FR, NL | ≈ 100,060 (≈ 100%) | 2015-05-20 | 0.27 – 0.31 | 0.23 – 0.36 |
| AT | 56,723 | 2018-10-01 | 0.20 | 0.41 |
| CZ, HR, HU, PL, RO, SI, SK | 38,300 (100% of Core) | 2022-06-09 | 0.18 – 0.32 | 0.32 – 0.50 |

## Caveats

- **AT in CWE (Oct 2018 – Jun 2022) is thin:** its only CWE neighbour is DE_LU and Austrian lines rarely appear in the CWE domain, so `cnec_avail` is 25–86% of hours in those years with typically 1–2 relevant constraints. Treat AT's constraint features as reliable from June 2022.
- **Minimum-RAM floor from 2019:** `min_ratio` can go below 0.05 in 2015–2018 but is floored at 0.20 from 2019 (EU minRAM rule, CWE then Core). See the `jao_cnec_hourly` doc; feed the year or a flag.
- **The other 26 project zones** (Nordics, Iberia, Italy, GB, SEE…) are never in the domain. They get no rows here; the model should give them `cnec_avail = 0` and zero features.
- **CWE hours are derived**, not published: CWE files give delivery day + period, mapped to UTC via Europe/Brussels. Core hours are published UTC. The bundle's `timestamp_status` records this; the two are contiguous at the switch.
- **Missing hours:** 120 hours in 2026 are absent from the source (`jao_hour = false`). The source covers 2015-05-20 → 2026-10-29, two days short of the ENTSO-E v1.2 datasets.
- **No shadow prices, no clearing net positions:** JAO does not publish them for these domains, so "binding" cannot be identified; tightness (`ram_fmax`) is the proxy.
- **No physical-asset matching across regimes:** the same line can appear under different names in CWE and Core. Irrelevant here because everything is aggregated per zone-hour.
- Source-level verification (counts, provenance, CWE border downloads) is in the bundle's `DATASET_GUIDE.md`, `build.json`, `verification.json`, kept with the raw bundle.
