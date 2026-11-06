# Interconnector table: hourly physical flows per directed zone pair

| | |
|---|---|
| Dataset id | `agg_zone_edges_hourly` |
| Version | **v1.0** |
| Bucket | `gs://cs224w-epgnn-datasets/agg_zone_edges_hourly/v1.0/` |
| Local file(s) | `datasets/agg_zone_edges_hourly_v1.0.md`, `datasets/agg_zone_edges_hourly_v1.0.parquet` |
| Source | derived from the dataset listed under *Built from*, by `scripts/build_agg_zone_edges_hourly.py` |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("agg_zone_edges_hourly", "agg_zone_edges_hourly.parquet"))
edges = df[["zone_from", "zone_to"]].drop_duplicates()        # the directed edge list
```

Version history:

| Version | What |
|---|---|
| v1.0 | first build: 174 directed pairs (87 interconnectors) x 94,967 hours from entsoe_physical_flows |

## Built from

| Source dataset | Version | File | md5 |
|---|---|---|---|
| `entsoe_physical_flows` | v1.2 | `entsoe_physical_flows.parquet` | `/KvYtuoN4B2akaT3ZOOq4g==` |

Rebuild with `scripts/build_agg_zone_edges_hourly.py`; the zone mapping is `epgnn/zones.py`.

## What it is

One row per **(zone_from -> zone_to, hour)** for every directed pair ENTSO-E reports between two agg zones: the zone-zone edges of the graph and their hourly flow feature. 16,002,644 rows = 174 directed pairs x 94,967 hours, 2016-01-01 00:00:00+00:00 to 2026-10-31 22:00:00+00:00 UTC.

The distinct (`zone_from`, `zone_to`) pairs **are** the edge list; there is no separate edge file. ENTSO-E reports each direction of a border as its own series, so both directions are present as separate rows and no netting is done. Which pairs a model uses (e.g. requiring both directions, a minimum coverage, or `first` <= the forecast day) is a modelling choice made at load time from the per-pair coverage below.

## Rules used to build it

1. **Zone id.** Both ends are mapped from the ENTSO-E display names to the canonical zone code (`epgnn.zones.ENTSOE_NAME_TO_ZONE`); the EIC codes are kept in `eic_from` / `eic_to`.
2. **Which pairs.** Every directed pair whose two ends are agg zones (the 44 zones that have both ENTSO-E prices and weather). Pairs with a non-zone end (country aggregates, control areas, virtual Italian areas, cable areas like NO2NSL / IT-SACODC, UA, GB-only neighbours) are not included. No coverage filter.
3. **Germany.** DE-AT-LU pairs before 2018-09-30 22:00:00+00:00 UTC and DE-LU pairs from then on are both mapped to `DE_LU`, so a border like NL -> DE_LU is one continuous series across the Oct 2018 split; `eic_from` / `eic_to` show which source area each row came from.
4. **Grid.** The source's full grid is kept: every pair has a row for every hour, with `flow_status = OUTSIDE_ZONE_LIFETIME` outside the period the pair is reported. Nothing is filled or masked beyond the source's own rules.
5. **Column names.** Source columns renamed with a `flow_` prefix (see schema); values untouched.

## Schema

| Column | From | Source column |
|---|---|---|
| `zone_from`, `zone_to` | derived | canonical zone codes of `OutArea`, `InArea` (flow direction from -> to) |
| `eic_from`, `eic_to` | entsoe_physical_flows | `OutAreaCode`, `InAreaCode` |
| `ts_utc` | entsoe_physical_flows | `DeliveryHourUTC` |
| `market_day_cet` | entsoe_physical_flows | `MarketDayCET` |
| `flow_mw` | entsoe_physical_flows | `FlowMW` (measured, MW, >= 0 in the stated direction) |
| `flow_mw_input` | entsoe_physical_flows | `FlowMW_input` (short gaps patched) |
| `flow_valid` | entsoe_physical_flows | `IsValidValue` |
| `flow_status` | entsoe_physical_flows | `row_status` |
| `flow_filled` | entsoe_physical_flows | `is_filled` |

## Directed pairs (period reported, share of valid hours in it)

| from | to | first | last | valid |
|---|---|---|---|---|
| AT | CH | 2016-01-01 | 2026-10-31 | 100.0% |
| AT | CZ | 2016-01-01 | 2026-10-31 | 100.0% |
| AT | DE_LU | 2018-09-30 | 2026-10-31 | 100.0% |
| AT | HU | 2016-01-01 | 2026-10-31 | 99.9% |
| AT | IT_NORD | 2018-09-30 | 2026-10-31 | 100.0% |
| AT | SI | 2016-01-01 | 2026-10-31 | 99.9% |
| BE | DE_LU | 2017-10-10 | 2026-10-31 | 100.0% |
| BE | FR | 2016-01-01 | 2026-10-31 | 100.0% |
| BE | GB | 2018-12-05 | 2026-10-31 | 100.0% |
| BE | NL | 2016-01-01 | 2026-10-31 | 100.0% |
| BG | GR | 2016-01-01 | 2026-10-31 | 100.0% |
| BG | MK | 2016-01-01 | 2026-10-31 | 99.5% |
| BG | RO | 2016-01-01 | 2026-10-31 | 100.0% |
| BG | RS | 2016-01-01 | 2026-10-31 | 100.0% |
| CH | AT | 2016-01-01 | 2026-10-31 | 100.0% |
| CH | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| CH | FR | 2016-01-01 | 2026-10-31 | 100.0% |
| CH | IT_NORD | 2016-01-01 | 2026-10-31 | 91.8% |
| CZ | AT | 2016-01-01 | 2026-10-31 | 100.0% |
| CZ | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| CZ | PL | 2016-01-01 | 2026-10-31 | 100.0% |
| CZ | SK | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | AT | 2018-09-30 | 2026-10-31 | 100.0% |
| DE_LU | BE | 2017-10-10 | 2026-10-31 | 99.8% |
| DE_LU | CH | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | CZ | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | DK_1 | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | DK_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | FR | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | HU | 2016-01-01 | 2018-09-30 | 100.0% |
| DE_LU | IT_NORD | 2016-01-01 | 2018-09-30 | 68.4% |
| DE_LU | NL | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | NO_2 | 2020-09-01 | 2026-10-31 | 100.0% |
| DE_LU | PL | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | SE_4 | 2016-01-01 | 2026-10-31 | 100.0% |
| DE_LU | SI | 2016-01-01 | 2018-09-30 | 100.0% |
| DK_1 | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| DK_1 | DK_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| DK_1 | GB | 2023-12-19 | 2026-10-31 | 99.9% |
| DK_1 | NL | 2019-09-07 | 2026-10-31 | 100.0% |
| DK_1 | NO_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| DK_1 | SE_3 | 2016-01-01 | 2026-10-31 | 100.0% |
| DK_2 | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| DK_2 | DK_1 | 2016-01-01 | 2026-10-31 | 100.0% |
| DK_2 | SE_4 | 2016-01-01 | 2026-10-31 | 100.0% |
| EE | FI | 2016-01-01 | 2026-10-31 | 99.7% |
| EE | LV | 2016-01-01 | 2026-10-31 | 98.0% |
| ES | FR | 2016-01-01 | 2026-10-31 | 100.0% |
| ES | PT | 2016-01-01 | 2026-10-31 | 100.0% |
| FI | EE | 2016-01-01 | 2026-10-31 | 99.7% |
| FI | NO_4 | 2016-01-01 | 2026-10-31 | 100.0% |
| FI | SE_1 | 2016-01-01 | 2026-10-31 | 100.0% |
| FI | SE_3 | 2016-01-01 | 2026-10-31 | 100.0% |
| FR | BE | 2016-01-01 | 2026-10-31 | 100.0% |
| FR | CH | 2016-01-01 | 2026-10-31 | 100.0% |
| FR | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| FR | ES | 2016-01-01 | 2026-10-31 | 100.0% |
| FR | GB | 2016-01-01 | 2026-10-31 | 99.3% |
| FR | IT_NORD | 2016-01-01 | 2026-10-31 | 95.2% |
| GB | BE | 2018-12-05 | 2026-10-31 | 99.8% |
| GB | DK_1 | 2023-12-19 | 2026-10-31 | 99.9% |
| GB | FR | 2016-01-01 | 2026-10-31 | 99.3% |
| GB | IE_SEM | 2016-01-01 | 2026-10-31 | 94.1% |
| GB | NL | 2016-01-01 | 2026-10-31 | 99.9% |
| GB | NO_2 | 2021-06-14 | 2026-10-31 | 100.0% |
| GR | BG | 2016-01-01 | 2026-10-31 | 100.0% |
| GR | IT_SUD | 2018-12-30 | 2026-10-31 | 100.0% |
| GR | MK | 2016-01-01 | 2026-10-31 | 99.9% |
| HR | HU | 2016-01-01 | 2026-10-31 | 100.0% |
| HR | RS | 2016-01-01 | 2026-10-31 | 100.0% |
| HR | SI | 2016-01-01 | 2026-10-31 | 100.0% |
| HU | AT | 2016-01-01 | 2026-10-31 | 99.9% |
| HU | DE_LU | 2016-01-01 | 2018-09-30 | 100.0% |
| HU | HR | 2016-01-01 | 2026-10-31 | 100.0% |
| HU | RO | 2016-01-01 | 2026-10-31 | 100.0% |
| HU | RS | 2016-01-01 | 2026-10-31 | 100.0% |
| HU | SI | 2022-06-29 | 2026-10-31 | 100.0% |
| HU | SK | 2016-01-01 | 2026-10-31 | 100.0% |
| IE_SEM | GB | 2016-01-01 | 2026-10-31 | 94.1% |
| IT_CALA | IT_SICI | 2020-12-31 | 2026-10-31 | 99.8% |
| IT_CALA | IT_SUD | 2020-12-31 | 2026-10-31 | 99.9% |
| IT_CNOR | IT_CSUD | 2017-12-31 | 2026-10-31 | 99.7% |
| IT_CNOR | IT_NORD | 2017-12-31 | 2026-10-31 | 99.8% |
| IT_CSUD | IT_CNOR | 2017-12-31 | 2026-10-31 | 99.8% |
| IT_CSUD | IT_SARD | 2017-12-31 | 2026-10-31 | 99.7% |
| IT_CSUD | IT_SUD | 2017-12-31 | 2026-10-31 | 99.7% |
| IT_CSUD | ME | 2019-12-26 | 2026-10-31 | 100.0% |
| IT_NORD | AT | 2018-09-30 | 2026-10-31 | 100.0% |
| IT_NORD | CH | 2016-01-01 | 2026-10-31 | 91.8% |
| IT_NORD | DE_LU | 2016-01-01 | 2018-09-30 | 68.4% |
| IT_NORD | FR | 2016-01-01 | 2026-10-31 | 92.0% |
| IT_NORD | IT_CNOR | 2017-12-31 | 2026-10-31 | 99.8% |
| IT_NORD | SI | 2016-01-01 | 2026-10-31 | 86.8% |
| IT_SARD | IT_CSUD | 2017-12-31 | 2026-10-31 | 99.8% |
| IT_SICI | IT_CALA | 2020-12-31 | 2026-10-31 | 99.8% |
| IT_SUD | GR | 2018-12-30 | 2026-10-31 | 100.0% |
| IT_SUD | IT_CALA | 2020-12-31 | 2026-10-31 | 99.7% |
| IT_SUD | IT_CSUD | 2017-12-31 | 2026-10-31 | 99.9% |
| LT | LV | 2016-01-01 | 2026-10-31 | 99.2% |
| LT | PL | 2016-01-01 | 2026-10-31 | 100.0% |
| LT | SE_4 | 2016-01-31 | 2026-10-31 | 99.9% |
| LV | EE | 2016-01-01 | 2026-10-31 | 98.0% |
| LV | LT | 2016-01-01 | 2026-10-31 | 99.2% |
| ME | IT_CSUD | 2019-12-26 | 2026-10-31 | 100.0% |
| ME | RS | 2016-01-01 | 2026-10-31 | 100.0% |
| MK | BG | 2016-01-01 | 2026-10-31 | 99.5% |
| MK | GR | 2016-01-01 | 2026-10-31 | 99.9% |
| MK | RS | 2016-01-01 | 2026-10-31 | 100.0% |
| NL | BE | 2016-01-01 | 2026-10-31 | 100.0% |
| NL | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| NL | DK_1 | 2019-09-07 | 2026-10-31 | 100.0% |
| NL | GB | 2016-01-01 | 2026-10-31 | 99.9% |
| NL | NO_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_1 | NO_2 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_1 | NO_3 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_1 | NO_5 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_1 | SE_3 | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_2 | DE_LU | 2020-09-01 | 2026-10-31 | 100.0% |
| NO_2 | DK_1 | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_2 | GB | 2021-06-14 | 2026-10-31 | 100.0% |
| NO_2 | NL | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_2 | NO_1 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_2 | NO_5 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_3 | NO_1 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_3 | NO_4 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_3 | NO_5 | 2016-02-14 | 2026-10-31 | 99.7% |
| NO_3 | SE_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_4 | FI | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_4 | NO_3 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_4 | SE_1 | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_4 | SE_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| NO_5 | NO_1 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_5 | NO_2 | 2016-01-01 | 2026-10-31 | 99.7% |
| NO_5 | NO_3 | 2016-02-14 | 2026-10-31 | 99.7% |
| PL | CZ | 2016-01-01 | 2026-10-31 | 100.0% |
| PL | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| PL | LT | 2016-01-01 | 2026-10-31 | 100.0% |
| PL | SE_4 | 2016-01-01 | 2026-10-31 | 100.0% |
| PL | SK | 2016-01-01 | 2026-10-31 | 100.0% |
| PT | ES | 2016-01-01 | 2026-10-31 | 100.0% |
| RO | BG | 2016-01-01 | 2026-10-31 | 100.0% |
| RO | HU | 2016-01-01 | 2026-10-31 | 100.0% |
| RO | RS | 2016-01-01 | 2026-10-31 | 100.0% |
| RS | BG | 2016-01-01 | 2026-10-31 | 100.0% |
| RS | HR | 2016-01-01 | 2026-10-31 | 100.0% |
| RS | HU | 2016-01-01 | 2026-10-31 | 100.0% |
| RS | ME | 2016-01-01 | 2026-10-31 | 100.0% |
| RS | MK | 2016-01-01 | 2026-10-31 | 100.0% |
| RS | RO | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_1 | FI | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_1 | NO_4 | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_1 | SE_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_2 | NO_3 | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_2 | NO_4 | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_2 | SE_1 | 2016-01-01 | 2026-10-31 | 99.9% |
| SE_2 | SE_3 | 2016-01-01 | 2026-10-31 | 99.9% |
| SE_3 | DK_1 | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_3 | FI | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_3 | NO_1 | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_3 | SE_2 | 2016-01-01 | 2026-10-31 | 99.9% |
| SE_3 | SE_4 | 2016-01-01 | 2026-10-31 | 99.9% |
| SE_4 | DE_LU | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_4 | DK_2 | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_4 | LT | 2016-01-31 | 2026-10-31 | 99.9% |
| SE_4 | PL | 2016-01-01 | 2026-10-31 | 100.0% |
| SE_4 | SE_3 | 2016-01-01 | 2026-10-31 | 99.9% |
| SI | AT | 2016-01-01 | 2026-10-31 | 99.9% |
| SI | DE_LU | 2016-01-01 | 2018-09-30 | 100.0% |
| SI | HR | 2016-01-01 | 2026-10-31 | 100.0% |
| SI | HU | 2022-06-29 | 2026-10-31 | 100.0% |
| SI | IT_NORD | 2016-01-01 | 2026-10-31 | 84.0% |
| SK | CZ | 2016-01-01 | 2026-10-31 | 100.0% |
| SK | HU | 2016-01-01 | 2026-10-31 | 100.0% |
| SK | PL | 2016-01-01 | 2026-10-31 | 100.0% |

## Caveats

- Flows are **realised** values for the hour, published after the fact; for a day-ahead forecast they are inputs only up to the cutoff (lagged). Capacities (NTC) are not in this table.
- Both directions can be positive in the same hour on some borders (source finding D); decide the netting convention in the model.
- Pairs that start late or end early (cables, zone changes) are visible in the table above; nothing is pruned here.
- Because of the German splice, before Oct 2018 `DE_LU` stands for the combined DE-AT-LU zone, so its edges in that period include Austria's borders (`DE_LU` <-> HU, SI, IT_NORD end on 2018-09-30 and continue as `AT` <-> HU, SI, IT_NORD). A model that starts in 2019 never sees this; one that uses 2016-18 should treat those three pairs as AT's.
