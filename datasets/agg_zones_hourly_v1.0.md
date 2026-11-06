# Zone-hour table: prices, load forecast and weather per bidding zone

| | |
|---|---|
| Dataset id | `agg_zones_hourly` |
| Version | **v1.0** |
| Bucket | `gs://cs224w-epgnn-datasets/agg_zones_hourly/v1.0/` |
| Local file(s) | `datasets/agg_zones_hourly_v1.0.md`, `datasets/agg_zones_hourly_v1.0.parquet` |
| Source | derived from the datasets listed under *Built from*, by `scripts/build_agg_zones_hourly.py` |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("agg_zones_hourly", "agg_zones_hourly.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | first build: 44 zones x 94,967 hours from entsoe_prices, entsoe_load_forecast, weather_hourly |

## Built from

| Source dataset | Version | File | md5 |
|---|---|---|---|
| `entsoe_prices` | v1.2 | `entsoe_prices.parquet` | `b/P5uhqjw38oKYqMyUoJhw==` |
| `entsoe_load_forecast` | v1.2 | `entsoe_load_forecast.parquet` | `WDNjm7PA8IHcw1lEPHWZMA==` |
| `weather_hourly` | v1.1 | `weather_hourly.parquet` | `Kyib4+sscdL9mxhRIuD2mg==` |

Rebuild with `scripts/build_agg_zones_hourly.py`; the zone mapping is `epgnn/zones.py`.

## What it is

One row per **(zone, hour)**: the zone-node inputs and targets, already joined. 4,178,548 rows = 44 zones x 94,967 hours, 2016-01-01 00:00:00+00:00 to 2026-10-31 22:00:00+00:00 UTC.

Nothing is filtered or re-cleaned: the price, load and weather values are the source datasets' values, with the source quality flags carried along under new names. Use the source docs for what the flags mean.

## Rules used to build it

1. **Zone id.** The canonical `zone` is the weather / JAO code (`DE_LU`, `NO_1`, `IT_NORD`). ENTSO-E areas are mapped by their display name to that code (`epgnn.zones.ENTSOE_NAME_TO_ZONE`); the ENTSO-E EIC code of the row is kept in `eic`.
2. **Which zones.** Every zone that has both an ENTSO-E price series and a weather series: 44 zones. Bosnia (BA) has weather but no ENTSO-E prices and is not included. No other zone-level filtering (GB, IE_SEM, ME, MK, IT_CALA, IT_SICI are in, with whatever coverage the sources have).
3. **Germany.** Before market day 2018-10-01 ENTSO-E reports the combined zone DE-AT-LU; from then on DE-LU. Both are spliced onto `zone = DE_LU`, switching at 2018-09-30 22:00:00+00:00 UTC (first hour of market day 2018-10-01, Brussels time); `eic` shows which one each row came from. The source rows that are lifetime filler on the other side of the split (DE-LU before, DE-AT-LU after) are dropped. AT is its own zone from Oct 2018; before that its prices are masked in the source.
4. **Grid and joins.** The row grid is the price table's grid: every zone x every ENTSO-E hour (2016-01-01 to 2026-10-31). Load forecast and weather are left-joined on (`zone`, `ts_utc`). A missing load row gives `load_status = NO_ROW`, `load_valid = False`; weather is absent after 2026-09-30 23:00:00+00:00 UTC (`weather_valid = False`).
5. **Column names.** Source columns are renamed only to avoid clashes between the three sources (see schema); values are untouched. `market_day_cet` is the price table's `MarketDayCET`.

## Schema

| Column | From | Source column |
|---|---|---|
| `zone` | derived | canonical zone code |
| `eic` | entsoe_prices | `AreaCode` |
| `ts_utc` | entsoe_prices | `DeliveryHourUTC` |
| `market_day_cet` | entsoe_prices | `MarketDayCET` |
| `price_eur` | entsoe_prices | `PriceEUR` (target; null when not valid) |
| `price_eur_input` | entsoe_prices | `PriceEUR_input` (history input, short gaps patched) |
| `price_valid` | entsoe_prices | `IsValidTarget` |
| `price_status` | entsoe_prices | `row_status` |
| `price_filled` | entsoe_prices | `is_filled` |
| `load_mw` | entsoe_load_forecast | `LoadForecastMW` |
| `load_mw_input` | entsoe_load_forecast | `LoadForecastMW_input` |
| `load_valid` | entsoe_load_forecast | `IsValidValue` (False when no row) |
| `load_status` | entsoe_load_forecast | `row_status` (`NO_ROW` when no row) |
| `load_filled` | entsoe_load_forecast | `is_filled` |
| `temp_pw`, `temp_anom`, `wind_cf`, `low_wind`, `solar_cf` | weather_hourly | same names |
| `weather_valid` | derived | weather row present for the zone-hour |

## Coverage per zone (share of hours)

| Zone | price valid | load valid | weather |
|---|---|---|---|
| AT | 74.6% | 100.0% | 99.2% |
| BE | 100.0% | 99.9% | 99.2% |
| BG | 91.9% | 99.9% | 99.2% |
| CH | 100.0% | 100.0% | 99.2% |
| CZ | 100.0% | 99.8% | 99.2% |
| DE_LU | 100.0% | 98.8% | 99.2% |
| DK_1 | 100.0% | 100.0% | 99.2% |
| DK_2 | 100.0% | 100.0% | 99.2% |
| EE | 100.0% | 99.1% | 99.2% |
| ES | 100.0% | 100.0% | 99.2% |
| FI | 100.0% | 99.9% | 99.2% |
| FR | 100.0% | 100.0% | 99.2% |
| GB | 46.1% | 50.2% | 99.2% |
| GR | 100.0% | 100.0% | 99.2% |
| HR | 83.4% | 100.0% | 99.2% |
| HU | 100.0% | 99.9% | 99.2% |
| IE_SEM | 99.7% | 50.5% | 99.2% |
| IT_CALA | 53.8% | 53.8% | 99.2% |
| IT_CNOR | 100.0% | 100.0% | 99.2% |
| IT_CSUD | 100.0% | 100.0% | 99.2% |
| IT_NORD | 100.0% | 100.0% | 99.2% |
| IT_SARD | 99.9% | 100.0% | 99.2% |
| IT_SICI | 100.0% | 100.0% | 99.2% |
| IT_SUD | 100.0% | 100.0% | 99.2% |
| LT | 100.0% | 99.3% | 99.2% |
| LV | 100.0% | 99.8% | 99.2% |
| ME | 32.5% | 99.8% | 99.2% |
| MK | 32.0% | 96.2% | 99.2% |
| NL | 100.0% | 100.0% | 99.2% |
| NO_1 | 100.0% | 100.0% | 99.2% |
| NO_2 | 100.0% | 100.0% | 99.2% |
| NO_3 | 100.0% | 100.0% | 99.2% |
| NO_4 | 100.0% | 100.0% | 99.2% |
| NO_5 | 100.0% | 100.0% | 99.2% |
| PL | 97.1% | 100.0% | 99.2% |
| PT | 100.0% | 100.0% | 99.2% |
| RO | 99.9% | 100.0% | 99.2% |
| RS | 91.0% | 99.9% | 99.2% |
| SE_1 | 100.0% | 100.0% | 99.2% |
| SE_2 | 100.0% | 100.0% | 99.2% |
| SE_3 | 100.0% | 100.0% | 99.2% |
| SE_4 | 100.0% | 100.0% | 99.2% |
| SI | 100.0% | 99.4% | 99.2% |
| SK | 99.9% | 100.0% | 99.2% |

## Caveats

- Validity differences between zones are inherited from the sources (e.g. AT and DE_LU prices before Oct 2018, Poland 2016-17, GB after 2020, IE_SEM load from 2022); nothing is masked or filled here beyond what the sources did.
- Weather ends on 2026-09-30; the ENTSO-E tables run to 2026-10-31.
- Weather is observed ERA5, not a forecast; using day-D weather as an input is look-ahead (see the weather_hourly doc).
