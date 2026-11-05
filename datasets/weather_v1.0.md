# Weather (PECD / ERA5)

| | |
|---|---|
| Dataset id | `weather` |
| Version | **v1.0** |
| Bucket | `gs://cs224w-epgnn-datasets/weather/v1.0/` |
| Local file(s) | `datasets/weather_hourly_v1.0.parquet`, `datasets/weather_daily_v1.0.parquet`, `datasets/weather_v1.0.md` |
| Source | Copernicus PECD 4.2 (ERA5-derived), aggregated to country level, 1991 – Sep 2026 |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("weather", "weather_hourly.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | country-level hourly and daily features; DK/IT/NO/SE bidding zones averaged into countries, DE merged with LU |

---

Hourly weather for 38 European zones (the country codes in the ENTSO-E load file), from January 1991 to September 2026, plus daily and hourly weather features derived from it.

## How it was sourced

The data comes from the **Pan-European Climate Database v4.2 (PECD)**, built by Copernicus (C3S) with ENTSO-E from the ERA5 reanalysis. It was downloaded through the Copernicus Climate Data Store API.

PECD already aggregates weather to ENTSO-E bidding zones, including a population-weighted temperature, so gridded data did not have to be matched to zone shapes. Bidding zones were combined into countries: **DK, IT, NO and SE with equal weights, and DE together with LU**.

Licence: CC BY 4.0. Map outlines come from entsoe-py and Natural Earth.

## Variables and coverage

- **Hourly, 1990 – Sep 2026:** population-weighted temperature, air temperature, solar radiation, precipitation.
- **Hourly, 2015 – Sep 2026:** 100 m wind speed, solar capacity factors (rooftop and utility-scale), onshore and offshore wind capacity factors.
- **Weekly, 2015 – Sep 2026:** hydro reservoir and run-of-river inflows (downloaded, not yet used).

## Missing values and limits

- No gaps inside each variable's date range. Wind and solar are empty before 2015.
- 2026 covers January to September only.
- Southern Norway (NO1, NO2, NO5) shares a single PECD series.
- Wind capacity factors come in sub-national wind regions and are averaged to each zone.
- UA_W (Ukraine's Burshtyn island) has data but no map outline. CS (Serbia and Montenegro) appears only in 2006 in the load file.
- Cooling bases are fitted from 2006–2015 load for 17 zones; the other 21 use 22 °C. The Latvia and Lithuania bases (16–17 °C) need checking.

## The 10 weather hazard metrics

| # | Metric | What it captures |
|---|---|---|
| 1 | Population-weighted temperature (hourly) | Heat where people live, which drives cooling demand |
| 2 | Temperature anomaly (hourly) | Departure from the zone's 1991–2020 normal for that hour and date |
| 3 | Daily max, min and mean temperature | Peak heat, and warm nights that keep demand high |
| 4 | Cooling degree hours (daily) | Total heat above the zone's cooling base |
| 5 | Heatwave flag, hot-day run length and magnitude | 3+ days above the zone's 90th percentile, May–Sep; run length is the forecast-safe version |
| 6 | Wind capacity factor and low-wind flag (hourly) | Wind output, and calm spells below the zone's 10th percentile |
| 7 | Solar capacity factor (hourly) | Solar output, including losses from hot panels |
| 8 | Heat persistence (3-day mean max temperature) | Heat building up over several days |
| 9 | 7-day air temperature and 30-day precipitation anomaly | River-cooling stress for thermal plants, and drought |
| 10 | Share of zones hot at once, and neighbour anomaly difference | Whether heat is synchronised, which limits how much neighbours can help |

## Outputs available

Two parquet files (see header for the versioned names).

**`weather_hourly.parquet`** — one row per `(zone, time_utc)`, 38 zones × 313,368 hours.

| Column | Type | Meaning |
|---|---|---|
| `time_utc` | timestamp (UTC) | hour start |
| `zone` | string | country code: `AT BA BE BG CH CS CY CZ DE DK DK_W EE ES FI FR GB GR HR HU IE IS IT LT LU LV ME MK NI NL NO PL PT RO RS SE SI SK UA_W` |
| `temp_pw` | float32 | population-weighted temperature, °C |
| `temp_anom` | float64 | departure from the 1991–2020 normal for that hour and date, °C |
| `wind_cf` | float32 | wind capacity factor 0–1 (onshore/offshore combined); null before 2015 |
| `low_wind` | float64 | 1 if below the zone's 10th-percentile wind; null before 2015 |
| `solar_cf` | float32 | solar capacity factor 0–1 (rooftop/utility combined); null before 2015 |

**`weather_daily.parquet`** — one row per `(zone, date)`, 38 zones × 13,058 days (includes a 2026-10-01 row built from partial data; filter by date).

| Column | Type | Meaning |
|---|---|---|
| `date` | date | calendar day |
| `zone` | string | as hourly |
| `tmax`, `tmin`, `tmean` | float32 | daily max / min / mean of `temp_pw`, °C |
| `cdh` | float64 | cooling degree hours above the zone's cooling base |
| `heatwave` | int8 | 1 if in a run of 3+ days above the zone's 90th percentile for that date. **Fires year-round**, not only May–Sep as metric 5 above suggests |
| `hot_run_days` | int64 | length of the current hot-day run |
| `hw_mag`, `hw_mag_cum` | float64 | heatwave magnitude that day / accumulated over the run; 0 outside May–Sep |
| `tmax_3d` | float64 | 3-day mean of `tmax` |
| `temp_anom_mean` | float64 | daily mean of `temp_anom` |
| `ta_7d` | float64 | 7-day mean air temperature |
| `precip_30d_anom` | float64 | 30-day precipitation anomaly, ratio (−1 = no rain) |
| `share_hot` | float64 | share of the 38 zones that are hot that day; same value for every zone on a date |

The data is ERA5 reanalysis, i.e. observed weather, not a forecast. Using day-D values as model inputs is look-ahead; see `docs/milestone-plan.md` §2.4.
