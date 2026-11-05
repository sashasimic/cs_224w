# Weather (PECD / ERA5), per bidding zone

| | |
|---|---|
| Dataset id | `weather` |
| Version | **v1.1** |
| Bucket | `gs://cs224w-epgnn-datasets/weather/v1.1/` |
| Local file(s) | `datasets/weather_v1.1.md`, `datasets/weather_hourly_v1.1.parquet` |
| Source | Copernicus PECD 4.2 (ERA5-derived), per ENTSO-E bidding zone, 1 Jan 2010 – 30 Sep 2026 |

```python
from epgnn.paths import dataset_path
df = pd.read_parquet(dataset_path("weather", "weather_hourly.parquet"))
```

Version history:

| Version | What |
|---|---|
| v1.0 | country-level hourly and daily features; DK/IT/NO/SE bidding zones averaged into countries, DE merged with LU; 1991 – 2026 |
| v1.1 | **hourly weather per ENTSO-E bidding zone** (45 zones; codes now `DE_LU`, `DK_1`, `NO_3`, `IT_NORD`, …, matching ENTSO-E); 2010 – Sep 2026. Hourly only: the daily file (`heatwave`, `cdh`, `tmax_3d`, …) has not yet been rebuilt per zone and is **not** in this version. Zones added: all DK/IT/NO/SE bidding zones, `DE_LU`, `IE_SEM`. Removed: country aggregates `DE DK DK_W IT NO SE`, and `CS CY IE IS LU NI UA_W`. |

---

## Hourly weather by bidding zone, 2010–2026

`weather_hourly.parquet`: hourly weather for 45 ENTSO-E bidding zones, 1 Jan 2010 to 30 Sep 2026 (6.6 million rows, 74 MB). There is one row per zone per hour.

| Column | Meaning |
| --- | --- |
| `time_utc` | Start of the hour, UTC (same convention as ENTSO-E) |
| `zone` | ENTSO-E bidding zone code (`DE_LU`, `DK_1`, `IT_NORD`, `NO_3`, `SE_4`, `IE_SEM`, ...) |
| `temp_pw` | Population-weighted temperature, °C |
| `temp_anom` | `temp_pw` minus its 1991–2020 mean for that day of year and hour |
| `wind_cf` | Onshore wind capacity factor (0–1), existing fleet |
| `low_wind` | 1 if `wind_cf` is at or below the zone's 10th percentile for that month (2016–2021) |
| `solar_cf` | PV capacity factor (0–1), mean of rooftop and utility-scale types |

Zones (45): `AT BA BE BG CH CZ DE_LU DK_1 DK_2 EE ES FI FR GB GR HR HU IE_SEM IT_CALA IT_CNOR IT_CSUD IT_NORD IT_SARD IT_SICI IT_SUD LT LV ME MK NL NO_1 NO_2 NO_3 NO_4 NO_5 PL PT RO RS SE_1 SE_2 SE_3 SE_4 SI SK`

**Notes**
- **Wind and solar start in 2015.** They are empty for 2010–2014, and for the final hour of the file.
- **Leakage:** values are actual weather for that hour. For day-ahead forecasting, lag them by at least one day, or treat same-day values as a perfect forecast.
- **Southern Norway:** NO_1, NO_2 and NO_5 have identical weather. The source data has only one zone covering all three.
- **Italy:** IT_CALA became a bidding zone only in 2021, so weather rows exist before then but there are no matching prices.
- `DE_LU` uses Germany's weather. `IE_SEM` combines Ireland and Northern Ireland, weighted by population.
- **Zone codes differ from the ENTSO-E price/load tables' `AreaName`** (`NO_1` vs `NO1`, `IT_NORD` vs `IT-North`, `DE_LU` vs `DE-LU`). The zone mapping file (`configs/zones.csv`) carries the weather code per zone.
- **No daily file in v1.1.** The v1.0 daily features (`heatwave`, `cdh`, `tmax_3d`, `ta_7d`, `precip_30d_anom`, `share_hot`) were country-level and are not carried over. Rebuild per zone from this hourly file, or pull `weather@v1.0` for the country-level daily file.

**Source:** Copernicus Climate Change Service, Pan-European Climate Database v4.2 (from ERA5), https://cds.climate.copernicus.eu/datasets/sis-energy-pecd. Licence CC BY 4.0; attribute Copernicus/C3S.
