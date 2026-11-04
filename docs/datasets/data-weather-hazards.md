# V1.0 Data — Weather Hazards

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

- `hourly.parquet` and `daily.parquet`: model features for the graph, in the team GDrive.
