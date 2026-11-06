"""Canonical zone ids and how each source maps onto them.

The canonical `zone` code is the weather_hourly / JAO code (DE_LU, NO_1, IT_NORD ...). ENTSO-E tables
identify areas by EIC `AreaCode`; the v1.x tables also carry a display name, which is what the mapping
below is keyed on (names are stable across the v1.x builds, EICs are looked up from the data).

The agg_* datasets contain every zone that has both weather and ENTSO-E prices (44 zones). Bosnia (BA)
has weather but no ENTSO-E price series and is left out.
"""
import pandas as pd

# ENTSO-E display name (as in entsoe_prices / entsoe_load_forecast / entsoe_physical_flows) -> zone
ENTSOE_NAME_TO_ZONE = {
    "Austria (AT)": "AT", "Belgium (BE)": "BE", "Bulgaria (BG)": "BG", "Switzerland (CH)": "CH",
    "Czech Republic (CZ)": "CZ", "DE-LU": "DE_LU", "DK1": "DK_1", "DK2": "DK_2", "Estonia (EE)": "EE",
    "Spain (ES)": "ES", "Finland (FI)": "FI", "France (FR)": "FR", "GB": "GB", "Greece (GR)": "GR",
    "Croatia (HR)": "HR", "Hungary (HU)": "HU", "IE(SEM)": "IE_SEM", "IT-Calabria": "IT_CALA",
    "IT-Centre-North": "IT_CNOR", "IT-Centre-South": "IT_CSUD", "IT-North": "IT_NORD", "IT-Sardinia": "IT_SARD",
    "IT-Sicily": "IT_SICI", "IT-South": "IT_SUD", "Lithuania (LT)": "LT", "Latvia (LV)": "LV",
    "Montenegro (ME)": "ME", "North Macedonia (MK)": "MK", "Netherlands (NL)": "NL",
    "NO1": "NO_1", "NO2": "NO_2", "NO3": "NO_3", "NO4": "NO_4", "NO5": "NO_5",
    "Poland (PL)": "PL", "Portugal (PT)": "PT", "Romania (RO)": "RO", "Serbia (RS)": "RS",
    "SE1": "SE_1", "SE2": "SE_2", "SE3": "SE_3", "SE4": "SE_4", "Slovenia (SI)": "SI", "Slovakia (SK)": "SK",
}

# Germany: before market day 2018-10-01 the bidding zone was DE-AT-LU; from then on DE-LU (and AT separate).
# The agg tables splice both onto zone DE_LU, switching at the first hour of market day 2018-10-01,
# and record which EIC each row came from in `eic`.
DE_SPLIT_TS = pd.Timestamp("2018-09-30 22:00", tz="UTC")
DE_PRE_SPLIT_NAME = "DE-AT-LU"
DE_ZONE = "DE_LU"

ZONES = sorted(ENTSOE_NAME_TO_ZONE.values())


def zone_of(name: pd.Series, ts: pd.Series) -> pd.Series:
    """Map ENTSO-E display names to zone codes, applying the German splice. NaN where unmapped."""
    z = name.map(ENTSOE_NAME_TO_ZONE)
    pre = ts < DE_SPLIT_TS
    z = z.mask((name == DE_PRE_SPLIT_NAME) & pre, DE_ZONE)     # DE-AT-LU rows before the split -> DE_LU
    z = z.mask((name == "DE-LU") & pre, pd.NA)                   # DE-LU rows before the split are lifetime filler
    z = z.mask((name == DE_PRE_SPLIT_NAME) & ~pre, pd.NA)       # DE-AT-LU rows after the split likewise
    return z
