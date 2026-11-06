"""Shared rules for the JAO flow-based constraint datasets (jao_cnec_hourly).

The build scripts in scripts/datasets/ import these so the rules live in one place; the dataset doc
describes the same rules in words. NEIGHBOURS is the AC-border table for zone-to-zone PTDF spreads.
"""
import re

# A constraint is synthetic (not a physical line) if its source name matches this: long-term
# allocation corners, import/export and external net-position limits, equality constraints.
# Their Fmax is a placeholder, so RAM/Fmax is meaningless; only physical rows are used.
SYNTHETIC = re.compile(r"LTA|IMPORT|EXPORT|EC_|EXTERNAL|CONSTRAINT|NET POSITION|NETPOSITION|MAXBEX")

# Zone codes, identical to the weather_hourly codes. DE_AT_LU is the combined German-Austrian zone
# before delivery day 2018-10-01; it is reported under DE_LU (AT is not in the domain then).
ZONES = ["AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK"]
SOURCE_ZONES = ZONES + ["DE_AT_LU"]

# AC-interconnected neighbours inside the flow-based domain, used for zone-to-zone PTDF spreads.
NEIGHBOURS = {
    "AT": ["DE_LU", "DE_AT_LU", "CZ", "HU", "SI"],
    "BE": ["FR", "NL", "DE_LU", "DE_AT_LU"],
    "CZ": ["DE_LU", "AT", "PL", "SK"],
    "DE_LU": ["AT", "BE", "CZ", "FR", "NL", "PL"],
    "DE_AT_LU": ["BE", "FR", "NL", "CZ"],
    "FR": ["BE", "DE_LU", "DE_AT_LU"],
    "HR": ["HU", "SI"],
    "HU": ["AT", "HR", "RO", "SI", "SK"],
    "NL": ["BE", "DE_LU", "DE_AT_LU"],
    "PL": ["DE_LU", "CZ", "SK"],
    "RO": ["HU"],
    "SI": ["AT", "HR", "HU"],
    "SK": ["CZ", "HU", "PL"],
}
