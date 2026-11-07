# Forecasting dataset construction

This package constructs data and audits forecasting eligibility. It does not
train models or modify the partner loaders, dataset pins, or original inputs.
Run from the repository root on `arsen_test`:

```bash
python -m epgnn.dataset_build audit --config configs/dataset_construction.json
python -m epgnn.dataset_build build --config configs/dataset_construction.json
python -m epgnn.dataset_build validate --config configs/dataset_construction.json
python -m unittest discover -s tests/dataset_build -v
```

For the supplied archives, first run
`python -m epgnn.dataset_build.import_archives --config configs/dataset_construction.json`.
The importer reads only the specified members and verifies their pinned checksums.

`audit --download` uses the existing GCS downloader with explicit release names
and `verify=True`, without pinning, pruning or uploading. Network or credential
failure is recorded and blocks a build when a required input is absent. Download
metadata is stored separately from `configs/datasets.json`.

## Inputs and identity

The dedicated construction configuration requests prices, load and flows at
v1.2 and weather at v1.1. The supplied ENTSO-E ZIP contains the three explicit
v1.2 filenames. Weather's supplied filename is unversioned: its schema and
coverage match the expected source, but its v1.1 release identity remains
unverified. Provided ZIP members are checked by ZIP CRC-32 on extraction, then
their MD5s are pinned in the construction configuration. SHA-256, sizes, archive
hashes and member names are preserved in the manifest. These checks establish
local reproducibility, not equality with a GCS release. That remaining limitation
is explicit in the findings. No original archive is overwritten.

The source paths can be updated to independently verified releases by editing
this dedicated configuration and recording the expected checksums. The partners'
existing `configs/datasets.json` is not changed. A new source identity creates a
new build fingerprint. The exact resolved original combined JAO generation is
recorded, including detailed CNECs, provenance, coverage and source manifests.

## Forecast and calendar contract

For every delivery hour in day D, the origin is **literal local noon in
Europe/Brussels on the previous calendar date**, converted to UTC. Winter noon is
11:00 UTC; summer noon is 10:00 UTC. Subtracting 24 elapsed hours from a local
delivery midnight or using a fixed UTC hour would mishandle this contract.
Hours are indexed in UTC; the local delivery days contain 23, 24 or 25 hours.
Both repeated autumn hours survive. Sources with microsecond and nanosecond
timestamps normalize explicitly to nanoseconds. Numeric epochs without an
explicit unit are rejected.

The output calendar is every UTC hour between the minimum and maximum source
price timestamps, including entirely missing source hours. Every distinct price
area has a row for every hour. Optional output_start/output_end bound only the
output, never the history reads. Bounds use inclusive start, exclusive end.
Other sources are left joins. There is no complete-case filtering or target
filling. October 2026 stays present wherever the source price calendar extends
there. The unchanged split dates label Brussels delivery dates, so rows after
2026-09-30 are `unassigned`. A UTC hour on September 30 may already belong to
October locally.

## Geography and structural availability

DE_AT_LU and DE_LU are separate identities at the 2018-10-01 Brussels delivery
boundary (2018-09-30 22:00 UTC). AT price availability is also structural before
that split. The source OUTSIDE_ZONE_LIFETIME status supplies additional
structural absences, including legacy areas and Italy. No structural lifetime
is inferred merely from a gap in observed prices. The dated crosswalk retains
original EICs, names, observed source ranges, and the German validity boundary.

Areas absent from the established canonical mapping are retained as
`ENTSOE:<original EIC>` with unresolved compatibility. They are not silently
dropped or substituted by a country proxy. Load/flow areas without a price area
survive in the source cache and audit; directed edges can have endpoints outside
the price node universe, flagged by `endpoints_in_node_grid`. Weather-only BA
and virtual JAO hubs remain in their original sources and are not given invented
price targets. Source/join reconciliation reports identify these unmatched rows.

Germany-only weather never attaches to DE_AT_LU. Weather labelled DE_LU is kept
on that distinct source identity with `germany_only_proxy_unresolved` geography.
Germany country/control-area load does not attach to the combined historical
zone. `price_absence_class` distinguishes structural absence, a missing source
row, invalid/missing value and a valid observation. Source flags remain nullable
where there was no source row; independent row-presence flags are always boolean.

## Values and admission

All physical values stay in their original units. Prices are EUR/MWh, flows,
load, scheduled/BEC exchanges and MaxBex are MW, temperature is degrees C,
capacity factors and RAM/Fmax are dimensionless. Negative prices are valid.
`price` uses original finite PriceEUR and IsValidTarget, excluding upstream
filled inputs and structural absences. `target_valid` is exactly price.notna().
Original price, validity, fill flags, repair reasons and `*_input` columns remain
under `price_source_*`, `load_source_*`, and `physical_source_*` for diagnosis.
No feature is derived from these upstream `*_input` fields.

The feature registry classifies **every master column** as admitted, conditional
or excluded. Default strict inputs are deterministic calendar columns and dated
CWE/CORE membership. Price lags are conditional by default because these archives
do not verify historical publication vintages, and corrected prices may reflect
later updates. To admit price lags, supply both an independently supported
publication-time column and its evidence in `price_publication`; absence of
either is rejected. Per-row observations must precede origin, interval completion
must be no later than origin, and publication must be no later than origin.
Ineligible values become NaN in the strict view, with an explicit eligibility
mask. Missing history remains NaN. No observation from a different zone or
bidding-zone identity is substituted.

Lags are exact elapsed-hour joins (24, 48, 168 hours for prices; 48 for weather;
48/168 for physical flows), not row shifts or same-local-slot joins. Lag columns
include observation, completion and publication timestamps and eligibility.
Even price_lag24 is not blanket admissible: many required intervals occur after
origin. A 48-hour lag alone does not establish availability either.

Load forecasts, ERA5/PECD weather, JAO summaries and borders remain conditional.
IsAvailabilityVerified and source update timestamps are preserved; they are not
reinterpreted as historical vintages. Weather anomaly climatology and low-wind
thresholds are upstream derived quantities and are not refitted or certified here.
JAO n_presolved and min_ram_ratio retain missing versus zero; membership is
independent of publication presence. Regional coverage fields remain null when
the original coverage record is absent, with a separate coverage-row indicator.

Physical flows are distinct from CWE allocated BEC and CORE scheduled exchanges.
Both directions are retained. A published border zero is a value; no publication
is missing. MaxBex is neither a physical interconnector rating nor a jointly
feasible import limit; it is never summed into one. No synthetic CNEC-name
filtering or persistent asset matching is performed. CNEC references point to
the original observations and provenance through content hashes and cnec_key.

Same-day price and physical-flow columns are targets/outcomes. Excluded source
diagnostics never become strict inputs automatically. The strict Parquet dataset
contains identifiers, origin, split, original target and validity mask alongside
admitted features. There is no fitted normalization, imputation or scaling;
future learned preprocessing must fit training data only. No leakage-free claim
is made for conditional experiments or for the archived sources themselves.

## Artifacts and inspection

`staging/forecast_dataset_v1.0/current` atomically selects a completed immutable
generation. `nodes/`, `edges/`, and `strict/` are monthly Parquet datasets, with
one file per UTC calendar month. Read with pandas or Arrow; the files are not
hive partitions. Source caches stream batches into monthly Parquets, bounding
join memory to at most two source months plus lag history. Detailed CNEC key
audits use external hash buckets. Reuse requires matching source/config/code/
runtime fingerprints; artifact and cache checksums guard corruption.

Each generation contains construction_config.json, build_fingerprint.json,
source_manifest.json, feature_registry.json, geography_crosswalk.json,
cnec_references.json, findings.json, coverage.parquet, join_reconciliation.json,
reference_comparison.json, DATASET_GUIDE.md and complete.json. Output key counts,
input row counts and actual joined-source counts are reconciled. Reference
comparisons quantify differences rather than asserting equality to v1.0.
The full calendar, historical German geography, unfilled values, exact lags and
CWE features deliberately differ from those reference tables.

The thin `notebooks/forecast_dataset_inspection.ipynb` calls package inspection
functions. It supports zone/date/variable selection, missingness by zone/month/
split/source, structural gaps, geography, DST, lag timelines, registry admission
and v1.0 comparisons. Structural coverage may inspect every partition. The
inspection selector rejects detailed test-target exploration. Default lag
examples use training dates. No test-target distributions are generated.

## Modelling blockers retained for the next phase

The audit reproduces the unchanged history loader's microsecond/nanosecond
comparison defect (integer timestamps and cutoff retaining inconsistent units), the
NaN-multiplication masking defect (`NaN * False` remains NaN), and the fixed UTC
noon mismatch in winter. These block use of the existing modelling path under
this contract. Their source hashes and numerical reproductions are recorded;
partner training code is intentionally unchanged in this construction phase.
The audit executes the original history function against both ns and us inputs;
the mismatch can empty the history or admit future observations, depending on
pandas' cutoff unit. It also executes the original masked loss when torch is
available, in addition to the dependency-independent numerical reproduction.
