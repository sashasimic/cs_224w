"""Build and validate a local relational catalog over immutable forecasting inputs."""

import json
import os
import shutil
import tempfile
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from epgnn.boosting import _check_artifacts
from epgnn.dataset_build.contracts import forecast_origin
from epgnn.dataset_build.storage import ROOT, canonical, digest, json_write

CORE = ("AT", "BE", "CZ", "DE_LU", "FR", "HR", "HU", "NL", "PL", "RO", "SI", "SK")
PTDF_ZONES = CORE + ("ALBE", "ALDE", "DE_AT_LU")
HOUR_NS = 3_600_000_000_000
RAW = {
    "price": "entsoe_prices",
    "load_forecast": "entsoe_load_forecast",
    "weather": "weather_hourly",
    "physical_flow": "entsoe_physical_flows",
    "jao_zone_summary": "jao_hourly",
    "jao_border": "jao_border_hourly",
    "cnec_coefficients": "jao_cnec_hourly",
    "cnec_provenance": "cnec_provenance",
}
COVERAGE = "jao_metadata/coverage.csv"
ZONE_HISTORY = "jao_metadata/zone_history.csv"


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def identity(value):
    return sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_settings(config):
    settings = read_json(config)
    for key in ("dataset", "benchmark_arrays", "staging", "runs"):
        settings[key] = str((ROOT / settings[key]).resolve())
    return settings


def connect(path, read_only=True):
    return duckdb.connect(
        str(path), read_only=read_only, config={"threads": 4, "memory_limit": "3GB"}
    )


def register(connection, name, frame):
    connection.register("frame", frame)
    connection.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
    connection.unregister("frame")


def views(connection, dataset):
    master = literal(dataset / "nodes" / "*.parquet")
    strict = literal(dataset / "strict" / "*.parquet")
    connection.execute(f"""
        CREATE VIEW forecast_target AS
        SELECT zone, ts_utc, price, target_valid, price_absence_class,
               price_structurally_available
        FROM read_parquet({master});
        CREATE VIEW strict_forecast AS SELECT * FROM read_parquet({strict});
        CREATE VIEW price AS
        SELECT r.*, m.zone, m.compatibility, DeliveryHourUTC AS effective_utc,
               DeliveryHourUTC + INTERVAL '1 hour' AS interval_end_utc,
               NULL::TIMESTAMPTZ AS published_utc, 'entsoe_prices' AS release_id,
               'excluded' AS predictor_admission
        FROM raw_price r LEFT JOIN source_area_map m
          ON m.source = 'entsoe_prices' AND m.area_code = r.AreaCode
         AND (m.valid_from_utc IS NULL OR r.DeliveryHourUTC >= m.valid_from_utc)
         AND (m.valid_until_utc_exclusive IS NULL OR r.DeliveryHourUTC < m.valid_until_utc_exclusive);
        CREATE VIEW load_forecast AS
        SELECT r.*, m.zone, m.compatibility, DeliveryHourUTC AS effective_utc,
               DeliveryHourUTC + INTERVAL '1 hour' AS interval_end_utc,
               NULL::TIMESTAMPTZ AS published_utc, 'entsoe_load_forecast' AS release_id,
               'conditional' AS predictor_admission
        FROM raw_load_forecast r LEFT JOIN source_area_map m
          ON m.source = 'entsoe_load_forecast' AND m.area_code = r.AreaCode
         AND (m.valid_from_utc IS NULL OR r.DeliveryHourUTC >= m.valid_from_utc)
         AND (m.valid_until_utc_exclusive IS NULL OR r.DeliveryHourUTC < m.valid_until_utc_exclusive);
        CREATE VIEW weather AS
        SELECT *, time_utc AS effective_utc, time_utc + INTERVAL '1 hour' AS interval_end_utc,
               NULL::TIMESTAMPTZ AS published_utc, 'weather_hourly' AS release_id,
               'conditional' AS predictor_admission
        FROM raw_weather;
        CREATE VIEW jao_zone_summary AS
        SELECT *, ts_utc + INTERVAL '1 hour' AS interval_end_utc,
               NULL::TIMESTAMPTZ AS published_utc, 'jao_hourly' AS release_id,
               'conditional' AS predictor_admission
        FROM raw_jao_zone_summary;
        CREATE VIEW border_observation AS
        SELECT DeliveryHourUTC AS ts_utc, OutAreaCode AS source_from, InAreaCode AS source_to,
               a.zone AS from_zone, b.zone AS to_zone,
               'physical_flow' AS kind, FlowMW AS value_mw, IsValidValue AS source_valid,
               'entsoe_physical_flows' AS release_id, NULL::VARCHAR AS regime
        FROM raw_physical_flow r LEFT JOIN source_area_map a
          ON a.source = 'entsoe_physical_flows' AND a.area_code = r.OutAreaCode
         AND (a.valid_from_utc IS NULL OR r.DeliveryHourUTC >= a.valid_from_utc)
         AND (a.valid_until_utc_exclusive IS NULL OR r.DeliveryHourUTC < a.valid_until_utc_exclusive)
        LEFT JOIN source_area_map b
          ON b.source = 'entsoe_physical_flows' AND b.area_code = r.InAreaCode
         AND (b.valid_from_utc IS NULL OR r.DeliveryHourUTC >= b.valid_from_utc)
         AND (b.valid_until_utc_exclusive IS NULL OR r.DeliveryHourUTC < b.valid_until_utc_exclusive)
        UNION ALL
        SELECT ts_utc, source_from_zone, source_to_zone, from_zone, to_zone,
               CASE WHEN regime = 'CWE' THEN 'allocated_BEC' ELSE 'scheduled_exchange' END,
               flow, flow IS NOT NULL, 'jao_border_hourly', regime FROM raw_jao_border
        UNION ALL
        SELECT ts_utc, source_from_zone, source_to_zone, from_zone, to_zone,
               'MaxBex', maxbex, maxbex IS NOT NULL, 'jao_border_hourly', regime FROM raw_jao_border;
        CREATE VIEW border_hour AS
        SELECT *, ts_utc + INTERVAL '1 hour' AS interval_end_utc, NULL::TIMESTAMPTZ AS published_utc,
               CASE WHEN kind = 'physical_flow' THEN 'excluded' ELSE 'conditional' END
                   AS predictor_admission FROM border_observation;
        CREATE VIEW cnec_wide AS
        SELECT c.cnec_key, c.ts_utc, c.regime, c.timestamp_status, c.presolved,
               TRY_CAST(CASE WHEN c.regime = 'CORE' THEN p.Ram
                             ELSE p.RemainingAvailableMargin END AS DOUBLE) AS ram_mw,
               TRY_CAST(p.Fmax AS DOUBLE) AS fmax_mw, c.ram_fmax,
               p.source_sha256, p._analysis_member AS source_member, p._analysis_row AS source_row,
               'jao_cnec_hourly' AS release_id, 'cnec_provenance' AS provenance_release_id,
               {", ".join(f"c.ptdf_{z}" for z in PTDF_ZONES)}
        FROM raw_cnec_coefficients c POSITIONAL JOIN raw_cnec_provenance p;
        CREATE VIEW cnec_observation AS
        SELECT * EXCLUDE ({", ".join(f"ptdf_{z}" for z in PTDF_ZONES)}) FROM cnec_wide;
        CREATE VIEW cnec_ptdf AS
        SELECT cnec_key, ts_utc, regime, zone, coefficient,
               CASE WHEN zone IN ('DE_LU', 'DE_AT_LU') THEN 'DE' ELSE zone END AS source_zone
        FROM raw_cnec_coefficients
        UNPIVOT (coefficient FOR zone IN ({", ".join(f"ptdf_{z} AS {z}" for z in PTDF_ZONES)}));
        CREATE VIEW core_forecast_domain AS
        SELECT h.ts_utc AS delivery_utc, h.forecast_origin_utc, h.split,
               d.ts_utc AS source_hour, d.domain_present, d.published_utc
        FROM delivery_hour h LEFT JOIN jao_domain_hour d
          ON d.ts_utc = h.ts_utc - INTERVAL '48 hours' AND d.regime = 'CORE'
         AND d.interval_end_utc <= h.forecast_origin_utc
         AND (d.published_utc IS NULL OR d.published_utc <= h.forecast_origin_utc);
        CREATE VIEW strict_core_forecast_domain AS
        SELECT * FROM core_forecast_domain WHERE published_utc IS NOT NULL AND domain_present;
    """)


def dimensions(connection, dataset, sources):
    files = literal(dataset / "nodes" / "*.parquet")
    connection.execute(f"""
        CREATE TABLE delivery_hour AS
        SELECT DISTINCT ts_utc, delivery_date, forecast_origin_utc, interval_end_utc,
                        hour_local, dow, month, utc_offset_hours, split
        FROM read_parquet({files});
    """)
    crosswalk = pd.read_json(dataset / "geography_crosswalk.json")
    for column in (
        "first_utc",
        "last_utc",
        "valid_from_utc",
        "valid_until_utc_exclusive",
    ):
        crosswalk[column] = pd.to_datetime(
            crosswalk[column], utc=True, format="ISO8601"
        )
    register(connection, "source_area_map", crosswalk)
    history = pd.read_csv(sources[ZONE_HISTORY]["path"])
    # Published JAO membership ranges use Brussels delivery days.
    for column, offset in (("first_delivery_date", 0), ("last_delivery_date", 1)):
        day = pd.to_datetime(history[column]) + pd.Timedelta(days=offset)
        name = "valid_from_utc" if not offset else "valid_until_utc_exclusive"
        history[name] = day.dt.tz_localize("Europe/Brussels").dt.tz_convert("UTC")
    register(connection, "jao_area_map", history)
    price_zones = connection.sql(
        f"SELECT DISTINCT zone FROM read_parquet({files})"
    ).df()
    register(connection, "priced_area", price_zones)
    connection.execute("""
        CREATE TABLE area AS
        SELECT zone, bool_or(in_price_grid) AS in_price_grid
        FROM (SELECT zone, true AS in_price_grid FROM priced_area
              UNION ALL SELECT zone, false FROM source_area_map
              UNION ALL SELECT DISTINCT zone, false FROM raw_weather
              UNION ALL SELECT zone, false FROM jao_area_map)
        GROUP BY zone;
    """)
    coverage = pd.read_csv(sources[COVERAGE]["path"])
    coverage["ts_utc"] = pd.to_datetime(coverage.ts_utc, utc=True, format="ISO8601")
    coverage["interval_end_utc"] = coverage.ts_utc + pd.Timedelta(hours=1)
    coverage["published_utc"] = pd.Series(
        pd.NaT, index=coverage.index, dtype="datetime64[ns, UTC]"
    )
    coverage["release_id"] = COVERAGE
    coverage["predictor_admission"] = "conditional"
    register(connection, "jao_domain_hour", coverage)
    build_metadata = read_json(sources["jao_metadata/build.json"]["path"])
    releases = []
    for name, info in sources.items():
        if name.startswith(("archive/", "reference/")):
            continue
        created = (
            build_metadata.get("created_at")
            if name.startswith("jao_metadata/")
            else None
        )
        releases.append(
            {
                "release_id": name,
                "path": info["path"],
                "sha256": info["sha256"],
                "version": info.get("version"),
                "archive_generated_at": created,
                "acquired_utc": None,
                "published_utc": None,
            }
        )
    register(connection, "source_release", pd.DataFrame(releases))


def verify(connection, sources):
    """Reconcile keys, source counts and exact calendar joins before publication."""
    checks = {
        "delivery_hour": "ts_utc",
        "area": "zone",
        "source_release": "release_id",
        "source_area_map": "row(source, area_code)",
        "forecast_target": "row(zone, ts_utc)",
        "price": "row(AreaCode, effective_utc)",
        "load_forecast": "row(AreaCode, effective_utc)",
        "weather": "row(zone, effective_utc)",
        "jao_zone_summary": "row(regime, zone, ts_utc)",
        "jao_domain_hour": "row(regime, ts_utc)",
        "raw_jao_border": "row(regime, from_zone, to_zone, ts_utc)",
        "raw_physical_flow": "row(OutAreaCode, InAreaCode, DeliveryHourUTC)",
        "raw_cnec_coefficients": "cnec_key",
    }
    counts = {}
    for name, key in checks.items():
        rows, keys = connection.execute(
            f"SELECT count(*), count(DISTINCT {key}) FROM {name}"
        ).fetchone()
        if rows != keys:
            raise ValueError(f"{name} has duplicate or null keys")
        counts[name] = rows
    # The audited source keys are non-null. Check the larger flow table too.
    for table, source in RAW.items():
        name = f"raw_{table}"
        rows = connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        if rows != pq.ParquetFile(sources[source]["path"]).metadata.num_rows:
            raise ValueError(f"{name} source row count differs from its pinned Parquet")
        counts[name] = rows
    integrity = {
        "target validity": "SELECT count(*) FROM forecast_target WHERE target_valid IS DISTINCT FROM coalesce(isfinite(price), false)",
        "target geography": "SELECT count(*) FROM forecast_target f ANTI JOIN priced_area a USING (zone)",
        "target calendar": "SELECT count(*) FROM forecast_target f ANTI JOIN delivery_hour h USING (ts_utc)",
        "border counts": """
            SELECT (SELECT count(*) FROM border_observation) <>
                   (SELECT count(*) FROM raw_physical_flow) + 2 * (SELECT count(*) FROM raw_jao_border)
        """,
        "paired CNEC provenance": """
            SELECT count(*) FROM raw_cnec_coefficients c POSITIONAL JOIN raw_cnec_provenance p
            WHERE c.cnec_key IS DISTINCT FROM p.cnec_key OR c.ts_utc IS DISTINCT FROM p.ts_utc
               OR c.regime IS DISTINCT FROM p.regime
        """,
        "CNEC domain": """
            SELECT count(*) FROM (SELECT DISTINCT regime, ts_utc FROM raw_cnec_coefficients) c
            ANTI JOIN (SELECT * FROM jao_domain_hour WHERE domain_present) d USING (regime, ts_utc)
        """,
        "CNEC domain row counts": """
            SELECT count(*) FROM
            (SELECT regime, ts_utc, count(*) AS n FROM raw_cnec_coefficients GROUP BY ALL) c
            FULL JOIN (SELECT * FROM jao_domain_hour WHERE domain_present) d USING (regime, ts_utc)
            WHERE coalesce(n, 0) <> coalesce(d.presolved_rows, 0)
        """,
        "CNEC PTDF area": """
            SELECT count(*) FROM (SELECT DISTINCT zone FROM cnec_ptdf) p ANTI JOIN area USING(zone)
        """,
    }
    for name, sql in integrity.items():
        if connection.execute(sql).fetchone()[0]:
            raise ValueError(f"Failed relational integrity check: {name}")
    clock = connection.sql(
        "SELECT ts_utc, forecast_origin_utc FROM delivery_hour ORDER BY ts_utc"
    ).df()
    actual = pd.DatetimeIndex(
        pd.to_datetime(clock.forecast_origin_utc, utc=True)
    ).as_unit("ns")
    if not actual.equals(pd.DatetimeIndex(forecast_origin(clock.ts_utc))):
        raise ValueError("Forecast calendar differs from Brussels local noon")
    if counts["forecast_target"] != counts["delivery_hour"] * len(
        connection.sql("FROM priced_area").df()
    ):
        raise ValueError("The canonical target scaffold is incomplete")
    return {
        "rows": counts,
        "checks": list(checks) + list(integrity) + ["local noon", "canonical grid"],
    }


def build(settings):
    dataset = Path(settings["dataset"])
    completion = read_json(dataset / "complete.json")
    if completion["fingerprint"] != settings["dataset_fingerprint"]:
        raise ValueError("Audited dataset fingerprint mismatch")
    hashes = _check_artifacts(
        dataset,
        completion,
        ["nodes", "strict", "source_manifest.json", "geography_crosswalk.json"],
    )
    sources = read_json(dataset / "source_manifest.json")
    required = set(RAW.values()) | {COVERAGE, ZONE_HISTORY, "jao_metadata/build.json"}
    for name in sorted(required):
        info = sources[name]
        if digest(info["path"])["sha256"] != info["sha256"]:
            raise ValueError(f"Source checksum differs from audited manifest: {name}")
    manifest = {
        "dataset": str(dataset),
        "artifacts": hashes,
        "sources": {name: sources[name] for name in sorted(required)},
        "schema": digest(__file__)["sha256"],
        "duckdb": duckdb.__version__,
    }
    fingerprint = identity(manifest)
    root = Path(settings["staging"])
    root.mkdir(parents=True, exist_ok=True)
    final = root / fingerprint
    if not final.exists():
        work = Path(tempfile.mkdtemp(prefix=".catalog-", dir=root))
        try:
            with closing(
                connect(work / "forecast.duckdb", read_only=False)
            ) as connection:
                for name, source in RAW.items():
                    connection.execute(
                        f"CREATE VIEW raw_{name} AS SELECT * FROM read_parquet({literal(sources[source]['path'])})"
                    )
                dimensions(connection, dataset, sources)
                views(connection, dataset)
                findings = verify(connection, sources)
            json_write(work / "manifest.json", manifest)
            json_write(work / "validation.json", findings)
            json_write(
                work / "complete.json",
                {p.name: digest(p)["sha256"] for p in work.iterdir() if p.is_file()},
            )
            try:
                work.rename(final)
            except OSError:
                if not final.exists():
                    raise
        finally:
            shutil.rmtree(work, ignore_errors=True)
    check_generation(final)
    with tempfile.TemporaryDirectory(prefix=".pointer-", dir=root) as temporary:
        pointer = Path(temporary) / "current"
        pointer.symlink_to(final.name)
        os.replace(pointer, root / "current")
    return final


def check_generation(directory):
    for name, checksum in read_json(directory / "complete.json").items():
        if digest(directory / name)["sha256"] != checksum:
            raise ValueError(f"Relational artifact checksum mismatch: {name}")
    return directory


def validate(settings):
    directory = check_generation((Path(settings["staging"]) / "current").resolve())
    manifest = read_json(directory / "manifest.json")
    if manifest["dataset"] != settings["dataset"]:
        raise ValueError("Catalog belongs to a different canonical generation")
    for name, info in manifest["sources"].items():
        if digest(info["path"])["sha256"] != info["sha256"]:
            raise ValueError(f"Pinned source changed: {name}")
    with closing(connect(directory / "forecast.duckdb")) as connection:
        return verify(connection, manifest["sources"])
