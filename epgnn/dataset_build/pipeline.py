"""Public audit/build/validate entry points. No model training or input mutation."""
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .construct import adapt, coverage_report, edge_month, node_month
from .contracts import (CALENDAR, EDGE_KEYS, NODE_KEYS, calendar, feature_registry,
                        forecast_origin, strict_view, unique, utc, zone_map)
from .findings import modelling_blockers
from .storage import (ROOT, SPECS, config, digest, download_missing, fingerprint,
                      identities, json_write, partition_sources, read_month,
                      resolve, source_paths)


def _cfg(value):
    return config(value) if not isinstance(value, dict) else value


def _monthly_audit(name, folder, cfg):
    ts, keys, validity = SPECS[name]
    result = dict(rows=0, duplicate_rows=0, conflicting_rows=0, null_keys=0,
                  non_hourly_rows=0, calendar_mismatches=0, months=[], status_counts={},
                  availability_verified_counts={}, source_identity_rows=[], canonical_duplicate_rows=0)
    seen_identity = {}
    for p in sorted(Path(folder).iterdir()):
        if not p.is_dir():
            continue
        f = read_month(folder, p.name)
        duplicate = f.duplicated(keys, keep=False)
        result["duplicate_rows"] += int(duplicate.sum())
        if duplicate.any():
            result["conflicting_rows"] += len(f.loc[duplicate].drop_duplicates()) - len(f.loc[duplicate, keys].drop_duplicates())
        result["null_keys"] += int(f[keys].isna().any(axis=1).sum())
        t = utc(f[ts])
        result["non_hourly_rows"] += int((t != t.dt.floor("h")).sum())
        if "MarketDayCET" in f:
            result["calendar_mismatches"] += int((f.MarketDayCET.astype("string") != t.dt.tz_convert("Europe/Brussels").dt.strftime("%Y-%m-%d")).fillna(True).sum())
        month = {"month": p.name, "rows": len(f), "hours": t.nunique(), "first": str(t.min()), "last": str(t.max())}
        if validity:
            month["valid_rows"] = int(f[validity].fillna(False).sum())
        result["months"].append(month)
        result["rows"] += len(f)
        for column, dest in [("row_status", "status_counts"), ("IsAvailabilityVerified", "availability_verified_counts")]:
            if column in f:
                for k, v in f[column].value_counts(dropna=False).items():
                    result[dest][str(k)] = result[dest].get(str(k), 0) + int(v)
        if name.startswith("entsoe"):
            endpoint_sets = [("OutAreaCode", "OutAreaName"), ("InAreaCode", "InAreaName")] if name == "entsoe_physical_flows" else [("AreaCode", "AreaName")]
            for code, area in endpoint_sets:
                for (eic, label), g in f.groupby([code, area], dropna=False):
                    identity = (str(eic), str(label))
                    r = seen_identity.setdefault(identity, {"area_code": str(eic), "area_name": str(label), "first_utc": str(g[ts].min()), "last_utc": str(g[ts].max())})
                    r["first_utc"] = min(r["first_utc"], str(g[ts].min()))
                    r["last_utc"] = max(r["last_utc"], str(g[ts].max()))
            # Check canonical collisions independently of original EIC uniqueness.
            kf = f[keys].rename(columns={ts: "ts_utc"}).copy()
            if name == "entsoe_physical_flows":
                kf["from_zone"] = zone_map(f.OutAreaName, f.OutAreaCode)
                kf["to_zone"] = zone_map(f.InAreaName, f.InAreaCode)
                canonical_keys = EDGE_KEYS
            else:
                kf["zone"] = zone_map(f.AreaName, f.AreaCode)
                canonical_keys = NODE_KEYS
            result["canonical_duplicate_rows"] += int(kf.duplicated(canonical_keys, keep=False).sum())
    result["source_identity_rows"] = list(seen_identity.values())
    return result


def _cnec_audit(manifest, cfg):
    """External hash-partition key check: memory bounded even for 17M observations.

    Detailed CNECs remain separate. Do not apply synthetic-name filtering or infer
    persistent constraint identities. Key/provenance equality is checked bucketwise.
    """
    result = {}
    parent = resolve(cfg, cfg["output_directory"])
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cnec-key-audit-", dir=parent) as d:
        d = Path(d)
        for name in ["jao_cnec_hourly", "cnec_provenance"]:
            q = pq.ParquetFile(manifest[name]["path"])
            writers = {}
            schema = pa.schema([("cnec_key", pa.string()), ("ts_utc", pa.timestamp("ns", tz="UTC"))])
            nulls = 0
            for batch in q.iter_batches(batch_size=131072, columns=["cnec_key", "ts_utc"]):
                f = batch.to_pandas()
                f["ts_utc"] = utc(f.ts_utc)
                nulls += int(f.cnec_key.isna().sum())
                bucket = pd.util.hash_pandas_object(f.cnec_key, index=False).to_numpy() % 64
                for b in np.unique(bucket):
                    if b not in writers:
                        writers[b] = pq.ParquetWriter(d / f"{name}-{b}.parquet", schema, compression="zstd")
                    writers[b].write_table(pa.Table.from_pandas(f.loc[bucket == b], schema=schema, preserve_index=False))
            for w in writers.values():
                w.close()
            result[name] = {"rows": q.metadata.num_rows, "null_keys": nulls, "duplicate_keys": 0}
        mismatches = 0
        for b in range(64):
            parts = []
            for name in ["jao_cnec_hourly", "cnec_provenance"]:
                p = d / f"{name}-{b}.parquet"
                f = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["cnec_key", "ts_utc"])
                result[name]["duplicate_keys"] += int(f.duplicated(["cnec_key"], keep=False).sum())
                parts.append(f.sort_values(["cnec_key", "ts_utc"]).reset_index(drop=True))
            mismatches += 0 if parts[0].equals(parts[1]) else 1
        result["key_timestamp_mismatching_buckets"] = mismatches
    return result


def audit(config_path, *, download=False):
    cfg = _cfg(config_path)
    acquisition = download_missing(config_path, cfg) if download else []
    manifest = identities(cfg)
    token = fingerprint(cfg, manifest)
    base = resolve(cfg, cfg["output_directory"])
    dest = base / "audits" / token
    findings = modelling_blockers()
    required = list(SPECS) + ["jao_cnec_hourly", "cnec_provenance", "jao_metadata/coverage.csv"]
    for name in required:
        if not manifest.get(name, {}).get("exists"):
            findings.append(dict(id="missing_source", source=name, severity="build_blocker", detail=manifest.get(name, {})))
    for name, spec in cfg["sources"].items():
        info = manifest[name]
        if info["exists"] and not info["checksum_verified"]:
            findings.append(dict(id="unverified_checksum", source=name, severity="build_blocker"))
        if spec.get("version_identity", "explicit_release_filename") != "explicit_release_filename":
            findings.append(dict(id="unresolved_release_identity", source=name, severity="provenance_limitation", detail=spec.get("version_identity")))
        if spec.get("checksum_authority") == "provided_archive":
            findings.append(dict(id="gcs_release_checksum_unverified", source=name, severity="provenance_limitation",
                detail="Provided archive CRC-32 and pinned MD5 checked; equality to the GCS release has not been independently verified"))
    report = dict(fingerprint=token, sources=manifest, findings=findings, acquisition=acquisition, tables={})
    if not any(r["severity"] == "build_blocker" for r in findings):
        # Cache completed audits only with a matching source/code/config identity.
        old = dest / "audit.json"
        if old.exists():
            previous = json.loads(old.read_text())
            if previous.get("fingerprint") == token and previous.get("status") == "passed_with_limitations":
                for name, sha in previous["source_cache_manifests"].items():
                    marker = Path(previous["source_cache"][name]) / "complete.json"
                    if not marker.exists() or digest(marker)["sha256"] != sha:
                        raise ValueError(f"Source cache manifest changed: {name}")
                partition_sources(cfg, manifest)  # checks fragment hashes and unexpected files
                return previous
        folders = partition_sources(cfg, manifest)
        for name, folder in folders.items():
            print(f"audit {name}", flush=True)
            r = _monthly_audit(name, folder, cfg)
            q = pq.ParquetFile(manifest[name]["path"])
            r["schema"] = {f.name: str(f.type) for f in q.schema_arrow}
            r["source_timestamp_unit"] = q.schema_arrow.field(SPECS[name][0]).type.unit
            r["metadata_rows"] = q.metadata.num_rows
            report["tables"][name] = r
            for check in ["duplicate_rows", "null_keys", "non_hourly_rows", "calendar_mismatches", "canonical_duplicate_rows"]:
                if r[check]:
                    findings.append(dict(id=check, source=name, severity="build_blocker", rows=r[check]))
            if r["rows"] != q.metadata.num_rows:
                findings.append(dict(id="row_reconciliation", source=name, severity="build_blocker"))
        print("audit CNEC observation/provenance keys", flush=True)
        cnec = _cnec_audit(manifest, cfg)
        report["cnec"] = cnec
        if cnec["key_timestamp_mismatching_buckets"] or any(cnec[n]["null_keys"] or cnec[n]["duplicate_keys"] for n in ["jao_cnec_hourly", "cnec_provenance"]):
            findings.append(dict(id="cnec_key_reconciliation", severity="build_blocker"))
        report["source_cache"] = {k: str(v) for k, v in folders.items()}
        report["source_cache_manifests"] = {k: digest(v / "complete.json")["sha256"] for k, v in folders.items()}
        pub = cfg.get("price_publication", {})
        if pub.get("column") and pub["column"] not in report["tables"]["entsoe_prices"]["schema"]:
            findings.append(dict(id="missing_verified_publication_column", severity="build_blocker"))
        for name in cfg["sources"]:
            findings.append(dict(id="historical_availability", source=name, severity="provenance_limitation",
                detail="No blanket availability inference from lags, observation completion, update times or archival extraction date"))
        findings.append(dict(id="jao_vintages_and_cwe_utc_anchor", source="combined original JAO", severity="provenance_limitation",
                             detail="Historical publication vintages and independent CWE UTC anchor remain unverified; source regime/timestamp status preserved"))
    report["status"] = "blocked" if any(r["severity"] == "build_blocker" for r in findings) else "passed_with_limitations"
    json_write(dest / "audit.json", report)
    json_write(base / "latest_audit.json", {"path": str(dest / "audit.json"), "status": report["status"], "fingerprint": token})
    return report


def _universe(report, cfg):
    price = report["tables"]["entsoe_prices"]
    rows = price["source_identity_rows"]
    zones = sorted(zone_map(pd.Series([r["area_name"] for r in rows]), pd.Series([r["area_code"] for r in rows])).unique())
    first = min(pd.Timestamp(r["first"]) for r in price["months"])
    last = max(pd.Timestamp(r["last"]) for r in price["months"])
    start = max(first, pd.Timestamp(cfg.get("output_start", str(first))))
    end = min(last + pd.Timedelta(hours=1), pd.Timestamp(cfg.get("output_end", str(last + pd.Timedelta(hours=1)))))
    if start >= end or start != start.floor("h") or end != end.floor("h"):
        raise ValueError("Output bounds must define a nonempty whole-hour UTC interval")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    # The calendar is continuous min..max, including any entirely missing source hour.
    pairs = []
    for name in ["entsoe_physical_flows", "jao_border_hourly"]:
        folder = Path(report["source_cache"][name])
        columns = ["OutAreaCode", "OutAreaName", "InAreaCode", "InAreaName"] if name.startswith("entsoe") else ["from_zone", "to_zone"]
        for p in sorted(folder.iterdir()):
            if p.is_dir():
                f = read_month(folder, p.name, columns).drop_duplicates()
                if name.startswith("entsoe"):
                    f = pd.DataFrame({"from_zone": zone_map(f.OutAreaName, f.OutAreaCode), "to_zone": zone_map(f.InAreaName, f.InAreaCode)})
                pairs.append(f[["from_zone", "to_zone"]])
    return zones, hours, pd.concat(pairs, ignore_index=True).drop_duplicates().sort_values(["from_zone", "to_zone"]).reset_index(drop=True)


def _write_parquet(frame, path):
    f = frame.copy()
    for c in f:
        if pd.api.types.is_object_dtype(f[c].dtype) or pd.api.types.is_string_dtype(f[c].dtype):
            f[c] = f[c].astype("string")
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(path, index=False, compression="zstd")


def _crosswalk(report):
    rows = []
    for source, table in report["tables"].items():
        for r in table["source_identity_rows"]:
            zone = zone_map(pd.Series([r["area_name"]]), pd.Series([r["area_code"]])).iloc[0]
            rows.append(dict(source=source, zone=zone, **r,
                valid_from_utc="2018-09-30T22:00:00Z" if zone in ["DE_LU", "AT"] else None,
                valid_until_utc_exclusive="2018-09-30T22:00:00Z" if zone == "DE_AT_LU" else None,
                compatibility="unresolved" if str(zone).startswith("ENTSOE:") else "exact_source_name_identity"))
    return rows


def _reference_comparison(nodes, edges, manifest, start, end):
    reports = []
    for label, out, keys, values in [("nodes", nodes, NODE_KEYS, ["price", "price_lag24", "price_lag168", "n_presolved", "min_ram_ratio"]),
                                      ("edges", edges, EDGE_KEYS, ["flow_phys", "maxbex", "flow_sched"])]:
        info = manifest.get(f"reference/{label}")
        if not info or not info["exists"]:
            continue
        ref = ds.dataset(info["path"]).to_table(columns=keys + values,
            filter=(ds.field("ts_utc") >= start.to_pydatetime()) & (ds.field("ts_utc") < end.to_pydatetime())).to_pandas()
        joined = out[keys + values].merge(ref, on=keys, how="outer", suffixes=("_new", "_v1"), indicator=True, validate="one_to_one")
        for c in values:
            a, b = joined[f"{c}_new"], joined[f"{c}_v1"]
            same = np.isclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-4)
            reports.append(dict(table=label, month=start.strftime("%Y-%m"), feature=c,
                aligned_rows=int((joined._merge == "both").sum()), new_only_rows=int((joined._merge == "left_only").sum()),
                reference_only_rows=int((joined._merge == "right_only").sum()), differing_rows=int((~same).sum()),
                explanation="Expected differences: full price calendar/areas, unfilled targets, exact lags, separate DE_AT_LU, CWE retained, no upstream *_input reuse"))
    return reports


def build(config_path):
    cfg = _cfg(config_path)
    report = audit(config_path)
    if report["status"] == "blocked":
        raise ValueError("Build blocked by input audit; see latest_audit.json")
    token = report["fingerprint"]
    base = resolve(cfg, cfg["output_directory"])
    final = base / "generations" / token
    if (final / "complete.json").exists():
        validate(config_path, generation=final)
        _publish_pointer(base, token)
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".building-", dir=final.parent))
    try:
        zones, hours, pairs = _universe(report, cfg)
        folders = report["source_cache"]
        coverage = pd.read_csv(report["sources"]["jao_metadata/coverage.csv"]["path"])
        coverage["ts_utc"] = utc(coverage.ts_utc)
        registry = feature_registry(bool(cfg.get("price_publication", {}).get("evidence")))
        coverage_parts, comparison = [], []
        counts = defaultdict(int)
        reconciliation = []
        months = hours.strftime("%Y-%m")
        for month in months.unique():
            h = hours[months == month]
            print(f"build {month}: {len(zones)} zones, {len(pairs)} directed pairs, {len(h)} hours", flush=True)
            n = node_month(cfg, folders, zones, h, coverage)
            e = edge_month(cfg, folders, pairs, h, coverage, zones)
            if not coverage_parts:
                # Every remaining master column is explicitly diagnostic/outcome
                # metadata, not an implicitly admitted feature.
                for table, f in [("node", n), ("edge", e)]:
                    known = {r["feature"] for r in registry if r["table"] == table}
                    for c in f:
                        if c not in known:
                            registry.append(dict(feature=c, table=table, source="source-preserving master metadata",
                                admission="excluded", timing_rule="diagnostic/key/outcome; no forecast admission rule",
                                transformation="source attribute or documented construction diagnostic",
                                units="source units", geography="retain original source identity",
                                normalization="unfitted; preserve NaNs"))
            s = strict_view(n, registry)
            for name, frame in [("nodes", n), ("edges", e), ("strict", s)]:
                _write_parquet(frame, work / name / f"{month}.parquet")
                counts[name] += len(frame)
            for table, frame in [("node", n), ("edge", e)]:
                coverage_parts.append(coverage_report(frame, table))
            comparison.extend(_reference_comparison(n, e, report["sources"], h.min(), h.max() + pd.Timedelta(hours=1)))
            reconciliation.append(dict(month=month, expected_nodes=len(zones) * len(h), nodes=len(n),
                expected_edges=len(pairs) * len(h), edges=len(e), strict=len(s),
                price_joined_rows=int(n.price_row_present.sum()), load_joined_rows=int(n.load_row_present.sum()),
                weather_joined_rows=int(n.weather_row_present.sum()), jao_hub_joined_rows=int(n.jao_row_present.sum()),
                physical_joined_rows=int(e.physical_row_present.sum()), border_joined_rows=int(e.border_row_present.sum()),
                source_rows={**n.attrs["reconciliation"], **e.attrs["reconciliation"]}))
        _write_parquet(pd.concat(coverage_parts, ignore_index=True), work / "coverage.parquet")
        json_write(work / "reference_comparison.json", comparison)
        json_write(work / "join_reconciliation.json", reconciliation)
        json_write(work / "feature_registry.json", registry)
        json_write(work / "source_manifest.json", report["sources"])
        json_write(work / "construction_config.json", cfg)
        json_write(work / "findings.json", report["findings"])
        json_write(work / "geography_crosswalk.json", _crosswalk(report))
        json_write(work / "cnec_references.json", {"observation_key": "cnec_key is source SHA256 + row, not a physical asset identity",
            "synthetic_name_filter_applied": False,
            **{k: report["sources"][k] for k in ["jao_cnec_hourly", "cnec_provenance"]}})
        shutil.copyfile(ROOT / "docs/forecast_dataset.md", work / "DATASET_GUIDE.md")
        # Source identities are checked again before publication, guarding mid-build changes.
        if identities(cfg) != report["sources"]:
            raise ValueError("Source identities changed during build")
        json_write(work / "build_fingerprint.json", {"fingerprint": token, "counts": dict(counts), "zones": zones,
            "directed_pairs": len(pairs), "hours": len(hours), "first_utc": str(hours.min()), "last_utc": str(hours.max()),
            "normalization": "unfitted", "strict_features": [r["feature"] for r in registry if r["admission"] == "admitted"]})
        checksums = {str(p.relative_to(work)): digest(p)["sha256"] for p in work.rglob("*") if p.is_file()}
        json_write(work / "complete.json", {"fingerprint": token, "artifacts": checksums})
        validate(config_path, generation=work)
        os.rename(work, final)
        # Relative symlink: the completed generation becomes visible atomically.
        _publish_pointer(base, token)
        return final
    finally:
        if work.exists():
            shutil.rmtree(work)


def _publish_pointer(base, token):
    tmp = base / f".current-{token[:16]}"
    if tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(Path("generations") / token, target_is_directory=True)
    os.replace(tmp, base / "current")


def validate(config_path, *, generation=None):
    cfg = _cfg(config_path)
    dest = Path(generation) if generation else resolve(cfg, cfg["output_directory"]) / "current"
    if not (dest / "complete.json").exists():
        raise ValueError(f"No completed generation at {dest}")
    complete = json.loads((dest / "complete.json").read_text())
    manifest = identities(cfg)
    if complete["fingerprint"] != fingerprint(cfg, manifest):
        raise ValueError("Build/config/code/source fingerprint mismatch; rebuild required")
    for rel, sha in complete["artifacts"].items():
        path = dest / rel
        if not path.exists() or digest(path)["sha256"] != sha:
            raise ValueError(f"Artifact checksum mismatch: {rel}")
    fp = json.loads((dest / "build_fingerprint.json").read_text())
    registry = json.loads((dest / "feature_registry.json").read_text())
    for r in json.loads((dest / "join_reconciliation.json").read_text()):
        if (r["nodes"] != r["expected_nodes"] or r["edges"] != r["expected_edges"] or r["strict"] != r["nodes"]):
            raise ValueError("Join row reconciliation failed")
        for source in r["source_rows"].values():
            if source["window_source_rows"] != source["joined_rows"] + source["unmatched_rows"]:
                raise ValueError("Source/join accounting mismatch")
    counts = defaultdict(int)
    zone_count, pair_count = len(fp["zones"]), fp["directed_pairs"]
    for p in sorted((dest / "nodes").glob("*.parquet")):
        n = pd.read_parquet(p)
        e = pd.read_parquet(dest / "edges" / p.name)
        s = pd.read_parquet(dest / "strict" / p.name)
        for label, f, keys in [("nodes", n, NODE_KEYS), ("edges", e, EDGE_KEYS), ("strict", s, NODE_KEYS)]:
            unique(f, keys, label)
            counts[label] += len(f)
            if not f.forecast_origin_utc.equals(forecast_origin(f.ts_utc)):
                raise ValueError("Forecast origin contract violation")
            expected_split = calendar(f[keys], cfg["splits"])["split"]
            if not f.split.astype("string").equals(expected_split):
                raise ValueError("Split boundary mismatch")
        if len(n) != zone_count * n.ts_utc.nunique() or len(e) != pair_count * n.ts_utc.nunique():
            raise ValueError("Canonical grid row reconciliation failed")
        if not n.target_valid.equals(n.price.notna()):
            raise ValueError("Target validity mismatch")
        if n.loc[~n.price_structurally_available, "price"].notna().any():
            raise ValueError("Target in structural absence")
        pd.testing.assert_frame_equal(s, strict_view(n, registry), check_dtype=False)
        for f in [n, e]:
            for c in [c for c in f if c.endswith("_eligible")]:
                prefix = c.removesuffix("_eligible")
                expected = (f[prefix].notna() & (f[f"{prefix}_observation_utc"] < f.forecast_origin_utc)
                    & (f[f"{prefix}_completion_utc"] <= f.forecast_origin_utc)
                    & (f[f"{prefix}_publication_utc"] <= f.forecast_origin_utc))
                if not f[c].equals(expected):
                    raise ValueError(f"Unsupported strict timing: {c}")
        if ((n.zone == "DE_AT_LU") & n.weather_row_present).any():
            raise ValueError("Weather substituted into historical combined geography")
    if dict(counts) != fp["counts"]:
        raise ValueError("Global row reconciliation failed")
    result = {"status": "passed", "fingerprint": fp["fingerprint"], "counts": dict(counts)}
    # Validation reports sit outside immutable generation checksums.
    json_write(resolve(cfg, cfg["output_directory"]) / "validation.json", result)
    return result
