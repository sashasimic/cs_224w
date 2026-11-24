"""Production scheduling, receipt capture, worker supervision and phase gates."""

import concurrent.futures
import json
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta

import numpy as np

from . import feed, models
from .contracts import (
    BRUSSELS,
    COMPONENTS,
    CPU_WORKERS,
    DAY,
    HISTORY,
    HOUR,
    MODELS,
    ROSTER,
    RULES,
    ZONE_NAMES,
    atomic_write,
    code_version,
    configure_threads,
    cutoff,
    deadline,
    digest,
    dump,
    file_hash,
    hour_grid,
    local_at,
    now,
)
from .training import model_version, read_artifact, runtime

FETCH_WORKERS = 12
FETCH_TIMEOUT_SECONDS = 22
CAPTURE_LEAD = timedelta(minutes=3)
FETCH_ATTEMPTS = 3
TERMINAL = {"success", "fallback", "missing", "late"}
RESEARCH = {"timesfm_joint", "timesfm_independent"}
SDAC_GROUPS = {
    "AT",
    "BE",
    "BG",
    "CZ",
    "DE-LU",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GR",
    "HR",
    "HU",
    "IT",
    "LT",
    "LV",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
}
PUBLICATION_RULE = {
    "sdac_groups": sorted(SDAC_GROUPS),
    "schedule_evidence": "https://www.entsoe.eu/documents/nc/NC%20CACM//SDAC%202021/210526_Coms_note_new_operational_timings_updated.pdf",
    "known": "complete hourly target captured before cutoff",
    "unknown": "scheduled SDAC coupling with good feed receipt and no observed target quarters",
    "unverified": "partial visibility, failed capture, or market with unverified cutoff schedule",
}


def encoded(array):
    return np.where(np.isfinite(array), array, None).tolist()


def fetch_batch(
    ledger, role, key, start, end, token, stop=None, fetcher=feed.fetch, roster=ROSTER
):
    done = {
        r["zone"]
        for r in ledger.rows(
            "SELECT zone FROM receipts WHERE role=? AND day=?", (role, key)
        )
    }
    remaining = [row for row in roster if row["zone"] not in done]
    if not remaining:
        return
    workers = concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS)
    try:
        tasks = {
            workers.submit(
                fetcher, row["eic"], start, end, token, FETCH_TIMEOUT_SECONDS
            ): row["zone"]
            for row in remaining
        }
        timeout = max(0, (stop - now()).total_seconds()) if stop else None
        for task in concurrent.futures.as_completed(tasks, timeout=timeout):
            receipt, raw = task.result()
            ledger.record_receipt(role, key, tasks[task], receipt, raw)
    except TimeoutError:
        pass
    finally:
        workers.shutdown(wait=stop is None, cancel_futures=True)


def receipt_rows(ledger, role, key, stop=None):
    selected = {}
    for row in ledger.rows(
        """SELECT zone,body FROM receipts WHERE role=? AND (day=? OR day LIKE ?)
           ORDER BY day""",
        (role, key, key + "/%"),
    ):
        receipt = json.loads(row["body"])
        if stop is None or receipt["received"] <= stop.isoformat():
            selected[row["zone"]] = receipt
    return selected


def retryable_receipt(receipt):
    if receipt is None or receipt["status"] == "fx_unavailable":
        return True
    if receipt["status"] not in ("http_error", "invalid_or_unavailable"):
        return False
    return receipt["http"] is None or receipt["http"] >= 500


def fetch_with_retries(
    ledger, role, key, start, end, token, stop=None, fetcher=feed.fetch, roster=ROSTER
):
    for attempt in range(FETCH_ATTEMPTS):
        if stop and now() >= stop:
            break
        records = receipt_rows(ledger, role, key, stop)
        pending = [row for row in roster if retryable_receipt(records.get(row["zone"]))]
        if not pending:
            break
        attempt_key = key if not attempt else f"{key}/{attempt + 1}"
        fetch_batch(
            ledger, role, attempt_key, start, end, token, stop, fetcher, pending
        )
    return receipt_rows(ledger, role, key, stop)


def capture(ledger, day, token):
    cut = cutoff(day)
    if not cut - CAPTURE_LEAD <= now() < cut:
        return
    start = hour_grid(day)[0]
    fetch_with_retries(
        ledger,
        "input",
        day.isoformat(),
        start - HISTORY * HOUR,
        local_at(day + DAY),
        token,
        stop=cut,
    )


def freeze_snapshot(ledger, day, artifacts, clock=now):
    key = day.isoformat()
    if existing := ledger.snapshot(key):
        return existing
    cut = cutoff(day)
    if clock() < cut:
        raise ValueError("Cannot freeze before 11:55 Brussels")
    times = hour_grid(day)
    first = times[0] - HISTORY * HOUR
    history = [[None] * len(ZONE_NAMES) for _ in range(HISTORY)]
    rows = receipt_rows(ledger, "input", key, cut)
    publication = {}
    provenance = {}
    for z, market in enumerate(ROSTER):
        zone = market["zone"]
        receipt = rows.get(zone)
        if receipt is not None and receipt["received"] <= cut.isoformat():
            sha = receipt["sha256"]
            if sha is not None and file_hash(ledger.state / "receipts" / sha) != sha:
                raise ValueError("Frozen input receipt hash mismatch")
            fx = receipt.get("fx")
            if (
                fx
                and fx["sha256"] is not None
                and digest(fx["body"].encode()) != fx["sha256"]
            ):
                raise ValueError("Frozen exchange-rate receipt hash mismatch")
            provenance[zone] = {k: v for k, v in receipt.items() if k != "hours"}
            for hour, info in receipt["hours"].items():
                i = (feed.timestamp(hour) - first) // HOUR
                if 0 <= i < HISTORY:
                    history[i][z] = info["price"]
        else:
            receipt = None
            provenance[zone] = {"status": "missing_before_cutoff"}
        flags = []
        for hour in times:
            item = receipt["hours"].get(hour.isoformat()) if receipt else None
            if item and (item["price"] is not None or item.get("native_complete")):
                flags.append("published_at_cutoff")
            elif (
                receipt
                and receipt["status"] == "ok"
                and not (item and item["published"])
                and not zone.startswith("ENTSOE:")
                and market["group"] in SDAC_GROUPS
            ):
                flags.append("unknown_at_cutoff")
            else:
                flags.append("publication_unverified")
        publication[zone] = flags
    snapshot = {
        "day": key,
        "cutoff": cut.isoformat(),
        "frozen_at": clock().isoformat(),
        "zones": list(ZONE_NAMES),
        "hours": [t.isoformat() for t in times],
        "history": history,
        "publication": publication,
        "receipts": provenance,
    }
    sha = ledger.save_snapshot(snapshot, artifacts, deadline(day))
    return sha, snapshot


def run_worker(state, day, name, snapshot_sha, artifact_sha, output):
    from .ledger import Ledger

    configure_threads()
    with Ledger(state) as ledger:
        sha, snapshot = ledger.snapshot(day)
        if sha != snapshot_sha or tuple(snapshot["zones"]) != ZONE_NAMES:
            raise ValueError("Unexpected snapshot hash or zone order")
    artifact = read_artifact(state, artifact_sha)
    result = models.predict(name, artifact, snapshot)
    if result.shape != (len(snapshot["hours"]), len(ZONE_NAMES)):
        raise ValueError("Unexpected forecast grid")
    atomic_write(
        output,
        dump({"snapshot": sha, "artifact": artifact_sha, "values": encoded(result)}),
    )


def spawn_worker(state, day, name, snapshot_sha, artifact_sha, output):
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "epgnn.forecast_service",
            "--state",
            str(state),
            "worker",
            day,
            name,
            snapshot_sha,
            artifact_sha,
            str(output),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def forecast(ledger, day, spec, clock=now, spawn=spawn_worker, pause=time.sleep):
    """Commit independent results and apply weekly fallback within the deadline."""
    key = day.isoformat()
    cut, due = cutoff(day), deadline(day)
    if clock() < cut:
        return
    sha, snapshot = freeze_snapshot(ledger, day, spec["artifacts"], clock)
    ledger.recover(key, clock())
    artifacts = spec["artifacts"]
    shape = len(snapshot["hours"]), len(ZONE_NAMES)
    try:
        read_artifact(ledger.state, artifacts["weekly"])
    except (ValueError, OSError):
        weekly = np.full(shape, np.nan, np.float32)
        weekly_error = "weekly artifact failed verification"
    else:
        weekly = models.predict("weekly", None, snapshot)
        weekly_error = None
    has_history = np.isfinite(np.asarray(snapshot["history"], np.float32)).any(axis=0)
    close = due - timedelta(seconds=10)
    finished = {}

    def commit(name, values, detail=None, degraded=None):
        valid = np.isfinite(values)
        # Learned models require a real price context for the given market.
        if name != "weekly":
            valid &= has_history[None, :]
        ok = valid if degraded is None else valid & ~degraded
        pred = np.where(valid, values, weekly)
        fallback = np.isfinite(pred) & ~ok
        status = (
            "missing"
            if not np.isfinite(pred).any()
            else "fallback"
            if fallback.any()
            else "success"
        )
        if not np.isfinite(pred).all():
            detail = f"{detail + '; ' if detail else ''}unavailable_hours={int((~np.isfinite(pred)).sum())}"
        status = ledger.finish(
            key,
            name,
            {
                "values": encoded(pred),
                "fallback": fallback.tolist(),
                "primary": ok.tolist(),
            },
            status,
            detail,
            clock,
        )
        if status != "late":
            finished[name] = np.where(ok, pred, np.nan)

    queued = []
    for row in ledger.rows("SELECT * FROM jobs WHERE day=?", (key,)):
        name = row["model"]
        if row["status"] in TERMINAL:
            if row["prediction"] is not None and row["status"] != "late":
                body = checked_prediction(row)
                values = np.asarray(body["values"], np.float32)
                finished[name] = np.where(body["primary"], values, np.nan)
        elif name == "weekly":
            ledger.start(key, name, clock())
            commit(name, weekly, weekly_error)
        else:
            queued.append(name)

    active = {}
    configure_threads()
    with tempfile.TemporaryDirectory(dir=ledger.state) as scratch:
        from pathlib import Path

        try:
            while queued or active:
                for name, (process, output) in list(active.items()):
                    if process.poll() is None:
                        continue
                    del active[name]
                    if process.returncode:
                        commit(
                            name,
                            np.full(shape, np.nan),
                            f"worker exited with status {process.returncode}",
                        )
                        continue
                    try:
                        result = json.loads(output.read_bytes())
                        if (
                            result["snapshot"] != sha
                            or result["artifact"] != artifacts[name]
                        ):
                            raise ValueError("Worker output binding mismatch")
                        values = np.asarray(result["values"], np.float32)
                        if values.shape != shape:
                            raise ValueError("Worker output shape mismatch")
                    except (ValueError, OSError, KeyError, TypeError):
                        commit(name, np.full(shape, np.nan), "invalid worker output")
                    else:
                        commit(name, values)

                if clock() >= close:
                    for name in [*queued, *active]:
                        if name in queued:
                            ledger.start(key, name, clock())
                        commit(
                            name,
                            np.full(shape, np.nan),
                            "worker exceeded forecast budget"
                            if name in active
                            else "forecast budget elapsed before dispatch",
                        )
                    break

                for name in queued[:]:
                    if name == "blend":
                        if any(n in active or n in queued for n in COMPONENTS):
                            continue
                        ledger.start(key, name, clock())
                        try:
                            weights = read_artifact(ledger.state, artifacts[name])
                            values = models.blend_predictions(weights, finished, shape)
                            missing = np.zeros(shape, bool)
                            for n, weight in zip(COMPONENTS, weights):
                                if weight > 1e-8:
                                    missing |= ~np.isfinite(
                                        finished.get(n, np.full(shape, np.nan))
                                    )
                            commit(name, values, degraded=missing)
                        except (ValueError, OSError, TypeError):
                            commit(
                                name,
                                np.full(shape, np.nan),
                                "blend artifact failed verification",
                            )
                    else:
                        gpu = name in RESEARCH
                        used = sum((n in RESEARCH) == gpu for n in active)
                        if used >= (spec["gpu_workers"] if gpu else CPU_WORKERS):
                            continue
                        output = Path(scratch) / name
                        ledger.start(key, name, clock())
                        try:
                            active[name] = (
                                spawn(
                                    ledger.state,
                                    key,
                                    name,
                                    sha,
                                    artifacts[name],
                                    output,
                                ),
                                output,
                            )
                        except OSError:
                            commit(
                                name, np.full(shape, np.nan), "worker could not start"
                            )
                    queued.remove(name)
                if queued or active:
                    pause(0.15)
        finally:
            for process, _ in active.values():
                process.kill()
            for process, _ in active.values():
                process.wait()


def checked_prediction(row):
    data = row["prediction"]
    if not isinstance(data, bytes) or digest(data) != row["prediction_sha"]:
        raise ValueError("Forecast payload hash mismatch")
    result = json.loads(data)
    if result["snapshot"] != row["snapshot"] or result["artifact"] != row["artifact"]:
        raise ValueError("Forecast input or model binding mismatch")
    return result["rows"]


def fetch_outcomes(ledger, day, token):
    if now() <= deadline(day):
        return
    key = day.isoformat()
    complete = {
        r[0]
        for r in ledger.rows(
            "SELECT zone FROM outcomes WHERE day=? GROUP BY zone HAVING count(*)=?",
            (key, len(hour_grid(day))),
        )
    }
    key_with_attempt = f"{key}/{now().strftime('%Y-%m-%dT%H:%M')}"
    pending = [z for z in ROSTER if z["zone"] not in complete]
    fetch_batch(
        ledger,
        "outcome",
        key_with_attempt,
        local_at(day),
        local_at(day + DAY),
        token,
        roster=pending,
    )
    with ledger.db:
        count = ledger.one("SELECT count(*) FROM outcomes WHERE day=?", (key,))[0]
        ledger.db.execute(
            "INSERT INTO outcome_checks VALUES (?,?,?)", (key, now().isoformat(), count)
        )


def preflight(ledger, token):
    today = now().astimezone(BRUSSELS).date()
    key = now().isoformat()
    selected = fetch_with_retries(
        ledger, "preflight", key, local_at(today - 3 * DAY), local_at(today), token
    )
    records = {}
    for zone, receipt in selected.items():
        records[zone] = {k: v for k, v in receipt.items() if k not in ("hours", "fx")}
        if "fx" in receipt:
            records[zone]["fx"] = {
                k: v for k, v in receipt["fx"].items() if k != "body"
            }
        records[zone]["complete_hours"] = sum(
            info["price"] is not None for info in receipt["hours"].values()
        )
    result = {
        "checked": key,
        "zones": records,
        "passed": set(records) == set(ZONE_NAMES)
        and all(
            r["status"] == "empty" or (r["status"] == "ok" and r["complete_hours"])
            for r in records.values()
        )
        and any(r["complete_hours"] for r in records.values()),
    }
    atomic_write(ledger.state / "preflight.json", dump(result))
    return result


def load_spec(state):
    spec = json.loads((state / "candidate.json").read_bytes())
    if spec["model_version"] != model_version() or spec["runtime"] != runtime():
        raise ValueError("Candidate code or Python model dependencies have changed")
    if set(spec["artifacts"]) != set(MODELS):
        raise ValueError("Candidate does not include the complete model lineup")
    if file_hash(state / "backtest.json") != spec["backtest_sha256"]:
        raise ValueError("Backtest hash mismatch")
    for sha in set(spec["artifacts"].values()):
        if file_hash(state / "artifacts" / sha) != sha:
            raise ValueError("Candidate artifact hash mismatch")
    return spec


def spec_for_phase(row):
    if digest(row["spec"]) != row["spec_hash"]:
        raise ValueError("Frozen phase spec hash mismatch")
    spec = json.loads(row["spec"])
    if (
        spec["service_code"] != code_version()
        or spec["model_version"] != model_version()
        or spec["runtime"] != runtime()
    ):
        raise ValueError("Frozen service code or model dependencies have changed")
    return spec


def check_arm_window(ledger, start):
    if now() >= cutoff(start) - CAPTURE_LEAD:
        raise ValueError("Pilot start must allow the first scheduled input acquisition")
    existing = ledger.one("SELECT start FROM phases WHERE name='pilot'")
    if existing and (
        existing["start"] != start.isoformat()
        or ledger.one("SELECT 1 FROM snapshots LIMIT 1")
        or ledger.one(
            "SELECT 1 FROM receipts WHERE role IN ('input','outcome') LIMIT 1"
        )
    ):
        raise ValueError(
            "Only an unstarted pilot can be refreshed at its existing date"
        )


def arm(ledger, start, token):
    check_arm_window(ledger, start)
    spec = load_spec(ledger.state)
    gpu = json.loads((ledger.state / "gpu.json").read_bytes())
    rehearsal = json.loads((ledger.state / "rehearsal.json").read_bytes())
    if (
        gpu["artifact"] != spec["artifacts"]["timesfm_joint"]
        or not rehearsal["passed"]
        or rehearsal["service_code"] != code_version()
        or rehearsal["candidate_sha256"] != file_hash(ledger.state / "candidate.json")
    ):
        raise ValueError(
            "Run the two-day fixture rehearsal and ROCm benchmark for the current candidate"
        )
    if gpu["gpu_workers"] not in (1, 2) or not gpu["validated"]:
        raise ValueError("ROCm test did not validate a worker configuration")
    checked = preflight(ledger, token)
    if not checked["passed"]:
        raise ValueError("ENTSO-E preflight failed; see preflight.json")
    check_arm_window(ledger, start)
    spec.update(
        gpu_workers=gpu["gpu_workers"],
        roster=ROSTER,
        rules={
            **RULES,
            "publication_evidence": PUBLICATION_RULE,
            "auction_series": "A44 day-ahead series with unspecified sequence or sequence 1.",
            "UAH_to_EUR": "Divide by the NBU official rate per euro effective on the Europe/Kyiv delivery date. Require an observed rate for each price hour.",
        },
        service_code=code_version(),
        preflight_sha256=file_hash(ledger.state / "preflight.json"),
        rehearsal_sha256=file_hash(ledger.state / "rehearsal.json"),
        gpu_sha256=file_hash(ledger.state / "gpu.json"),
    )
    data = dump(spec)
    with ledger.db:
        ledger.db.execute(
            """INSERT INTO phases VALUES ('pilot',?,2,?,?)
               ON CONFLICT(name) DO UPDATE SET spec=excluded.spec, spec_hash=excluded.spec_hash""",
            (start.isoformat(), data, digest(data)),
        )


def schedule(ledger):
    rows = ledger.rows("SELECT * FROM phases ORDER BY start")
    return [
        (date.fromisoformat(row["start"]) + i * DAY, row)
        for row in rows
        for i in range(row["days"])
    ]


def pilot_result(ledger):
    result = {}
    for day, phase in schedule(ledger):
        if phase["name"] != "pilot":
            continue
        key = day.isoformat()
        snap = ledger.snapshot(key)
        jobs = ledger.rows("SELECT * FROM jobs WHERE day=?", (key,))
        spec = json.loads(phase["spec"])
        ok = bool(snap)
        if snap:
            ok &= set(snap[1]["zones"]) == set(ZONE_NAMES)
            ok &= any(r["status"] == "ok" for r in snap[1]["receipts"].values())
            for job in jobs:
                if job["prediction"] is not None:
                    checked_prediction(job)
                ok &= (
                    job["snapshot"] == snap[0]
                    and job["artifact"] == spec["artifacts"][job["model"]]
                )
                ok &= job["status"] in TERMINAL
                if job["status"] in ("success", "fallback"):
                    ok &= bool(
                        job["started"]
                        and job["committed"]
                        and cutoff(day).isoformat()
                        <= job["committed"]
                        <= deadline(day).isoformat()
                    )
        ok &= {j["model"] for j in jobs} == set(MODELS)
        ok &= any(j["model"] == "weekly" and j["status"] == "success" for j in jobs)
        ok &= any(j["model"] == "pooled_xgb" and j["status"] == "success" for j in jobs)
        outcomes = ledger.one("SELECT count(*) FROM outcomes WHERE day=?", (key,))[0]
        result[key] = {
            "passed": bool(ok and outcomes),
            "outcome_hours": outcomes,
            "jobs": {j["model"]: j["status"] for j in jobs},
        }
    return result


def advance(ledger):
    pilot = ledger.one("SELECT * FROM phases WHERE name='pilot'")
    if not pilot or ledger.one("SELECT 1 FROM phases WHERE name='experiment'"):
        return
    results = pilot_result(ledger)
    if len(results) != 2 or any(not r["passed"] for r in results.values()):
        return
    spec_for_phase(pilot)
    start = max(
        date.fromisoformat(pilot["start"]) + 2 * DAY,
        now().astimezone(BRUSSELS).date() + DAY,
    )
    if now() >= cutoff(start) - CAPTURE_LEAD:
        start += DAY
    with ledger.db:
        ledger.db.execute(
            "INSERT OR IGNORE INTO phases VALUES ('experiment',?,14,?,?)",
            (start.isoformat(), pilot["spec"], pilot["spec_hash"]),
        )
