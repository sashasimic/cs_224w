"""Exercise two fixture delivery days through the actual forecast workers."""

import json
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from . import feed, report, service
from .contracts import (
    HISTORY,
    HOUR,
    MODELS,
    ROSTER,
    atomic_write,
    code_version,
    cutoff,
    deadline,
    digest,
    dump,
    file_hash,
    hour_grid,
    now,
)
from .ledger import Ledger


def receipt(day, market, role, *, late=False, published=False):
    hours = hour_grid(day)
    cut = cutoff(day)
    start = hours[0] - HISTORY * HOUR if role == "input" else hours[0]
    end = hours[-1] + HOUR
    stamp = (
        cut - timedelta(seconds=20)
        if role == "input" and not late
        else deadline(day) + HOUR
    )
    prices = []
    n = HISTORY if role == "input" else len(hours)
    for i in range(n * 4):
        prices.append(
            f"<Point><position>{i + 1}</position><price.amount>{40 + i % 36 - i % 5}</price.amount></Point>"
        )
    if role == "input" and published:
        prices.extend(
            f"<Point><position>{n * 4 + i + 1}</position><price.amount>909</price.amount></Point>"
            for i in range(len(hours) * 4)
        )
    identity = market["eic"]
    payload = f"""<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
    <type>A44</type><TimeSeries><in_Domain.mRID>{identity}</in_Domain.mRID>
    <out_Domain.mRID>{identity}</out_Domain.mRID><contract_MarketAgreement.type>A01</contract_MarketAgreement.type>
    <currency_Unit.name>EUR</currency_Unit.name><price_Measure_Unit.name>MWH</price_Measure_Unit.name>
    <curveType>A01</curveType><Period><timeInterval><start>{start.isoformat()}</start>
    <end>{end.isoformat()}</end></timeInterval><resolution>PT15M</resolution>{"".join(prices)}
    </Period></TimeSeries></Publication_MarketDocument>""".encode()
    values, status = feed.parse(payload, identity, start, end)
    return {
        "eic": identity,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "started": (stamp - timedelta(seconds=1)).isoformat(),
        "received": stamp.isoformat(),
        "sha256": digest(payload),
        "http": 200,
        "status": status,
        "hours": values,
    }, payload


def rehearse_day(root, day, spec, data):
    key = day.isoformat()
    cut = cutoff(day)
    timers = {}
    with Ledger(root) as ledger:
        with ledger.db:
            ledger.db.execute(
                "INSERT INTO phases VALUES ('pilot',?,1,?,?)", (key, data, digest(data))
            )
        for z, market in enumerate(ROSTER):
            r, payload = receipt(day, market, "input", late=z == 2, published=z == 1)
            # No capture for the first market, and a post-cutoff arrival for the third.
            if z:
                ledger.record_receipt("input", key, market["zone"], r, payload)
        sha, snap = service.freeze_snapshot(ledger, day, spec["artifacts"], lambda: cut)
        ledger.start(key, "pooled_xgb", cut)
    for z in (0, 2):
        if any(row[z] is not None for row in snap["history"]):
            raise AssertionError("Missing or late data entered the frozen snapshot")
    if any(value == 909 for row in snap["history"] for value in row) or any(
        label != "published_at_cutoff"
        for label in snap["publication"][ROSTER[1]["zone"]]
    ):
        raise AssertionError(
            "Known targets must be classified and excluded from input prices"
        )
    base = time.monotonic()
    clock_anchor = cut + timedelta(seconds=1)
    if len(snap["hours"]) == 23:
        slow = "timesfm_independent"
        late_case = "budget_fallback"
    else:
        slow = None
        late_case = "committed_late"

    def clock():
        spent = time.monotonic() - base
        if slow and spent >= 15:
            return deadline(day) - timedelta(seconds=9) + timedelta(seconds=spent - 15)
        return clock_anchor + timedelta(seconds=spent)

    def spawn(worker_state, worker_day, model, snapshot, artifact, output):
        timers[model] = clock().isoformat()
        if model == slow:
            return subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            )
        return service.spawn_worker(
            worker_state, worker_day, model, snapshot, artifact, output
        )

    with Ledger(root) as ledger:
        if not slow:
            # Include a real durable late job and verify that reporting excludes it.
            ledger.start(key, "timesfm_independent", cut)
            shape = [[60.0] * len(ROSTER) for _ in snap["hours"]]
            flags = [[False] * len(ROSTER) for _ in snap["hours"]]
            ledger.finish(
                key,
                "timesfm_independent",
                {"values": shape, "primary": flags, "fallback": flags},
                "success",
                "deliberate late fixture",
                lambda: deadline(day) + timedelta(seconds=1),
            )
        service.forecast(ledger, day, spec, clock=clock, spawn=spawn)
        for market in ROSTER:
            r, payload = receipt(day, market, "outcome")
            ledger.record_receipt(
                "outcome", key + "/fixture", market["zone"], r, payload
            )
        with ledger.db:
            ledger.db.execute(
                "INSERT INTO outcome_checks VALUES (?,?,?)",
                (
                    key,
                    (deadline(day) + HOUR).isoformat(),
                    len(snap["hours"]) * len(ROSTER),
                ),
            )
        recovered = ledger.rows(
            "SELECT status FROM attempts WHERE day=? AND model='pooled_xgb' ORDER BY started",
            (key,),
        )
        jobs = {
            j["model"]: j for j in ledger.rows("SELECT * FROM jobs WHERE day=?", (key,))
        }
        results = service.pilot_result(ledger)
        scores = report.score(ledger, "pilot", True, clock=lambda: deadline(day) + HOUR)
        baseline = report.score(ledger, "pilot", clock=lambda: deadline(day) + HOUR)
        if (
            len(jobs) != len(MODELS)
            or not all(j["snapshot"] == sha for j in jobs.values())
            or [r["status"] for r in recovered] != ["interrupted", "success"]
            or jobs["timesfm_independent"]["status"] != ("fallback" if slow else "late")
            or not results[key]["passed"]
            or not baseline["scores"]
        ):
            raise AssertionError("Operational fixture failed")
        lineup = scores["operations"]["timesfm_independent"]["jobs"]
        if not slow and lineup != {"late": 1}:
            raise AssertionError("Late result was not excluded")
        cpu_workers = sorted(
            (stamp, name)
            for name, stamp in timers.items()
            if name not in service.RESEARCH
        )[:3]
        cpu = [r[0] for r in cpu_workers]
        finishes = [jobs[name]["committed"] for _, name in cpu_workers]
        if len(cpu_workers) != 3 or max(cpu) >= min(finishes):
            raise AssertionError("Three CPU workers did not overlap")
        return {
            "hours": len(snap["hours"]),
            "snapshot": sha,
            "recovery": [r["status"] for r in recovered],
            "late_case": late_case,
            "parallel_cpu_workers": 3,
            "outcome_hours": results[key]["outcome_hours"],
            "job_status": results[key]["jobs"],
        }


def rehearse(state):
    spec = service.load_spec(state)
    gpu = json.loads((state / "gpu.json").read_bytes())
    if not gpu["validated"] or spec["artifacts"]["timesfm_joint"] != gpu["artifact"]:
        raise ValueError("Run the ROCm benchmark against this candidate first")
    spec["gpu_workers"] = gpu["gpu_workers"]
    data = dump(spec)
    details = {}
    with tempfile.TemporaryDirectory(dir=state) as temp:
        root = Path(temp)
        for day in (date(2026, 3, 29), date(2026, 10, 25)):
            path = root / day.isoformat()
            path.mkdir()
            (path / "artifacts").symlink_to(
                state / "artifacts", target_is_directory=True
            )
            details[day.isoformat()] = rehearse_day(path, day, spec, data)

    result = {
        "passed": True,
        "kind": "simulated_cutoffs_and_fixture_feeds",
        "service_code": code_version(),
        "candidate_sha256": file_hash(state / "candidate.json"),
        "finished_at": now().isoformat(),
        "days": details,
    }
    atomic_write(state / "rehearsal.json", dump(result))
    return result
