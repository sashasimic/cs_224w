"""Scoring keyed by delivery day, bidding zone, UTC hour and cutoff evidence."""

import csv
import io
import json
from collections import Counter
from datetime import datetime
from statistics import mean, median

from .contracts import (
    CORE_MODELS,
    DAY,
    PRIVATE_MODELS,
    ROSTER,
    atomic_write,
    cutoff,
    deadline,
    digest,
    dump,
    hour_grid,
    local_at,
    now,
)
from .service import TERMINAL, checked_prediction, schedule

OUTCOME_WINDOW = 3
PUBLICATION_CLASSES = (
    "unknown_at_cutoff",
    "published_at_cutoff",
    "publication_unverified",
)
GROUP_OF = {row["zone"]: row["group"] for row in ROSTER}


def empty():
    return {
        "expected": 0,
        "observed": 0,
        "forecasts": 0,
        "scored": 0,
        "matched": 0,
        "fallback": 0,
        "scored_fallback": 0,
        "absolute_error": 0.0,
        "matched_error": 0.0,
        "weekly_error": 0.0,
    }


def add_count(totals, received, prediction, baseline, fallback):
    totals["expected"] += 1
    totals["observed"] += received is not None
    totals["forecasts"] += prediction is not None
    totals["fallback"] += fallback
    if received is None or prediction is None:
        return
    totals["scored"] += 1
    totals["scored_fallback"] += fallback
    error = abs(received - prediction)
    totals["absolute_error"] += error
    if baseline is not None:
        totals["matched"] += 1
        totals["matched_error"] += error
        totals["weekly_error"] += abs(received - baseline)


def summarize(values):
    row = {k: v for k, v in values.items() if not k.endswith("error")}
    for key, a, b in (
        ("coverage", "forecasts", "expected"),
        ("outcome_coverage", "observed", "expected"),
        ("scored_coverage", "scored", "observed"),
        ("mae", "absolute_error", "scored"),
        ("matched_mae", "matched_error", "matched"),
        ("matched_weekly_mae", "weekly_error", "matched"),
    ):
        row[key] = values[a] / values[b] if values[b] else None
    row["skill"] = (
        1 - values["matched_error"] / values["weekly_error"]
        if values["weekly_error"]
        else None
    )
    return row


def average(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return mean(values) if values else None


def score(ledger, phase_name, private=False, clock=now):
    lineup = PRIVATE_MODELS if private else CORE_MODELS
    phase_days = [(d, p) for d, p in schedule(ledger) if p["name"] == phase_name]
    model_names = tuple(lineup)
    specs = {p["spec_hash"] for _, p in phase_days}
    by_key = {}
    latency = {m: [] for m in model_names}
    worker = {m: [] for m in model_names}
    statuses = {m: Counter() for m in model_names}
    attempts = {m: Counter() for m in model_names}
    clock_now = clock()
    processed = checks = 0

    for day, phase in phase_days:
        if clock_now < cutoff(day):
            continue
        processed += 1
        key = day.isoformat()
        if digest(phase["spec"]) != phase["spec_hash"]:
            raise ValueError("Scoring requires the frozen phase manifest")
        artifact = json.loads(phase["spec"])["artifacts"]
        jobs = {
            j["model"]: j for j in ledger.rows("SELECT * FROM jobs WHERE day=?", (key,))
        }
        checked = ledger.one("SELECT 1 FROM outcome_checks WHERE day=?", (key,))
        checks += bool(checked)
        snapshot = ledger.snapshot(key)
        sha, snap = snapshot if snapshot else (None, None)
        hours = [t.isoformat() for t in hour_grid(day)]
        outcomes = {
            (r["zone"], r["hour"]): r["price"]
            for r in ledger.rows(
                "SELECT * FROM outcomes WHERE day=? AND received>?",
                (key, deadline(day).isoformat()),
            )
        }
        predictions = {}
        for name in set(model_names) | {"weekly"}:
            job = jobs.get(name)
            if name in statuses:
                label = job["status"] if job else "unscheduled"
                statuses[name][label] += 1
                if job and job["committed"]:
                    latency[name].append(
                        (
                            datetime.fromisoformat(job["committed"]) - cutoff(day)
                        ).total_seconds()
                    )
            if job and job["status"] in ("success", "fallback"):
                if (
                    job["snapshot"] != sha
                    or job["artifact"] != artifact[name]
                    or not (
                        cutoff(day).isoformat()
                        <= job["committed"]
                        <= deadline(day).isoformat()
                    )
                ):
                    raise ValueError(
                        "Scoring requires the frozen snapshot, pinned model and timely commit"
                    )
                predictions[name] = checked_prediction(job)
        for row in ledger.rows("SELECT * FROM attempts WHERE day=?", (key,)):
            name = row["model"]
            if name in attempts:
                attempts[name][row["status"]] += 1
                if row["ended"]:
                    worker[name].append(
                        (
                            datetime.fromisoformat(row["ended"])
                            - datetime.fromisoformat(row["started"])
                        ).total_seconds()
                    )

        for z, zone in enumerate(GROUP_OF):
            for h, hour in enumerate(hours):
                category = (
                    snap["publication"][zone][h] if snap else "publication_unverified"
                )
                outcome = outcomes.get((zone, hour))
                baseline = predictions.get("weekly")
                base_price = baseline["values"][h][z] if baseline else None
                for name in model_names:
                    p = predictions.get(name)
                    price, fallback = (
                        (p["values"][h][z], p["fallback"][h][z]) if p else (None, False)
                    )
                    for level, group in (
                        ("overall", "all"),
                        ("country_or_market", GROUP_OF[zone]),
                        ("zone", zone),
                    ):
                        values = by_key.setdefault(
                            (name, category, level, group), empty()
                        )
                        add_count(values, outcome, price, base_price, fallback)

    rows = []
    for (name, category, level, group), totals in sorted(by_key.items()):
        rows.append(
            dict(
                model=name,
                publication=category,
                level=level,
                group=group,
                **summarize(totals),
            )
        )
    for name in model_names:
        for category in PUBLICATION_CLASSES:
            countries = [
                r
                for r in rows
                if r["model"] == name
                and r["publication"] == category
                and r["level"] == "country_or_market"
            ]
            if countries:
                fields = (
                    "coverage",
                    "outcome_coverage",
                    "scored_coverage",
                    "mae",
                    "matched_mae",
                    "matched_weekly_mae",
                    "skill",
                )
                rows.append(
                    dict(
                        model=name,
                        publication=category,
                        level="equal_weight_country_or_market",
                        group="all",
                        groups=len(countries),
                        countries_with_scores=sum(
                            r["mae"] is not None for r in countries
                        ),
                        countries_with_skill=sum(
                            r["skill"] is not None for r in countries
                        ),
                        **{k: average(countries, k) for k in fields},
                    )
                )

    def timing(values):
        return {
            "median": median(values) if values else None,
            "max": max(values) if values else None,
        }

    complete = bool(
        phase_days
        and processed == len(phase_days)
        and checks == len(phase_days)
        and clock_now >= local_at(phase_days[-1][0] + (OUTCOME_WINDOW + 1) * DAY)
        and all(
            sum(statuses[m][s] for s in TERMINAL) == len(phase_days)
            for m in model_names
        )
    )
    return {
        "phase": phase_name,
        "scope": "private_research" if private else "core",
        "frozen_spec": next(iter(specs), None),
        "scheduled_delivery_days": len(phase_days),
        "processed_delivery_days": processed,
        "delivery_days_with_outcome_checks": checks,
        "complete": complete,
        "scores": rows,
        "operations": {
            m: {
                "jobs": dict(statuses[m]),
                "attempts": dict(attempts[m]),
                "commit_seconds_after_cutoff": timing(latency[m]),
                "worker_seconds": timing(worker[m]),
            }
            for m in model_names
        },
    }


def write_reports(ledger, phase, private=False):
    result = score(ledger, phase, private)
    prefix = phase + ("-private" if private else "")
    atomic_write(ledger.state / (prefix + ".json"), dump(result))
    fields = list(dict.fromkeys(k for row in result["scores"] for k in row))
    csv_buffer = io.StringIO()
    if fields:
        writer = csv.DictWriter(csv_buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["scores"])
    atomic_write(ledger.state / (prefix + ".csv"), csv_buffer.getvalue().encode())
    return result
