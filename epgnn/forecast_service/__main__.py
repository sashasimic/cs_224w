"""Entrypoints for the day-ahead forecast experiment."""

import argparse
import json
import time
from datetime import date
from pathlib import Path

from .contracts import (
    BRUSSELS,
    DAY,
    MODELS,
    configure_threads,
    cutoff,
    deadline,
    local_at,
    now,
)

configure_threads()

from . import feed, private_backtest, rehearsal, report, rocm, service, training
from .ledger import Ledger

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "runs/forecast_service"


def forecast_tick(ledger):
    with ledger.lock("forecast") as locked:
        if not locked:
            return
        today = now().astimezone(BRUSSELS).date()
        planned = service.schedule(ledger)
        # Give today's cutoff precedence over recovering any overdue records.
        for day, phase in sorted(
            planned, key=lambda item: (item[0] != today + DAY, item[0])
        ):
            if now() < cutoff(day) - service.CAPTURE_LEAD:
                continue
            complete = ledger.one(
                """SELECT count(*) FROM jobs WHERE day=? AND status IN ('success','fallback','late','missing')""",
                (day.isoformat(),),
            )[0]
            if complete == len(MODELS):
                continue
            spec = service.spec_for_phase(phase)
            if now() < cutoff(day):
                try:
                    token = feed.credential()
                except (OSError, RuntimeError):
                    token = None
                if token:
                    service.capture(ledger, day, token)
                delay = (cutoff(day) - now()).total_seconds()
                while delay > 0:
                    time.sleep(min(0.25, delay))
                    delay = (cutoff(day) - now()).total_seconds()
            service.forecast(ledger, day, spec)


def outcome_tick(ledger):
    clock = now()
    # Leave a clear CPU and GPU scheduling window before the forecast commitment.
    local = clock.astimezone(BRUSSELS)
    if (local.hour, local.minute) >= (11, 50) and local.hour < 12:
        return
    with ledger.lock("outcome") as locked:
        if not locked:
            return
        scheduled = service.schedule(ledger)
        if not scheduled:
            return
        for day, phase in reversed(scheduled):
            if (
                deadline(day)
                < clock
                <= local_at(day + (report.OUTCOME_WINDOW + 1) * DAY)
            ):
                service.spec_for_phase(phase)
                service.fetch_outcomes(ledger, day, feed.credential())
        service.advance(ledger)
        for phase in {p["name"] for _, p in service.schedule(ledger)}:
            for private in (False, True):
                report.write_reports(ledger, phase, private)


def status(ledger):
    phases = [
        {
            "phase": r["name"],
            "start": r["start"],
            "days": r["days"],
            "spec": r["spec_hash"],
        }
        for r in ledger.rows("SELECT * FROM phases ORDER BY start")
    ]
    jobs = [
        dict(r)
        for r in ledger.rows(
            """SELECT day, model, status, started, committed, snapshot, artifact, detail
           FROM jobs ORDER BY day,model"""
        )
    ]
    outcomes = [
        dict(r)
        for r in ledger.rows(
            "SELECT day, max(checked) AS checked, max(observed_hours) AS observed_hours FROM outcome_checks GROUP BY day"
        )
    ]
    return {
        "phases": phases,
        "jobs": jobs,
        "outcomes": outcomes,
        "pilot": service.pilot_result(ledger),
        "readiness": {
            n: (ledger.state / f"{n}.json").is_file()
            for n in ("candidate", "backtest", "gpu", "rehearsal", "preflight")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("train")
    command.add_argument("--archive", type=Path, required=True)
    command.add_argument(
        "--timesfm-repo", type=Path, default=ROOT.parent / "cs_224w_timesfm"
    )
    command = sub.add_parser("rocm")
    command.add_argument("--archive", type=Path, required=True)
    command.add_argument("--vram-file", type=Path)
    command = sub.add_parser("private-backtest")
    command.add_argument("--archive", type=Path, required=True)
    command = sub.add_parser("arm")
    command.add_argument(
        "--start",
        required=True,
        type=date.fromisoformat,
        help="First pilot delivery date",
    )
    command = sub.add_parser("report")
    command.add_argument(
        "--phase", choices=("pilot", "experiment"), default="experiment"
    )
    command.add_argument("--private", action="store_true")
    command = sub.add_parser("worker", help=argparse.SUPPRESS)
    for name in ("day", "model", "snapshot", "artifact"):
        command.add_argument(name)
    command.add_argument("output", type=Path)
    for name in ("forecast-tick", "outcome-tick", "status", "preflight", "rehearse"):
        sub.add_parser(name)
    args = parser.parse_args()
    state = args.state.resolve()
    if args.command == "worker":
        service.run_worker(
            state, args.day, args.model, args.snapshot, args.artifact, args.output
        )
        return
    with Ledger(state) as ledger:
        if args.command in (
            "train",
            "rocm",
            "arm",
            "preflight",
            "rehearse",
            "private-backtest",
        ):
            with ledger.lock("setup") as locked:
                if not locked:
                    parser.error("Another setup task has the ledger lock")
                if args.command in (
                    "train",
                    "rocm",
                    "private-backtest",
                ) and service.schedule(ledger):
                    parser.error(
                        "Use a separate state directory for setup after the pilot has been armed"
                    )
                if args.command == "train":
                    result = training.train(
                        state, args.archive.resolve(), args.timesfm_repo.resolve()
                    )
                elif args.command == "rocm":
                    paths = list(
                        Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")
                    )
                    if args.vram_file is None and len(paths) != 1:
                        parser.error(
                            "Choose the sysfs VRAM counter for the selected ROCm device with --vram-file"
                        )
                    result = rocm.benchmark(
                        state, args.archive.resolve(), args.vram_file or paths[0]
                    )
                    if not result["validated"]:
                        print(json.dumps(result, indent=2))
                        parser.exit(1)
                elif args.command == "private-backtest":
                    result = private_backtest.run(state, args.archive.resolve())
                elif args.command == "rehearse":
                    pilot = ledger.one("SELECT start FROM phases WHERE name='pilot'")
                    if pilot:
                        service.check_arm_window(
                            ledger, date.fromisoformat(pilot["start"])
                        )
                    result = rehearsal.rehearse(state)
                elif args.command == "arm":
                    service.arm(ledger, args.start, feed.credential())
                    result = status(ledger)
                else:
                    result = service.preflight(ledger, feed.credential())
        elif args.command == "outcome-tick":
            outcome_tick(ledger)
            return
        elif args.command == "forecast-tick":
            forecast_tick(ledger)
            return
        elif args.command == "report":
            results = report.write_reports(ledger, args.phase, args.private)
            result = {k: v for k, v in results.items() if k != "scores"}
        else:
            result = status(ledger)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
