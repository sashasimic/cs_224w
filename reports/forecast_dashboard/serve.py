"""Serve the local forecast dashboard and its read-only data API."""

import argparse
import json
import mimetypes
import sqlite3
from contextlib import closing
from datetime import date, datetime, time, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
STATE = REPO / "runs" / "forecast_service"
BRUSSELS = ZoneInfo("Europe/Brussels")
ASSETS = {
    "/": "index.html",
    "/dashboard.css": "dashboard.css",
    "/dashboard.js": "dashboard.js",
}
REPORTS = ("pilot", "pilot-private", "experiment", "experiment-private")
CSV = {f"/data/{name}.csv": f"{name}.csv" for name in REPORTS}


def read_json(path):
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return None


def timetable(day):
    prior = day - timedelta(days=1)
    return {
        key: datetime.combine(prior, time(h, m), BRUSSELS).isoformat()
        for key, h, m in (
            ("capture", 11, 52),
            ("freeze", 11, 55),
            ("commit", 12, 0),
            ("outcomes", 12, 45),
        )
    }


def ledger_status(state):
    path = state / "ledger.sqlite3"
    if not path.is_file():
        return [], []
    with closing(
        sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)
    ) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        jobs = [
            dict(row)
            for row in db.execute(
                """SELECT day, model, status, started, committed, detail
               FROM jobs ORDER BY day, model"""
            )
        ]
        outcomes = {
            row["day"]: dict(row)
            for row in db.execute(
                """SELECT day, max(checked) AS checked, max(observed_hours) AS observed
               FROM outcome_checks GROUP BY day"""
            )
        }
        receipts = dict(
            db.execute(
                """SELECT substr(day,1,10), count(DISTINCT zone)
               FROM receipts WHERE role='input' GROUP BY substr(day,1,10)"""
            )
        )
        snapshots = {row[0] for row in db.execute("SELECT day FROM snapshots")}
        phases = []
        for row in db.execute("SELECT name,start,days,spec FROM phases ORDER BY start"):
            start = date.fromisoformat(row["start"])
            models = list(json.loads(row["spec"])["artifacts"])
            days = []
            for i in range(row["days"]):
                day = start + timedelta(days=i)
                key = day.isoformat()
                daily_jobs = [job for job in jobs if job["day"] == key]
                days.append(
                    {
                        "date": key,
                        "times": timetable(day),
                        "receipts": receipts.get(key, 0),
                        "frozen": key in snapshots,
                        "checked": outcomes.get(key, {}).get("checked"),
                        "observed": outcomes.get(key, {}).get("observed"),
                        "committed": sum(bool(job["committed"]) for job in daily_jobs),
                    }
                )
            phases.append({"name": row["name"], "models": models, "days": days})
        return phases, jobs


def dashboard(state):
    phases, jobs = ledger_status(state)
    roster = read_json(REPO / "epgnn" / "forecast_service" / "zones.json")
    preflight = read_json(state / "preflight.json")
    if preflight:
        preflight = {
            "checked": preflight["checked"],
            "passed": preflight["passed"],
            "zones": {
                key: {k: row[k] for k in ("status", "http", "complete_hours")}
                for key, row in preflight["zones"].items()
            },
        }
    return {
        "updated": datetime.now(BRUSSELS).isoformat(),
        "phases": phases,
        "jobs": jobs,
        "zones": roster,
        "preflight": preflight,
        "reports": {name: read_json(state / (name + ".json")) for name in REPORTS},
        "backtests": {
            name: read_json(state / (name + ".json"))
            for name in ("backtest", "backtest-private")
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        port = self.server.server_port
        if self.headers.get("Host") not in {
            "localhost",
            "127.0.0.1",
            f"localhost:{port}",
            f"127.0.0.1:{port}",
        }:
            return self.send_error(403)
        path = urlsplit(self.path).path
        try:
            if path == "/api/dashboard":
                data = json.dumps(
                    dashboard(self.server.state), allow_nan=False
                ).encode()
                return self.respond(data, "application/json; charset=utf-8")
            if path in ASSETS:
                filename = ASSETS[path]
                mime = mimetypes.guess_type(filename)[0]
                return self.respond(
                    (ROOT / filename).read_bytes(), f"{mime}; charset=utf-8"
                )
            if path in CSV:
                filename = CSV[path]
                data = (self.server.state / filename).read_bytes()
                return self.respond(data, "text/csv; charset=utf-8", filename)
        except FileNotFoundError:
            pass
        except (OSError, sqlite3.Error, ValueError):
            return self.respond(
                b'{"error":"The forecast data is temporarily unavailable."}',
                "application/json",
                status=503,
            )
        self.send_error(404)

    def respond(self, data, mime, filename=None, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; object-src 'none'; frame-ancestors 'none'",
        )
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.state = args.state.resolve()
        print(
            f"Forecast monitor is listening at http://127.0.0.1:{server.server_port}",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
