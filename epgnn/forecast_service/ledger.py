"""Durable snapshots, idempotent job keys and a history of worker attempts."""

import fcntl
import json
import sqlite3
from contextlib import contextmanager

from .contracts import atomic_write, digest, dump

SCHEMA = """
CREATE TABLE IF NOT EXISTS phases (
    name TEXT PRIMARY KEY, start TEXT NOT NULL, days INTEGER NOT NULL,
    spec TEXT NOT NULL, spec_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    role TEXT NOT NULL, day TEXT NOT NULL, zone TEXT NOT NULL,
    body TEXT NOT NULL, PRIMARY KEY (role, day, zone)
);
CREATE TABLE IF NOT EXISTS snapshots (
    day TEXT PRIMARY KEY, sha TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    day TEXT NOT NULL, model TEXT NOT NULL, snapshot TEXT NOT NULL, artifact TEXT NOT NULL,
    deadline TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
    started TEXT, committed TEXT, prediction TEXT, prediction_sha TEXT, detail TEXT,
    PRIMARY KEY (day, model)
);
CREATE TABLE IF NOT EXISTS attempts (
    day TEXT NOT NULL, model TEXT NOT NULL, started TEXT NOT NULL,
    ended TEXT, status TEXT NOT NULL, detail TEXT, PRIMARY KEY (day, model, started)
);
CREATE TABLE IF NOT EXISTS outcomes (
    day TEXT NOT NULL, zone TEXT NOT NULL, hour TEXT NOT NULL, price REAL NOT NULL,
    receipt_sha TEXT NOT NULL, received TEXT NOT NULL, PRIMARY KEY (day, zone, hour)
);
CREATE TABLE IF NOT EXISTS outcome_checks (
    day TEXT NOT NULL, checked TEXT NOT NULL, observed_hours INTEGER NOT NULL,
    PRIMARY KEY(day, checked)
);
CREATE INDEX IF NOT EXISTS attempt_day ON attempts(day);
"""


class Ledger:
    def __init__(self, state):
        self.state = state.resolve()
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.state / "ledger.sqlite3", timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()

    def rows(self, query, args=()):
        return self.db.execute(query, args).fetchall()

    def one(self, query, args=()):
        return self.db.execute(query, args).fetchone()

    @contextmanager
    def lock(self, name):
        with (self.state / (name + ".lock")).open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
            else:
                yield True

    def record_receipt(self, role, key, zone, receipt, payload):
        sha = receipt["sha256"]
        if payload is not None:
            if digest(payload) != sha:
                raise ValueError("Receipt checksum mismatch")
            path = self.state / "receipts" / sha
            if not path.exists():
                atomic_write(path, payload)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO receipts VALUES (?,?,?,?)",
                (role, key, zone, dump(receipt)),
            )
            if role == "outcome" and receipt["status"] == "ok":
                day = key.split("/")[0]
                for hour, info in receipt["hours"].items():
                    if info["price"] is not None:
                        self.db.execute(
                            "INSERT OR IGNORE INTO outcomes VALUES (?,?,?,?,?,?)",
                            (day, zone, hour, info["price"], sha, receipt["received"]),
                        )

    def save_snapshot(self, snapshot, artifacts, deadline):
        data = dump(snapshot)
        sha = digest(data)
        day = snapshot["day"]
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO snapshots VALUES (?,?,?)", (day, sha, data)
            )
            saved = self.one("SELECT sha FROM snapshots WHERE day=?", (day,))
            if saved["sha"] != sha:
                raise ValueError("Delivery day already has a different frozen snapshot")
            self.db.executemany(
                "INSERT OR IGNORE INTO jobs(day,model,snapshot,artifact,deadline) VALUES (?,?,?,?,?)",
                [
                    (day, name, sha, artifact, deadline.isoformat())
                    for name, artifact in artifacts.items()
                ],
            )
        return sha

    def snapshot(self, day):
        saved = self.one("SELECT * FROM snapshots WHERE day=?", (day,))
        if saved is None:
            return None
        if digest(saved["body"]) != saved["sha"]:
            raise ValueError("Snapshot hash mismatch")
        return saved["sha"], json.loads(saved["body"])

    def recover(self, day, now):
        with self.db:
            self.db.execute(
                """UPDATE attempts SET ended=?,status='interrupted'
                            WHERE day=? AND status='running'""",
                (now.isoformat(), day),
            )
            self.db.execute(
                "UPDATE jobs SET status='pending',started=NULL WHERE day=? AND status='running'",
                (day,),
            )

    def start(self, day, model, now):
        stamp = now.isoformat()
        with self.db:
            self.db.execute(
                "UPDATE jobs SET started=?,status='running' WHERE day=? AND model=? AND status='pending'",
                (stamp, day, model),
            )
            self.db.execute(
                "INSERT INTO attempts(day,model,started,status) VALUES (?,?,?,'running')",
                (day, model, stamp),
            )

    def finish(self, day, model, data, status, detail, clock):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.one(
                "SELECT snapshot,artifact,deadline FROM jobs WHERE day=? AND model=?",
                (day, model),
            )
            body = (
                dump(
                    {
                        "snapshot": row["snapshot"],
                        "artifact": row["artifact"],
                        "rows": data,
                    }
                )
                if data is not None
                else None
            )
            stamp = clock().isoformat()
            if stamp > row["deadline"]:
                status = "late"
            self.db.execute(
                """UPDATE jobs SET status=?,committed=?,prediction=?,prediction_sha=?,detail=?
                            WHERE day=? AND model=? AND status IN ('pending','running')""",
                (
                    status,
                    stamp,
                    body,
                    digest(body) if body is not None else None,
                    detail,
                    day,
                    model,
                ),
            )
            self.db.execute(
                """UPDATE attempts SET ended=?,status=?,detail=?
                            WHERE day=? AND model=? AND status='running'""",
                (stamp, status, detail, day, model),
            )
        finished = clock().isoformat()
        if stamp <= row["deadline"] < finished:
            with self.db:
                self.db.execute(
                    "UPDATE jobs SET status='late',committed=? WHERE day=? AND model=?",
                    (finished, day, model),
                )
                self.db.execute(
                    "UPDATE attempts SET status='late',ended=? WHERE day=? AND model=? AND ended=?",
                    (finished, day, model, stamp),
                )
        return status if finished <= row["deadline"] else "late"
