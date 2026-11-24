"""Shared market calendar, immutable artifacts and model roster."""

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

UTC = timezone.utc
BRUSSELS = ZoneInfo("Europe/Brussels")
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
HISTORY = 336
THREADS = 4
CPU_WORKERS = 3
CORE_MODELS = (
    "weekly",
    "pooled_xgb",
    "zone_xgb",
    "lear",
    "factor",
    "sequence",
    "blend",
)
PRIVATE_MODELS = ("timesfm_joint", "timesfm_independent")
MODELS = CORE_MODELS + PRIVATE_MODELS
COMPONENTS = CORE_MODELS[:-1]
ROSTER = tuple(json.loads(Path(__file__).with_name("zones.json").read_text()))
ZONE_NAMES = tuple(zone["zone"] for zone in ROSTER)
RULES = {
    "version": 1,
    "timezone": BRUSSELS.key,
    "snapshot": "11:55 D-1",
    "deadline": "12:00 D-1",
    "history_hours": HISTORY,
    "persistence": "168 elapsed hours, hourly EUR/MWh",
    "outcome": "complete EUR/MWh hours from A44 day-ahead, first captured complete hour after deadline",
    "publication": "cutoff visibility on the ENTSO-E feed, with unresolved evidence reported separately",
    "skill": "1 - model MAE / weekly MAE on matched hours",
}


def now():
    return datetime.now(UTC)


def local_at(day: date, hour=0, minute=0):
    return datetime.combine(day, time(hour, minute), BRUSSELS).astimezone(UTC)


def cutoff(day):
    return local_at(day - DAY, 11, 55)


def deadline(day):
    return local_at(day - DAY, 12)


def hour_grid(day):
    start = local_at(day)
    length = (local_at(day + DAY) - start) // HOUR
    return [start + i * HOUR for i in range(length)]


def dump(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        hash_ = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)


def code_version():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    return digest(dump({path.name: file_hash(path) for path in paths}))


def configure_threads():
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(THREADS)
