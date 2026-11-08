"""Restartable official-ranking and METAR downloads; never model-fill."""

import gzip
import io
import json
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from epgnn.dataset_build.storage import ROOT, digest
from epgnn.experiment_suite.artifacts import write_json

RANKING_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/avia_paoa?"
    "lang=en&time=2023&freq=A&unit=PAS&tra_meas=PAS_CRD&schedule=TOTAL&tra_cov=TOTAL"
)
COUNTRIES = [
    "AT",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LV",
    "ME",
    "MK",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "RS",
    "SE",
    "SI",
    "SK",
]
ME_SOURCE = "https://montenegroairports.com/novosti/finalne-brojke-tokom-2023-godine-opsluzeno-preko-dva-i-po-miliona-putnika/"


def download(url):
    """Bounded retries and conservative pacing respect the IEM rate limit."""
    for attempt in range(4):
        try:
            request = Request(
                url, headers={"User-Agent": "CS224W-airport-temperature-study/1.0"}
            )
            with urlopen(request, timeout=180) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def freeze_mapping(config):
    """Freeze rankings once; no station selection using weather or price scores."""
    mapping_path = ROOT / config["airport_mapping"]
    source = ROOT / config["source_directory"]
    source.mkdir(parents=True, exist_ok=True)
    rankings_path = source / "eurostat_2023.json"
    if not rankings_path.exists():
        payload = download(RANKING_URL)
        json.loads(payload)
        temporary = rankings_path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(rankings_path)
    document = json.loads(rankings_path.read_text())
    dimensions = document["id"]
    airports = document["dimension"]["rep_airp"]["category"]
    rows = []
    for code, airport_index in airports["index"].items():
        if not re.fullmatch(r"[A-Z]{2}_[A-Z]{4}", code):
            continue
        coordinates = [0] * len(dimensions)
        coordinates[dimensions.index("rep_airp")] = airport_index
        flat = int(np.ravel_multi_index(tuple(coordinates), document["size"]))
        value = document["value"].get(str(flat))
        if value is None:
            continue
        country, station = code.split("_")
        rows.append(
            {
                "country": {"EL": "GR"}.get(country, country),
                "station": station,
                "airport": airports["label"][code],
                "passengers_2023": int(value),
                "ranking_source": RANKING_URL,
            }
        )
    ranked = (
        pd.DataFrame(rows)
        .sort_values(
            ["country", "passengers_2023", "station"], ascending=[True, False, True]
        )
        .drop_duplicates("country")
    )
    # Official operator lists both Montenegrin airports; Podgorica > Tivat.
    ranked = ranked[ranked.country.ne("ME")]
    ranked = pd.concat(
        [
            ranked,
            pd.DataFrame(
                [
                    {
                        "country": "ME",
                        "station": "LYPG",
                        "airport": "Podgorica",
                        "passengers_2023": 1657522,
                        "ranking_source": ME_SOURCE,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ranked = (
        ranked[ranked.country.isin(COUNTRIES)]
        .sort_values("country")
        .reset_index(drop=True)
    )
    if set(ranked.country) != set(COUNTRIES):
        raise ValueError(
            f"Missing official rankings: {set(COUNTRIES) - set(ranked.country)}"
        )
    if mapping_path.exists():
        pd.testing.assert_frame_equal(
            pd.read_csv(mapping_path), ranked, check_dtype=False
        )
    else:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(mapping_path, index=False)
    write_json(
        source / "ranking_manifest.json",
        {
            "url": RANKING_URL,
            "file": digest(rankings_path),
            "mapping_sha256": digest(mapping_path)["sha256"],
            "year": 2023,
            "rule": "largest annual passengers, ties by ICAO identifier",
            "montenegro_source": ME_SOURCE,
        },
    )
    return ranked


def observation_url(station, config):
    return "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urlencode(
        {
            "station": station,
            "data": ["tmpf", "metar"],
            "sts": config["observation_start"],
            "ets": config["observation_end"],
            "tz": "UTC",
            "format": "onlycomma",
            "latlon": "yes",
            "elev": "yes",
            "missing": "M",
            "report_type": [3, 4],
            "direct": "no",
        },
        doseq=True,
    )


def fetch(config):
    source = ROOT / config["source_directory"]
    mapping_path = ROOT / config["airport_mapping"]
    mapping = freeze_mapping(config)
    raw = source / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for row in mapping.itertuples():
        path = raw / f"{row.station}.csv.gz"
        metadata = path.with_suffix(".json")
        url = observation_url(row.station, config)
        if path.exists() and metadata.exists():
            record = json.loads(metadata.read_text())
            if record["url"] != url or record["sha256"] != digest(path)["sha256"]:
                raise ValueError(f"Source identity changed: {path}")
        else:
            print(f"FETCH {row.country}: {row.station} ({row.airport})", flush=True)
            payload = download(url)
            frame = pd.read_csv(io.BytesIO(payload), comment="#", na_values=["M"])
            if not {"station", "valid", "tmpf", "metar"}.issubset(frame):
                raise ValueError(
                    f"Invalid IEM response for {row.station}: {payload[:200]!r}"
                )
            if set(frame.station.dropna().astype(str)) - {row.station}:
                raise ValueError(f"Unexpected station identifiers for {row.station}")
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(gzip.compress(payload, mtime=0))
            temporary.replace(path)
            record = {
                "station": row.station,
                "country": row.country,
                "url": url,
                "sha256": digest(path)["sha256"],
                "rows": len(frame),
                "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "availability": "conditional: original receipt/revision timestamps absent",
            }
            write_json(metadata, record)
            time.sleep(1.1)
        print(f"READY {row.station}: {record['rows']} reports", flush=True)
        records.append(record)
    write_json(
        source / "manifest.json",
        {
            "mapping_sha256": digest(mapping_path)["sha256"],
            "files": records,
            "ranking_manifest": json.loads(
                (source / "ranking_manifest.json").read_text()
            ),
        },
    )
    return records


def checked_sources(config):
    if config["ranking_year"] != 2023:
        raise ValueError("This mapping is pinned to the 2023 ranking")
    source = ROOT / config["source_directory"]
    mapping = ROOT / config["airport_mapping"]
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["mapping_sha256"] != digest(mapping)["sha256"]:
        raise ValueError("Airport mapping changed after download")
    if {item["station"] for item in manifest["files"]} != set(
        pd.read_csv(mapping).station
    ):
        raise ValueError("Incomplete observation manifest")
    ranking = manifest["ranking_manifest"]
    if (
        ranking["file"] != digest(source / "eurostat_2023.json")
        or ranking["mapping_sha256"] != manifest["mapping_sha256"]
    ):
        raise ValueError("Official ranking provenance changed")
    if len(manifest["files"]) != pd.read_csv(mapping).station.nunique():
        raise ValueError("Duplicate source manifest records")
    for item in manifest["files"]:
        path = source / "raw" / f"{item['station']}.csv.gz"
        if item["sha256"] != digest(path)["sha256"] or item["url"] != observation_url(
            item["station"], config
        ):
            raise ValueError(f"Observation checksum/request mismatch: {path}")
    return manifest
