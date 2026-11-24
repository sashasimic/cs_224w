"""Capture ENTSO-E A44 auctions, retaining exact payload and receipt provenance."""

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .contracts import HOUR, UTC, digest, now

ENDPOINT = "https://web-api.tp.entsoe.eu/api"
UKRAINE_ZONE = "10Y1001C--000182"
AUCTION_SEQUENCE = "classificationSequence_AttributeInstanceComponent.position"
KYIV = ZoneInfo("Europe/Kyiv")
NBU_ENDPOINT = "https://bank.gov.ua/NBU_Exchange/exchange_site"
QUARTER = timedelta(minutes=15)
RESOLUTIONS = {"PT15M": QUARTER, "PT30M": 2 * QUARTER, "PT60M": HOUR}


def timestamp(text):
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Expected an offset in the source timestamp")
    return dt.astimezone(UTC)


def parse(payload, eic, start, end, rates=None):
    """Return hourly value/publication evidence and the quality of the query."""
    root = ET.fromstring(payload)
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag == "Acknowledgement_MarketDocument":
        messages = [el.text or "" for el in root.findall(".//Reason/text")]
        absent = any(
            re.search(
                r"no matching data|no data found|no data available", msg, re.IGNORECASE
            )
            for msg in messages
        )
        return {}, "empty" if absent else "rejected"
    if root.tag != "Publication_MarketDocument" or root.findtext("type") != "A44":
        raise ValueError("Expected an A44 publication document")

    # A response header can be generated for a query; keep it for diagnostics.
    # Cutoff classification only uses the separate pre-cutoff feed observation.
    all_points = {}
    rates = rates or {}
    for series in root.findall("TimeSeries"):
        if series.findtext("contract_MarketAgreement.type", "A01") != "A01":
            continue
        if series.findtext(AUCTION_SEQUENCE) not in (None, "1"):
            continue
        if any(series.findtext(f"{side}_Domain.mRID") != eic for side in ("in", "out")):
            raise ValueError("Response domain differs from requested EIC")
        currency = series.findtext("currency_Unit.name")
        if series.findtext("price_Measure_Unit.name") != "MWH" or (
            currency != "EUR" and (currency, eic) != ("UAH", UKRAINE_ZONE)
        ):
            raise ValueError("Unsupported price currency or unit")
        for period in series.findall("Period"):
            resolution = RESOLUTIONS.get(period.findtext("resolution"))
            if resolution is None:
                raise ValueError("Unsupported price resolution")
            a = timestamp(period.findtext("timeInterval/start", ""))
            b = timestamp(period.findtext("timeInterval/end", ""))
            if (
                b <= a
                or (b - a) % resolution
                or (a - datetime(1970, 1, 1, tzinfo=UTC)) % resolution
            ):
                raise ValueError("Invalid price period")
            n = (b - a) // resolution
            points = {}
            for point in period.findall("Point"):
                k = int(point.findtext("position", "0")) - 1
                price = float(point.findtext("price.amount", "nan"))
                if not 0 <= k < n or k in points:
                    raise ValueError("Invalid or duplicate price position")
                points[k] = price if math.isfinite(price) else None
            if not points:
                continue
            curve = series.findtext("curveType", "A01")
            if curve not in ("A01", "A03"):
                raise ValueError("Unsupported price curve")
            offered = all_points.setdefault(resolution, {})
            previous = None
            for k in range(n):
                if k in points:
                    previous = points[k]
                t = a + k * resolution
                if start <= t < end:
                    value = previous if curve == "A03" else points.get(k)
                    value = (value, currency)
                    if t in offered and value != offered[t]:
                        offered[t] = (None, currency)
                    else:
                        offered[t] = value
    hours = {}
    t = start
    while t < end:
        for resolution, points in sorted(all_points.items()):
            slots = [t + i * resolution for i in range(HOUR // resolution)]
            if any(slot in points for slot in slots):
                records = [points.get(slot, (None, "EUR")) for slot in slots]
                values = []
                for slot, (amount, currency) in zip(slots, records, strict=True):
                    rate = (
                        1
                        if currency == "EUR"
                        else rates.get(slot.astimezone(KYIV).date())
                    )
                    values.append(
                        amount / rate if amount is not None and rate else None
                    )
                # A finer partial hour cannot be filled by a coarser series.
                hour = {
                    "price": sum(values) / len(values) if None not in values else None,
                    "published": any(amount is not None for amount, _ in records),
                }
                if any(currency == "UAH" for _, currency in records):
                    hour["native_complete"] = all(
                        amount is not None for amount, _ in records
                    )
                hours[t.isoformat()] = hour
                break
        t += HOUR
    return hours, "ok" if all_points else "empty"


def parse_rates(body):
    rates = {}
    for row in json.loads(body):
        day = datetime.strptime(row["exchangedate"], "%d.%m.%Y").date()
        rate = float(row["rate_per_unit"])
        if row["cc"] != "EUR" or not math.isfinite(rate) or rate <= 0 or day in rates:
            raise ValueError("Invalid official EUR/UAH exchange rate")
        rates[day] = rate
    return rates


def fetch_rates(start, end):
    first = start.astimezone(KYIV).date()
    last = (end - timedelta(microseconds=1)).astimezone(KYIV).date()
    query = f"start={first:%Y%m%d}&end={last:%Y%m%d}&valcode=eur&sort=exchangedate&order=asc&json"
    url = f"{NBU_ENDPOINT}?{query}"
    body = None
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            raw = response.read(65_537)
            if len(raw) > 65_536:
                raise ValueError("Unexpected size of official exchange-rate response")
        body = raw.decode("utf-8")
        rates = parse_rates(raw)
    except (OSError, ValueError, KeyError, TypeError):
        rates = {}
    return rates, {
        "url": url,
        "received": now().isoformat(),
        "days": len(rates),
        "sha256": digest(body.encode()) if body is not None else None,
        "body": body,
    }


def fetch(eic, start, end, token, timeout):
    """Return the raw response and normalized receipt without retaining credentials."""
    query = urllib.parse.urlencode(
        {
            "documentType": "A44",
            "contract_MarketAgreement.Type": "A01",
            "in_Domain": eic,
            "out_Domain": eic,
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
            "securityToken": token,
        }
    )
    started = now()
    payload = None
    fx = None
    http = None
    try:
        with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=timeout) as response:
            http = response.status
            payload = response.read(4_000_001)
        if len(payload) > 4_000_000:
            raise ValueError("Response exceeds expected length")
        rates = None
        if eic == UKRAINE_ZONE and b">UAH<" in payload:
            rates, fx = fetch_rates(start, end)
        values, status = parse(payload, eic, start, end, rates)
        if (
            fx
            and any(hour.get("native_complete") for hour in values.values())
            and not any(hour["price"] is not None for hour in values.values())
        ):
            status = "fx_unavailable"
    except urllib.error.HTTPError as error:
        http = error.code
        with error:
            payload = error.read(65_536)
        values = {}
        status = "http_error"
        if http == 400:
            try:
                _, parsed = parse(payload, eic, start, end)
                if parsed == "empty":
                    status = parsed
            except (ValueError, ET.ParseError):
                pass
    except (OSError, ValueError, ET.ParseError):
        values, status = {}, "invalid_or_unavailable"
    if payload is not None and token.encode() in payload:
        # Keep credentials out of retained error pages or echoed request URLs.
        payload = None
        values, status = {}, "invalid_or_unavailable"
    receipt = {
        "eic": eic,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "started": started.isoformat(),
        "received": now().isoformat(),
        "sha256": digest(payload) if payload is not None else None,
        "http": http,
        "status": status,
        "hours": values,
    }
    if fx is not None:
        receipt["fx"] = fx
    return receipt, payload


def credential():
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    path = os.environ.get("ENTSOE_TOKEN_FILE") or (
        os.path.join(directory, "entsoe_token") if directory else None
    )
    token = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if not token and path:
        with open(path, encoding="utf-8") as stream:
            token = stream.read().strip()
    if not token:
        raise RuntimeError("Configure ENTSOE_API_TOKEN or a local ENTSO-E token file")
    return token
