import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from epgnn.forecast_service import feed, models, report, service
from epgnn.forecast_service.contracts import (
    DAY,
    HISTORY,
    HOUR,
    MODELS,
    ROSTER,
    ZONE_NAMES,
    code_version,
    cutoff,
    deadline,
    digest,
    dump,
    hour_grid,
)
from epgnn.forecast_service.ledger import Ledger
from epgnn.forecast_service.rehearsal import receipt
from epgnn.forecast_service.training import (
    model_version,
    read_artifact,
    runtime,
    store_artifact,
)


class FeedTests(unittest.TestCase):
    def test_regular_and_variable_quarters(self):
        day = date(2026, 3, 29)
        market = ROSTER[0]
        response, raw = receipt(day, market, "outcome")
        start, end = hour_grid(day)[0], hour_grid(day)[0] + 2 * HOUR
        self.assertEqual(response["hours"][start.isoformat()]["price"], 40)
        self.assertEqual(len(response["hours"]), 23)
        missing = raw.replace(
            b"<Point><position>2</position><price.amount>40</price.amount></Point>", b""
        )
        normal, _ = feed.parse(missing, market["eic"], start, end)
        self.assertIsNone(normal[start.isoformat()]["price"])
        self.assertTrue(normal[start.isoformat()]["published"])
        variable, _ = feed.parse(
            missing.replace(b"<curveType>A01", b"<curveType>A03"),
            market["eic"],
            start,
            end,
        )
        self.assertEqual(variable[start.isoformat()]["price"], 40)
        with self.assertRaises(ValueError):
            feed.parse(raw, ROSTER[1]["eic"], start, end)

    def test_secondary_auction_does_not_replace_or_conflict_with_primary_prices(self):
        day = date(2026, 10, 25)
        market = ROSTER[0]
        _, raw = receipt(day, market, "outcome")
        start, end = hour_grid(day)[0], hour_grid(day)[-1] + HOUR
        primary = raw.split(b"<TimeSeries>", 1)[1].split(b"</TimeSeries>", 1)[0]
        secondary = primary.replace(
            b"<price.amount>40</price.amount>", b"<price.amount>400</price.amount>"
        )
        sequence = feed.AUCTION_SEQUENCE.encode()
        for primary_sequence in (b"", b"<" + sequence + b">1</" + sequence + b">"):
            combined = raw.replace(
                b"<TimeSeries>" + primary + b"</TimeSeries>",
                b"<TimeSeries>" + primary_sequence + primary + b"</TimeSeries>"
                b"<TimeSeries><"
                + sequence
                + b">2</"
                + sequence
                + b">"
                + secondary
                + b"</TimeSeries>",
            )
            prices, status = feed.parse(combined, market["eic"], start, end)
            self.assertEqual(status, "ok")
            self.assertEqual(prices[start.isoformat()]["price"], 40)
        secondary_only = raw.replace(
            b"<TimeSeries>", b"<TimeSeries><" + sequence + b">2</" + sequence + b">"
        )
        self.assertEqual(
            feed.parse(secondary_only, market["eic"], start, end), ({}, "empty")
        )

    def test_ukrainian_hryvnia_uses_kyiv_delivery_date_and_requires_rate(self):
        day = date(2026, 3, 29)
        market = next(row for row in ROSTER if row["eic"] == feed.UKRAINE_ZONE)
        _, raw = receipt(day, market, "outcome")
        raw = raw.replace(b">EUR<", b">UAH<")
        start, end = hour_grid(day)[0], hour_grid(day)[-1] + HOUR
        # Brussels delivery day ends one hour into the next calendar day in Kyiv.
        rates = feed.parse_rates(
            b'[{"exchangedate":"29.03.2026","cc":"EUR","rate_per_unit":50},'
            b'{"exchangedate":"30.03.2026","cc":"EUR","rate_per_unit":100}]'
        )
        eur, status = feed.parse(raw, market["eic"], start, end, rates)
        native, _ = feed.parse(raw, market["eic"], start, end)
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(eur[start.isoformat()]["price"], 0.8)
        self.assertAlmostEqual(eur[(end - HOUR).isoformat()]["price"], 0.555)
        self.assertIsNone(native[start.isoformat()]["price"])
        self.assertTrue(native[start.isoformat()]["native_complete"])

    def test_no_data_is_distinct_from_invalid_response(self):
        start = hour_grid(date(2026, 10, 25))[0]
        doc = b"<Acknowledgement_MarketDocument><Reason><text>No matching data found</text></Reason></Acknowledgement_MarketDocument>"
        self.assertEqual(
            feed.parse(doc, ROSTER[0]["eic"], start, start + HOUR), ({}, "empty")
        )
        self.assertEqual(
            feed.parse(
                doc.replace(b"No matching data found", b"Bad parameter"),
                ROSTER[0]["eic"],
                start,
                start + HOUR,
            ),
            ({}, "rejected"),
        )

    def test_dst_uses_elapsed_hour_persistence(self):
        for day, hours in [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25)]:
            with self.subTest(day=day):
                panel = np.arange(HISTORY, dtype=np.float32)[:, None]
                inputs = models.daily_inputs(day, panel)
                self.assertEqual(len(set(hour_grid(day))), hours)
                np.testing.assert_array_equal(
                    models.weekly(inputs)[:, 0],
                    np.arange(HISTORY - 168, HISTORY - 168 + hours),
                )


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.day = date(2026, 10, 25)
        self.key = self.day.isoformat()
        self.cutoff = cutoff(self.day)
        self.deadline = deadline(self.day)
        self.artifact = store_artifact(self.state, {"algorithm": "weekly_168h"})
        self.spec = {"artifacts": {"weekly": self.artifact}}

    def test_frozen_snapshot_and_outcomes_stay_immutable(self):
        with Ledger(self.state) as ledger:
            known = ROSTER[0]
            late = ROSTER[1]
            r, payload = receipt(self.day, known, "input", published=True)
            ledger.record_receipt("input", self.key, known["zone"], r, payload)
            with self.assertRaises(ValueError):
                service.freeze_snapshot(
                    ledger, self.day, self.spec["artifacts"], lambda: self.cutoff - HOUR
                )
            sha, snapshot = service.freeze_snapshot(
                ledger, self.day, self.spec["artifacts"], lambda: self.cutoff
            )
            r2, raw = receipt(self.day, late, "input", late=True)
            ledger.record_receipt("input", self.key, late["zone"], r2, raw)
            self.assertEqual(
                service.freeze_snapshot(
                    ledger, self.day, self.spec["artifacts"], lambda: self.cutoff + HOUR
                ),
                (sha, snapshot),
            )
            self.assertEqual(
                set(snapshot["publication"][known["zone"]]), {"published_at_cutoff"}
            )
            self.assertEqual(
                set(snapshot["publication"][late["zone"]]), {"publication_unverified"}
            )
            self.assertTrue(
                all(row[1] is None and 909 not in row for row in snapshot["history"])
            )

            r, payload = receipt(self.day, known, "outcome")
            ledger.record_receipt(
                "outcome", self.key + "/one", known["zone"], r, payload
            )
            item = r["hours"][hour_grid(self.day)[0].isoformat()]
            item["price"] = 999
            ledger.record_receipt(
                "outcome", self.key + "/two", known["zone"], r, payload
            )
            row = ledger.one(
                "SELECT price FROM outcomes WHERE day=? ORDER BY hour", (self.key,)
            )
            self.assertEqual(row[0], 40)

    def test_deadline_crossing_is_excluded_and_hashes_checked(self):
        with Ledger(self.state) as ledger:
            data = dump(self.spec)
            with ledger.db:
                ledger.db.execute(
                    "INSERT INTO phases VALUES ('pilot',?,1,?,?)",
                    (self.key, data, digest(data)),
                )
            sha, _ = service.freeze_snapshot(
                ledger, self.day, self.spec["artifacts"], lambda: self.cutoff
            )
            ledger.start(self.key, "weekly", self.cutoff)
            shape = (len(hour_grid(self.day)), len(ZONE_NAMES))
            values = {
                "values": np.ones(shape).tolist(),
                "fallback": np.zeros(shape, bool).tolist(),
                "primary": np.ones(shape, bool).tolist(),
            }
            times = iter((self.deadline, self.deadline + timedelta(microseconds=1)))
            self.assertEqual(
                ledger.finish(
                    self.key, "weekly", values, "success", None, lambda: next(times)
                ),
                "late",
            )
            row = ledger.one("SELECT * FROM jobs")
            self.assertEqual((row["status"], row["snapshot"]), ("late", sha))
            summary = report.score(ledger, "pilot", clock=lambda: self.deadline + HOUR)
            baseline = next(
                r
                for r in summary["scores"]
                if r["model"] == "weekly" and r["level"] == "overall"
            )
            self.assertEqual(baseline["forecasts"], 0)
            with ledger.db:
                ledger.db.execute(
                    "UPDATE jobs SET prediction=? WHERE model='weekly'",
                    (json.dumps(values),),
                )
            with self.assertRaises(ValueError):
                service.checked_prediction(ledger.one("SELECT * FROM jobs"))

    def test_two_pilot_days_open_fourteen_consecutive_days_with_same_manifest(self):
        spec = {
            "artifacts": dict.fromkeys(MODELS, self.artifact),
            "service_code": code_version(),
            "model_version": model_version(),
            "runtime": runtime(),
        }
        data = dump(spec)
        with Ledger(self.state) as ledger:
            with ledger.db:
                ledger.db.execute(
                    "INSERT INTO phases VALUES ('pilot',?,2,?,?)",
                    (self.key, data, digest(data)),
                )
            for offset in range(2):
                day = self.day + offset * DAY
                key = day.isoformat()
                market = ROSTER[0]
                r, raw = receipt(day, market, "input")
                ledger.record_receipt("input", key, market["zone"], r, raw)
                service.freeze_snapshot(
                    ledger, day, spec["artifacts"], lambda day=day: cutoff(day)
                )
                shape = (len(hour_grid(day)), len(ZONE_NAMES))
                body = {
                    "values": np.ones(shape).tolist(),
                    "fallback": np.zeros(shape, bool).tolist(),
                    "primary": np.ones(shape, bool).tolist(),
                }
                for name in MODELS:
                    ledger.start(key, name, cutoff(day))
                    ledger.finish(
                        key, name, body, "success", None, lambda day=day: cutoff(day)
                    )
                r, raw = receipt(day, market, "outcome")
                ledger.record_receipt(
                    "outcome", key + "/fixture", market["zone"], r, raw
                )
                with patch.object(service, "now", return_value=deadline(day) + HOUR):
                    service.advance(ledger)
                if not offset:
                    self.assertIsNone(
                        ledger.one("SELECT 1 FROM phases WHERE name='experiment'")
                    )
            experiment = [
                (day, phase)
                for day, phase in service.schedule(ledger)
                if phase["name"] == "experiment"
            ]
            first = self.day + 2 * DAY
            self.assertEqual(
                [day for day, _ in experiment], [first + i * DAY for i in range(14)]
            )
            self.assertTrue(
                all(
                    row["spec_hash"] == digest(data) and row["spec"] == data
                    for _, row in experiment
                )
            )

    def test_pre_cutoff_retries_only_retryable_zones_and_ignores_later_receipts(self):
        tries = {}

        def fetch(eic, start, end, token, timeout):
            tries[eic] = tries.get(eic, 0) + 1
            market = next(r for r in ROSTER if r["eic"] == eic)
            if market == ROSTER[0] and tries[eic] == 1:
                return {
                    "sha256": None,
                    "status": "http_error",
                    "http": 599,
                    "received": (self.cutoff - HOUR).isoformat(),
                }, None
            return receipt(self.day, market, "input")

        with (
            Ledger(self.state) as ledger,
            patch.object(service, "now", return_value=self.cutoff - HOUR),
        ):
            selected = service.fetch_with_retries(
                ledger,
                "input",
                self.key,
                self.cutoff - HOUR,
                self.cutoff,
                "",
                stop=self.cutoff,
                fetcher=fetch,
                roster=ROSTER[:2],
            )
            self.assertEqual(tries, {ROSTER[0]["eic"]: 2, ROSTER[1]["eic"]: 1})
            self.assertTrue(all(row["status"] == "ok" for row in selected.values()))
            late, raw = receipt(self.day, ROSTER[0], "input", late=True)
            ledger.record_receipt(
                "input", self.key + "/3", ROSTER[0]["zone"], late, raw
            )
            self.assertEqual(
                service.receipt_rows(ledger, "input", self.key, self.cutoff), selected
            )

    def test_only_unstarted_pilot_can_be_refreshed(self):
        with Ledger(self.state) as ledger:
            with ledger.db:
                ledger.db.execute(
                    "INSERT INTO phases VALUES ('pilot',?,2,?,?)",
                    (self.key, b"{}", digest(b"{}")),
                )
            with patch.object(service, "now", return_value=self.cutoff - HOUR):
                service.check_arm_window(ledger, self.day)
                with self.assertRaises(ValueError):
                    service.check_arm_window(ledger, self.day + DAY)
                market = ROSTER[0]
                captured, payload = receipt(self.day, market, "input")
                ledger.record_receipt(
                    "input", self.key, market["zone"], captured, payload
                )
                with self.assertRaises(ValueError):
                    service.check_arm_window(ledger, self.day)

    def test_model_artifact_is_verified_before_loading(self):
        (self.state / "artifacts" / self.artifact).write_bytes(b"wrong")
        with self.assertRaises(ValueError):
            read_artifact(self.state, self.artifact)


if __name__ == "__main__":
    unittest.main()
