import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from epgnn.dataset_build import audit, build, validate
from epgnn.dataset_build.construct import adapt, coverage_join, regional_membership
from epgnn.dataset_build.contracts import WEATHER
from epgnn.dataset_build.inspection import select
from epgnn.dataset_build.storage import digest, json_write


def fixture(root):
    zones = ["BE", "FR", "DE_LU", "DE_AT_LU"]
    names = {"BE": "Belgium (BE)", "FR": "France (FR)", "DE_LU": "DE-LU", "DE_AT_LU": "DE-AT-LU"}
    t = pd.date_range("2026-09-21", "2026-10-03", inclusive="left", freq="h", tz="UTC")
    n = pd.MultiIndex.from_product([zones, t], names=["zone", "ts"]).to_frame(index=False)
    common = pd.DataFrame({"AreaCode": n.zone, "AreaName": n.zone.map(names), "DeliveryHourUTC": n.ts,
        "MarketDayCET": n.ts.dt.tz_convert("Europe/Brussels").dt.strftime("%Y-%m-%d"), "is_filled": False,
        "row_status": np.where(n.zone == "DE_AT_LU", "OUTSIDE_ZONE_LIFETIME", "VALID")})
    price = common.assign(PriceEUR=-5., IsValidTarget=n.zone != "DE_AT_LU", PriceEUR_input=-5.)
    missing = (price.AreaCode == "BE") & (price.DeliveryHourUTC == pd.Timestamp("2026-09-30T12:00Z"))
    price.loc[missing, ["PriceEUR", "IsValidTarget"]] = [np.nan, False]
    price.loc[missing, "PriceEUR_input"] = 1234.
    price.loc[missing, "is_filled"] = True
    # A genuinely absent source hour remains on the canonical grid.
    price = price.loc[~((price.AreaCode == "FR") & (price.DeliveryHourUTC == pd.Timestamp("2026-10-01T00:00Z")))]
    load = common.assign(LoadForecastMW=100., IsValidValue=True, IsAvailabilityVerified=False,
        SourceUpdateTimeMaxUTC=pd.Timestamp("2026-11-01", tz="UTC"))
    weather = n.loc[(n.zone != "DE_AT_LU") & (n.ts < pd.Timestamp("2026-10-01", tz="UTC"))].rename(columns={"ts": "time_utc"})
    for c in WEATHER:
        weather[c] = 0.5
    flow = pd.DataFrame({"OutAreaCode": "BE", "OutAreaName": "Belgium (BE)", "InAreaCode": "FR", "InAreaName": "France (FR)",
        "DeliveryHourUTC": t, "FlowMW": 10., "IsValidValue": True, "is_filled": False,
        "MarketDayCET": t.tz_convert("Europe/Brussels").strftime("%Y-%m-%d"), "IsAvailabilityVerified": False})
    sources = {}
    for name, f in [("entsoe_prices", price), ("entsoe_load_forecast", load), ("entsoe_physical_flows", flow), ("weather_hourly", weather)]:
        path = root / f"{name}.parquet"
        if name in ["entsoe_prices", "entsoe_physical_flows"]:
            f["DeliveryHourUTC"] = f.DeliveryHourUTC.dt.as_unit("us")
        f.to_parquet(path, index=False)
        sources[name] = {"path": str(path), "version": "fixture", "md5": digest(path)["md5"]}
    jao = root / "jao"
    jao.mkdir()
    jt = t[t < pd.Timestamp("2026-09-30T00:00Z")]
    hub = pd.DataFrame({"ts_utc": jt, "zone": "BE", "n_presolved": 0, "min_ram_ratio": np.nan,
                        "regime": "CORE", "source_zone": "BE", "timestamp_status": "source_utc"})
    border = pd.DataFrame({"ts_utc": jt, "from_zone": "BE", "to_zone": "FR", "maxbex": 0., "flow": 0.,
        "regime": "CORE", "source_from_zone": "BE", "source_to_zone": "FR", "timestamp_status": "source_utc"})
    hub.to_parquet(jao / "jao_hourly.parquet", index=False)
    border.to_parquet(jao / "jao_border_hourly.parquet", index=False)
    cnec = pd.DataFrame({"ts_utc": jt, "cnec_key": [f"source:{i}" for i in range(len(jt))]})
    for name in ["jao_cnec_hourly", "cnec_provenance"]:
        cnec.to_parquet(jao / f"{name}.parquet", index=False)
    pd.DataFrame({"ts_utc": jt, "domain_present": True, "maxbex_present": True, "flow_present": True}).to_csv(jao / "coverage.csv", index=False)
    split = root / "splits.json"
    json_write(split, {"splits": {"test": ["2025-04-01", "2026-09-30"]}})
    cfg = {"root": str(root), "output_directory": "output", "sources": sources, "splits_file": str(split),
        "jao_directory": str(jao), "jao_metadata": ["coverage.csv"], "references": {},
        "output_start": "2026-09-29T00:00Z", "output_end": "2026-10-03T00:00Z", "batch_size": 127}
    path = root / "config.json"
    json_write(path, cfg)
    return path


class IntegrationTests(unittest.TestCase):
    def test_full_build_missingness_reproducibility_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(Path(temp))
            self.assertEqual(audit(cfg)["status"], "passed_with_limitations")
            generation = build(cfg)
            self.assertEqual(validate(cfg)["status"], "passed")
            self.assertEqual(build(cfg), generation)
            n = pd.read_parquet(generation / "nodes")
            e = pd.read_parquet(generation / "edges")
            strict = pd.read_parquet(generation / "strict")
            self.assertEqual(len(n), 4 * 96)
            self.assertEqual(len(e), 96)
            self.assertFalse(n.loc[n.zone == "DE_AT_LU", "price"].notna().any())
            self.assertFalse(n.loc[n.zone == "DE_AT_LU", "weather_row_present"].any())
            self.assertTrue(n.loc[n.delivery_date >= "2026-10-01", "split"].eq("unassigned").all())
            missing = n[(n.zone == "BE") & (n.ts_utc == pd.Timestamp("2026-09-30T12:00Z"))].iloc[0]
            self.assertTrue(pd.isna(missing.price))
            self.assertEqual(missing.price_source_PriceEUR_input, 1234)
            self.assertTrue(n.loc[n.ts_utc >= pd.Timestamp("2026-10-01T00:00Z"), "temp_pw"].isna().all())
            self.assertIn("missing_source_row", n.price_absence_class.unique())
            self.assertTrue(n.price_lag168.notna().any())  # reads before output start
            self.assertFalse(any(c.startswith("price_lag") for c in strict))
            self.assertTrue((e.loc[e.border_row_present, "flow_sched"] == 0).all())
            self.assertTrue(e.loc[~e.border_row_present, "flow_sched"].isna().all())
            self.assertTrue(n.loc[n.jao_row_present, "n_presolved"].eq(0).all())
            self.assertTrue(n.loc[~n.jao_row_present, "n_presolved"].isna().all())
            with self.assertRaisesRegex(ValueError, "test-target"):
                select(cfg, ["BE"], "2026-09-29T00:00Z", "2026-09-30T00:00Z", ["price"])
            p = next((generation / "strict").glob("*.parquet"))
            with p.open("ab") as out:
                out.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate(cfg)

    def test_missing_and_changed_inputs_block_build(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(Path(temp))
            (Path(temp) / "entsoe_prices.parquet").unlink()
            report = audit(cfg)
            self.assertEqual(report["status"], "blocked")
            with self.assertRaisesRegex(ValueError, "blocked"):
                build(cfg)

    def test_absent_coverage_is_not_published_false_or_zero(self):
        f = pd.DataFrame({"ts_utc": pd.to_datetime(["2021-11-10T00:00Z", "2021-11-11T00:00Z"])})
        c = pd.DataFrame({"ts_utc": pd.to_datetime(["2021-11-10T00:00Z"]),
            "domain_present": False, "maxbex_present": False, "flow_present": False})
        got = coverage_join(f, c)
        self.assertFalse(got.jao_domain_present.iloc[0])
        self.assertTrue(pd.isna(got.jao_domain_present.iloc[1]))
        self.assertEqual(got.jao_coverage_row_present.tolist(), [True, False])

    def test_cwe_core_zero_values_and_dated_membership(self):
        border = pd.DataFrame({"ts_utc": pd.to_datetime(["2018-09-29T22:00Z", "2022-06-08T22:00Z"]),
            "from_zone": ["BE", "BE"], "to_zone": ["DE_AT_LU", "DE_LU"],
            "maxbex": [0., np.nan], "flow": [0., np.nan], "regime": ["CWE", "CORE"]})
        got = adapt(border, "jao_border_hourly")
        self.assertEqual(got.exchange_definition.tolist(), ["allocated_BEC", "scheduled_exchange"])
        self.assertEqual(got.exchange_value_present.tolist(), [True, False])
        self.assertEqual(got.flow_sched.iloc[0], 0)
        member = regional_membership(pd.DataFrame({"zone": ["DE_AT_LU", "DE_LU", "AT", "DE_LU"],
            "ts_utc": pd.to_datetime(["2018-09-29T22:00Z", "2018-09-29T22:00Z", "2018-09-29T22:00Z", "2022-06-08T22:00Z"])}))
        self.assertEqual(member.in_cwe.tolist(), [True, False, False, False])
        self.assertEqual(member.in_core.tolist(), [False, False, False, True])

    def test_conflicting_source_rows_and_checksum_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(Path(temp))
            p = Path(temp) / "entsoe_prices.parquet"
            f = pd.read_parquet(p)
            f = pd.concat([f, f.iloc[[0]].assign(PriceEUR=999.)], ignore_index=True)
            f.to_parquet(p, index=False)
            # Checksum mismatch is rejected before attempting source ingestion.
            self.assertEqual(audit(cfg)["status"], "blocked")
            c = json.loads(cfg.read_text())
            c["sources"]["entsoe_prices"]["md5"] = digest(p)["md5"]
            json_write(cfg, c)
            r = audit(cfg)
            self.assertEqual(r["status"], "blocked")
            self.assertGreater(r["tables"]["entsoe_prices"]["conflicting_rows"], 0)


if __name__ == "__main__":
    unittest.main()
