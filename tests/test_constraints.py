"""Checks for the JAO constraint nodes (epgnn.pipeline.constraints): PTDF reference and synthetic filter.

    python -m unittest tests/test_constraints.py -v

Hub PTDFs are zone-to-slack: adding one constant to all hubs of a constraint changes no flow, so every
zone-level quantity built from them must not change either. Synthetic constraints (equality / external /
default limits) are not grid elements and must not become nodes.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from epgnn.pipeline import constraints as C

Zc = len(C.CORE_ZONES)


def hourly_frame(seed: int = 0, n_c: int = 40, hours: int = 6, shift: np.ndarray | float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rows = len(ts) * n_c
    p = rng.normal(0, 0.08, size=(rows, Zc))
    p += np.broadcast_to(np.asarray(shift, dtype=np.float64).reshape(-1, 1), (rows, 1)) if np.ndim(shift) else shift
    df = pd.DataFrame({"ts_utc": np.repeat(ts, n_c), "constraint_id": np.tile([f"c{i}" for i in range(n_c)], len(ts)),
                       "ram_fmax": rng.uniform(0.1, 1.3, rows).astype(np.float32)})
    for k in C.NODE_FEATURES[1:]:
        df[k] = np.float32(1.0)
    for i, z in enumerate(C.CORE_ZONES):
        df[f"ptdf_{z}"] = p[:, i].astype(np.float32)
    return df


class TestRestPtdf(unittest.TestCase):
    def test_formula(self):
        p = np.zeros(Zc)
        p[C.CORE_ZONES.index("FR")] = 0.3
        r = C.rest_ptdf(p)
        self.assertAlmostEqual(r[C.CORE_ZONES.index("FR")], 0.3)
        self.assertTrue(np.allclose(np.delete(r, C.CORE_ZONES.index("FR")), -0.3 / (Zc - 1)))
        # = 12/11 x the PTDF de-meaned over the hubs
        q = np.random.default_rng(1).normal(size=(5, Zc))
        np.testing.assert_allclose(C.rest_ptdf(q), Zc / (Zc - 1) * (q - q.mean(1, keepdims=True)))

    def test_invariant_to_a_constant_per_constraint(self):
        q = np.random.default_rng(2).normal(size=(50, Zc))
        c = np.random.default_rng(3).normal(size=(50, 1))
        np.testing.assert_allclose(C.rest_ptdf(q + c), C.rest_ptdf(q), atol=1e-12)

    def test_missing_hub(self):
        q = np.random.default_rng(4).normal(size=Zc)
        q[2] = np.nan
        r = C.rest_ptdf(q)
        self.assertTrue(np.isnan(r[2]))
        ok = np.isfinite(q)
        np.testing.assert_allclose(r[ok], C.rest_ptdf(q[ok]))


class TestReferenceInvariance(unittest.TestCase):
    def setUp(self):
        self.base = hourly_frame()
        shift = np.random.default_rng(9).normal(0, 0.2, size=len(self.base))   # a different slack per row
        self.shifted = hourly_frame(shift=shift)

    def test_zone_exposure(self):
        a, b = C.zone_exposure(self.base), C.zone_exposure(self.shifted)
        pd.testing.assert_frame_equal(a, b, atol=1e-5, rtol=1e-5)
        # the published (raw) reference is not invariant, which was the bug
        ra, rb = C.zone_exposure(self.base, reference="raw"), C.zone_exposure(self.shifted, reference="raw")
        self.assertFalse(np.allclose(ra.jao_tightness, rb.jao_tightness))

    def test_constraint_graph(self):
        hours = np.array(self.base.ts_utc.drop_duplicates().astype("int64").tolist() + [-1] * 19, dtype=np.int64)
        zones = sorted(C.CORE_ZONES + ["NO_1"])
        ga, gb = C.constraint_graph(self.base, hours, zones), C.constraint_graph(self.shifted, hours, zones)
        np.testing.assert_array_equal(ga["edges"], gb["edges"])
        np.testing.assert_allclose(ga["weight"], gb["weight"], atol=1e-5)
        np.testing.assert_allclose(ga["ptdf"], gb["ptdf"], atol=1e-5)
        self.assertEqual(ga["reference"], "rest")
        self.assertFalse((ga["edges"][0] == zones.index("NO_1")).any())          # only Core zones get edges

    def test_catalog_summary(self):
        cat = pd.DataFrame({"constraint_id": [f"c{i}" for i in range(40)]})
        a, b = C.with_ptdf_summary(cat, self.base), C.with_ptdf_summary(cat, self.shifted)
        pd.testing.assert_series_equal(a.main_zone, b.main_zone)
        np.testing.assert_allclose(a.max_abs_ptdf, b.max_abs_ptdf, atol=1e-5)


def fake_jao_dir(rows: list[dict]) -> Path:
    """A minimal jao_core_hourly_views folder: the columns constraints.load reads, one row per dict."""
    d = Path(tempfile.mkdtemp())
    ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    cnec = pd.DataFrame({"ts_utc": [ts + pd.Timedelta(hours=r.get("h", 0)) for r in rows],
                         "cnec_key": [f"k{i}" for i in range(len(rows))],
                         "ram_fmax": [r.get("ram_fmax", 0.5) for r in rows]})
    for h in C.HUBS:
        cnec[f"ptdf_{h}"] = 0.1 if h == "FR" else 0.0
    prov = pd.DataFrame({"cnec_key": cnec.cnec_key, "CneEic": [r["eic"] for r in rows], "CneName": [r["name"] for r in rows],
                         "Direction": "DIRECT", "ContName": "", "Tso": "RTE", "ElementType": [r.get("type", "Line") for r in rows],
                         "Ram": [str(r.get("ram", 100.0)) for r in rows], "Fmax": "200.0", "Frm": "10.0", "F0core": "5.0",
                         "FtotalLtn": "8.0", "Amr": [str(r.get("amr", 0.0)) for r in rows], "Iva": "0.0", "MinRamFactor": "70.0"})
    cnec.to_parquet(d / "jao_cnec_hourly.parquet", index=False)
    prov.iloc[::-1].to_parquet(d / "cnec_provenance.parquet", index=False)      # different row order on purpose
    return d


class TestSyntheticConstraints(unittest.TestCase):
    def test_is_synthetic(self):
        syn = ["Equality Constraint Core hubs 01", "External Constraint DE_AL_import", "External Constraint NL_NO2_NorNed_export",
               "External Constraint NL_export", "Default_FR_Import", "Default_CZ_Export"]
        real = ["St. Peter 2 - Pleinting 258", "Obersielach - Podlog 247", "400 kV Cirkovce - Zerjavinec",
                "Y-Ensdorf (-Diefflen - Uchtelfangen) ENSDF S", "Mintia - Timisoara", "Kerepes-Zuglo 2"]
        got = C.is_synthetic(pd.Series(syn + real))
        np.testing.assert_array_equal(got, [True] * len(syn) + [False] * len(real))

    def test_load_drops_synthetic_and_keeps_lines_without_eic_apart(self):
        rows = [
            {"eic": "NA", "name": "External Constraint DE_AL_import", "type": ""},
            {"eic": "NA", "name": "Equality Constraint Core hubs 01", "type": ""},
            {"eic": "NA", "name": "Default_FR_Import", "type": ""},
            {"eic": "NA", "name": "Kerepes-Zuglo 2"},               # real line, no EIC
            {"eic": "NA", "name": "Mintia - Timisoara"},            # another real line, no EIC
            {"eic": "10T-AAA", "name": "Line A", "ram_fmax": 0.7},
            {"eic": "10T-AAA", "name": "Line A", "ram_fmax": 0.4},  # same hour, a variant: keep the tightest
            {"eic": "10T-AAA", "name": "Line A", "h": 1},
        ]
        hourly, cat = C.load(fake_jao_dir(rows))
        self.assertEqual(C.load.synthetic_dropped, 3)
        self.assertEqual(C.load.duplicates_dropped, 1)
        self.assertFalse(C.is_synthetic(cat.cne_name).any())
        self.assertEqual(sorted(cat.cne_name.astype(str)), ["Kerepes-Zuglo 2", "Line A", "Mintia - Timisoara"])
        self.assertEqual(len(hourly), 4)
        a = hourly[(hourly.constraint_id == cat.constraint_id[cat.cne_name == "Line A"].astype(str).iat[0])
                   & (hourly.ts_utc == pd.Timestamp("2024-01-01", tz="UTC"))]
        self.assertAlmostEqual(float(a.ram_fmax.iat[0]), 0.4, places=5)
        self.assertEqual(cat.main_zone.tolist(), ["FR"] * 3)


class TestPolicyAdjustedRam(unittest.TestCase):
    def test_load_adds_grid_margin_and_uplift(self):
        rows = [{"eic": "10T-A", "name": "Line A", "ram": 140.0, "amr": 60.0},   # lifted to 70 % of Fmax 200
                {"eic": "10T-B", "name": "Line B", "ram": 150.0, "amr": 0.0}]
        hourly, cat = C.load(fake_jao_dir(rows))
        self.assertEqual(list(hourly.columns[2:2 + len(C.NODE_FEATURES)]), C.NODE_FEATURES)
        name = dict(zip(cat.constraint_id.astype(str), cat.cne_name.astype(str).str[-1]))
        a = hourly.set_index(hourly.constraint_id.map(name))
        self.assertAlmostEqual(float(a.loc["A", "ram_grid_fmax"]), (140 - 60) / 200, places=5)
        self.assertAlmostEqual(float(a.loc["A", "amr_fmax"]), 60 / 200, places=5)
        self.assertAlmostEqual(float(a.loc["B", "ram_grid_fmax"]), 150 / 200, places=5)
        self.assertAlmostEqual(float(a.loc["A", "min_ram_factor"]), 0.7, places=5)

    def test_zone_exposure_policy_features(self):
        h = hourly_frame(n_c=4, hours=1)
        for i, z in enumerate(C.CORE_ZONES):
            h[f"ptdf_{z}"] = 0.0
        h["ptdf_FR"] = np.float32(0.3)                      # every constraint loads FR exports
        h["ram_fmax"] = np.float32([0.7, 0.7, 0.9, 0.2])
        h["amr"] = np.float32([50, 0, 0, 0])
        h["ram_grid_fmax"] = np.float32([0.45, 0.7, 0.9, 0.2])
        e = C.zone_exposure(h).set_index("zone").loc["FR"]
        self.assertEqual(e.jao_n_exposed, 4)
        self.assertAlmostEqual(e.jao_amr_share_exposed, 0.25)
        self.assertAlmostEqual(e.jao_min_ram_grid_exposed, 0.2, places=6)
        self.assertGreater(e.jao_tightness_grid, e.jao_tightness)          # the uplift hid scarcity
        # an older table without the policy columns still works; those features are NaN
        old = C.zone_exposure(h.drop(columns=["amr", "ram_grid_fmax"]))
        self.assertTrue(old.jao_tightness_grid.isna().all())
        self.assertTrue(set(C.EXPOSURE_FEATURES) <= set(old.columns))



class TestTightnessIsAMean(unittest.TestCase):
    def test_hand_value_and_missing_ram(self):
        h = hourly_frame(n_c=3, hours=1)
        for z in C.CORE_ZONES:
            h[f"ptdf_{z}"] = np.float32(0.0)
        h["ptdf_FR"] = np.float32([0.11, 0.33, 0.22])        # zone-to-rest PTDFs 0.12, 0.36, 0.24 (x 12/11)
        h["ram_fmax"] = np.float32([0.5, 1.2, np.nan])        # scarcity 0.5, 0 (capped), missing
        e = C.zone_exposure(h).set_index("zone").loc["FR"]
        self.assertAlmostEqual(float(e.jao_tightness), (0.12 * 0.5 + 0.36 * 0.0) / (0.12 + 0.36), places=5)
        self.assertEqual(e.jao_n_exposed, 3)

    def test_independent_of_the_number_of_constraints(self):
        h = hourly_frame(n_c=30, hours=4)
        twice = pd.concat([h, h.assign(constraint_id=h.constraint_id + "_copy")], ignore_index=True)
        a, b = C.zone_exposure(h), C.zone_exposure(twice)
        np.testing.assert_allclose(a.jao_tightness, b.jao_tightness, rtol=1e-5)
        np.testing.assert_allclose(a.jao_tightness_grid, b.jao_tightness_grid, rtol=1e-5)
        np.testing.assert_array_equal(2 * a.jao_n_exposed, b.jao_n_exposed)
        for c in ("jao_tightness", "jao_tightness_grid"):
            self.assertTrue(((b[c] >= 0) & (b[c] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
