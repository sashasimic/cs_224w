"""Fixtures for hourly graph isolation, timing and missing domains."""

import unittest
from contextlib import closing
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from epgnn.dataset_build.contracts import forecast_origin
from epgnn.relational_forecast.catalog import CORE, connect
from epgnn.relational_forecast.graph import ConstraintGAT, Snapshots


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.active = np.ones((3, len(CORE)), dtype=bool)
        self.active[2, 2] = False
        coefficient = np.zeros((3, len(CORE)), dtype=np.float32)
        coefficient[0, :3] = [-0.5, 0.25, np.nan]
        coefficient[1, :3] = [0.5, 0, -0.3]
        coefficient[2, :3] = [0, 0.2, -0.2]
        cache = {
            "offsets": np.array([0, 2, 2, 3]),
            "attributes": np.array(
                [[0, 0, np.nan], [1, 2, 0.5], [3, 4, 0.75]], np.float32
            ),
            "coefficients": coefficient,
        }
        self.graph = Snapshots(cache, self.active, CORE, np.array([0]), 901)
        self.device = torch.device("cpu")

    def test_edge_masks_and_permutation(self):
        real = self.graph.batch(
            np.array([2, 1, 0]), self.active[[2, 1, 0]], self.device, False
        )
        shuffled = self.graph.batch(
            np.array([2, 1, 0]), self.active[[2, 1, 0]], self.device, True
        )
        self.assertEqual(real[1].tolist(), [[1, 24, 25, 24, 26], [0, 1, 1, 2, 2]])
        self.assertEqual(real[2].flatten().tolist(), shuffled[2].flatten().tolist())
        self.assertEqual(real[1][1].tolist(), shuffled[1][1].tolist())
        self.assertTrue((real[2] < 0).any())
        self.assertFalse(self.graph.connected[1].any())
        self.assertFalse(self.graph.connected[2, 2])

    def test_no_messages_between_hours(self):
        model = ConstraintGAT(2, hidden=8).eval()
        x = torch.zeros((3, len(CORE), 2))
        graph = self.graph.batch(np.arange(3), self.active, self.device, False)
        prediction = model(x, graph)
        changed = x.clone()
        changed[2] = 400
        after = model(changed, graph)
        torch.testing.assert_close(prediction[:2], after[:2])
        empty = self.graph.batch(np.array([1]), self.active[[1]], self.device, False)
        torch.testing.assert_close(prediction[1:2], model(x[1:2], empty))
        prediction.square().mean().backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )


class PublishedCatalogTests(unittest.TestCase):
    def test_cutoff_with_future_revisions_and_dst(self):
        database = Path("staging/relational_forecast/current/forecast.duckdb")
        if not database.exists():
            self.skipTest("Build the catalog before running the integration fixture")
        with (
            closing(connect(database)) as published,
            closing(duckdb.connect()) as fixture,
        ):
            ddl = published.execute("""
                SELECT sql FROM duckdb_views()
                WHERE view_name IN ('core_forecast_domain', 'strict_core_forecast_domain')
                ORDER BY view_name
            """).fetchall()
            hours = pd.Series(
                pd.to_datetime(
                    [
                        "2024-03-31T01:00:00Z",
                        "2024-10-27T00:00:00Z",
                        "2024-10-27T01:00:00Z",
                    ],
                    utc=True,
                )
            )
            clock = pd.DataFrame(
                {
                    "ts_utc": hours,
                    "forecast_origin_utc": forecast_origin(hours),
                    "split": "val",
                }
            )
            fixture.register("delivery_hour", clock)
            fixture.execute("""
                CREATE TABLE jao_domain_hour (
                    ts_utc TIMESTAMPTZ, regime VARCHAR, interval_end_utc TIMESTAMPTZ,
                    domain_present BOOLEAN, published_utc TIMESTAMPTZ
                );
                INSERT INTO jao_domain_hour VALUES
                    ('2024-03-29T01:00:00Z', 'CORE', '2024-03-29T02:00:00Z', true, NULL),
                    ('2024-10-25T00:00:00Z', 'CORE', '2024-10-25T01:00:00Z', true, '2024-10-26T09:59:59Z'),
                    ('2024-10-25T01:00:00Z', 'CORE', '2024-10-25T02:00:00Z', true, '2024-10-26T10:00:01Z');
            """)
            self.assertEqual(len(ddl), 2)
            for (view,) in ddl:
                fixture.execute(view)
            query = """SELECT epoch_ns(delivery_utc), epoch_ns(source_hour)
                FROM core_forecast_domain ORDER BY delivery_utc"""
            before = fixture.execute(query).fetchall()
            self.assertEqual(
                [source is not None for _, source in before], [True, True, False]
            )
            self.assertEqual(
                fixture.execute(
                    "SELECT count(*) FROM strict_core_forecast_domain"
                ).fetchone(),
                (1,),
            )
            fixture.execute("""
                UPDATE jao_domain_hour SET domain_present = false
                WHERE published_utc > '2024-10-26T10:00:00Z'
            """)
            self.assertEqual(before, fixture.execute(query).fetchall())
            states = dict(
                published.execute("""
                SELECT zone, count(*) FROM cnec_ptdf WHERE zone IN ('DE_LU', 'DE_AT_LU') GROUP BY zone
            """).fetchall()
            )
            self.assertTrue(all(states[z] > 0 for z in ("DE_LU", "DE_AT_LU")))
            self.assertGreater(
                published.execute(
                    "SELECT count(*) FROM raw_jao_border WHERE maxbex = 0"
                ).fetchone()[0],
                0,
            )


if __name__ == "__main__":
    unittest.main()
