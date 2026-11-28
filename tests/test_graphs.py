"""Checks for the zone graphs (epgnn.pipeline.graphs): only interconnectors of the current zone layout.

    python -m unittest tests/test_graphs.py -v

Needs datasets/zone_edges_hourly; skipped without it.
"""
import unittest

from epgnn.paths import dataset_path

try:
    HAVE_EDGES = dataset_path("zone_edges_hourly", "zone_edges_hourly.parquet").exists()
except Exception:
    HAVE_EDGES = False

PHANTOM = {("DE_LU", "HU"), ("DE_LU", "IT_NORD"), ("DE_LU", "SI")}


@unittest.skipUnless(HAVE_EDGES, "zone_edges_hourly not available")
class TestActiveInterconnects(unittest.TestCase):
    def setUp(self):
        from epgnn.pipeline import graphs
        self.G = graphs
        self.zl = graphs.zones()
        e = graphs.interconnect_edges(self.zl)
        self.pairs = {(self.zl[a], self.zl[b]) for a, b in e.T}

    def test_phantom_links_dropped_both_directions(self):
        for a, b in PHANTOM:
            self.assertNotIn((a, b), self.pairs)
            self.assertNotIn((b, a), self.pairs)
        dropped = {tuple(r) for r in self.G.dropped_pairs().values.tolist()}
        self.assertEqual(dropped, PHANTOM | {(b, a) for a, b in PHANTOM})

    def test_real_links_kept(self):
        for p in [("DE_LU", "AT"), ("AT", "HU"), ("AT", "SI"), ("AT", "IT_NORD"), ("GB", "DK_1"), ("DE_LU", "FR")]:
            self.assertIn(p, self.pairs)
        self.assertEqual(len(self.pairs), 174)
        self.assertEqual(self.G.spread_pairs(self.zl).shape[1], 87)

    def test_graph_still_connected_and_symmetric(self):
        A = self.G.adjacency(self.zl)
        self.assertTrue((A == A.T).all())
        self.assertEqual(len(self.G.components(self.zl)), 1)


if __name__ == "__main__":
    unittest.main()
