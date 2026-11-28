"""Checks for epgnn.models.asof: the input swap is scoped to the base model's run.

    python -m unittest tests/test_asof_adapter.py -v
"""
import unittest
from unittest import mock

import pandas as pd

from epgnn.data import features as F
from epgnn.data import tables
from epgnn.models import asof


class TestAsofAdapter(unittest.TestCase):
    def test_feature_sets_have_no_lag24_weather(self):
        for name in ("own_asof", "own+jao_asof"):
            cols = F.day_d_columns(name)
            self.assertFalse([c for c in cols if c.endswith("_lag24") and not c.startswith("price")], name)
            self.assertIn("temp_pw_asof", cols)
            self.assertIn("price_lag24", cols)                 # D-1 price is known at the cutoff

    def test_swap_is_restored_even_on_error(self):
        fake = pd.DataFrame({"zone": ["Z"], "ts_utc": [0]})
        before = (tables.nodes, tables.edge_pairs)
        with mock.patch.object(asof, "nodes_asof", return_value=fake):
            with self.assertRaises(RuntimeError):
                with asof.safe_inputs():
                    self.assertIs(tables.nodes(), fake)
                    self.assertIs(tables.edge_pairs, asof.graphs.active_pairs)
                    raise RuntimeError("boom")
        self.assertEqual((tables.nodes, tables.edge_pairs), before)

    def test_refuses_leaky_feature_set(self):
        with self.assertRaises(AssertionError):
            asof.run({"base_model": "xgb", "feature_set": "own"}, [], [])


if __name__ == "__main__":
    unittest.main()
