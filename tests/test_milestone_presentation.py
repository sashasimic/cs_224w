"""Check benchmark normalization and explicit reporting conventions."""

import json
import unittest
from pathlib import Path

from epgnn.reporting.milestone import comparison_rows, render_table

ROOT = Path(__file__).resolve().parents[1]


class MilestonePresentationTests(unittest.TestCase):
    def test_ratios_use_identical_evaluation_benchmark(self):
        summary = json.loads((ROOT / "results/controlled_study_full.json").read_text())
        rows = comparison_rows(summary)
        self.assertEqual(rows[0]["test_rmae"], 1)
        self.assertIsNone(rows[0]["test_sd"])
        graph = next(row for row in rows if row["variant"] == "gat_rich_real")
        self.assertAlmostEqual(graph["test_rmae"], graph["test"] / rows[0]["test"])
        self.assertAlmostEqual(graph["test_rmae"], 0.72, places=2)
        table = render_table(summary)
        self.assertIn(r"\textbf{20.74", table)
        self.assertIn(r"\textbf{20.97", table)
        self.assertIn(r"\textbf{0.720}", table)
        self.assertLess(table.index("Shuffled GAT"), table.index("GRU + MLP"))
        self.assertIn("not statistical significance", table)

    def test_changed_benchmark_is_rejected(self):
        summary = json.loads((ROOT / "results/controlled_study_full.json").read_text())
        for row in summary["scores"]:
            if row["variant"] == "mlp_rich_none" and row["scope"] == "overall":
                row["persistence_mae_mean"] = 99
        with self.assertRaises(ValueError):
            comparison_rows(summary)
