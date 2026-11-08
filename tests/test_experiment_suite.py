"""Temporal, graph-control, masked-loss and score invariants for the suite."""
import unittest

import numpy as np
import pandas as pd
import torch

from epgnn.experiment_suite.data import FEATURES, guard_price24, neighbour_features, shuffle_edges
from epgnn.experiment_suite.neural import Preprocessor, PriceGAT
from epgnn.experiment_suite.runner import grouped_scores
from epgnn.train import masked_loss


class SuiteTests(unittest.TestCase):
    def test_masked_nan_loss_and_spread_have_finite_correct_gradients(self):
        pred = torch.tensor([[2., 3.], [5., 8.]], requires_grad=True)
        target = torch.tensor([[1., float('nan')], [3., 4.]])
        mask = torch.tensor([[True, False], [True, True]])
        pairs = torch.tensor([[0], [1]])
        loss = masked_loss(pred, target, mask, pairs, spread_weight=.5)
        self.assertAlmostEqual(loss.item(), 7/3 + .5, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[0, 1], 0.)
        empty = masked_loss(pred, torch.full_like(pred, float('nan')), torch.zeros_like(mask))
        self.assertEqual(empty.item(), 0.)
        empty.backward()

    def test_prior_day_auction_and_autumn_target_day_are_distinct(self):
        frame = pd.DataFrame({'ts_utc': pd.to_datetime(['2024-01-15T22:00Z', '2024-10-27T22:00Z']),
            'delivery_date': ['2024-01-15', '2024-10-27'], 'price_lag24': [7., 9.]})
        guarded = guard_price24(frame)
        self.assertEqual(guarded.iloc[0], 7.)
        self.assertTrue(pd.isna(guarded.iloc[1]))

    def test_shuffle_preserves_in_and_out_degrees_without_duplicates(self):
        edges = np.array([(a,b) for a in range(12) for b in [(a+1)%12, (a+3)%12]], np.int64).T
        shuffled = shuffle_edges(edges)
        self.assertFalse(np.array_equal(edges, shuffled))
        for row in [0, 1]:
            np.testing.assert_array_equal(np.bincount(edges[row]), np.bincount(shuffled[row]))
        self.assertTrue((shuffled[0] != shuffled[1]).all())
        self.assertEqual(len(set(map(tuple, shuffled.T))), edges.shape[1])

    def test_neighbours_keep_missing_zero_and_inactive_identity_distinct(self):
        x = np.full((3, 3, len(FEATURES)), np.nan, np.float32)
        col = FEATURES.index('price_lag24_guarded')
        x[:, 0, col] = [0., 3., 100.]
        active = np.ones((3, 3), bool); active[1, 0] = False
        edges = np.array([[0], [1]])
        before = neighbour_features(x, active, edges)
        self.assertEqual(before[0, 1, 0], 0.)
        self.assertTrue(np.isnan(before[1, 1, 0]))
        self.assertTrue(np.isnan(before[0, 2, 0]))
        x[2, 0, col] = 9999.
        np.testing.assert_array_equal(before[:2], neighbour_features(x, active, edges)[:2])

    def test_inactive_graph_nodes_cannot_send_messages(self):
        torch.manual_seed(224)
        model = PriceGAT(4).eval()
        x = torch.randn(1, 3, 4)
        edges = torch.tensor([[0, 1, 2], [1, 0, 1]])
        active = torch.tensor([[True, True, False]])
        with torch.no_grad():
            before = model(x, edges, active)
            x[0, 2] = 10000.
            after = model(x, edges, active)
        torch.testing.assert_close(before[:, :2], after[:, :2])

    def test_neural_preprocessing_uses_frozen_training_statistics(self):
        prep = Preprocessor().fit(np.array([[0., np.nan], [2., np.nan]], np.float32))
        np.testing.assert_array_equal(prep.median, [1., 0.])
        saved = prep.state()
        transformed = prep.transform(np.array([[1000., np.nan]], np.float32))
        self.assertEqual(transformed[0, -1], 1.)
        self.assertEqual(saved, prep.state())
        np.testing.assert_array_equal(Preprocessor.restore(saved).transform(np.zeros((1,2))), prep.transform(np.zeros((1,2))))

    def test_country_and_overall_pool_squared_errors(self):
        rows = [{'experiment': 'm', 'split': 'val', 'coverage': 'all', 'zone': z,
            'hours': n, 'paired_hours': n, 'mae': a, 'rmse': r, 'paired_mae': a, 'persistence_mae': a*2}
            for z,n,a,r in [('IT_NORD',1,2.,2.),('IT_SUD',3,6.,8.)]]
        scores = grouped_scores(rows)
        for row in scores:
            self.assertEqual(row['hours'],4)
            self.assertEqual(row['mae'],5.)
            self.assertEqual(row['rmse'],7.)
            self.assertEqual(row['relative_mae'],.5)


if __name__ == '__main__':
    unittest.main()
