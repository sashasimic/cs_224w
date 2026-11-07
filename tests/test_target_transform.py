import unittest

import numpy as np

from epgnn.boosting import errors, transform_target


class TargetTransformTests(unittest.TestCase):
    def test_round_trip_preserves_extremes_negative_prices_zero_and_nan(self):
        prices = np.array([-10000., -500., -50., -1., 0., 1., 50., 500., 10000., np.nan])
        transformed = transform_target(prices, "asinh", 50.)
        np.testing.assert_allclose(transform_target(transformed, "asinh", 50., inverse=True), prices, equal_nan=True)
        np.testing.assert_array_equal(np.sign(transformed[:-1]), np.sign(prices[:-1]))
        self.assertLess(abs(transformed[7]), abs(prices[7]))

    def test_identity_retains_values_and_shape(self):
        prices = np.array([[np.nan, -30.], [0., 40.]])
        np.testing.assert_array_equal(transform_target(prices), prices)
        np.testing.assert_array_equal(transform_target(prices, inverse=True), prices)

    def test_transform_requires_supported_kind_and_fixed_positive_scale(self):
        for kind, scale in [("unknown", 50.), ("asinh", 0.), ("asinh", -1.), ("asinh", np.inf), ("asinh", np.nan)]:
            with self.subTest(kind=kind, scale=scale), self.assertRaises(ValueError):
                transform_target([10.], kind, scale)

    def test_reported_price_error_must_use_inverse_transform(self):
        truth = np.array([-50., 100., np.nan])
        predicted_prices = np.array([-25., 150., 0.])
        latent = transform_target(predicted_prices, "asinh", 50.)
        restored = transform_target(latent, "asinh", 50., inverse=True)
        metric = errors(truth, restored, np.isfinite(truth))
        self.assertAlmostEqual(metric["mae"], 37.5)
        self.assertEqual(metric["n_hours"], 2)


if __name__ == "__main__":
    unittest.main()
