from itertools import product

import numpy as np

from neupy import algorithms, layers
from neupy.utils import iters
from neupy.utils.iters import average_batch_errors, count_samples

from base import BaseTestCase


class MinibatchGDTestCase(BaseTestCase):
    def setUp(self):
        super(MinibatchGDTestCase, self).setUp()
        self.network_classes = [
            algorithms.GradientDescent,
            algorithms.Momentum,
        ]

    def test_minibatch_valid_values(self):
        valid_values = [None, 1, 10, 1000]

        for net_class, value in product(self.network_classes, valid_values):
            net_class(
                [
                    layers.Input(10),
                    layers.Sigmoid(20),
                    layers.Sigmoid(1)
                ],
                batch_size=value,
            )

    def test_minibatch_invalid_values(self):
        invalid_values = [-10, 3.50, 'invalid values', [10]]

        for net_class, value in product(self.network_classes, invalid_values):
            msg = "Network: {}, Value: {}".format(net_class.__name__, value)
            with self.assertRaises((TypeError, ValueError), msg=msg):
                net_class(
                    [
                        layers.Input(10),
                        layers.Sigmoid(20),
                        layers.Sigmoid(1)
                    ],
                    batch_size=value,
                )

    def test_iterbatches(self):
        n_samples = 50
        batch_size = 20
        expected_shapes = [(20, 2), (20, 2), (10, 2)]

        data = np.random.random((n_samples, 2))
        batch_slices = list(iters.minibatches(data, batch_size))

        for batch, expected_shape in zip(batch_slices, expected_shapes):
            self.assertEqual(batch.shape, expected_shape)

    def test_batch_average(self):
        expected_error = 0.9  # or 225 / 250
        actual_error = average_batch_errors([1, 1, 0.5], 250, 100)
        self.assertAlmostEqual(expected_error, actual_error)

        expected_error = 0.8  # or 240 / 300
        actual_error = average_batch_errors([1, 1, 0.4], 300, 100)
        self.assertAlmostEqual(expected_error, actual_error)

    def test_count_samples_function(self):
        x = np.random.random((10, 5))

        self.assertEqual(count_samples(x), 10)
        self.assertEqual(count_samples([x, x]), 10)
