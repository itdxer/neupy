from itertools import product

import numpy as np

from neupy import algorithms, init, layers
from neupy.algorithms.gd.base import (
    BatchSizeProperty, iter_batches,
    average_batch_errors, count_samples,
    cannot_divide_into_batches,
)

from data import simple_classification
from base import BaseTestCase


class MinibatchGDTestCase(BaseTestCase):
    def setUp(self):
        super(MinibatchGDTestCase, self).setUp()
        self.network_classes = [
            algorithms.MinibatchGradientDescent,
            algorithms.Momentum,
        ]

    def test_minibatch_valid_values(self):
        valid_values = [None, 1, 10, 1000]

        for net_class, value in product(self.network_classes, valid_values):
            net_class((10, 20, 1), batch_size=value)

    def test_minibatch_invalid_values(self):
        invalid_values = [-10, 3.50, 'invalid values', [10]]

        for net_class, value in product(self.network_classes, invalid_values):
            msg = "Network: {}, Value: {}".format(net_class.__name__, value)
            with self.assertRaises((TypeError, ValueError), msg=msg):
                net_class((10, 20, 1), batch_size=value)

    def test_full_batch_training(self):
        fullbatch_identifiers = BatchSizeProperty.fullbatch_identifiers
        x_train, _, y_train, _ = simple_classification()

        xavier_normal = init.XavierNormal()
        weight1 = xavier_normal.sample((10, 20), return_array=True)
        weight2 = xavier_normal.sample((20, 1), return_array=True)

        for network_class in self.network_classes:
            errors = []

            for fullbatch_value in fullbatch_identifiers:
                net = network_class(
                    [
                        layers.Input(10),
                        layers.Sigmoid(20, weight=weight1),
                        layers.Sigmoid(1, weight=weight2),
                    ],
                    batch_size=fullbatch_value,
                )
                net.train(x_train, y_train, epochs=10)

                errors.append(net.errors.last())

            self.assertTrue(
                np.all(np.abs(errors - errors[0]) < 1e-3),
                msg=errors,
            )

    def test_iterbatches(self):
        n_samples = 50
        batch_size = 20
        expected_shapes = [(20, 2), (20, 2), (10, 2)]

        data = np.random.random((n_samples, 2))
        batch_slices = list(iter_batches(n_samples, batch_size))
        for batch, expected_shape in zip(batch_slices, expected_shapes):
            self.assertEqual(data[batch].shape, expected_shape)

    def test_batch_average(self):
        expected_error = 0.9  # or 225 / 250
        actual_error = average_batch_errors([1, 1, 0.5], 250, 100)
        self.assertAlmostEqual(expected_error, actual_error)

        expected_error = 0.8  # or 240 / 300
        actual_error = average_batch_errors([1, 1, 0.4], 300, 100)
        self.assertAlmostEqual(expected_error, actual_error)

    def test_cannot_divide_into_batches(self):
        x = np.random.random(10)

        self.assertTrue(cannot_divide_into_batches(x, batch_size=None))
        self.assertTrue(cannot_divide_into_batches(x, batch_size=100))
        self.assertTrue(cannot_divide_into_batches(x, batch_size=10))

        self.assertFalse(cannot_divide_into_batches(x, batch_size=1))
        self.assertFalse(cannot_divide_into_batches(x, batch_size=2))
        self.assertFalse(cannot_divide_into_batches(x, batch_size=3))
        self.assertFalse(cannot_divide_into_batches(x, batch_size=9))
        self.assertFalse(cannot_divide_into_batches((x, x), batch_size=9))

    def test_count_samples_function(self):
        x = np.random.random((10, 5))

        self.assertEqual(count_samples(x), 10)
        self.assertEqual(count_samples([x, x]), 10)
