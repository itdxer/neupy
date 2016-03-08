from itertools import product

from neupy import algorithms
from neupy.algorithms.gd.base import BatchSizeProperty

from data import simple_classification
from base import BaseTestCase


class MinibatchGDTestCase(BaseTestCase):
    network_classes = [
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

        for network_class in self.network_classes:
            errors = []
            for fullbatch_value in fullbatch_identifiers:
                self.setUp()

                net = network_class((10, 20, 1), batch_size=fullbatch_value)
                net.train(x_train, y_train, epochs=10)

                errors.append(net.errors.last())

            self.assertTrue(all(e == errors[0] for e in errors))
