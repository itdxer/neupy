from sklearn import datasets
from neupy import algorithms

from base import BaseTestCase


class NetworkMainTestCase(BaseTestCase):
    def test_training_epoch(self):
        data, target = datasets.make_classification(30, n_features=10,
                                                    n_classes=2)
        network = algorithms.GradientDescent((10, 3, 1))

        self.assertEqual(network.last_epoch, 0)

        network.train(data, target, epochs=10)
        self.assertEqual(network.last_epoch, 10)

        network.train(data, target, epochs=5)
        self.assertEqual(network.last_epoch, 15)

    def test_train_and_test_dataset_training(self):
        data, target = datasets.make_classification(30, n_features=10,
                                                    n_classes=2)
        network = algorithms.GradientDescent((10, 3, 1))
        network.train(data, target, epochs=2)
        network.train(data, target, data, target, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, data, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, target_test=target, epochs=2)
