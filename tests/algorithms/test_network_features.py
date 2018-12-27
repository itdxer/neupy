# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from sklearn import datasets
from neupy import algorithms, layers
from neupy.algorithms.base import preformat_value

from helpers import catch_stdout
from base import BaseTestCase


class NetworkMainTestCase(BaseTestCase):
    def test_training_epoch_accumulation(self):
        data, target = datasets.make_classification(
            30, n_features=10, n_classes=2)

        network = algorithms.GradientDescent([
            layers.Input(10),
            layers.Sigmoid(3),
            layers.Sigmoid(1),
        ])
        self.assertEqual(network.last_epoch, 0)

        network.train(data, target, epochs=10)
        self.assertEqual(network.last_epoch, 10)

        network.train(data, target, epochs=5)
        self.assertEqual(network.last_epoch, 15)

    def test_train_and_test_dataset_training(self):
        data, target = datasets.make_classification(
            30, n_features=10, n_classes=2)

        network = algorithms.GradientDescent([
                layers.Input(10),
                layers.Sigmoid(3),
                layers.Sigmoid(1),
            ],
            batch_size=None,
        )

        # Should work fine without exceptions
        network.train(data, target, epochs=2)
        network.train(data, target, data, target, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, data, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, y_test=target, epochs=2)

    def test_network_training_summary(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                verbose=False,
                batch_size=None,
            )

            x = np.zeros((5, 2))
            y = np.zeros((5, 1))

            network.verbose = True
            n_epochs = 10
            network.train(x, y, epochs=n_epochs)

            terminal_output = out.getvalue().strip()

        # `n_epochs - 1` because \n appears only between
        # inline summary lines.
        # Also network prints 5 additional lines at the beggining
        self.assertEqual(terminal_output.count('\n'), n_epochs - 1)

    def test_wrong_number_of_training_epochs(self):
        network = algorithms.GradientDescent(
            layers.Input(2) > layers.Sigmoid(1),
            verbose=False,
            batch_size=None,
        )

        with self.assertRaisesRegexp(ValueError, "a positive number"):
            network.train(np.zeros((4, 2)), np.zeros((4, 1)), epochs=0)

        with self.assertRaisesRegexp(ValueError, "a positive number"):
            network.train(np.zeros((4, 2)), np.zeros((4, 1)), epochs=-1)

    def test_preformat_value(self):
        def my_func():
            pass

        class MyClass(object):
            pass

        self.assertEqual('my_func', preformat_value(my_func))
        self.assertEqual('MyClass', preformat_value(MyClass))

        expected = ['my_func', 'MyClass', 1]
        actual = preformat_value((my_func, MyClass, 1))
        np.testing.assert_array_equal(expected, actual)

        expected = ['my_func', 'MyClass', 1]
        actual = preformat_value([my_func, MyClass, 1])
        np.testing.assert_array_equal(expected, actual)

        expected = sorted(['my_func', 'MyClass', 'x'])
        actual = sorted(preformat_value({my_func, MyClass, 'x'}))
        np.testing.assert_array_equal(expected, actual)

        self.assertEqual(1, preformat_value(1))

        expected = (3, 2)
        actual = preformat_value(np.ones((3, 2)))
        np.testing.assert_array_equal(expected, actual)

        expected = (1, 2)
        actual = preformat_value(np.matrix([[1, 1]]))
        np.testing.assert_array_equal(expected, actual)

    def test_init_logging(self):
        with catch_stdout() as out:
            algorithms.GradientDescent(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                verbose=False,
            )
            terminal_output = out.getvalue()
            self.assertEqual("", terminal_output.strip())

        with catch_stdout() as out:
            algorithms.GradientDescent(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                verbose=True,
            )
            terminal_output = out.getvalue()

            self.assertNotEqual("", terminal_output.strip())
            self.assertIn("verbose = True", terminal_output)
