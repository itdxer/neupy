from functools import partial

import numpy as np
from sklearn import datasets

from neupy import algorithms, init
from neupy.exceptions import NotTrained

from base import BaseTestCase
from utils import compare_networks
from data import xor_input_train, xor_target_train


class LVQTestCase(BaseTestCase):
    def setUp(self):
        self.data = np.concatenate(
            [
                xor_input_train,
                xor_input_train + 0.1,
                xor_input_train + np.array([[0.1, -0.1]]),
                xor_input_train - 0.1,
                xor_input_train - np.array([[0.1, -0.1]]),
            ],
            axis=0
        )
        target = np.concatenate(
            [
                xor_target_train,
                xor_target_train,
                xor_target_train,
                xor_target_train,
                xor_target_train,
            ],
            axis=0
        )
        self.target = np.where(target == -1, 0, target)
        super(LVQTestCase, self).setUp()

    def test_lvq_initialization_exceptions(self):
        with self.assertRaises(ValueError):
            # n_sublcasses < n_classes
            algorithms.LVQ(n_inputs=2, n_subclasses=2, n_classes=3)

        with self.assertRaises(ValueError):
            # sum(prototypes_per_class) != n_subclasses
            algorithms.LVQ(n_inputs=2, n_subclasses=10, n_classes=3,
                           prototypes_per_class=[5, 3, 3])

        with self.assertRaises(ValueError):
            # len(prototypes_per_class) != n_classes
            algorithms.LVQ(n_inputs=2, n_subclasses=10, n_classes=3,
                           prototypes_per_class=[5, 5])

    def test_lvq_training_exceptions(self):
        lvqnet = algorithms.LVQ(n_inputs=2, n_subclasses=4, n_classes=2)

        with self.assertRaises(NotTrained):
            lvqnet.predict(np.array([1, 2]))

        with self.assertRaises(ValueError):
            input_train = np.array([
                [1, 2],
                [3, 4],
                [4, 5],
            ])
            target_train = np.array([0, 1, 0])
            # len(input_train) <= n_subclasses
            lvqnet.train(input_train, target_train)

        with self.assertRaises(ValueError):
            input_train = np.array([
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
            ])
            target_train = np.array([0, 0, 0, 0, 1])
            # there are should be 3 or more samples for
            # class 1, got only 1
            lvqnet.train(input_train, target_train)

        with self.assertRaises(ValueError):
            input_train = np.array([
                [1, 2],
                [3, 4],
                [4, 5],
                [5, 6],
                [67, 8],
            ])
            target_train = np.array([0, 1, 0, 1, 2])
            # 3 unique classes instead of 2 expected
            lvqnet.train(input_train, target_train)

    def test_lvq_weight_initialization_state(self):
        lvqnet = algorithms.LVQ(n_inputs=2, n_classes=2)
        self.assertFalse(lvqnet.initialized)

        lvqnet.train(np.random.random((10, 2)),
                     np.random.random(10).round(),
                     epochs=1)
        self.assertTrue(lvqnet.initialized)

        lvqnet = algorithms.LVQ(n_inputs=2, n_classes=3,
                                weight=np.random.random((2, 3)))
        self.assertTrue(lvqnet.initialized)

        lvqnet = algorithms.LVQ(n_inputs=2, n_classes=3,
                                weight=init.Normal())
        self.assertTrue(lvqnet.initialized)
        self.assertEqual(lvqnet.weight.shape, (2, 3))

    def test_lvq_with_odd_number_of_subclasses(self):
        lvqnet = algorithms.LVQ(
            n_inputs=2,
            n_subclasses=3,
            n_classes=2,
        )
        self.assertIn(lvqnet.prototypes_per_class, ([2, 1], [1, 2]))

    def test_simple_lvq(self):
        lvqnet = algorithms.LVQ(
            n_inputs=2,
            n_subclasses=4,
            n_classes=2,
            shuffle_data=True,
        )

        lvqnet.train(self.data, self.target, epochs=100)
        predicted_target = lvqnet.predict(self.data)

        self.assertEqual(lvqnet.errors[-1], 0)
        np.testing.assert_array_equal(
            predicted_target,
            self.target[:, 0])

        self.assertPickledNetwork(lvqnet, self.data)

    def test_lvq_step_reduction(self):
        lvqnet = algorithms.LVQ(
            n_inputs=2,
            n_subclasses=4,
            n_classes=2,

            minstep=0.1,
            step=1.1,
            n_updates_to_stepdrop=200,
        )

        n_expected_updates = 0
        for i in range(10):
            n_expected_updates += len(self.data)
            lvqnet.train(self.data, self.target, epochs=1)
            expected_step = 1.1 - (i + 1) * 0.1
            self.assertAlmostEqual(lvqnet.training_step, expected_step)

        self.assertEqual(n_expected_updates, lvqnet.n_updates)
        self.assertEqual(lvqnet.training_step, lvqnet.minstep)

    def test_lvq_with_disabled_step_reduction(self):
        lvqnet = algorithms.LVQ(
            n_inputs=2,
            n_subclasses=4,
            n_classes=2,
            n_updates_to_stepdrop=None,
        )

        n_expected_updates = 0
        for i in range(10):
            n_expected_updates += len(self.data)
            lvqnet.train(self.data, self.target, epochs=1)
            self.assertAlmostEqual(lvqnet.training_step, lvqnet.step)

        self.assertEqual(n_expected_updates, lvqnet.n_updates)

    def test_lvq_with_custom_set_of_prototypes(self):
        lvqnet = algorithms.LVQ(
            n_inputs=2,
            n_subclasses=4,
            n_classes=2,

            prototypes_per_class=[3, 1],
        )

        lvqnet.train(self.data, self.target, epochs=3)
        self.assertGreater(lvqnet.errors[-1], 0)

        lvqnet.train(self.data, self.target, epochs=30)
        self.assertEqual(lvqnet.errors[-1], 0)

    def test_compare_lvq_and_lvq2(self):
        dataset = datasets.load_iris()
        data, target = dataset.data, dataset.target

        # Prepare the same weights for the fair comparison
        lvq = algorithms.LVQ(n_inputs=4, n_subclasses=3, n_classes=3)
        lvq.train(data, target, epochs=1)
        prepared_lvq_weights = lvq.weight

        compare_networks(
            algorithms.LVQ,
            partial(algorithms.LVQ2, epsilon=0.1),

            data=[data, target],
            epochs=10,
            show_comparison_plot=False,

            n_inputs=4,
            n_subclasses=3,
            n_classes=3,
            weight=prepared_lvq_weights,
        )

    def test_compare_lvq_and_lvq21(self):
        dataset = datasets.load_iris()
        data, target = dataset.data, dataset.target

        # Prepare the same weights for the fair comparison
        lvq = algorithms.LVQ(n_inputs=4, n_subclasses=3, n_classes=3)
        lvq.train(data, target, epochs=1)
        prepared_lvq_weights = lvq.weight

        compare_networks(
            algorithms.LVQ,
            partial(algorithms.LVQ21, epsilon=0.1),

            data=[data, target],
            epochs=10,
            show_comparison_plot=False,

            n_inputs=4,
            n_subclasses=3,
            n_classes=3,
            weight=prepared_lvq_weights,
        )

    def test_compare_lvq_and_lvq3(self):
        dataset = datasets.load_iris()
        data, target = dataset.data, dataset.target

        # Prepare the same weights for the fair comparison
        lvq = algorithms.LVQ(n_inputs=4, n_subclasses=6, n_classes=3)
        lvq.train(data, target, epochs=1)
        prepared_lvq_weights = lvq.weight

        compare_networks(
            algorithms.LVQ,
            partial(algorithms.LVQ3, epsilon=0.4),

            data=[data, target],
            epochs=100,
            show_comparison_plot=False,

            n_inputs=4,
            n_subclasses=6,
            n_classes=3,
            prototypes_per_class=[4, 1, 1],
            step=0.001,
            weight=prepared_lvq_weights,
        )
