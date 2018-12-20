import numpy as np

import warnings
from neupy import algorithms

from algorithms.memory.data import (
    zero, one, two, half_one,
    half_zero, half_two,
)
from base import BaseTestCase


class DiscreteHopfieldNetworkTestCase(BaseTestCase):
    def test_check_limit_option_for_iterative_updates(self):
        data = np.matrix([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
        ])

        dhnet = algorithms.DiscreteHopfieldNetwork(check_limit=True)
        dhnet.train(data[0])
        dhnet.train(data[1])

        with self.assertRaises(ValueError):
            dhnet.train(data[2])

        # The same must be OK without validation
        dhnet = algorithms.DiscreteHopfieldNetwork(check_limit=False)
        dhnet.train(data[0])
        dhnet.train(data[1])
        dhnet.train(data[2])

    def test_check_limit_option(self):
        data = np.matrix([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
        ])

        with self.assertRaises(ValueError):
            # To many data samples compare to the number of feature
            dhnet = algorithms.DiscreteHopfieldNetwork(check_limit=True)
            dhnet.train(data)

        # The same must be OK without validation
        dhnet = algorithms.DiscreteHopfieldNetwork(check_limit=False)
        dhnet.train(data)

    def test_input_data_validation(self):
        dhnet = algorithms.DiscreteHopfieldNetwork()
        dhnet.weight = np.array([[0, 1], [1, 0]])

        # Invalid discrete input values
        with self.assertRaises(ValueError):
            dhnet.train(np.array([-1, 1]))

        with self.assertRaises(ValueError):
            dhnet.energy(np.array([-1, 1]))

        with self.assertRaises(ValueError):
            dhnet.predict(np.array([-1, 1]))

    def test_discrete_hopfield_sync(self):
        data = np.concatenate([zero, one, two], axis=0)
        data_before = data.copy()
        dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
        dhnet.train(data)

        half_zero_before = half_zero.copy()
        np.testing.assert_array_almost_equal(zero, dhnet.predict(half_zero))
        np.testing.assert_array_almost_equal(two, dhnet.predict(half_two))

        # Test predicition for the 1d array
        np.testing.assert_array_almost_equal(
            one,
            dhnet.predict(half_one.ravel())
        )

        multiple_inputs = np.vstack([zero, one, two])
        np.testing.assert_array_almost_equal(
            multiple_inputs, dhnet.predict(multiple_inputs)
        )

        np.testing.assert_array_equal(data_before, data)
        np.testing.assert_array_equal(half_zero, half_zero_before)

        self.assertPickledNetwork(dhnet, data)

    def test_discrete_hopfield_async(self):
        data = np.concatenate([zero, one, two], axis=0)
        data_before = data.copy()
        dhnet = algorithms.DiscreteHopfieldNetwork(mode='async', n_times=1000)
        dhnet.train(data)

        half_zero_before = half_zero.copy()
        np.testing.assert_array_almost_equal(zero, dhnet.predict(half_zero))
        np.testing.assert_array_almost_equal(one, dhnet.predict(half_one))
        np.testing.assert_array_almost_equal(two, dhnet.predict(half_two))

        multiple_outputs = np.vstack([zero, one, two])
        multiple_inputs = np.vstack([half_zero, half_one, half_two])
        np.testing.assert_array_almost_equal(
            multiple_outputs,
            dhnet.predict(multiple_inputs),
        )

        np.testing.assert_array_equal(data_before, data)
        np.testing.assert_array_equal(half_zero, half_zero_before)

    def test_energy_function(self):
        input_vector = np.array([[1, 0, 0, 1, 1, 0, 0]])
        dhnet = algorithms.DiscreteHopfieldNetwork()
        dhnet.train(input_vector)

        self.assertEqual(-21, dhnet.energy(input_vector))
        self.assertEqual(3, dhnet.energy(np.array([[0, 0, 0, 0, 0, 0, 0]])))
        self.assertEqual(-21, dhnet.energy(np.array([0, 1, 1, 0, 0, 1, 1])))

        # Test energy calculatin for the 1d array
        self.assertEqual(3, dhnet.energy(np.array([0, 0, 0, 0, 0, 0, 0])))

        np.testing.assert_array_almost_equal(
            np.array([-21, 3]),
            dhnet.energy(
                np.array([
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ])
            )
        )

    def test_argument_in_predict_method(self):
        data = np.concatenate([zero, one, two], axis=0)
        dhnet = algorithms.DiscreteHopfieldNetwork(mode='async', n_times=1)
        dhnet.train(data)

        self.assertTrue(np.any(zero != dhnet.predict(half_zero)))
        np.testing.assert_array_almost_equal(
            zero,
            dhnet.predict(half_zero, n_times=100)
        )

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.DiscreteHopfieldNetwork(check_limit=False),
            np.array([1, 0, 0, 1]),
            is_feature1d=False
        )

    def test_predict_different_inputs(self):
        dhnet = algorithms.DiscreteHopfieldNetwork()
        data = np.array([[1, 0, 0, 1]])
        dhnet.train(data)
        self.assertInvalidVectorPred(dhnet, np.array([1, 0, 0, 1]), data,
                                     is_feature1d=False)

    def test_discrete_hn_warning(self):
        with warnings.catch_warnings(record=True) as warns:
            algorithms.DiscreteHopfieldNetwork(
                verbose=True,
                n_times=100,
                mode='sync'
            )
            self.assertEqual(len(warns), 1)
            self.assertIn('only in `async` mode', str(warns[0].message))

    def test_iterative_updates(self):
        data = np.concatenate([zero, one, two], axis=0)
        dhnet_full = algorithms.DiscreteHopfieldNetwork(mode='sync')
        dhnet_full.train(data)

        dhnet_iterative = algorithms.DiscreteHopfieldNetwork(mode='sync')
        for digit in [zero, one, two]:
            dhnet_iterative.train(digit)

        np.testing.assert_array_almost_equal(
            dhnet_iterative.weight, dhnet_full.weight)

    def test_iterative_updates_wrong_feature_shapes_exception(self):
        dhnet = algorithms.DiscreteHopfieldNetwork()
        dhnet.train(np.ones((1, 10)))

        with self.assertRaisesRegexp(ValueError, "invalid number of features"):
            dhnet.train(np.ones((1, 7)))
