import numpy as np

from neupy import algorithms

from memory.data import *
from base import BaseTestCase


class DiscreteHopfieldNetworkTestCase(BaseTestCase):
    def test_check_limit_option(self):
        data = np.matrix([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
        ])

        with self.assertRaises(ValueError):
            # To many data samples comparison to number of feature
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
            rows1d=True
        )

    def test_predict_different_inputs(self):
        dhnet = algorithms.DiscreteHopfieldNetwork()
        data = np.array([[1, 0, 0, 1]])
        dhnet.train(data)
        self.assertInvalidVectorPred(dhnet, np.array([1, 0, 0, 1]), data,
                                     rows1d=True)
