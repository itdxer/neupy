import numpy as np

from neupy import algorithms

from memory.data import *
from base import BaseTestCase


class DiscreteHopfieldNetworkTestCase(BaseTestCase):
    def test_errors(self):
        data = np.matrix([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 1, 1],
        ])

        with self.assertRaises(ValueError):
            # Wrong discrete values for data
            dhnet = algorithms.DiscreteHopfieldNetwork()
            dhnet.train(data)

        with self.assertRaises(ValueError):
            # To many data samples comparison to number of feature
            dhnet = algorithms.DiscreteHopfieldNetwork()
            dhnet.train(data)

    def test_discrete_hopfield_full(self):
        data = np.concatenate([zero, one, two], axis=0)
        dhnet = algorithms.DiscreteHopfieldNetwork(mode='full')
        dhnet.train(data)

        np.testing.assert_array_almost_equal(zero, dhnet.predict(half_zero))
        np.testing.assert_array_almost_equal(one, dhnet.predict(half_one))
        np.testing.assert_array_almost_equal(two, dhnet.predict(half_two))

        multiple_inputs = np.vstack([zero, one, two])
        np.testing.assert_array_almost_equal(
            multiple_inputs, dhnet.predict(multiple_inputs)
        )

    def test_discrete_hopfield_random(self):
        data = np.concatenate([zero, one, two], axis=0)
        dhnet = algorithms.DiscreteHopfieldNetwork(mode='random', n_nodes=1000)
        dhnet.train(data)

        np.testing.assert_array_almost_equal(zero, dhnet.predict(half_zero))
        np.testing.assert_array_almost_equal(one, dhnet.predict(half_one))
        np.testing.assert_array_almost_equal(two, dhnet.predict(half_two))

        multiple_inputs = np.vstack([zero, one, two])
        np.testing.assert_array_almost_equal(
            multiple_inputs, dhnet.predict(multiple_inputs)
        )

    def test_energy_function(self):
        input_vector = np.array([[1, 0, 0, 1, 1, 0, 0]])
        dhnet = algorithms.DiscreteHopfieldNetwork()
        dhnet.train(input_vector)

        self.assertEqual(-21, dhnet.energy(input_vector))
        self.assertEqual(3, dhnet.energy(np.array([[0, 0, 0, 0, 0, 0, 0]])))
        self.assertEqual(-21, dhnet.energy(np.array([0, 1, 1, 0, 0, 1, 1])))

        np.testing.assert_array_almost_equal(
            np.array([-21, 3]),
            dhnet.energy(
                np.array([
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ])
            )
        )
