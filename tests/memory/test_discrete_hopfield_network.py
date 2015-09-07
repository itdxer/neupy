import numpy as np

from neupy.algorithms import DiscreteHopfieldNetwork

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
            dhnet = DiscreteHopfieldNetwork()
            dhnet.train(data)

        with self.assertRaises(ValueError):
            # To many data samples comparison to number of feature
            dhnet = DiscreteHopfieldNetwork()
            dhnet.train(data)

    def test_discrete_hopfield_full(self):
        data = np.concatenate([zero, one, two], axis=0)
        dhnet = DiscreteHopfieldNetwork(mode='full')
        dhnet.train(data)

        self.assertTrue(np.all(zero == dhnet.predict(half_zero)))
        self.assertTrue(np.all(one == dhnet.predict(half_one)))
        self.assertTrue(np.all(two == dhnet.predict(half_two)))

    def test_discrete_hopfield_random(self):
        data = np.concatenate([zero, one, two], axis=0)
        dhnet = DiscreteHopfieldNetwork(mode='random', n_nodes=1000)
        dhnet.train(data)

        self.assertTrue(np.all(zero == dhnet.predict(half_zero)))
        self.assertTrue(np.all(one == dhnet.predict(half_one)))
        self.assertTrue(np.all(two == dhnet.predict(half_two)))
