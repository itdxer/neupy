import numpy as np

from neupy import algorithms

from memory.data import *
from base import BaseTestCase


zero_hint = np.matrix([[0, 1, 0, 0]])
one_hint = np.matrix([[1, 0, 0, 0]])


class BAMTestCase(BaseTestCase):
    def setUp(self):
        super(BAMTestCase, self).setUp()
        self.data = np.concatenate([zero, one], axis=0)
        self.hints = np.concatenate([zero_hint, one_hint], axis=0)

    def test_discrete_bam(self):
        bamnet = algorithms.DiscreteBAM()
        bamnet.train(self.data.copy(), self.hints.copy())

        np.testing.assert_array_almost_equal(
            bamnet.predict(half_zero),
            zero_hint
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_output(half_one),
            one_hint
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_input(zero_hint),
            zero
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_input(one_hint),
            one
        )

        input_matrix = np.vstack([one, zero])
        output_matrix = np.vstack([one_hint, zero_hint])

        np.testing.assert_array_almost_equal(
            bamnet.predict_input(output_matrix),
            input_matrix
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict(input_matrix),
            output_matrix
        )

    def test_energy_function(self):
        input_vector = np.array([[1, 0, 0, 1, 1, 0, 0]])
        output_vector = np.array([[1, 0]])
        dhnet = algorithms.DiscreteBAM()
        dhnet.train(input_vector, output_vector)

        self.assertEqual(-7, dhnet.energy(input_vector, output_vector))
        self.assertEqual(0, dhnet.energy(
            np.array([[0, 0, 0, 0, 0, 0, 0]]),
            np.array([[0, 0]])
        ))
        self.assertEqual(-7, dhnet.energy(
            np.array([[0, 1, 1, 0, 0, 1, 1]]),
            np.array([[0, 1]])
        ))

        np.testing.assert_array_almost_equal(
            np.array([-7, 0]),
            dhnet.energy(
                np.array([
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]),
                np.array([
                    [0, 1],
                    [0, 0],
                ])
            )
        )
