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

    def test_input_data_validation(self):
        dhnet = algorithms.DiscreteBAM()
        dhnet.weight = np.array([[0, 1], [1, 0]])

        # Invalid discrete input values
        with self.assertRaises(ValueError):
            dhnet.train(np.array([-1, 1]), np.array([0, 1]))

        with self.assertRaises(ValueError):
            dhnet.train(np.array([0, 1]), np.array([-1, 1]))

        with self.assertRaises(ValueError):
            dhnet.energy(np.array([-1, 1]), np.array([0, 1]))

        with self.assertRaises(ValueError):
            dhnet.energy(np.array([0, 1]), np.array([-1, 1]))

        with self.assertRaises(ValueError):
            dhnet.predict(np.array([-1, 1]))

    def test_discrete_bam_sync(self):
        bamnet = algorithms.DiscreteBAM(mode='sync')
        bamnet.train(self.data, self.hints)

        np.testing.assert_array_almost_equal(
            bamnet.predict(half_zero)[1],
            zero_hint
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_output(half_one)[1],
            one_hint
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_input(zero_hint)[0],
            zero
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_input(one_hint)[0],
            one
        )

        input_matrix = np.vstack([one, zero])
        output_matrix = np.vstack([one_hint, zero_hint])

        np.testing.assert_array_almost_equal(
            bamnet.predict_input(output_matrix)[0],
            input_matrix
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict(input_matrix)[1],
            output_matrix
        )

    def test_discrete_bam_async(self):
        bamnet = algorithms.DiscreteBAM(mode='async', n_times=400)
        bamnet.train(self.data, self.hints)

        input_matrix = np.vstack([one, zero])
        output_matrix = np.vstack([one_hint, zero_hint])

        np.testing.assert_array_almost_equal(
            bamnet.predict_input(output_matrix)[0],
            input_matrix
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_output(input_matrix)[1],
            output_matrix
        )

    def test_argument_in_predict_method(self):
        dhnet = algorithms.DiscreteBAM(mode='async', n_times=1)
        dhnet.train(self.data, self.hints)

        self.assertTrue(np.any(one != dhnet.predict_output(half_one)[0]))
        np.testing.assert_array_almost_equal(
            one,
            dhnet.predict_output(half_one, n_times=100)[0]
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
