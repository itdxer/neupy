import numpy as np

from neupy import algorithms
from neupy.exceptions import NotTrained

from algorithms.memory.data import zero, one, half_one, half_zero
from base import BaseTestCase
from utils import vectors_for_testing


zero_hint = np.array([[0, 1, 0, 0]])
one_hint = np.array([[1, 0, 0, 0]])


class BAMTestCase(BaseTestCase):
    def setUp(self):
        super(BAMTestCase, self).setUp()
        self.data = np.concatenate([zero, one], axis=0)
        self.hints = np.concatenate([zero_hint, one_hint], axis=0)

    def test_bam_prediction_method(self):
        dbnet = algorithms.DiscreteBAM()
        dbnet.weight = np.array([[0, 1], [1, 0]])

        with self.assertRaises(ValueError):
            dbnet.prediction()

        with self.assertRaises(ValueError):
            dbnet.prediction(np.array([0, 1]), np.array([0, 1]))

    def test_bam_exceptions(self):
        with self.assertRaises(NotTrained):
            dbnet = algorithms.DiscreteBAM()
            dbnet.predict(np.array([-1, 1]))

        with self.assertRaises(ValueError):
            dbnet = algorithms.DiscreteBAM()
            dbnet.weight = np.array([[0, 1], [1, 0]])
            dbnet.train(np.array([0, 1, 1]), np.array([0, 1]))

    def test_bam_input_data_validation(self):
        dbnet = algorithms.DiscreteBAM()
        dbnet.weight = np.array([[0, 1], [1, 0]])

        with self.assertRaises(ValueError):
            # Invalid discrete input values
            dbnet.train(np.array([-1, 1]), np.array([0, 1]))

        with self.assertRaises(ValueError):
            dbnet.train(np.array([0, 1]), np.array([-1, 1]))

        with self.assertRaises(ValueError):
            dbnet.energy(np.array([-1, 1]), np.array([0, 1]))

        with self.assertRaises(ValueError):
            dbnet.energy(np.array([0, 1]), np.array([-1, 1]))

        with self.assertRaises(ValueError):
            dbnet.predict(np.array([-1, 1]))

    def test_discrete_bam_sync(self):
        bamnet = algorithms.DiscreteBAM(mode='sync')
        bamnet.train(self.data, self.hints)
        data_before = self.data.copy()
        hints_before = self.hints.copy()

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

        # Test 1d input array prediction
        np.testing.assert_array_almost_equal(
            bamnet.predict_input(one_hint.ravel())[0],
            one
        )

        # Test 1d output array input prediction
        np.testing.assert_array_almost_equal(
            bamnet.predict_output(half_one.ravel())[1],
            one_hint
        )

        # Test multiple input values prediction
        input_matrix = np.vstack([one, zero])
        output_matrix = np.vstack([one_hint, zero_hint])
        output_matrix_before = output_matrix.copy()
        input_matrix_before = input_matrix.copy()

        np.testing.assert_array_almost_equal(
            bamnet.predict_input(output_matrix)[0],
            input_matrix
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict(input_matrix)[1],
            output_matrix
        )

        np.testing.assert_array_equal(self.data, data_before)
        np.testing.assert_array_equal(self.hints, hints_before)
        np.testing.assert_array_equal(output_matrix, output_matrix_before)
        np.testing.assert_array_equal(input_matrix, input_matrix_before)

    def test_discrete_bam_async(self):
        bamnet = algorithms.DiscreteBAM(mode='async', n_times=400)
        data_before = self.data.copy()
        hints_before = self.hints.copy()
        bamnet.train(self.data, self.hints)

        input_matrix = np.vstack([one, zero])
        output_matrix = np.vstack([one_hint, zero_hint])
        output_matrix_before = output_matrix.copy()
        input_matrix_before = input_matrix.copy()

        np.testing.assert_array_almost_equal(
            bamnet.predict_input(output_matrix)[0],
            input_matrix
        )
        np.testing.assert_array_almost_equal(
            bamnet.predict_output(input_matrix)[1],
            output_matrix
        )

        np.testing.assert_array_equal(self.data, data_before)
        np.testing.assert_array_equal(self.hints, hints_before)
        np.testing.assert_array_equal(output_matrix, output_matrix_before)
        np.testing.assert_array_equal(input_matrix, input_matrix_before)

    def test_bam_argument_in_predict_method(self):
        dbnet = algorithms.DiscreteBAM(mode='async', n_times=1)
        dbnet.train(self.data, self.hints)

        self.assertTrue(np.any(one != dbnet.predict_output(half_one)[0]))
        np.testing.assert_array_almost_equal(
            one,
            dbnet.predict_output(half_one, n_times=100)[0]
        )

    def test_bam_energy_function(self):
        input_vector = np.array([[1, 0, 0, 1, 1, 0, 0]])
        output_vector = np.array([[1, 0]])
        dbnet = algorithms.DiscreteBAM()
        dbnet.train(input_vector, output_vector)

        self.assertEqual(-7, dbnet.energy(input_vector, output_vector))
        self.assertEqual(0, dbnet.energy(
            np.array([[0, 0, 0, 0, 0, 0, 0]]),
            np.array([[0, 0]])
        ))
        self.assertEqual(-7, dbnet.energy(
            np.array([[0, 1, 1, 0, 0, 1, 1]]),
            np.array([[0, 1]])
        ))

        # Test 1d array
        self.assertEqual(-7, dbnet.energy(
            np.array([0, 1, 1, 0, 0, 1, 1]),
            np.array([0, 1])
        ))

        # Test multiple input values energy calculation
        np.testing.assert_array_almost_equal(
            np.array([-7, 0]),
            dbnet.energy(
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

    def test_bam_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.DiscreteBAM(),
            np.array([1, 0, 0, 1]),
            np.array([1, 0]),
            is_feature1d=False
        )

    def test_bam_predict_different_inputs(self):
        bamnet = algorithms.DiscreteBAM()

        data = np.array([[1, 0, 0, 1]])
        target = np.array([[1, 0]])

        bamnet.train(data, target)
        test_vectors = vectors_for_testing(data.reshape(data.size),
                                           is_feature1d=False)

        for test_vector in test_vectors:
            np.testing.assert_array_almost_equal(
                bamnet.predict(test_vector)[1],
                target
            )
