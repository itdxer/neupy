import numpy as np
from sklearn import metrics

from neupy import algorithms
from base import BaseTestCase


class CMACTestCase(BaseTestCase):
    def test_cmac(self):
        input_train = np.reshape(np.linspace(0, 2 * np.pi, 100), (100, 1))
        input_train_before = input_train.copy()

        input_test = np.reshape(np.linspace(np.pi, 2 * np.pi, 50), (50, 1))

        target_train = np.sin(input_train)
        target_train_before = target_train.copy()
        target_test = np.sin(input_test)

        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
            verbose=False,
        )
        cmac.train(input_train, target_train, epochs=100)

        predicted_test = cmac.predict(input_test)
        predicted_test = predicted_test.reshape((len(predicted_test), 1))
        error = metrics.mean_absolute_error(target_test, predicted_test)

        self.assertAlmostEqual(error, 0.0024, places=4)

        # Test that algorithm didn't modify data samples
        np.testing.assert_array_equal(input_train, input_train_before)
        np.testing.assert_array_equal(input_train, input_train_before)
        np.testing.assert_array_equal(target_train, target_train_before)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            network=algorithms.CMAC(),
            input_vector=np.array([1, 2, 3]),
            target=np.array([1, 2, 3])
        )

    def test_predict_different_inputs(self):
        cmac = algorithms.CMAC()

        data = np.array([[1, 2, 3]]).T
        target = np.array([[1, 2, 3]]).T

        cmac.train(data, target, epochs=100)
        self.assertInvalidVectorPred(
            network=cmac,
            input_vector=np.array([1, 2, 3]),
            target=target,
            decimal=2
        )

    def test_cmac_multi_output(self):
        input_train = np.linspace(0, 2 * np.pi, 100)
        input_train = np.vstack([input_train, input_train])

        input_test = np.linspace(0, 2 * np.pi, 100)
        input_test = np.vstack([input_test, input_test])

        target_train = np.sin(input_train)
        target_test = np.sin(input_test)

        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
        )
        cmac.train(input_train, target_train,
                   input_test, target_test, epochs=100)
        predicted_test = cmac.predict(input_test)
        error = metrics.mean_absolute_error(target_test, predicted_test)

        self.assertAlmostEqual(error, 0, places=6)

    def test_cmac_training_exceptions(self):
        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
        )

        with self.assertRaises(ValueError):
            cmac.train(input_train=True, target_train=True,
                       input_test=None, target_test=True)
