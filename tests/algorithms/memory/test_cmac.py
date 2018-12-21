import numpy as np
from sklearn import metrics

from neupy import algorithms
from base import BaseTestCase


class CMACTestCase(BaseTestCase):
    def test_cmac(self):
        X_train = np.reshape(np.linspace(0, 2 * np.pi, 100), (100, 1))
        X_train_before = X_train.copy()

        X_test = np.reshape(np.linspace(np.pi, 2 * np.pi, 50), (50, 1))

        y_train = np.sin(X_train)
        y_train_before = y_train.copy()
        y_test = np.sin(X_test)

        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
            verbose=False,
        )
        cmac.train(X_train, y_train, epochs=100)

        predicted_test = cmac.predict(X_test)
        predicted_test = predicted_test.reshape((len(predicted_test), 1))
        error = metrics.mean_absolute_error(y_test, predicted_test)

        self.assertAlmostEqual(error, 0.0024, places=4)

        # Test that algorithm didn't modify data samples
        np.testing.assert_array_equal(X_train, X_train_before)
        np.testing.assert_array_equal(X_train, X_train_before)
        np.testing.assert_array_equal(y_train, y_train_before)

        self.assertPickledNetwork(cmac, X_train)

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
        X_train = np.linspace(0, 2 * np.pi, 100)
        X_train = np.vstack([X_train, X_train])

        X_test = np.linspace(0, 2 * np.pi, 100)
        X_test = np.vstack([X_test, X_test])

        y_train = np.sin(X_train)
        y_test = np.sin(X_test)

        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
        )
        cmac.train(X_train, y_train,
                   X_test, y_test, epochs=100)
        predicted_test = cmac.predict(X_test)
        error = metrics.mean_absolute_error(y_test, predicted_test)

        self.assertAlmostEqual(error, 0, places=6)

    def test_cmac_training_exceptions(self):
        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
        )

        with self.assertRaises(ValueError):
            cmac.train(X_train=True, y_train=True,
                       X_test=None, y_test=True)
