import numpy as np

from neuralpy import algorithms
from neuralpy.functions import errors
from base import BaseTestCase


class CMACTestCase(BaseTestCase):
    def test_cmac(self):
        input_train = np.reshape(np.linspace(0, 2 * np.pi, 100), (100, 1))
        input_test = np.reshape(np.linspace(np.pi, 2 * np.pi, 50), (50, 1))

        target_train = np.sin(input_train)
        target_test = np.sin(input_test)

        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
        )
        cmac.train(input_train, target_train, epochs=100)
        predicted_test = cmac.predict(input_test)
        error = errors.mae(target_test, predicted_test)

        self.assertEqual(round(error, 4), 0.0024)

    def test_cmac_multi_putput(self):
        input_train = np.linspace(0, 2 * np.pi, 100)
        input_train = np.reshape(
            np.concatenate([input_train, input_train], axis=0), (100, 2)
        )
        input_test = np.linspace(0, 2 * np.pi, 100)
        input_test = np.reshape(
            np.concatenate([input_test, input_test], axis=0), (100, 2)
        )

        target_train = np.sin(input_train)
        target_test = np.sin(input_test)

        cmac = algorithms.CMAC(
            quantization=100,
            associative_unit_size=32,
            step=0.2,
        )
        cmac.train(input_train, target_train, epochs=100)
        predicted_test = cmac.predict(input_test)
        error = errors.mae(target_test, predicted_test)

        self.assertEqual(round(error, 6), 0)
