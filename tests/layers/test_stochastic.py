from scipy import stats
import numpy as np

from neupy import layers

from base import BaseTestCase


class DropoutLayerTestCase(BaseTestCase):
    def test_dropout_layer(self):
        test_input = np.ones((50, 20))
        dropout_layer = layers.Dropout(proba=0.5)

        y = dropout_layer.output(test_input, training=True)
        layer_output = self.eval(y)

        self.assertGreater(layer_output.sum(), 900)
        self.assertLess(layer_output.sum(), 1100)

        self.assertTrue(np.all(
            np.bitwise_or(layer_output == 0, layer_output == 2)
        ))

    def test_dropout_disable_training_state(self):
        test_input = np.ones((50, 20))
        dropout_layer = layers.Dropout(proba=0.5)
        layer_output = dropout_layer.output(test_input)
        np.testing.assert_array_equal(layer_output, test_input)

    def test_dropout_repr(self):
        layer = layers.Dropout(0.5)
        self.assertEqual(
            "Dropout(proba=0.5, name='dropout-1')",
            str(layer))


class GaussianNoiseLayerTestCase(BaseTestCase):
    def test_gaussian_noise_layer(self):
        test_input = np.zeros((50, 20))
        gauss_noise = layers.GaussianNoise(std=0.5)

        layer_output = self.eval(gauss_noise.output(test_input, training=True))
        self.assertTrue(stats.mstats.normaltest(layer_output))

    def test_gaussian_noise_disable_training_state(self):
        test_input = np.ones((50, 20))
        gauss_noise = layers.GaussianNoise(std=1)
        layer_output = gauss_noise.output(test_input)
        np.testing.assert_array_equal(layer_output, test_input)

    def test_gaussian_noise_repr(self):
        layer = layers.GaussianNoise(0, 1)
        self.assertEqual(
            "GaussianNoise(mean=0, std=1, name='gaussian-noise-1')",
            str(layer))
