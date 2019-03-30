from scipy import stats
import numpy as np

from neupy import layers
from neupy.exceptions import LayerConnectionError

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


class DropBlockTestCase(BaseTestCase):
    def test_drop_block(self):
        test_input = np.ones((2, 100, 100, 1))
        dropblock_layer = layers.DropBlock(keep_proba=0.9, block_size=(2, 10))
        output = dropblock_layer.output(test_input, training=True)
        actual_output = self.eval(output)

        fraction_masked = np.mean(actual_output != 0)

        self.assertTrue(0.88 <= fraction_masked <= 0.92)
        self.assertGreater(actual_output.max(), 1)
        self.assertAlmostEqual(
            actual_output.max(),
            1 / fraction_masked,
        )

    def test_drop_block_during_inference(self):
        test_input = np.ones((1, 20, 20, 1))
        dropblock_layer = layers.DropBlock(keep_proba=0.9, block_size=5)
        actual_output = dropblock_layer.output(test_input, training=False)

        self.assertIs(test_input, actual_output)
        self.assertEqual(dropblock_layer.block_size, (5, 5))

    def test_drop_block_shape(self):
        network = layers.join(
            layers.Input((10, 10, 3)),
            layers.DropBlock(0.9, block_size=5),
        )
        self.assertShapesEqual(network.output_shape, (None, 10, 10, 3))

    def test_drop_block_invalid_connection(self):
        err_message = "DropBlock layer expects input with 4 dimensions"
        with self.assertRaisesRegexp(LayerConnectionError, err_message):
            layers.join(
                layers.Input((10, 10)),
                layers.DropBlock(0.9, block_size=5),
            )
