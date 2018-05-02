import numpy as np

from neupy.algorithms.regularization.max_norm import max_norm_clip
from neupy.utils import asfloat
from neupy import algorithms, layers

from base import BaseTestCase
from data import simple_classification


class MaxNormRegularizationTestCase(BaseTestCase):
    def test_max_norm_clip(self):
        random_vector = asfloat(np.random.random(100))

        clipped_random_vector = self.eval(
            max_norm_clip(
                random_vector,
                max_norm=100
            )
        )
        self.assertEqual(
            np.linalg.norm(random_vector),
            np.linalg.norm(clipped_random_vector),
        )

        clipped_random_vector = self.eval(
            max_norm_clip(
                random_vector,
                max_norm=0.001
            )
        )
        self.assertAlmostEqual(
            np.linalg.norm(clipped_random_vector),
            0.001,
        )

    def test_max_norm_regularizer(self):
        def on_epoch_end(network):
            layer = network.layers[1]

            weight = self.eval(layer.weight)
            weight_norm = np.round(np.linalg.norm(weight), 5)

            bias = self.eval(layer.bias)
            bias_norm = np.round(np.linalg.norm(bias), 5)

            error_message = "Epoch #{}".format(network.last_epoch)
            self.assertLessEqual(weight_norm, 2, msg=error_message)
            self.assertLessEqual(bias_norm, 2, msg=error_message)

        mnet = algorithms.Momentum(
            [
                layers.Input(10),
                layers.Relu(20),
                layers.Sigmoid(1),
            ],

            step=0.1,
            momentum=0.95,
            verbose=False,
            epoch_end_signal=on_epoch_end,

            max_norm=2,
            addons=[algorithms.MaxNormRegularization],
        )

        x_train, _, y_train, _ = simple_classification()
        mnet.train(x_train, y_train, epochs=100)
