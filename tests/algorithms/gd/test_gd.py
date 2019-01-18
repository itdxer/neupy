import pickle

import numpy as np
import tensorflow as tf

from neupy import algorithms, layers
from neupy.algorithms.gd import objectives
from neupy.exceptions import InvalidConnection

from base import BaseTestCase
from helpers import simple_classification


class GradientDescentTestCase(BaseTestCase):
    def test_large_network_representation(self):
        optimizer = algorithms.GradientDescent([
            layers.Input(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(2),
        ])
        self.assertIn(
            "(?, 1) -> [... 6 layers ...] -> (?, 2)",
            str(optimizer))

    def test_raise_exception_for_multioutputs(self):
        network = layers.join(
            layers.Input(5),
            layers.parallel(
                layers.Relu(1),
                layers.Relu(2),
            )
        )
        error_message = "should have one output layer"
        with self.assertRaisesRegexp(InvalidConnection, error_message):
            algorithms.GradientDescent(network)

    def test_network_initializations(self):
        possible_networks = (
            # as a list
            [layers.Input(2), layers.Sigmoid(3), layers.Tanh(1)],

            # as forward sequence with inline operators
            layers.Input(2) > layers.Relu(10) > layers.Tanh(1),
            layers.Input(2) >> layers.Relu(10) >> layers.Tanh(1),
        )

        for i, network in enumerate(possible_networks, start=1):
            optimizer = algorithms.GradientDescent(network)
            message = "[Test #{}] Network: {}".format(i, network)
            self.assertEqual(len(optimizer.network.layers), 3, msg=message)

    def test_gd_get_params_method(self):
        optimizer = algorithms.GradientDescent([
            layers.Input(2),
            layers.Sigmoid(3),
            layers.Sigmoid(1),
        ])

        self.assertIn(
            'network',
            optimizer.get_params(with_network=True),
        )
        self.assertNotIn(
            'network',
            optimizer.get_params(with_network=False),
        )

    def test_gd_storage(self):
        optimizer = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Sigmoid(3),
                layers.Sigmoid(1),
            ],
            step=0.2,
            shuffle_data=True,
        )
        recovered_optimizer = pickle.loads(pickle.dumps(optimizer))

        self.assertAlmostEqual(self.eval(recovered_optimizer.step), 0.2)
        self.assertEqual(recovered_optimizer.shuffle_data, True)

    def test_optimizer_with_bad_input_shape_passed(self):
        optimizer = algorithms.GradientDescent(
            [
                layers.Input((10, 10, 3)),
                layers.Convolution((3, 3, 7)),
                layers.Reshape(),
                layers.Sigmoid(1),
            ],
            batch_size=None,
            verbose=False,
            loss='mse',
        )

        image = np.random.random((10, 10, 3))
        optimizer.train(image, [1], epochs=1)

        retrieved_score = optimizer.score(image, [1])
        self.assertLessEqual(0, retrieved_score)
        self.assertGreaterEqual(1, retrieved_score)

        prediction = optimizer.predict(image)
        self.assertEqual(prediction.ndim, 2)

    def test_optimizer_with_bad_input_shape_passed(self):
        optimizer = algorithms.GradientDescent(
            [
                layers.Input((10, 10, 3)),
                layers.Convolution((3, 3, 3), padding='same'),
            ],
            batch_size=None,
            verbose=False,
            loss='mse',
        )

        image = np.random.random((10, 10, 3))
        optimizer.train(image, image, epochs=1)

        retrieved_score = optimizer.score(image, image)
        self.assertLessEqual(0, retrieved_score)
        self.assertGreaterEqual(1, retrieved_score)

    def test_invalid_number_of_inputs(self):
        optimizer = algorithms.GradientDescent(
            [
                layers.parallel(
                    layers.Input((10, 10, 3)),
                    layers.Input((10, 10, 3)),
                ),
                layers.Concatenate(),
                layers.Convolution((3, 3, 3), padding='same'),
            ],
            batch_size=None,
            verbose=False,
            loss='mse',
        )

        image = np.random.random((10, 10, 3))
        optimizer.train([image, image], image, epochs=1)

        message = (
            "Number of inputs doesn't match number "
            "of input layers in the network."
        )
        with self.assertRaisesRegexp(ValueError, message):
            optimizer.train(image, image, epochs=1)

    def test_gd_custom_target(self):
        def custom_loss(actual, predicted):
            actual_shape = tf.shape(actual)
            n_samples = actual_shape[0]
            actual = tf.reshape(actual, (n_samples, 1))
            return objectives.rmse(actual, predicted)

        optimizer = algorithms.GradientDescent(
            layers.Input(10) >> layers.Sigmoid(1),

            step=0.2,
            shuffle_data=True,
            batch_size=None,

            loss=custom_loss,
            target=tf.placeholder(tf.float32, shape=(None, 1, 1)),
        )
        x_train, _, y_train, _ = simple_classification()

        error_message = "Cannot feed value of shape \(60, 1\) for Tensor"
        with self.assertRaisesRegexp(ValueError, error_message):
            optimizer.train(x_train, y_train, epochs=1)

        optimizer.train(x_train, y_train.reshape(-1, 1, 1), epochs=1)
