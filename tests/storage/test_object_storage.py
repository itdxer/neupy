import tempfile

import dill
import numpy as np
from sklearn import datasets, preprocessing
from six.moves import cPickle as pickle

from neupy import algorithms, layers, init

from base import BaseTestCase
from helpers import catch_stdout


class BasicStorageTestCase(BaseTestCase):
    def test_simple_dill_storage(self):
        bpnet = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Sigmoid(3),
                layers.Sigmoid(1),
            ],
            step=0.25,
            batch_size=None,
        )
        data, target = datasets.make_regression(n_features=2, n_targets=1)

        data = preprocessing.MinMaxScaler().fit_transform(data)
        target_scaler = preprocessing.MinMaxScaler()
        target = target_scaler.fit_transform(target.reshape(-1, 1))

        with tempfile.NamedTemporaryFile() as temp:
            original_layers = bpnet.network.layers
            test_layer_weights = self.eval(original_layers[1].weight)

            dill.dump(bpnet, temp)
            temp.file.seek(0)

            restored_bpnet = dill.load(temp)
            temp.file.seek(0)

            restored_layers = restored_bpnet.network.layers

            self.assertEqual(0.25, self.eval(restored_bpnet.step))
            self.assertEqual(3, len(restored_layers))
            np.testing.assert_array_equal(
                test_layer_weights,
                self.eval(restored_layers[1].weight)
            )

            # Check if it's possible to recover training state
            bpnet.train(data, target, epochs=5)
            real_bpnet_error = bpnet.score(data, target)
            updated_input_weight = self.eval(original_layers[1].weight)

            dill.dump(bpnet, temp)
            temp.file.seek(0)

            restored_bpnet_2 = dill.load(temp)
            temp.file.seek(0)

            restored_layers_2 = restored_bpnet_2.network.layers
            np.testing.assert_array_equal(
                updated_input_weight,
                self.eval(restored_layers_2[1].weight)
            )

            restored_bpnet_error = restored_bpnet_2.score(data, target)
            self.assertEqual(real_bpnet_error, restored_bpnet_error)

    def test_non_initialized_graph_storage(self):
        network = layers.Relu(10) >> layers.Relu(2)  # no input layer

        with tempfile.NamedTemporaryFile() as temp:
            pickle.dump(network, temp)
            temp.file.seek(0)

            network_restored = pickle.load(temp)

            self.assertFalse(network_restored.layers[0].frozen)
            self.assertIsInstance(
                network_restored.layers[0].weight,
                init.Initializer,
            )

            # Loaded parmeters are not variables and we
            # expect layer not to be frozen
            self.assertFalse(network_restored.layers[1].frozen)
            self.assertIsInstance(
                network_restored.layers[1].weight,
                init.Initializer,
            )

    def test_basic_storage(self):
        X = np.random.random((100, 2))
        y = np.random.random(100) > 0.5

        # We keep verbose=True in order to see if value will
        # be True when we restore it from the pickle object.
        pnn = algorithms.PNN(std=0.123, verbose=True)
        pnn.train(X, y)

        stored_pnn = pickle.dumps(pnn)
        loaded_pnn = pickle.loads(stored_pnn)

        testcases = [
            ('pnn', pnn),
            ('loaded_pnn', loaded_pnn),
        ]

        for name, network in testcases:
            print("Test case name: {}".format(name))

            self.assertAlmostEqual(network.std, 0.123)
            self.assertAlmostEqual(network.verbose, True)

            with catch_stdout() as out:
                network.logs.stdout = out
                network.logs.write("Test message")
                terminal_output = out.getvalue()
                self.assertIn("Test message", terminal_output)

        pnn_prediction = pnn.predict(X)
        loaded_pnn_prediction = loaded_pnn.predict(X)

        np.testing.assert_array_almost_equal(
            loaded_pnn_prediction, pnn_prediction)
