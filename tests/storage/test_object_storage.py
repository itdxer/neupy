import tempfile

import dill
import theano
import numpy as np
from sklearn import datasets, preprocessing
from six.moves import cPickle as pickle

from neupy import algorithms, layers, init

from base import BaseTestCase
from data import simple_classification
from utils import catch_stdout


class BasicStorageTestCase(BaseTestCase):
    def test_simple_dill_storage(self):
        bpnet = algorithms.GradientDescent((2, 3, 1), step=0.25)
        data, target = datasets.make_regression(n_features=2, n_targets=1)

        data = preprocessing.MinMaxScaler().fit_transform(data)
        target_scaler = preprocessing.MinMaxScaler()
        target = target_scaler.fit_transform(target.reshape(-1, 1))

        with tempfile.NamedTemporaryFile() as temp:
            test_layer_weights = self.eval(bpnet.layers[1].weight)
            dill.dump(bpnet, temp)
            temp.file.seek(0)

            restored_bpnet = dill.load(temp)
            temp.file.seek(0)
            layers_sizes = [layer.size for layer in restored_bpnet.layers]

            self.assertEqual(0.25, restored_bpnet.step)
            self.assertEqual([2, 3, 1], layers_sizes)
            np.testing.assert_array_equal(
                test_layer_weights,
                self.eval(restored_bpnet.layers[1].weight)
            )

            bpnet.train(data, target, epochs=5)
            real_bpnet_error = bpnet.prediction_error(data, target)
            updated_input_weight = self.eval(bpnet.layers[1].weight)

            dill.dump(bpnet, temp)
            temp.file.seek(0)

            restored_bpnet2 = dill.load(temp)
            temp.file.seek(0)
            restored_bpnet_error = restored_bpnet2.prediction_error(
                data, target
            )

            np.testing.assert_array_equal(
                updated_input_weight,
                self.eval(restored_bpnet2.layers[1].weight)
            )

            # Error must be big, because we didn't normalize data
            self.assertEqual(real_bpnet_error, restored_bpnet_error)

    def test_dynamic_classes(self):
        test_classes = {
            algorithms.GradientDescent: {},
            algorithms.MinibatchGradientDescent: {'batch_size': 10},
            algorithms.Momentum: {'momentum': 0.5},
        }

        for algorithm_class, algorithm_params in test_classes.items():
            optimization_classes = [algorithms.WeightDecay,
                                    algorithms.SearchThenConverge]

            bpnet = algorithm_class(
                (3, 5, 1),
                addons=optimization_classes,
                verbose=False,
                **algorithm_params
            )
            data, target = datasets.make_regression(n_features=3, n_targets=1)

            data = preprocessing.MinMaxScaler().fit_transform(data)
            target_scaler = preprocessing.MinMaxScaler()
            target = target_scaler.fit_transform(target.reshape(-1, 1))

            with tempfile.NamedTemporaryFile() as temp:
                valid_class_name = bpnet.__class__.__name__
                dill.dump(bpnet, temp)
                temp.file.seek(0)

                restored_bpnet = dill.load(temp)
                restored_class_name = restored_bpnet.__class__.__name__
                temp.file.seek(0)

                self.assertEqual(valid_class_name, restored_class_name)
                self.assertEqual(optimization_classes,
                                 restored_bpnet.addons)

                bpnet.train(data, target, epochs=10)
                real_bpnet_error = bpnet.prediction_error(data, target)
                updated_input_weight = self.eval(bpnet.layers[1].weight)

                dill.dump(bpnet, temp)
                temp.file.seek(0)

                restored_bpnet2 = dill.load(temp)
                temp.file.seek(0)
                restored_bpnet_error = restored_bpnet2.prediction_error(
                    data, target
                )

                np.testing.assert_array_equal(
                    updated_input_weight,
                    self.eval(restored_bpnet2.layers[1].weight)
                )
                # Error must be big, because we didn't normalize data
                self.assertEqual(real_bpnet_error, restored_bpnet_error)

    def test_non_initialized_graph_storage(self):
        network = layers.Relu(10) > layers.Relu(2)  # no input layer

        with tempfile.NamedTemporaryFile() as temp:
            pickle.dump(network, temp)
            temp.file.seek(0)

            network_restored = pickle.load(temp)

            self.assertFalse(network_restored.layers[0].initialized)
            self.assertIsInstance(
                network_restored.layers[0].weight,
                init.Initializer,
            )

            self.assertTrue(network_restored.layers[1].initialized)
            np.testing.assert_array_equal(
                self.eval(network.layers[1].weight),
                self.eval(network_restored.layers[1].weight)
            )


    def test_basic_storage(self):
        input_data = np.random.random((100, 2))
        target_data = np.random.random(100) > 0.5

        pnn = algorithms.PNN(std=0.123, verbose=True)
        pnn.train(input_data, target_data)

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

        pnn_prediction = pnn.predict(input_data)
        loaded_pnn_prediction = loaded_pnn.predict(input_data)

        np.testing.assert_array_almost_equal(
            loaded_pnn_prediction, pnn_prediction)
