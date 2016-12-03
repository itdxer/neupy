import os
import tempfile

import dill
import theano
import numpy as np
from sklearn import datasets, preprocessing
from six.moves import cPickle as pickle

from neupy import algorithms, layers, storage
from neupy.exceptions import StopTraining

from base import BaseTestCase
from data import simple_classification


class StorageTestCase(BaseTestCase):
    def test_simple_dill_storage(self):
        bpnet = algorithms.GradientDescent((2, 3, 1), step=0.25)
        data, target = datasets.make_regression(n_features=2, n_targets=1)

        data = preprocessing.MinMaxScaler().fit_transform(data)
        target_scaler = preprocessing.MinMaxScaler()
        target = target_scaler.fit_transform(target.reshape(-1, 1))

        with tempfile.NamedTemporaryFile() as temp:
            test_layer_weights = bpnet.layers[1].weight.get_value().copy()
            dill.dump(bpnet, temp)
            temp.file.seek(0)

            restored_bpnet = dill.load(temp)
            temp.file.seek(0)
            layers_sizes = [layer.size for layer in restored_bpnet.layers]

            self.assertEqual(0.25, restored_bpnet.step)
            self.assertEqual([2, 3, 1], layers_sizes)
            np.testing.assert_array_equal(
                test_layer_weights,
                restored_bpnet.layers[1].weight.get_value()
            )

            bpnet.train(data, target, epochs=5)
            real_bpnet_error = bpnet.prediction_error(data, target)
            updated_input_weight = bpnet.layers[1].weight.get_value().copy()

            dill.dump(bpnet, temp)
            temp.file.seek(0)

            restored_bpnet2 = dill.load(temp)
            temp.file.seek(0)
            restored_bpnet_error = restored_bpnet2.prediction_error(
                data, target
            )

            np.testing.assert_array_equal(
                updated_input_weight,
                restored_bpnet2.layers[1].weight.get_value()
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
                updated_input_weight = (
                    bpnet.layers[1].weight.get_value().copy()
                )

                dill.dump(bpnet, temp)
                temp.file.seek(0)

                restored_bpnet2 = dill.load(temp)
                temp.file.seek(0)
                restored_bpnet_error = restored_bpnet2.prediction_error(
                    data, target
                )

                np.testing.assert_array_equal(
                    updated_input_weight,
                    restored_bpnet2.layers[1].weight.get_value()
                )
                # Error must be big, because we didn't normalize data
                self.assertEqual(real_bpnet_error, restored_bpnet_error)

    def test_storage_with_custom_theano_float_config(self):
        theano.config.floatX = 'float32'

        x_train, x_test, y_train, y_test = simple_classification()
        bpnet = algorithms.GradientDescent((10, 20, 1), step=0.25)
        bpnet.train(x_train, y_train, x_test, y_test)

        with tempfile.NamedTemporaryFile() as temp:
            test_layer_weights = bpnet.layers[1].weight.get_value().copy()
            dill.dump(bpnet, temp)
            temp.file.seek(0)

            theano.config.floatX = 'float64'
            restored_bpnet = dill.load(temp)
            np.testing.assert_array_equal(
                test_layer_weights,
                restored_bpnet.layers[1].weight.get_value()
            )


class LayerStorageTestCase(BaseTestCase):
    def test_storage_save_conection_from_network(self):
        network = algorithms.GradientDescent([
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        ])

        with tempfile.NamedTemporaryFile() as temp:
            storage.save(network, temp.name)
            temp.file.seek(0)

            filesize_after = os.path.getsize(temp.name)
            self.assertGreater(filesize_after, 0)

    def test_simple_storage(self):
        connection = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )

        with tempfile.NamedTemporaryFile() as temp:
            storage.save(connection, temp.name)
            temp.file.seek(0)

            filesize_after = os.path.getsize(temp.name)
            self.assertGreater(filesize_after, 0)

            data = pickle.load(temp.file)

            self.assertIn('sigmoid-1', data)
            self.assertIn('sigmoid-2', data)

            self.assertIn('weight', data['sigmoid-1'])
            self.assertIn('bias', data['sigmoid-1'])
            self.assertIn('weight', data['sigmoid-2'])
            self.assertIn('bias', data['sigmoid-2'])

            self.assertEqual(data['sigmoid-1']['weight'].shape, (10, 5))
            self.assertEqual(data['sigmoid-1']['bias'].shape, (5,))
            self.assertEqual(data['sigmoid-2']['weight'].shape, (5, 2))
            self.assertEqual(data['sigmoid-2']['bias'].shape, (2,))

    def test_storage_save_load_save(self):
        connection = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )

        with tempfile.NamedTemporaryFile() as temp:
            storage.save(connection, temp.name)
            temp.file.seek(0)

            filesize_first = os.path.getsize(temp.name)

            storage.load(connection, temp.name)

        with tempfile.NamedTemporaryFile() as temp:
            storage.save(connection, temp.name)
            temp.file.seek(0)

            filesize_second = os.path.getsize(temp.name)

        self.assertEqual(filesize_first, filesize_second)

    def test_storage_load_invalid_source(self):
        connection = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )

        with self.assertRaisesRegexp(TypeError, "Source type is unknown"):
            storage.load(connection, object)

    def test_storage_load_unknown_parameter(self):
        connection = layers.join(
            layers.Input(10),
            layers.Relu(1),
        )

        with self.assertRaisesRegexp(ValueError, "Cannot load parameters"):
            storage.load(connection, {}, ignore_missed=False)

        # Nothing happens in case if we ignore it
        storage.load(connection, {}, ignore_missed=True)

    def test_storage_load_from_dict(self):
        relu = layers.Relu(2, name='relu')
        connection = layers.Input(10) > relu

        weight = np.ones((10, 2))
        bias = np.ones((2,))

        storage.load(connection, {
            'relu': {
                'weight': weight,
                'bias': bias,
            }
        })

        np.testing.assert_array_almost_equal(weight, relu.weight.get_value())
        np.testing.assert_array_almost_equal(bias, relu.bias.get_value())

    def test_storage_save_and_load_during_the_training(self):
        tempdir = tempfile.mkdtemp()
        x_train, x_test, y_train, y_test = simple_classification()

        errors = {}

        def on_epoch_end(network):
            epoch = network.last_epoch
            errors[epoch] = network.prediction_error(x_test, y_test)

            if epoch == 4:
                storage.load(
                    network.connection,
                    os.path.join(tempdir, 'training-epoch-2'))
                raise StopTraining('Stop training process after 4th epoch')
            else:
                storage.save(
                    network.connection,
                    os.path.join(tempdir, 'training-epoch-{}'.format(epoch)))

        gdnet = algorithms.GradientDescent(
            connection=(10, 4, 1),
            epoch_end_signal=on_epoch_end,
            step=0.5
        )
        gdnet.train(x_train, y_train)

        validation_error = gdnet.prediction_error(x_test, y_test)

        self.assertGreater(errors[2], errors[4])
        self.assertAlmostEqual(validation_error, errors[2])
        self.assertNotAlmostEqual(validation_error, errors[4])
