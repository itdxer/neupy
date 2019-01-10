import copy

import numpy as np

from neupy.utils import asfloat
from neupy import layers, storage
from neupy.storage import (
    validate_data_structure, InvalidFormat,
    ParameterLoaderError, validate_layer_compatibility,
    load_layer_parameter,
)

from base import BaseTestCase


class DictStorageTestCase(BaseTestCase):
    maxDiff = 10000

    def test_storage_save_dict(self):
        network = layers.join(
            [[
                layers.Input(2, name='input-1'),
                layers.PRelu(1, name='prelu')
            ], [
                layers.Input(1, name='input-2'),
                layers.Sigmoid(4, name='sigmoid'),
                layers.BatchNorm(name='batch-norm'),
            ]],
            layers.Concatenate(name='concatenate'),
            layers.Softmax(3, name='softmax'),
        )
        dict_network = storage.save_dict(network)

        expected_keys = ('metadata', 'layers', 'graph')
        self.assertItemsEqual(expected_keys, dict_network.keys())

        expected_metadata_keys = ('created', 'language', 'library', 'version')
        actual_metadata_keys = dict_network['metadata'].keys()
        self.assertItemsEqual(expected_metadata_keys, actual_metadata_keys)

        self.assertEqual(len(dict_network['layers']), 7)

        expected_layers = [{
            'class_name': 'Input',
            'configs': {'name': 'input-2', 'size': 1},
            'input_shape': (1,),
            'name': 'input-2',
            'output_shape': (1,)
        }, {
            'class_name': 'Sigmoid',
            'configs': {'name': 'sigmoid', 'size': 4},
            'input_shape': (1,),
            'name': 'sigmoid',
            'output_shape': (4,)
        }, {
            'class_name': 'BatchNorm',
            'configs': {
                'alpha': 0.1,
                'axes': (0,),
                'epsilon': 1e-05,
                'name': 'batch-norm'
            },
            'input_shape': (4,),
            'name': 'batch-norm',
            'output_shape': (4,)
        }, {
            'class_name': 'Input',
            'configs': {'name': 'input-1', 'size': 2},
            'input_shape': (2,),
            'name': 'input-1',
            'output_shape': (2,)
        }, {
            'class_name': 'PRelu',
            'configs': {'alpha_axes': (-1,), 'name': 'prelu', 'size': 1},
            'input_shape': (2,),
            'name': 'prelu',
            'output_shape': (1,)
        }, {
            'class_name': 'Concatenate',
            'configs': {'axis': -1, 'name': 'concatenate'},
            'input_shape': [(1,), (4,)],
            'name': 'concatenate',
            'output_shape': (5,)
        }, {
            'class_name': 'Softmax',
            'configs': {'name': 'softmax', 'size': 3},
            'input_shape': (5,),
            'name': 'softmax',
            'output_shape': (3,)
        }]
        actual_layers = []
        for i, layer in enumerate(dict_network['layers']):
            self.assertIn('parameters', layer, msg="Layer #" + str(i))

            layer = copy.deepcopy(layer)
            del layer['parameters']
            actual_layers.append(layer)

        self.assertEqual(actual_layers, expected_layers)

    def test_storage_load_dict_using_names(self):
        relu = layers.Relu(2, name='relu')
        network = layers.Input(10) > relu

        weight = np.ones((10, 2))
        bias = np.ones((2,))

        storage.load_dict(network, {
            'metadata': {},  # avoided for simplicity
            'graph': {},  # avoided for simplicity
            # Input layer was avoided on purpose
            'layers': [{
                'name': 'relu',
                'class_name': 'Relu',
                'input_shape': (10,),
                'output_shape': (2,),
                'configs': {},
                'parameters': {
                    'weight': {'trainable': True, 'value': weight},
                    'bias': {'trainable': True, 'value': bias},
                }
            }]
        })

        np.testing.assert_array_almost_equal(weight, self.eval(relu.weight))
        np.testing.assert_array_almost_equal(bias, self.eval(relu.bias))

    def test_storage_load_dict_using_wrong_names(self):
        network = layers.join(
            layers.Input(3),
            layers.Relu(4, name='relu'),
            layers.Linear(5, name='linear') > layers.Relu(),
            layers.Softmax(6, name='softmax'),
        )

        storage.load_dict(network, {
            'metadata': {},  # avoided for simplicity
            'graph': {},  # avoided for simplicity
            # Input layer was avoided on purpose
            'layers': [{
                'name': 'name-1',
                'class_name': 'Relu',
                'input_shape': (3,),
                'output_shape': (4,),
                'configs': {},
                'parameters': {
                    'weight': {'trainable': True, 'value': np.ones((3, 4))},
                    'bias': {'trainable': True, 'value': np.ones((4,))},
                }
            }, {
                'name': 'name-2',
                'class_name': 'Relu',
                'input_shape': (4,),
                'output_shape': (5,),
                'configs': {},
                'parameters': {
                    'weight': {'trainable': True, 'value': np.ones((4, 5))},
                    'bias': {'trainable': True, 'value': np.ones((5,))},
                }
            }, {
                'name': 'name-3',
                'class_name': 'Softmax',
                'input_shape': (5,),
                'output_shape': (6,),
                'configs': {},
                'parameters': {
                    'weight': {'trainable': True, 'value': np.ones((5, 6))},
                    'bias': {'trainable': True, 'value': np.ones((6,))},
                }
            }]
        }, load_by='order', skip_validation=False)

        relu = network.layer('relu')
        self.assertEqual(12, np.sum(self.eval(relu.weight)))
        self.assertEqual(4, np.sum(self.eval(relu.bias)))

        linear = network.layer('linear')
        self.assertEqual(20, np.sum(self.eval(linear.weight)))
        self.assertEqual(5, np.sum(self.eval(linear.bias)))

        softmax = network.layer('softmax')
        self.assertEqual(30, np.sum(self.eval(softmax.weight)))
        self.assertEqual(6, np.sum(self.eval(softmax.bias)))

    def test_storage_load_dict_invalid_number_of_paramters(self):
        network = layers.join(
            layers.Input(3),
            layers.Relu(4, name='relu'),
            layers.Linear(5, name='linear') > layers.Relu(),
            layers.Softmax(6, name='softmax'),
        )
        data = {
            'metadata': {},  # avoided for simplicity
            'graph': {},  # avoided for simplicity
            # Input layer was avoided on purpose
            'layers': [{
                'name': 'name-1',
                'class_name': 'Relu',
                'input_shape': (3,),
                'output_shape': (4,),
                'configs': {},
                'parameters': {
                    'weight': {
                        'trainable': True,
                        'value': np.ones((3, 4))
                    },
                    'bias': {'trainable': True, 'value': np.ones((4,))},
                }
            }]
        }

        with self.assertRaises(ParameterLoaderError):
            storage.load_dict(network, data, ignore_missing=False)

    def test_failed_loading_mode_for_storage(self):
        network = layers.Input(2) > layers.Sigmoid(1)

        with self.assertRaisesRegexp(ValueError, "Invalid value"):
            storage.load_dict(network, {}, load_by='unknown')

    def test_failed_load_parameter_invalid_type(self):
        sigmoid = layers.Sigmoid(1, bias=None)
        layers.Input(2) > sigmoid

        with self.assertRaisesRegexp(ParameterLoaderError, "equal to None"):
            load_layer_parameter(sigmoid, {
                'parameters': {
                    'bias': {
                        'value': np.array([[0]]),
                        'trainable': True,
                    },
                },
            })


class StoredDataValidationTestCase(BaseTestCase):
    def test_stored_data_dict_format_basics(self):
        with self.assertRaises(InvalidFormat):
            validate_data_structure([])

        with self.assertRaises(InvalidFormat):
            validate_data_structure({})

        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': {}})

        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': []})

    def test_stored_data_layers_format(self):
        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': [[]]})

        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': [{
                'parameters': {},
            }]})

        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': [{
                'parameters': {},
                'input_shape': (2,),
            }]})

        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': [{
                'parameters': {},
                'input_shape': (2,),
                'output_shape': (3,),
            }]})

        with self.assertRaises(InvalidFormat):
            validate_data_structure({
                'layers': [{
                    'parameters': {},
                    'input_shape': 2,  # wrong type
                    'output_shape': (3,),
                    'name': 'name',
                }]
            })

        with self.assertRaises(InvalidFormat):
            validate_data_structure({
                'layers': [{
                    'parameters': [],  # wrong type
                    'input_shape': (2,),
                    'output_shape': (3,),
                    'name': 'name',
                }]
            })

        result = validate_data_structure({
            'layers': [{
                'parameters': {},
                'input_shape': (2,),
                'output_shape': (3,),
                'name': 'name',
            }]
        })
        self.assertIsNone(result)

    def test_stored_data_parameters_format(self):
        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': [{
                'name': 'name',
                'input_shape': (2,),
                'output_shape': (3,),
                'parameters': {
                    'weight': np.ones((2, 3)),
                }
            }]})

        with self.assertRaises(InvalidFormat):
            validate_data_structure({'layers': [{
                'name': 'name',
                'input_shape': (2,),
                'output_shape': (3,),
                'parameters': {
                    'weight': {
                        'data': np.ones((2, 3)),
                    },
                }
            }]})

        result = validate_data_structure({'layers': [{
            'name': 'name',
            'input_shape': (2,),
            'output_shape': (3,),
            'parameters': {
                'weight': {
                    'value': np.ones((2, 3)),
                    'trainable': True,
                },
            }
        }]})
        self.assertIsNone(result)

    def test_storage_data_layer_compatibility(self):
        network = layers.Input(2) > layers.Sigmoid(3, name='sigm')
        sigmoid = network.layer('sigm')

        with self.assertRaises(ParameterLoaderError):
            validate_layer_compatibility(sigmoid, {
                'name': 'sigm',
                'class_name': 'Sigmoid',
                'input_shape': (3,),  # wrong input shape
                'output_shape': (3,),
                'configs': {},
                'parameters': {
                    'weight': {'trainable': True, 'value': np.ones((2, 3))},
                    'bias': {'trainable': True, 'value': np.ones((3,))},
                }
            })

        with self.assertRaises(ParameterLoaderError):
            validate_layer_compatibility(sigmoid, {
                'name': 'sigm',
                'class_name': 'Sigmoid',
                'input_shape': (2,),
                'output_shape': (2,),  # wrong output shape
                'configs': {},
                'parameters': {
                    'weight': {'trainable': True, 'value': np.ones((2, 3))},
                    'bias': {'trainable': True, 'value': np.ones((3,))},
                }
            })

        result = validate_layer_compatibility(sigmoid, {
            'name': 'sigm',
            'class_name': 'Sigmoid',
            'input_shape': (2,),
            'output_shape': (3,),
            'configs': {},
            'parameters': {
                'weight': {'trainable': True, 'value': np.ones((2, 3))},
                'bias': {'trainable': True, 'value': np.ones((3,))},
            }
        })
        self.assertIsNone(result)

    def test_basic_skip_validation(self):
        network = layers.Input(10) > layers.Relu(1)

        with self.assertRaises(InvalidFormat):
            storage.load_dict(network, {}, skip_validation=False)


class TransferLearningTestCase(BaseTestCase):
    def test_transfer_learning_using_position(self):
        network_pretrained = layers.join(
            layers.Input(10),
            layers.Relu(5),
            layers.Relu(2, name='relu-2'),
            layers.Sigmoid(1),
        )
        network_new = layers.join(
            layers.Input(10),
            layers.Relu(5),
            layers.Relu(2),
        )
        pretrained_layers_stored = storage.save_dict(network_pretrained)

        with self.assertRaises(ParameterLoaderError):
            storage.load_dict(
                network_new,
                pretrained_layers_stored,
                load_by='names_or_order',
                ignore_missing=False)

        storage.load_dict(
            network_new,
            pretrained_layers_stored,
            load_by='names_or_order',
            ignore_missing=True)

        random_input = asfloat(np.random.random((12, 10)))
        new_network_output = self.eval(network_new.output(random_input))
        pretrained_output = self.eval(
            network_pretrained.end('relu-2').output(random_input))

        np.testing.assert_array_almost_equal(
            pretrained_output, new_network_output)

    def test_transfer_learning_using_names(self):
        network_pretrained = layers.join(
            layers.Input(10),
            layers.Relu(5, name='relu-1'),
            layers.Relu(2, name='relu-2'),
            layers.Sigmoid(1),
        )
        network_new = layers.join(
            layers.Input(10),
            layers.Relu(5, name='relu-1'),
            layers.Relu(2, name='relu-2'),
            layers.Relu(8, name='relu-3'),  # new layer
        )
        pretrained_layers_stored = storage.save_dict(network_pretrained)

        storage.load_dict(
            network_new,
            pretrained_layers_stored,
            load_by='names',
            skip_validation=False,
            ignore_missing=True)

        random_input = asfloat(np.random.random((12, 10)))

        pretrained_output = self.eval(
            network_pretrained.end('relu-2').output(random_input))
        new_network_output = self.eval(
            network_new.end('relu-2').output(random_input))

        np.testing.assert_array_almost_equal(
            pretrained_output, new_network_output)

        pred = self.eval(network_new.output(random_input))
        self.assertEqual(pred.shape, (12, 8))
