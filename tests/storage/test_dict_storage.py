import copy

import numpy as np

from neupy import layers, storage
from neupy.storage import (validate_data_structure, InvalidFormat,
                           ParameterLoaderError, validate_layer_compatibility)

from base import BaseTestCase


class DictStorageTestCase(BaseTestCase):
    def test_storage_save_dict(self):
        connection = layers.join(
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
        dict_connection = storage.save_dict(connection)

        expected_keys = ('metadata', 'layers', 'graph')
        self.assertItemsEqual(expected_keys, dict_connection.keys())

        expected_metadata_keys = ('created', 'language', 'library',
                                  'version', 'theano_float')
        actual_metadata_keys = dict_connection['metadata'].keys()
        self.assertItemsEqual(expected_metadata_keys, actual_metadata_keys)

        self.assertEqual(len(dict_connection['layers']), 7)

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
            'configs': {'alpha_axes': (1,), 'name': 'prelu', 'size': 1},
            'input_shape': (2,),
            'name': 'prelu',
            'output_shape': (1,)
        }, {
            'class_name': 'Concatenate',
            'configs': {'axis': 1, 'name': 'concatenate'},
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
        for i, layer in enumerate(dict_connection['layers']):
            self.assertIn('parameters', layer, msg="Layer #" + str(i))

            layer = copy.deepcopy(layer)
            del layer['parameters']
            actual_layers.append(layer)

        self.assertEqual(actual_layers, expected_layers)

    def test_storage_load_dict_using_names(self):
        relu = layers.Relu(2, name='relu')
        connection = layers.Input(10) > relu

        weight = np.ones((10, 2))
        bias = np.ones((2,))

        storage.load_dict(connection, {
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

        np.testing.assert_array_almost_equal(weight, relu.weight.get_value())
        np.testing.assert_array_almost_equal(bias, relu.bias.get_value())

    def test_storage_load_dict_using_wrong_names(self):
        connection = layers.join(
            layers.Input(3),
            layers.Relu(4, name='relu'),
            layers.Linear(5, name='linear') > layers.Relu(),
            layers.Softmax(6, name='softmax'),
        )

        storage.load_dict(connection, {
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
        })

        relu = connection.layer('relu')
        self.assertEqual(12, np.sum(relu.weight.get_value()))
        self.assertEqual(4, np.sum(relu.bias.get_value()))

        linear = connection.layer('linear')
        self.assertEqual(20, np.sum(linear.weight.get_value()))
        self.assertEqual(5, np.sum(linear.bias.get_value()))

        softmax = connection.layer('softmax')
        self.assertEqual(30, np.sum(softmax.weight.get_value()))
        self.assertEqual(6, np.sum(softmax.bias.get_value()))

    def test_storage_load_dict_invalid_number_of_paramters(self):
        connection = layers.join(
            layers.Input(3),
            layers.Relu(4, name='relu'),
            layers.Linear(5, name='linear') > layers.Relu(),
            layers.Softmax(6, name='softmax'),
        )

        with self.assertRaises(ParameterLoaderError):
            storage.load_dict(connection, {
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
            }, ignore_missed=False)


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

        result = validate_data_structure({
            'layers': [{
                'parameters': {},
                'input_shape': (2,),
                'output_shape': (3,),
                'name': 'name',
            }]
        })
        self.assertIsNone(result)

    def test_storage_data_layer_compatibility(self):
        connection = layers.Input(2) > layers.Sigmoid(3, name='sigm')
        sigmoid = connection.layer('sigm')

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
