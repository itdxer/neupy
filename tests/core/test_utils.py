from collections import namedtuple

import theano
import numpy as np
from scipy.sparse import csr_matrix

from neupy.utils import (preformat_value, as_tuple, AttributeKeyDict, asint,
                         asfloat, format_data)
from neupy.algorithms.utils import shuffle, iter_until_converge
from neupy.layers.utils import preformat_layer_shape
from neupy import algorithms

from base import BaseTestCase
from utils import catch_stdout


class UtilsTestCase(BaseTestCase):
    def test_preformat_value(self):
        def my_func():
            pass

        class MyClass(object):
            pass

        self.assertEqual('my_func', preformat_value(my_func))
        self.assertEqual('MyClass', preformat_value(MyClass))

        expected = ['my_func', 'MyClass', 1]
        actual = preformat_value((my_func, MyClass, 1))
        np.testing.assert_array_equal(expected, actual)

        expected = ['my_func', 'MyClass', 1]
        actual = preformat_value([my_func, MyClass, 1])
        np.testing.assert_array_equal(expected, actual)

        expected = sorted(['my_func', 'MyClass', 'x'])
        actual = sorted(preformat_value({my_func, MyClass, 'x'}))
        np.testing.assert_array_equal(expected, actual)

        self.assertEqual(1, preformat_value(1))

        expected = (3, 2)
        actual = preformat_value(np.ones((3, 2)))
        np.testing.assert_array_equal(expected, actual)

        expected = (1, 2)
        actual = preformat_value(np.matrix([[1, 1]]))
        np.testing.assert_array_equal(expected, actual)

    def test_preformat_layer_shape(self):
        self.assertEqual((2, 3, 1), preformat_layer_shape((2, 3, 1)))
        self.assertEqual(10, preformat_layer_shape((10,)))

    def test_attribute_key_dict(self):
        attrdict = AttributeKeyDict(val1='hello', val2='world')

        # Get
        self.assertEqual(attrdict.val1, 'hello')
        self.assertEqual(attrdict.val2, 'world')

        with self.assertRaises(KeyError):
            attrdict.unknown_variable

        # Set
        attrdict.new_value = 'test'
        self.assertEqual(attrdict.new_value, 'test')

        # Delete
        del attrdict.val1
        with self.assertRaises(KeyError):
            attrdict.val1

    def test_format_data(self):
        # None input
        self.assertEqual(format_data(None), None)

        # Sparse data
        sparse_matrix = csr_matrix((3, 4), dtype=np.int8)
        formated_sparce_matrix = format_data(sparse_matrix)
        np.testing.assert_array_equal(formated_sparce_matrix, sparse_matrix)
        self.assertEqual(formated_sparce_matrix.dtype, sparse_matrix.dtype)

        # Vector input
        x = np.random.random(10)
        formated_x = format_data(x, is_feature1d=True)
        self.assertEqual(formated_x.shape, (10, 1))

        x = np.random.random(10)
        formated_x = format_data(x, is_feature1d=False)
        self.assertEqual(formated_x.shape, (1, 10))

    def test_asfloat(self):
        float_type = theano.config.floatX

        # Sparse matrix
        sparse_matrix = csr_matrix((3, 4), dtype=np.int8)
        self.assertIs(sparse_matrix, asfloat(sparse_matrix))

        # Numpy array-like elements
        x = np.array([1, 2, 3], dtype=float_type)
        self.assertIs(x, asfloat(x))

        x = np.array([1, 2, 3], dtype=np.int8)
        self.assertIsNot(x, asfloat(x))

        # Python list
        x = [1, 2, 3]
        self.assertEqual(asfloat(x).shape, (3,))

        # Theano variables
        x = theano.tensor.imatrix()
        self.assertNotEqual(x.dtype, float_type)
        self.assertEqual(asfloat(x).dtype, float_type)

    def test_asint(self):
        int2float_types = {
            'float32': 'int32',
            'float64': 'int64',
        }
        int_type = int2float_types[theano.config.floatX]

        # Sparse matrix
        sparse_matrix = csr_matrix((3, 4), dtype=np.int8)
        self.assertIs(sparse_matrix, asint(sparse_matrix))

        # Numpy array-like elements
        x = np.array([1, 2, 3], dtype=int_type)
        self.assertIs(x, asint(x))

        x = np.array([1, 2, 3], dtype=np.int8)
        self.assertIsNot(x, asint(x))

        # Python list
        x = [1, 2, 3]
        self.assertEqual(asint(x).shape, (3,))

        # Theano variables
        x = theano.tensor.fmatrix()
        self.assertNotEqual(x.dtype, int_type)
        self.assertEqual(asint(x).dtype, int_type)

    def test_as_tuple(self):
        Case = namedtuple("Case", "input_args expected_output")
        testcases = (
            Case(input_args=(1, 2, 3),
                 expected_output=(1, 2, 3)),
            Case(input_args=(None, (1, 2, 3), None),
                 expected_output=(None, 1, 2, 3, None)),
            Case(input_args=((1, 2, 3), (4, 5, 3)),
                 expected_output=(1, 2, 3, 4, 5, 3)),
        )

        for testcase in testcases:
            actual_output = as_tuple(*testcase.input_args)
            self.assertEqual(actual_output, testcase.expected_output,
                             msg="Input args: {}".format(testcase.input_args))


class IterUntilConvergeTestCase(BaseTestCase):
    def test_iter_until_converge_critical_cases(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=True)
            iterator = iter_until_converge(network, epsilon=1e-5, max_epochs=5)

            for epoch in iterator:
                network.errors.append(np.nan)

            terminal_output = out.getvalue()
            self.assertIn('NaN or Inf', terminal_output)


class ShuffleTestCase(BaseTestCase):
    def test_shuffle_basic(self):
        input_data = np.arange(10)
        shuffeled_data = shuffle(input_data, input_data)
        np.testing.assert_array_equal(*shuffeled_data)

    def test_shuffle_empty_input(self):
        np.testing.assert_array_equal(tuple(), shuffle())

    def test_shuffle_single_input(self):
        input_data = np.ones(10)
        shuffeled_data = shuffle(input_data)
        # Output suppose to be a shuffled array, but
        # not a tuple with shuffled array
        np.testing.assert_array_equal(input_data, shuffeled_data)

    def test_shuffle_invalid_shapes_exception(self):
        input_data = np.arange(10)
        with self.assertRaisesRegexp(ValueError, r'\(10,\), \(9,\)'):
            shuffle(input_data, input_data[:len(input_data) - 1])

    def test_shuffle_with_nones(self):
        input_with_nones = (None, None)
        actual_output = shuffle(*input_with_nones)
        self.assertEqual(tuple(), actual_output)
