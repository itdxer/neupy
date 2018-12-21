# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from collections import namedtuple

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from neupy.utils.misc import as_tuple, AttributeKeyDict
from neupy.utils.iters import iter_until_converge
from neupy.utils.processing import shuffle, format_data, asfloat
from neupy.algorithms.base import preformat_value
from neupy import algorithms, layers

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
        self.assertIs(formated_sparce_matrix, sparse_matrix)
        self.assertEqual(formated_sparce_matrix.dtype, sparse_matrix.dtype)

        # Vector input
        x = np.random.random(10)
        formated_x = format_data(x, is_feature1d=True)
        self.assertEqual(formated_x.shape, (10, 1))

        x = np.random.random(10)
        formated_x = format_data(x, is_feature1d=False)
        self.assertEqual(formated_x.shape, (1, 10))

    def test_asfloat(self):
        # Sparse matrix
        sparse_matrix = csr_matrix((3, 4), dtype=np.int8)
        self.assertIs(sparse_matrix, asfloat(sparse_matrix))

        # Numpy array-like elements
        x = np.array([1, 2, 3], dtype=np.float32)
        self.assertIs(x, asfloat(x))

        x = np.array([1, 2, 3], dtype=np.int8)
        self.assertIsNot(x, asfloat(x))

        # Python list
        x = [1, 2, 3]
        self.assertEqual(asfloat(x).shape, (3,))

        # Tensorfow variables
        x = tf.placeholder(dtype=tf.int32)
        self.assertNotEqual(x.dtype, tf.float32)
        self.assertEqual(asfloat(x).dtype, tf.float32)

    def test_as_tuple(self):
        Case = namedtuple("Case", "input_args expected_output")
        testcases = (
            Case(
                input_args=(1, 2, 3),
                expected_output=(1, 2, 3),
            ),
            Case(
                input_args=(None, (1, 2, 3), None),
                expected_output=(None, 1, 2, 3, None),
            ),
            Case(
                input_args=((1, 2, 3), tuple()),
                expected_output=(1, 2, 3),
            ),
            Case(
                input_args=((1, 2, 3), (4, 5, 3)),
                expected_output=(1, 2, 3, 4, 5, 3),
            ),
        )

        for testcase in testcases:
            actual_output = as_tuple(*testcase.input_args)
            self.assertEqual(
                actual_output, testcase.expected_output,
                msg="Input args: {}".format(testcase.input_args)
            )


class IterUntilConvergeTestCase(BaseTestCase):
    def test_iter_until_converge_critical_cases(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                verbose=True,
            )
            iterator = iter_until_converge(network, epsilon=1e-5, max_epochs=5)

            for epoch in iterator:
                network.training_errors.append(np.nan)

            terminal_output = out.getvalue()
            self.assertIn('NaN or Inf', terminal_output)


class ShuffleTestCase(BaseTestCase):
    def test_shuffle_basic(self):
        X = np.arange(10)
        shuffeled_data = shuffle(X, X)
        np.testing.assert_array_equal(*shuffeled_data)

    def test_shuffle_invalid_shapes_exception(self):
        X = np.arange(10)
        with self.assertRaisesRegexp(ValueError, r'\(10,\), \(9,\)'):
            shuffle(X, X[:len(X) - 1])

    def test_shuffle_with_nones(self):
        input_with_nones = (None, None)
        actual_output = shuffle(*input_with_nones)
        self.assertEqual(input_with_nones, actual_output)
