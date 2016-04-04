import numpy as np

from neupy.utils import (preformat_value, as_array2d, AttributeKeyDict,
                         smallest_positive_number, asfloat)
from neupy.network.utils import shuffle

from base import BaseTestCase


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

    def test_shuffle(self):
        input_data = np.arange(10)
        shuffeled_data = shuffle(input_data, input_data)
        np.testing.assert_array_equal(*shuffeled_data)

        np.testing.assert_array_equal(tuple(), shuffle())

        with self.assertRaises(ValueError):
            shuffle(input_data, input_data[:len(input_data) - 1])

    def test_as_array2d(self):
        test_input = np.ones(5)
        actual_output = as_array2d(test_input)
        self.assertEqual((1, 5), actual_output.shape)

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

    def test_smallest_positive_number(self):
        epsilon = smallest_positive_number()
        self.assertNotEqual(0, asfloat(1) - (asfloat(1) - asfloat(epsilon)))
        self.assertEqual(0, asfloat(1) - (asfloat(1) - asfloat(epsilon / 10)))
