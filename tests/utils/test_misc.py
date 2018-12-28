# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import random
import pickle
from collections import namedtuple

import numpy as np

from neupy.utils.misc import as_tuple, AttributeKeyDict, reproducible

from base import BaseTestCase


class MiscUtilsTestCase(BaseTestCase):
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

        # Storage
        recovered = pickle.loads(pickle.dumps(attrdict))
        self.assertDictEqual(dict(recovered), {
            'new_value': 'test', 'val2': 'world'})

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
                input_args=((1, 2, 3), [], (4, 5)),
                expected_output=(1, 2, 3, 4, 5),
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
                msg="Input args: {}".format(testcase.input_args))

    def test_reproducible_utils_math_library(self):
        reproducible(seed=0)
        x1 = random.random()

        reproducible(seed=0)
        x2 = random.random()

        self.assertAlmostEqual(x1, x2)

    def test_reproducible_utils_numpy_library(self):
        reproducible(seed=0)
        x1 = np.random.random((10, 10))

        reproducible(seed=0)
        x2 = np.random.random((10, 10))

        np.testing.assert_array_almost_equal(x1, x2)
