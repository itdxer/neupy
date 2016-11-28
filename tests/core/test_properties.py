from collections import namedtuple

import numpy as np

from neupy.core.config import Configurable
from neupy.core.properties import (Property, ArrayProperty, TypedListProperty,
                                   BoundedProperty, ProperFractionProperty,
                                   NumberProperty, IntProperty, ChoiceProperty,
                                   WithdrawProperty)

from base import BaseTestCase


class PropertiesBasicsTestCase(BaseTestCase):
    def test_basic_properties(self):
        class A(Configurable):
            int_property = Property(expected_type=int)

        a = A()
        a.int_property = 1
        a.int_property = -10
        with self.assertRaises(TypeError):
            a.int_property = '1'

        class B(Configurable):
            int_property = Property(expected_type=(str, set))

        b = B()
        b.int_property = {1, 2, 3}
        b.int_property = 'hello'
        with self.assertRaises(TypeError):
            b.int_property = [5, 4]

    def test_required_properties(self):
        class A(Configurable):
            required_prop = Property(required=True)

        A(required_prop='defined')
        with self.assertRaises(ValueError):
            A()

    def test_properties_with_specified_typse(self):
        Case = namedtuple("Case", "property_class valid invalid")
        test_cases = (
            Case(property_class=IntProperty,
                 valid=[1, 2, 0, -10, np.int32(23), np.int64(11)],
                 invalid=[1.0, 12.34, 'invalid']),
            Case(property_class=NumberProperty,
                 valid=[1, 1.0, 12.34, -345],
                 invalid=['invalid', [4, 5]]),
            Case(property_class=ArrayProperty,
                 valid=[np.array([]), np.ones((5, 3)), np.matrix([1, 2])],
                 invalid=[[1, 2, 3], 'invalid']),
        )

        for test_case in test_cases:
            class A(Configurable):
                value = test_case.property_class()

            a = A()
            for valid_value in test_case.valid:
                a.value = valid_value

            for invalid_value in test_case.invalid:
                with self.assertRaises(TypeError):
                    a.value = invalid_value

    def test_property_get_method(self):
        prop = Property(default=3)
        self.assertEqual(None, prop.__get__(None, None))

    def test_property_repr(self):
        prop = Property(default=3)
        self.assertEqual('Property()', repr(prop))

    def test_property_repr_with_name(self):
        prop = Property(default=3)
        prop.name = 'test'

        self.assertEqual('Property(name="test")', repr(prop))


class BoundedPropertiesTestCase(BaseTestCase):
    def test_bounded_properties(self):
        class A(Configurable):
            bounded_property = BoundedProperty(minval=-1, maxval=1)

        a = A()
        a.bounded_property = 0
        with self.assertRaises(ValueError):
            a.bounded_property = -2

    def test_propert_raciton_property(self):
        class A(Configurable):
            fraction = ProperFractionProperty()

        a = A()

        valid_values = [0, 0.5, 1 / 3., 1]
        for valid_value in valid_values:
            a.fraction = valid_value

        invalid_values = [10, 1 + 1e-10, 0 - 1e-10, 'test']
        for invalid_value in invalid_values:
            msg = "Specified value: {}".format(invalid_value)
            with self.assertRaises((ValueError, TypeError), msg=msg):
                a.fraction = invalid_value


class TypedListPropertiesTestCase(BaseTestCase):
    def test_typed_list_properties(self):
        class A(Configurable):
            list_of_properties = TypedListProperty(element_type=str,
                                                   n_elements=3)

        a = A()
        a.list_of_properties = ('1', '2', '3')

        with self.assertRaises(TypeError):
            # Invalid types
            a.list_of_properties = (1, '2', '3')

        with self.assertRaises(ValueError):
            # Invalid number of elements
            a.list_of_properties = ('1', '2', '3', '4')

    def test_multiple_types(self):
        class A(Configurable):
            list_of_properties = TypedListProperty(element_type=(int, float),
                                                   n_elements=3)

        a = A(list_of_properties=[1, -2.0, 3.4])

        with self.assertRaises(TypeError):
            a.list_of_properties = (1, 1.23, 'x')


class ChoicesPropertiesTestCase(BaseTestCase):
    def test_choice_property_from_dict(self):
        class A(Configurable):
            choice = ChoiceProperty(choices={'one': 1, 'two': 2, 'three': 3})

        a = A(choice='three')
        self.assertEqual(a.choice, 3)
        a.choice = 'one'
        self.assertEqual(a.choice, 1)

    def test_choice_property_invalid_values(self):
        class A(Configurable):
            choice = ChoiceProperty(choices={'one': 1, 'two': 2, 'three': 3})

        a = A(choice='three')
        invalid_values = [2, None, "2"]
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                a.choice = invalid_value

    def test_choice_property_from_list(self):
        class A(Configurable):
            choice = ChoiceProperty(choices=['one', 'two', 'three'],
                                    default='two')

        a = A()
        self.assertEqual(a.choice, 'two')
        a.choice = 'three'
        self.assertEqual(a.choice, 'three')

        with self.assertRaises(ValueError):
            a.choice = 1

    def test_choice_property_exceptions(self):
        with self.assertRaises(ValueError):
            class A(Configurable):
                choice = ChoiceProperty(choices='test')

        with self.assertRaises(ValueError):
            class B(Configurable):
                choice = ChoiceProperty(choices=[])

    def test_choice_property_on_unknown_instance(self):
        prop = ChoiceProperty(choices=[1, 2, 3])
        self.assertEqual(None, prop.__get__(None, None))


class WithdrawPropertyTestCase(BaseTestCase):
    def test_withdraw_property(self):
        class A(Configurable):
            prop = Property(default=3)

        class B(A):
            prop = WithdrawProperty()

        a = A()
        self.assertIn('prop', a.options)
        self.assertEqual(a.prop, 3)

        b = B()
        self.assertNotIn('prop', b.options)
