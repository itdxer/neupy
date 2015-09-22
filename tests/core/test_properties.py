from neupy.core.config import Configurable
from neupy.core.properties import *

from base import BaseTestCase


class ParametersTestCase(BaseTestCase):
    def test_base_properties(self):
        class A(Configurable):
            basic_propepty = Property(expected_type=int)
            size_propepty = CheckSizeProperty(min_size=-1, max_size=1)
            list_of_properties = ListOfTypesProperty(inner_list_type=str,
                                                     count=3)

        a = A()
        a.basic_propepty = 1
        a.size_propepty = 0
        a.list_of_properties = ('1', '2', '3')

        with self.assertRaises(TypeError):
            a.basic_propepty = '1'

        with self.assertRaises(TypeError):
            a.list_of_properties = (1, '2', '3')

        with self.assertRaises(ValueError):
            a.size_propepty = -2

        with self.assertRaises(ValueError):
            a.list_of_properties = ('1', '2', '3', '4')

    def test_choice_property(self):
        class A(Configurable):
            choice = ChoiceProperty(choices={
                'one': 1, 'two': 2, 'three': 3
            })

        a = A(choice='three')
        self.assertEqual(a.choice, 3)

        a.choice = 'one'
        self.assertEqual(a.choice, 1)

        with self.assertRaises(ValueError):
            a.choice = 2

        with self.assertRaises(ValueError):
            a.choice = None

        with self.assertRaises(ValueError):
            a.choice = "2"
