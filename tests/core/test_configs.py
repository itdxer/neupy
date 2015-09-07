from neupy.core.config import Configurable
from neupy.core.properties import *

from base import BaseTestCase


class ConfigsTestCase(BaseTestCase):
    def test_config_options(self):
        class A(Configurable):
            property_a = Property()

        class B(A):
            property_b = Property()

        class C(B):
            property_c = Property()

        class D(A):
            property_d = Property()

        self.assertEqual(sorted(A.options.keys()), ['property_a'])
        self.assertEqual(sorted(B.options.keys()),
                         ['property_a', 'property_b'])
        self.assertEqual(sorted(D.options.keys()),
                         ['property_a', 'property_d'])
        self.assertEqual(sorted(C.options.keys()),
                         ['property_a', 'property_b', 'property_c'])
