from neupy import algorithms

from base import BaseTestCase


class LVQTestCase(BaseTestCase):
    def test_simple_lvq(self):
        lvqnet = algorithms.LVQ(
            n_inputs=2,
            n_subclasses=4,
            n_classes=2,
        )
