import numpy as np

from neuralpy import algorithms

from memory.data import *
from base import BaseTestCase


zero_hint = np.matrix([[0, 1, 0, 0]])
one_hint = np.matrix([[1, 0, 0, 0]])


class BAMTestCase(BaseTestCase):
    def setUp(self):
        super(BAMTestCase, self).setUp()
        self.data = np.concatenate([zero, one], axis=0)
        self.hints = np.concatenate([zero_hint, one_hint], axis=0)

    def test_discrete_bam(self):
        bamnet = algorithms.DiscreteBAM()
        bamnet.train(self.data.copy(), self.hints.copy())

        self.assertTrue(np.all(bamnet.predict(half_zero) == zero_hint))
        self.assertTrue(np.all(bamnet.predict_output(half_one) == one_hint))

        self.assertTrue(np.all(bamnet.predict_input(zero_hint) == zero))
        self.assertTrue(np.all(bamnet.predict_input(one_hint) == one))
