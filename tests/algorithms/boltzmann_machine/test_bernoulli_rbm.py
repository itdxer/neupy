import theano
import numpy as np

from neupy import algorithms
from neupy.utils import asfloat

from base import BaseTestCase


class BernoulliRBMTestCase(BaseTestCase):
    def setUp(self):
        super(BernoulliRBMTestCase, self).setUp()
        self.data = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 0],  # incomplete sample
            [1, 0, 1, 0],

            [0, 1, 0, 1],
            [0, 0, 0, 1],  # incomplete sample
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ], dtype=theano.config.floatX)

    def test_simple_bernoulli_rbm(self):
        data = self.data

        rbm = algorithms.RBM(n_hidden=1, n_visible=4, verbose=True, step=0.1)
        rbm.train(data, epochs=500)

        output = rbm.transform(data)
        np.testing.assert_array_equal(
            output.round(),
            np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T
        )

        typical_class1_sample = output[0]
        incomplete_class1_sample = output[2]
        # Check that probability for a typical case is
        # closer to 0 (because 0 is a class defined by RBM)
        self.assertLess(typical_class1_sample, incomplete_class1_sample)

        typical_class2_sample = output[4]
        incomplete_class2_sample = output[5]
        # Check that probability for a typical case is
        # closer to 1 (because 1 is a class defined by RBM)
        self.assertGreater(typical_class2_sample, incomplete_class2_sample)
