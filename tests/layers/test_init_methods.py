import math

from scipy import stats
import numpy as np

from neupy import init

from base import BaseTestCase


class LayersInitializationTestCase(BaseTestCase):
    def assertUniformlyDistributed(self, value):
        self.assertTrue(stats.kstest(value, 'uniform'),
                        msg="Sampled distribution is not uniformal")

    def assertNormalyDistributed(self, value):
        self.assertTrue(stats.mstats.normaltest(value),
                        msg="Sampled distribution is not normal")

    def test_constant_initializer(self):
        const = init.Constant(value=0)
        np.testing.assert_array_almost_equal(
            const.sample(shape=(2, 3)),
            np.zeros((2, 3))
        )

        const = init.Constant(value=1.5)
        np.testing.assert_array_almost_equal(
            const.sample(shape=(2, 3)),
            np.ones((2, 3)) * 1.5
        )

    def test_normal_initializer(self):
        norm = init.Normal(mean=0, std=0.01)
        weight = norm.sample((30, 30))

        self.assertNormalyDistributed(weight)

    def test_uniformal_initializer(self):
        uniform = init.Uniform(minval=-10, maxval=10)
        weight = uniform.sample((30, 30))

        self.assertUniformlyDistributed(weight)
        self.assertAlmostEqual(-10, np.min(weight), places=1)
        self.assertAlmostEqual(10, np.max(weight), places=1)

    def test_orthogonal_matrix_initializer_errors(self):
        with self.assertRaises(ValueError):
            ortho = init.Orthogonal()
            # More than 2 dimensions
            ortho.sample((5, 5, 5))

    def test_orthogonal_matrix_initializer(self):
        # Note: Matrix can't be orthogonal for row and column space
        # at the same time in case if matrix rectangular

        ortho = init.Orthogonal(scale=1)
        # Matrix that have more rows than columns
        weight = ortho.sample((30, 10))
        np.testing.assert_array_almost_equal(
            np.eye(10),
            weight.T.dot(weight),
            decimal=5
        )

        ortho = init.Orthogonal(scale=1)
        # Matrix that have more columns than rows
        weight = ortho.sample((10, 30))
        np.testing.assert_array_almost_equal(
            np.eye(10),
            weight.dot(weight.T),
            decimal=5
        )

    def test_he_normal(self):
        he_normal = init.HeNormal()
        weight = he_normal.sample((10, 30))

        self.assertNormalyDistributed(weight)
        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertAlmostEqual(weight.std(), math.sqrt(2. / 10),
                               places=2)

    def test_he_uniform(self):
        n_inputs = 30
        bound = math.sqrt(6. / n_inputs)

        he_uniform = init.HeUniform()
        weight = he_uniform.sample((n_inputs, 30))

        self.assertUniformlyDistributed(weight)
        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertGreaterEqual(weight.min(), -bound)
        self.assertLessEqual(weight.max(), bound)

    def test_xavier_normal(self):
        n_inputs, n_outputs = 30, 30

        xavier_normal = init.XavierNormal()
        weight = xavier_normal.sample((n_inputs, n_outputs))

        self.assertNormalyDistributed(weight)
        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertAlmostEqual(weight.std(),
                               math.sqrt(2. / (n_inputs + n_outputs)),
                               places=2)

    def test_xavier_uniform(self):
        n_inputs, n_outputs = 10, 30
        n_inputs, n_outputs = 30, 30
        xavier_uniform = init.XavierUniform()
        weight = xavier_uniform.sample((n_inputs, n_outputs))
        bound = math.sqrt(6. / (n_inputs + n_outputs))

        self.assertUniformlyDistributed(weight)

        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertGreaterEqual(weight.min(), -bound)
        self.assertLessEqual(weight.max(), bound)
