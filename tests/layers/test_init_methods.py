import math

from scipy import stats
import numpy as np

from neupy import init

from base import BaseTestCase


class BaseInitializerTestCase(BaseTestCase):
    def assertUniformlyDistributed(self, value):
        self.assertTrue(stats.kstest(value.ravel(), 'uniform'),
                        msg="Sampled distribution is not uniformal")

    def assertNormalyDistributed(self, value):
        self.assertTrue(stats.mstats.normaltest(value.ravel()),
                        msg="Sampled distribution is not normal")


class ConstantInitializationTestCase(BaseInitializerTestCase):
    def test_constant_initializer(self):
        const = init.Constant(value=0)
        np.testing.assert_array_almost_equal(
            self.eval(const.sample(shape=(2, 3))),
            np.zeros((2, 3))
        )

        const = init.Constant(value=1.5)
        np.testing.assert_array_almost_equal(
            self.eval(const.sample(shape=(2, 3))),
            np.ones((2, 3)) * 1.5
        )

        const = init.Constant(value=1.5)
        np.testing.assert_array_almost_equal(
            const.sample(shape=(2, 3), return_array=True),
            np.ones((2, 3)) * 1.5
        )

    def test_constant_initialize_repr(self):
        const_initializer = init.Constant(value=3)
        self.assertEqual("Constant(3)", str(const_initializer))


class NormalInitializeTestCase(BaseInitializerTestCase):
    def test_normal_initializer(self):
        norm = init.Normal(mean=0, std=0.01)
        weight = self.eval(norm.sample((30, 30)))
        self.assertNormalyDistributed(weight)

        weight = norm.sample((30, 30), return_array=True)
        self.assertNormalyDistributed(weight)

    def test_normal_reprodusible_with_outside_seed(self):
        norm = init.Normal(mean=0, std=0.01)

        np.random.seed(0)
        weight1 = norm.sample((10, 4), return_array=True)

        np.random.seed(0)
        weight2 = norm.sample((10, 4), return_array=True)

        np.testing.assert_array_almost_equal(weight1, weight2)

    def test_normal_initialize_repr(self):
        hormal_initializer = init.Normal(mean=0, std=0.01)
        self.assertEqual("Normal(mean=0, std=0.01)", str(hormal_initializer))


class UniformInitializeTestCase(BaseInitializerTestCase):
    def test_uniformal_initializer(self):
        uniform = init.Uniform(minval=-10, maxval=10)
        weight = self.eval(uniform.sample((30, 30)))

        self.assertUniformlyDistributed(weight)
        self.assertAlmostEqual(-10, np.min(weight), places=1)
        self.assertAlmostEqual(10, np.max(weight), places=1)

        weight = uniform.sample((30, 30), return_array=True)
        self.assertUniformlyDistributed(weight)
        self.assertAlmostEqual(-10, np.min(weight), places=1)
        self.assertAlmostEqual(10, np.max(weight), places=1)

    def test_uniform_reprodusible_with_outside_seed(self):
        uniform = init.Uniform(minval=-10, maxval=10)

        np.random.seed(0)
        weight1 = uniform.sample((10, 4), return_array=True)

        np.random.seed(0)
        weight2 = uniform.sample((10, 4), return_array=True)

        np.testing.assert_array_almost_equal(weight1, weight2)

    def test_uniform_initializer_repr(self):
        uniform_initializer = init.Uniform(minval=0, maxval=1)
        self.assertEqual("Uniform(0, 1)", str(uniform_initializer))


class InitializerWithGainTestCase(BaseInitializerTestCase):
    def test_gain_relu(self):
        he_initializer = init.HeNormal(gain='relu')
        self.assertEqual(he_initializer.gain, math.sqrt(2))

    def test_gain_relu_he_normal_scale(self):
        he_initializer = init.HeNormal(gain=1, seed=0)
        sample_1 = self.eval(he_initializer.sample((4, 4)))

        he_initializer = init.HeNormal(gain='relu', seed=0)
        sample_2 = self.eval(he_initializer.sample((4, 4)))

        self.assertAlmostEqual(
            np.mean(sample_2 / sample_1),
            math.sqrt(2),
            places=5
        )


class HeInitializeTestCase(BaseInitializerTestCase):
    def test_he_normal(self):
        he_normal = init.HeNormal()
        weight = self.eval(he_normal.sample((40, 40)))

        self.assertNormalyDistributed(weight)
        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertAlmostEqual(weight.std(), math.sqrt(2. / 40),
                               places=2)

    def test_he_uniform(self):
        n_inputs = 30
        bound = math.sqrt(6. / n_inputs)

        he_uniform = init.HeUniform()
        weight = self.eval(he_uniform.sample((n_inputs, 30)))

        self.assertUniformlyDistributed(weight)
        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertGreaterEqual(weight.min(), -bound)
        self.assertLessEqual(weight.max(), bound)

    def test_he_initializer_repr(self):
        he_initializer = init.HeNormal()
        self.assertEqual("HeNormal(gain=1.0)", str(he_initializer))


class XavierInitializeTestCase(BaseInitializerTestCase):
    def test_xavier_normal(self):
        n_inputs, n_outputs = 30, 30

        xavier_normal = init.XavierNormal()
        weight = self.eval(xavier_normal.sample((n_inputs, n_outputs)))

        self.assertNormalyDistributed(weight)
        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertAlmostEqual(
            weight.std(),
            math.sqrt(2. / (n_inputs + n_outputs)),
            places=2)

    def test_xavier_uniform(self):
        n_inputs, n_outputs = 10, 30

        xavier_uniform = init.XavierUniform()
        weight = self.eval(xavier_uniform.sample((n_inputs, n_outputs)))

        bound = math.sqrt(6. / (n_inputs + n_outputs))

        self.assertUniformlyDistributed(weight)
        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertGreaterEqual(weight.min(), -bound)
        self.assertLessEqual(weight.max(), bound)


class OrthogonalInitializeTestCase(BaseInitializerTestCase):
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
        weight = self.eval(ortho.sample((30, 10)))
        np.testing.assert_array_almost_equal(
            np.eye(10),
            weight.T.dot(weight),
            decimal=5
        )

        ortho = init.Orthogonal(scale=1)
        # Matrix that have more columns than rows
        weight = self.eval(ortho.sample((10, 30)))
        np.testing.assert_array_almost_equal(
            np.eye(10),
            weight.dot(weight.T),
            decimal=5
        )

    def test_orthogonal_init_repr(self):
        ortho_initializer = init.Orthogonal(scale=1)
        self.assertEqual("Orthogonal(scale=1)", str(ortho_initializer))

    def test_orthogonal_1d_shape(self):
        ortho = init.Orthogonal(scale=1)
        sampled_data = self.eval(ortho.sample(shape=(1,)))
        self.assertEqual((1,), sampled_data.shape)
