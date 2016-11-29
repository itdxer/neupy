import numpy as np
import theano
import theano.tensor as T
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

from neupy import algorithms, layers
from neupy.utils import asfloat
from neupy.algorithms.gd.lev_marq import compute_jacobian

from base import BaseTestCase


class LevenbergMarquardtTestCase(BaseTestCase):
    def test_jacobian_for_levenberg_marquardt(self):
        w1 = theano.shared(name='w1', value=asfloat(np.array([[1]])))
        b1 = theano.shared(name='b1', value=asfloat(np.array([0])))
        w2 = theano.shared(name='w2', value=asfloat(np.array([[2]])))
        b2 = theano.shared(name='b2', value=asfloat(np.array([1])))

        x = T.matrix('x')
        y = T.matrix('y')
        output = ((x.dot(w1.T) + b1) ** 2).dot(w2.T) + b2
        error_func = T.mean((y - output), axis=1)

        x_train = asfloat(np.array([[1, 2, 3]]).T)
        y_train = asfloat(np.array([[1, 2, 3]]).T)
        output_expected = asfloat(np.array([[3, 9, 19]]).T)

        np.testing.assert_array_almost_equal(
            output.eval({x: x_train}),
            output_expected
        )

        jacobian_expected = asfloat(np.array([
            [-4, -4, -1, -1],
            [-16, -8, -4, -1],
            [-36, -12, -9, -1],
        ]))
        jacobian_actual = compute_jacobian(error_func, [w1, b1, w2, b2])
        np.testing.assert_array_almost_equal(
            jacobian_expected,
            jacobian_actual.eval({x: x_train, y: y_train})
        )

    def test_levenberg_marquardt_invalid_error_exceptions(self):
        with self.assertRaises(ValueError):
            algorithms.LevenbergMarquardt((2, 3, 1),
                                          error='categorical_crossentropy')

    def test_levenberg_marquardt(self):
        dataset = datasets.make_regression(n_samples=50, n_features=2)
        data, target = dataset

        data_scaler = preprocessing.MinMaxScaler()
        target_scaler = preprocessing.MinMaxScaler()

        x_train, x_test, y_train, y_test = train_test_split(
            data_scaler.fit_transform(data),
            target_scaler.fit_transform(target.reshape(-1, 1)),
            train_size=0.85
        )

        lmnet = algorithms.LevenbergMarquardt(
            connection=[
                layers.Input(2),
                layers.Sigmoid(6),
                layers.Sigmoid(1),
            ],
            mu_update_factor=2,
            mu=0.1,
            verbose=False,
            show_epoch=1,
        )
        lmnet.train(x_train, y_train, epochs=4)
        error = lmnet.prediction_error(x_test, y_test)

        self.assertAlmostEqual(0.006, error, places=3)

    def test_levenberg_marquardt_assign_step_exception(self):
        with self.assertRaises(ValueError):
            algorithms.LevenbergMarquardt((2, 3, 1), step=0.01)
