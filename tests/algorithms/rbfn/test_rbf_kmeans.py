import numpy as np

from neupy import algorithms

from base import BaseTestCase


data = np.array([
    [0.11, 0.20],
    [0.25, 0.32],
    [0.64, 0.60],
    [0.12, 0.42],
    [0.70, 0.73],
    [0.30, 0.27],
    [0.43, 0.81],
    [0.44, 0.87],
    [0.12, 0.92],
    [0.56, 0.67],
    [0.36, 0.35],
])


class RBFKMeansTestCase(BaseTestCase):
    def test_rbfk_exceptions(self):
        with self.assertRaises(ValueError):
            # More clusters than samples
            nw = algorithms.RBFKMeans(n_clusters=1000, verbose=False)
            nw.train(data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # Number of clusters the same as number of samples
            nw = algorithms.RBFKMeans(n_clusters=data.shape[0],
                                      verbose=False)
            nw.train(data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # One cluster
            nw = algorithms.RBFKMeans(n_clusters=1, verbose=False)
            nw.train(data, epsilon=1e-5)

    def test_rbfk_classification(self):
        expected_centers = np.array([
            [0.228, 0.312],
            [0.482,  0.767],
        ])

        nw = algorithms.RBFKMeans(n_clusters=2, verbose=False)
        nw.train(data, epsilon=1e-5)
        np.testing.assert_array_almost_equal(expected_centers, nw.centers,
                                             decimal=3)

    def test_rbfk_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.RBFKMeans(n_clusters=2, verbose=False),
            np.array([1, 2, 10]),
        )

    def test_rbfk_predict_different_inputs(self):
        kmnet = algorithms.RBFKMeans(verbose=False, n_clusters=2)

        data = np.array([[1, 2, 10]]).T
        target = np.array([[0, 0, 1]]).T

        kmnet.train(data)
        self.assertInvalidVectorPred(kmnet, data.ravel(), target, decimal=2)

    def test_rbfk_means_assign_step_exception(self):
        with self.assertRaises(ValueError):
            algorithms.RBFKMeans(n_cluster=2, step=0.01)
