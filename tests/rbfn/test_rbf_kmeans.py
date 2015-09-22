import numpy as np
import matplotlib.pyplot as plt

from neupy.algorithms import RBFKMeans

from base import BaseTestCase


class RBFKMeansTestCase(BaseTestCase):
    def setUp(self):
        super(RBFKMeansTestCase, self).setUp()
        self.draw_plot = False
        self.data = np.array([
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

    def test_validation(self):
        with self.assertRaises(ValueError):
            # More clusters than samples
            nw = RBFKMeans(n_clusters=1000)
            nw.train(self.data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # Number of clusters the same as number of samples
            nw = RBFKMeans(n_clusters=self.data.shape[0])
            nw.train(self.data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # One cluster
            nw = RBFKMeans(n_clusters=1)
            nw.train(self.data, epsilon=1e-5)

    def test_classification(self):
        result = np.array([
            [0.228, 0.312],
            [0.48166667,  0.76666667],
        ])

        nw = RBFKMeans(n_clusters=2)
        nw.train(self.data, epsilon=1e-5)
        self.assertTrue(np.all(result == np.round(nw.centers, 8)))

        if self.draw_plot:
            classes = nw.predict(self.data)

            for i, center in enumerate(nw.centers):
                positions = np.argwhere(classes[:, 0] == i)
                class_data = np.take(self.data, positions[:, 0], axis=0)
                plt.plot(class_data[:, 0], class_data[:, 1], 'o')

            for center in nw.centers:
                plt.plot(center[0], center[1], 'kx')

            plt.axis([0, 1, 0, 1])
            plt.show()
