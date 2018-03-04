import numpy as np
import pandas as pd
from sklearn import preprocessing
from neupy import algorithms

from base import BaseTestCase
from data import lenses


data = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
])
lenses = np.array([
    [1, 1, 1, 1, 1, 3],
    [2, 1, 1, 1, 2, 2],
    [3, 1, 1, 2, 1, 3],
    [4, 1, 1, 2, 2, 1],
    [5, 1, 2, 1, 1, 3],
    [6, 1, 2, 1, 2, 2],
    [7, 1, 2, 2, 1, 3],
    [8, 1, 2, 2, 2, 1],
    [9, 2, 1, 1, 1, 3],
    [10, 2, 1, 1, 2, 2],
    [11, 2, 1, 2, 1, 3],
    [12, 2, 1, 2, 2, 1],
    [13, 2, 2, 1, 1, 3],
    [14, 2, 2, 1, 2, 2],
    [15, 2, 2, 2, 1, 3],
    [16, 2, 2, 2, 2, 3],
    [17, 3, 1, 1, 1, 3],
    [18, 3, 1, 1, 2, 3],
    [19, 3, 1, 2, 1, 3],
    [20, 3, 1, 2, 2, 1],
    [21, 3, 2, 1, 1, 3],
    [22, 3, 2, 1, 2, 2],
    [23, 3, 2, 2, 1, 3],
    [24, 3, 2, 2, 2, 3],
])


class ARTTestCase(BaseTestCase):
    def test_art_handle_errors(self):
        with self.assertRaises(ValueError):
            # Invalid input data dimension
            artnet = algorithms.ART1(step=0.4, rho=0.1, n_clusters=3,
                                     verbose=False)
            artnet.predict(np.array([[[0]]]))

        with self.assertRaises(ValueError):
            # Non-binary input values
            artnet = algorithms.ART1(step=0.4, rho=0.1, n_clusters=3,
                                     verbose=False)
            artnet.predict(np.array([[0.5]]))

        with self.assertRaises(ValueError):
            # Invalid data size for second input
            artnet = algorithms.ART1(step=0.4, rho=0.1, n_clusters=3,
                                     verbose=False)
            artnet.predict(np.array([[0]]))
            artnet.predict(np.array([[0, 1]]))

    def test_simple_art1(self):
        ann = algorithms.ART1(step=2, rho=0.7, n_clusters=2, verbose=False)
        classes = ann.predict(data)

        for answer, result in zip([0, 1, 1], classes):
            self.assertEqual(result, answer)

        self.assertPickledNetwork(ann, data)

    def test_art1_on_real_problem(self):
        data = pd.DataFrame(lenses)

        encoder = preprocessing.OneHotEncoder()
        enc_data = encoder.fit_transform(data.values[:, 1:]).toarray()

        artnet = algorithms.ART1(step=1.5, rho=0.7, n_clusters=3,
                                 verbose=False)
        classes = artnet.predict(enc_data)

        unique_classes = list(np.sort(np.unique(classes)))
        self.assertEqual(unique_classes, [0, 1, 2])
