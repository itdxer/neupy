import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from neuralpy.algorithms import ART1

from base import BaseTestCase


BASEDIR = os.path.abspath(os.path.dirname(__file__))

data = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
])


class ARTTestCase(BaseTestCase):
    def test_art_handle_errors(self):
        with self.assertRaises(ValueError):
            # Invalid input data dimention
            artnet = ART1(step=0.4, rho=0.1, n_clusters=3)
            artnet.predict(np.array([0.5]))

        with self.assertRaises(ValueError):
            # Non-binary input values
            artnet = ART1(step=0.4, rho=0.1, n_clusters=3)
            artnet.predict(np.array([[0.5]]))

        with self.assertRaises(ValueError):
            # Invalid data size for second input
            artnet = ART1(step=0.4, rho=0.1, n_clusters=3)
            artnet.predict(np.array([[0.5]]))
            artnet.predict(np.array([[0.5, 1]]))

    def test_simple_art1(self):
        ann = ART1(step=2, rho=0.7, n_clusters=2)
        classes = ann.predict(data)

        for answer, result in zip([0, 1, 1], classes):
            self.assertEqual(result, answer)

    def test_art1_on_real_problem(self):
        path_to_data = os.path.join(BASEDIR, '..', 'data', 'lenses.csv')
        data = pd.read_csv(path_to_data, index_col=[0], sep=' ', header=None)

        encoder = preprocessing.OneHotEncoder()
        enc_data = encoder.fit_transform(data.values[:, 1:]).toarray()

        artnet = ART1(step=1.5, rho=0.7, n_clusters=3)
        classes = artnet.predict(enc_data)

        self.assertEqual(list(np.sort(np.unique(classes))), [0, 1, 2])
