import numpy as np

from neupy import datasets

from base import BaseTestCase


class DiscreteDigitsDatasetTestCase(BaseTestCase):
    def test_load_digits(self):
        data, labels = datasets.load_digits()

        self.assertEqual(data.shape, (10, 24))
        self.assertEqual(labels.shape, (10,))

    def test_make_digits_exceptions(self):
        with self.assertRaises(ValueError):
            datasets.make_digits(noise_level=-1)

        with self.assertRaises(ValueError):
            datasets.make_digits(noise_level=1)

        with self.assertRaises(ValueError):
            datasets.make_digits(n_samples=0)

    def test_make_digits(self):
        data, labels = datasets.load_digits()
        noisy_data, noisy_labels = datasets.make_digits(noise_level=0.3,
                                                        n_samples=1)

        diff = np.abs(data[noisy_labels] - noisy_data).sum()
        self.assertNotEqual(diff, 0)
