import numpy as np

from neupy import datasets

from base import BaseTestCase


class DiscreteDigitsDatasetTestCase(BaseTestCase):
    def test_load_digits(self):
        data, labels = datasets.load_digits()

        self.assertEqual(data.shape, (10, 24))
        self.assertEqual(labels.shape, (10,))

    def test_make_digits_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "from \[0\, 1\) range"):
            datasets.make_digits(noise_level=-1)

        with self.assertRaisesRegexp(ValueError, "from \[0\, 1\) range"):
            datasets.make_digits(noise_level=1)

        with self.assertRaisesRegexp(ValueError, "greater or equal to 1"):
            datasets.make_digits(n_samples=0)

        with self.assertRaisesRegexp(ValueError, "Unknown mode"):
            datasets.make_digits(n_samples=0, mode='unknown')

    def test_make_digits_remove_mode(self):
        data, labels = datasets.load_digits()
        noisy_data, noisy_labels = datasets.make_digits(
            noise_level=0.3, n_samples=100, mode='remove')

        diff = data[noisy_labels] - noisy_data
        diff_signs = np.sign(diff).flatten()

        self.assertNotEqual(np.abs(diff).sum(), 0)
        self.assertEqual({0, 1}, set(diff_signs))

    def test_make_digits_flip_mode(self):
        data, labels = datasets.load_digits()
        noisy_data, noisy_labels = datasets.make_digits(
            noise_level=0.5, n_samples=100, mode='flip')

        diff_signs = np.sign(data[noisy_labels] - noisy_data).flatten()
        self.assertEqual({-1, 0, 1}, set(diff_signs))
