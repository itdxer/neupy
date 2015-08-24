import math

from neuralpy.datasets import (make_reber, is_valid_by_reber,
                               make_reber_classification)

from base import BaseTestCase


class ReberTestCase(BaseTestCase):
    def test_reber_wrod_generation(self):
        words = make_reber(50)
        self.assertEqual(50, len(words))

        for word in words:
            self.assertTrue(is_valid_by_reber(word))

    def test_reber_classification_data(self):
        invalid_data_ratio = 0.5
        n_words = 100

        words, labels = make_reber_classification(
            n_words, invalid_size=invalid_data_ratio
        )

        self.assertEqual(n_words, len(labels))
        self.assertEqual(math.ceil(n_words * invalid_data_ratio), sum(labels))

        for word, label in zip(words, labels):
            self.assertEqual(bool(label), is_valid_by_reber(word))
