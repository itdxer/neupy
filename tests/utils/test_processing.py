import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from neupy.utils.processing import format_data, asfloat

from base import BaseTestCase


class ProcessingUtilsTestCase(BaseTestCase):
    def test_format_data(self):
        # None input
        self.assertEqual(format_data(None), None)

        # Sparse data
        sparse_matrix = csr_matrix((3, 4), dtype=np.int8)
        formated_sparce_matrix = format_data(sparse_matrix)
        self.assertIs(formated_sparce_matrix, sparse_matrix)
        self.assertEqual(formated_sparce_matrix.dtype, sparse_matrix.dtype)

        # Vector input
        x = np.random.random(10)
        formated_x = format_data(x, is_feature1d=True)
        self.assertEqual(formated_x.shape, (10, 1))

        x = np.random.random(10)
        formated_x = format_data(x, is_feature1d=False)
        self.assertEqual(formated_x.shape, (1, 10))

    def test_asfloat(self):
        # Sparse matrix
        sparse_matrix = csr_matrix((3, 4), dtype=np.int8)
        self.assertIs(sparse_matrix, asfloat(sparse_matrix))

        # Numpy array-like elements
        x = np.array([1, 2, 3], dtype=np.float32)
        self.assertIs(x, asfloat(x))

        x = np.array([1, 2, 3], dtype=np.int8)
        self.assertIsNot(x, asfloat(x))

        # Python list
        x = [1, 2, 3]
        self.assertEqual(asfloat(x).shape, (3,))

        # Tensorfow variables
        x = tf.placeholder(dtype=tf.int32)
        self.assertNotEqual(x.dtype, tf.float32)
        self.assertEqual(asfloat(x).dtype, tf.float32)
