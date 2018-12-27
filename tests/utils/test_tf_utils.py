import pytest
import numpy as np

from neupy.utils import tf_utils

from base import BaseTestCase


@pytest.mark.parametrize("in_shape,out_shape", [
    ((10,), (10,)),
    ((10, 2), (20,)),
    ((10, 2, 4), (80,)),
])
def test_flatten(in_shape, out_shape):
    X = np.random.random(in_shape)
    Y = tf_utils.tensorflow_eval(tf_utils.flatten(X))
    assert Y.shape == out_shape


class TFUtilsTestCase(BaseTestCase):
    def test_outer(self):
        pass

    def test_dot(self):
        pass

    def test_make_single_vector(self):
        pass

    def test_setup_parameter_updates(self):
        pass

    def test_function_name_scope(self):
        pass

    def test_class_method_name_scope(self):
        pass

    def test_function(self):
        pass

    def test_tensorflow_session_function(self):
        pass

    def test_initialize_uninitialized_variables(self):
        pass
