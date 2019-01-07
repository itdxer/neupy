import numpy as np
import tensorflow as tf

from neupy.utils import as_tuple, tf_utils
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import TypedListProperty
from .base import BaseLayer


__all__ = ('Reshape', 'Transpose')


class Reshape(BaseLayer):
    """
    Reshapes input tensor.

    Parameters
    ----------
    shape : tuple or list
        New feature shape. The ``-1`` value means that this value
        will be computed from the total size that remains. If you need
        to get the output feature with more that 2 dimensions then you can
        set up new feature shape using tuples or list. Defaults to ``-1``.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------

    Covert 4D input to 2D

    >>> from neupy.layers import *
    >>> conn = Input((2, 5, 5)) >> Reshape()
    >>> conn.input_shape
    (2, 5, 5)
    >>> conn.output_shape
    (50,)

    Convert 3D to 4D

    >>> from neupy.layers import *
    >>> conn = Input((5, 4)) >> Reshape((5, 2, 2))
    >>> conn.input_shape
    (5, 4)
    >>> conn.output_shape
    (5, 2, 2)
    """
    shape = TypedListProperty()

    def __init__(self, shape=-1, name=None):
        super(Reshape, self).__init__(name=name)
        self.shape = as_tuple(shape)

        if self.shape.count(-1) >= 2:
            raise ValueError("Only single -1 value can be specified")

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        if -1 not in self.shape:
            return tf.TensorShape(self.shape)

        if input_shape.is_fully_defined():
            known_shape_values = [val for val in self.shape if val != -1]

            flatten_shape = np.prod(input_shape)
            expected_shape_parts = np.prod(known_shape_values)

            if flatten_shape % expected_shape_parts != 0:
                raise ValueError(
                    "Cannot derive values for shape {} from the input "
                    "shape {}".format(self.shape, input_shape))

            missing_value = int(flatten_shape // expected_shape_parts)
        else:
            missing_value = None

        return tf.TensorShape([
            missing_value if val == -1 else val for val in self.shape])

    def output(self, input_value):
        """
        Reshape the feature space for the input value.

        Parameters
        ----------
        input_value : array-like or Tensorfow variable
        """
        input_shape = tf.shape(input_value)
        n_samples = input_shape[0]
        output_shape = as_tuple(n_samples, self.shape)
        return tf.reshape(input_value, output_shape)


class Transpose(BaseLayer):
    """
    Transposes input. Permutes the dimensions according to ``perm``.

    Parameters
    ----------
    perm : tuple or list
        A permutation of the dimensions of the input tensor. Layer cannot
        transpose batch dimension and using ``0`` in the list of
        permuted dimensions is not allowed.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> conn = Input((7, 11)) >> Transpose((2, 1))
    >>> conn.input_shape
    (7, 11)
    >>> conn.output_shape
    (11, 7)
    """
    perm = TypedListProperty()

    def __init__(self, perm, name=None):
        super(Transpose, self).__init__(name=name)

        if 0 in perm:
            raise ValueError(
                "Batch dimension has fixed position and 0 "
                "index cannot be used.")

        self.perm = perm

    def fail_if_shape_invalid(self, input_shape):
        if len(input_shape) < 2:
            raise LayerConnectionError(
                "Transpose expects input with at least 3 dimensions.")

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.fail_if_shape_invalid(input_shape)

        # Input shape doesn't have information about the batch size and perm
        # indeces require to have this dimension on zero's position.
        input_shape = tf_utils.add_batch_dim(input_shape)
        input_shape = np.array(input_shape.dims)

        return tf.TensorShape(input_shape[self.perm])

    def output(self, input_value):
        # Input value has batch dimension, but perm will never have it
        # specified as (zero index), so we need to add it in order to
        # fix batch dimesnion in place.
        return tf.transpose(input_value, [0] + list(self.perm))
