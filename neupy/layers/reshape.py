import numpy as np
import tensorflow as tf

from neupy.utils import as_tuple
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import TypedListProperty
from .base import BaseLayer


__all__ = ('Reshape', 'Transpose')


class Reshape(BaseLayer):
    """
    Layer reshapes input tensor.

    Parameters
    ----------
    shape : tuple
        New feature shape. If one dimension specified with the ``-1`` value
        that this dimension will be computed from the total size that remains.
        Defaults to ``-1``.

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
    >>> network = Input((2, 5, 5)) >> Reshape()
    (?, 2, 5, 5) -> [... 2 layers ...] -> (?, 50)

    Convert 3D to 4D

    >>> from neupy.layers import *
    >>> network = Input((5, 4)) >> Reshape((5, 2, 2))
    (?, 5, 4) -> [... 2 layers ...] -> (?, 5, 2, 2)
    """
    shape = TypedListProperty()

    def __init__(self, shape=-1, name=None):
        super(Reshape, self).__init__(name=name)
        self.shape = as_tuple(shape)

        if self.shape.count(-1) >= 2:
            raise ValueError("Only single -1 value can be specified")

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        feature_shape = input_shape[1:]
        missing_value = None

        if -1 in self.shape and feature_shape.is_fully_defined():
            known_shape_values = [val for val in self.shape if val != -1]

            n_feature_values = np.prod(feature_shape.dims)
            n_expected_values = np.prod(known_shape_values)

            if n_feature_values % n_expected_values != 0:
                raise ValueError(
                    "Input shape and specified shape are incompatible Shape: "
                    "{}, Input shape: {}".format(self.shape, input_shape))

            missing_value = int(n_feature_values // n_expected_values)

        n_samples = input_shape[0]
        new_feature_shape = [
            missing_value if val == -1 else val for val in self.shape]

        return tf.TensorShape([n_samples] + new_feature_shape)

    def output(self, input, **kwargs):
        """
        Reshape the feature space for the input value.

        Parameters
        ----------
        input : array-like or Tensorfow variable
        """
        input = tf.convert_to_tensor(input, dtype=tf.float32)
        input_shape = tf.shape(input)

        n_samples = input_shape[0]
        expected_shape = self.get_output_shape(input.shape)
        feature_shape = expected_shape[1:]

        if feature_shape.is_fully_defined():
            # For cases when we have -1 in the shape and feature shape
            # can be precomputed from the input we want to be explicit about
            # expected output shape. Because of the unknown batch dimension
            # it won't be possible for tensorflow to derive exact output
            # shape from the -1
            output_shape = as_tuple(n_samples, feature_shape.dims)
        else:
            output_shape = as_tuple(n_samples, self.shape)

        return tf.reshape(input, output_shape)

    def __repr__(self):
        return self._repr_arguments(self.shape, name=self.name)


class Transpose(BaseLayer):
    """
    Layer transposes input tensor. Permutes the dimensions according
    to the ``perm`` parameter.

    Parameters
    ----------
    perm : tuple or list
        A permutation of the dimensions of the input tensor.

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
    >>> network = Input((7, 11)) >> Transpose((0, 2, 1))
    (?, 7, 11) -> [... 2 layers ...] -> (?, 11, 7)
    """
    perm = TypedListProperty()

    def __init__(self, perm, name=None):
        super(Transpose, self).__init__(name=name)
        self.perm = perm

    def fail_if_shape_invalid(self, input_shape):
        if input_shape and max(self.perm) >= input_shape.ndims:
            raise LayerConnectionError(
                "Cannot apply transpose operation to the "
                "input. Permutation: {}, Input shape: {}"
                "".format(self.perm, input_shape))

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.fail_if_shape_invalid(input_shape)

        if input_shape.ndims is None:
            n_dims_expected = len(self.perm)
            return tf.TensorShape([None] * n_dims_expected)

        input_shape = np.array(input_shape.dims)
        perm = list(self.perm)

        return tf.TensorShape(input_shape[perm])

    def output(self, input, **kwargs):
        # Input value has batch dimension, but perm will never have it
        # specified as (zero index), so we need to add it in order to
        # fix batch dimension in place.
        return tf.transpose(input, list(self.perm))

    def __repr__(self):
        return self._repr_arguments(self.perm, name=self.name)
