import tensorflow as tf

from neupy import init
from neupy.utils import as_tuple
from neupy.core.properties import IntProperty, ParameterProperty
from .base import BaseLayer


__all__ = ('Embedding',)


class Embedding(BaseLayer):
    """
    Embedding layer accepts indices as an input and returns
    rows from the weight matrix associated with these indices.
    It's useful when inputs are categorical features or for the
    word embedding tasks.

    Parameters
    ----------
    input_size : int
        Layer's input vector dimension. It's, typically, associated with
        number of categories or number of unique words that input vector has.

    output_size : int
        Layer's output vector dimension.

    weight : array-like, Tensorfow variable, scalar or Initializer
        Defines layer's weights. Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    {BaseLayer.name}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}

    Examples
    --------

    This example converts dataset that has only categorical
    variables into format that suitable for Embedding layer.

    >>> import numpy as np
    >>> from neupy.layers import *
    >>>
    >>> dataset = np.array([
    ...     ['cold', 'high'],
    ...     ['hot',  'low'],
    ...     ['cold', 'low'],
    ...     ['hot',  'low'],
    ... ])
    >>>
    >>> unique_value, dataset_indices = np.unique(
    ...     dataset, return_inverse=True
    ... )
    >>> dataset_indices = dataset_indices.reshape((4, 2))
    >>> dataset_indices
    array([[0, 1],
           [2, 3],
           [0, 3],
           [2, 3]])
    >>>
    >>> n_features = dataset.shape[1]
    >>> n_unique_categories = len(unique_value)
    >>> embedded_size = 1
    >>>
    >>> network = join(
    ...     Input(n_features),
    ...     Embedding(n_unique_categories, embedded_size),
    ...     # Output from the embedding layer is 3D
    ...     # To make output 2D we need to reshape dimensions
    ...     Reshape(),
    ... )
    """
    input_size = IntProperty(minval=1)
    output_size = IntProperty(minval=1)
    weight = ParameterProperty()

    def __init__(self, input_size, output_size,
                 weight=init.HeNormal(), name=None):

        super(Embedding, self).__init__(name=name)

        self.input_size = input_size
        self.output_size = output_size
        self.weight = weight

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape.concatenate(self.output_size)

    def create_variables(self, input_shape):
        self.input_shape = input_shape
        self.weight = self.variable(
            value=self.weight, name='weight',
            shape=as_tuple(self.input_size, self.output_size))

    def output(self, input_value, **kwargs):
        input_value = tf.cast(input_value, tf.int32)
        return tf.gather(self.weight, input_value)

    def __repr__(self):
        return self._repr_arguments(
            self.input_size,
            self.output_size,
            name=self.name,
            weight=self.weight,
        )
