from neupy import init
from neupy.utils import as_tuple, asint
from neupy.core.properties import IntProperty, ParameterProperty
from .base import BaseLayer


__all__ = ('Embedding',)


class Embedding(BaseLayer):
    """
    Embedding layer accepts indeces as an input and returns
    rows from the weight matrix associated with these indeces.
    Useful in case of categorical features or for the word
    embedding tasks.

    Parameters
    ----------
    input_size : int
        Layer's input vector dimension. Usualy associated with number
        of categories or number of unique words that input vector has.

    output_size : int
        Layer's output vector dimension.

    weight : array-like, Theano variable, scalar or Initializer
        Defines layer's weights. Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`XavierNormal() <neupy.init.XavierNormal>`.

    {BaseLayer.Parameters}

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
    >>> from neupy import layers
    >>>
    >>> dataset = np.array([
    ...     ['cold', 'high'],
    ...     ['hot',  'low'],
    ...     ['cold', 'low'],
    ...     ['hot',  'low'],
    ... ])
    >>>
    >>> unique_value, dataset_indeces = np.unique(
    ...     dataset, return_inverse=True
    ... )
    >>> dataset_indeces = dataset_indeces.reshape((4, 2))
    >>> dataset_indeces
    array([[0, 1],
           [2, 3],
           [0, 3],
           [2, 3]])
    >>>
    >>> n_features = dataset.shape[1]
    >>> n_unique_categories = len(unique_value)
    >>> embedded_size = 1
    >>>
    >>> connection = layers.join(
    ...     layers.Input(n_features),
    ...     layers.Embedding(n_unique_categories, embedded_size),
    ...     # Output from the embedding layer is 3D
    ...     # To make output 2D we need to reshape dimensions
    ...     layers.Reshape(),
    ... )
    """
    input_size = IntProperty(minval=1)
    output_size = IntProperty(minval=1)
    weight = ParameterProperty(default=init.XavierNormal())

    def __init__(self, input_size, output_size, **options):
        super(Embedding, self).__init__(input_size=input_size,
                                        output_size=output_size, **options)

    @property
    def output_shape(self):
        if self.input_shape is not None:
            return as_tuple(self.input_shape, self.output_size)

    def initialize(self):
        super(Embedding, self).initialize()
        self.add_parameter(
            value=self.weight, name='weight',
            shape=as_tuple(self.input_size, self.output_size),
            trainable=True)

    def output(self, input_value):
        return self.weight[asint(input_value)]

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({input_size}, {output_size})'.format(
            name=classname,
            input_size=self.input_size,
            output_size=self.output_size,
        )
