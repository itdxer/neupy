import numpy as np
import theano.tensor as T

from neupy import init
from neupy.utils import as_tuple
from neupy.core.properties import (TypedListProperty, IntProperty,
                                   ParameterProperty)
from .base import BaseLayer


__all__ = ('Reshape', 'Embedding')


class Reshape(BaseLayer):
    """
    Gives a new shape to an input value without changing
    its data.

    Parameters
    ----------
    shape : tuple or list
        New feature shape. ``None`` value means that feature
        will be flatten in 1D vector. If you need to get the
        output feature with more that 2 dimensions then you can
        set up new feature shape using tuples. Defaults to ``None``.
    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    shape = TypedListProperty()

    def __init__(self, shape=None, **options):
        if shape is not None:
            options['shape'] = shape
        super(Reshape, self).__init__(**options)

    @property
    def output_shape(self):
        if self.shape is not None:
            return as_tuple(self.shape)

        n_output_features = np.prod(self.input_shape)
        return as_tuple(n_output_features)

    def output(self, input_value):
        """
        Reshape the feature space for the input value.

        Parameters
        ----------
        input_value : array-like or Theano variable
        """
        n_samples = input_value.shape[0]
        output_shape = as_tuple(n_samples, self.output_shape)
        return T.reshape(input_value, output_shape)


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
        Defaults to :class:`XavierNormal() <neupy.core.init.XavierNormal>`.
    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    input_size = IntProperty(minval=1)
    output_size = IntProperty(minval=1)
    weight = ParameterProperty(default=init.XavierNormal())

    def __init__(self, input_size, output_size, **options):
        super(Embedding, self).__init__(input_size=input_size,
                                        output_size=output_size, **options)

    @property
    def output_shape(self):
        return as_tuple(self.output_size)

    def initialize(self):
        super(Embedding, self).initialize()
        self.add_parameter(
            value=self.weight, name='weight',
            shape=as_tuple(self.input_size, self.output_size)
        )

    def output(self, input_value):
        return self.weight[input_value]

    def __repr__(self):
        classname = self.__class__.__name__
        return '{name}({input_size}, {output_size})'.format(
            name=classname,
            input_size=self.input_size,
            output_size=self.output_size,
        )
