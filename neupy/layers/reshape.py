import numpy as np
import theano.tensor as T

from neupy.utils import as_tuple
from neupy.core.properties import TypedListProperty
from .base import BaseLayer


__all__ = ('Reshape',)


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

    Examples
    --------

    Covert 4D input to 2D

    >>> from neupy import layers
    >>>
    >>> connection = layers.join(
    ...     layers.Input((2, 5, 5)),
    ...     layers.Reshape()
    ... )
    >>>
    >>> print("Input shape: {{}}".format(connection.input_shape))
    Input shape: (2, 5, 5)
    >>>
    >>> print("Output shape: {{}}".format(connection.output_shape))
    Output shape: (50,)

    Convert 3D to 4D

    >>> from neupy import layers
    >>>
    >>> connection = layers.join(
    ...     layers.Input((5, 4)),
    ...     layers.Reshape((5, 2, 2))
    ... )
    >>>
    >>> print("Input shape: {{}}".format(connection.input_shape))
    Input shape: (5, 4)
    >>>
    >>> print("Output shape: {{}}".format(connection.output_shape))
    Output shape: (5, 2, 2)
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
