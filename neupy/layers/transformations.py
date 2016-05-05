import numpy as np
import theano.tensor as T

from neupy.utils import as_tuple, cached_property
from neupy.core.properties import ProperFractionProperty, TypedListProperty
from .base import BaseLayer


__all__ = ('Dropout', 'Reshape')


class Dropout(BaseLayer):
    """ Dropout layer

    Parameters
    ----------
    proba : float
        Fraction of the input units to drop. Value needs to be
        between 0 and 1.
    """
    proba = ProperFractionProperty(required=True)

    def __init__(self, proba, **options):
        options['proba'] = proba
        super(Dropout, self).__init__(**options)

    @cached_property
    def size(self):
        return self.relate_to_layer.size

    def output(self, input_value):
        # Use NumPy seed to make Theano code easely reproducible
        max_possible_seed = 4e9
        seed = np.random.randint(max_possible_seed)
        theano_random = T.shared_randomstreams.RandomStreams(seed)

        proba = (1.0 - self.proba)
        mask = theano_random.binomial(n=1, p=proba,
                                      size=input_value.shape,
                                      dtype=input_value.dtype)
        return (mask * input_value) / proba

    def __repr__(self):
        return "{name}(proba={proba})".format(
            name=self.__class__.__name__,
            proba=self.proba
        )


class Reshape(BaseLayer):
    """ Gives a new shape to an input value without changing
    its data.

    Parameters
    ----------
    shape : tuple or list
        New feature shape. ``None`` value means that feature
        will be flatten in 1D vector. If you need to get the
        output feature with more that 2 dimensions then you can
        set up new feature shape using tuples. Defaults to ``None``.
    """
    shape = TypedListProperty()

    def __init__(self, shape=None, **options):
        if shape is not None:
            options['shape'] = shape
        super(Reshape, self).__init__(**options)

    @cached_property
    def output_shape(self):
        if self.shape is not None:
            return self.shape

        output_shape = self.input_shape[1:]
        return np.prod(output_shape)

    def output(self, input_value):
        """ Reshape the feature space for the input value.

        Parameters
        ----------
        input_value : array-like or Theano variable
        """
        output_shape = as_tuple(input_value.shape[0], self.output_shape)
        return T.reshape(input_value, output_shape)
