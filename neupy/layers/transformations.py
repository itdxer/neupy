import numpy as np
import theano.tensor as T

from neupy.core.properties import ProperFractionProperty
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

    @property
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
    """ Reshape layer makes a simple transformation that
    changes input data shape from tensor (>= 3 features) to
    matrix (2 features).
    """
    def output(self, input_value):
        input_shape = input_value.shape[0]
        output_shape = input_value.shape[1:]
        flattened_shape = (input_shape, T.prod(output_shape))
        return T.reshape(input_value, flattened_shape)
