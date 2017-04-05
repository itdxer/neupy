import theano.tensor as T

from neupy.core.properties import NumberProperty
from .base import WeightUpdateConfigurable


__all__ = ('MaxNormRegularization',)


def max_norm_clip(array, max_norm=3):
    array_norm = T.nlinalg.norm(array, ord=None)
    return T.switch(
        T.ge(array_norm, max_norm),
        T.mul(max_norm, array) / array_norm,
        array,
    )


class MaxNormRegularization(WeightUpdateConfigurable):
    """
    Max-norm regularization.

    Parameters
    ----------
    max_norm : int, float
    """
    max_norm = NumberProperty(default=3, minval=0)

    def init_param_updates(self, layer, parameter):
        updates = super(MaxNormRegularization, self).init_param_updates(
            layer, parameter)

        updates_mapper = dict(updates)
        updated_value = updates_mapper[parameter]
        updates_mapper[parameter] = max_norm_clip(updated_value, self.max_norm)

        return list(updates_mapper.items())
