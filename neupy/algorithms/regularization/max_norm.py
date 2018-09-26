import tensorflow as tf

from neupy.utils import asfloat
from neupy.layers.utils import iter_parameters
from neupy.core.properties import NumberProperty
from .base import WeightUpdateConfigurable


__all__ = ('MaxNormRegularization',)


def max_norm_clip(array, max_norm):
    """
    Clips norm of the array in case if it will exceed
    specified limit.

    Parameters
    ----------
    array : array-like
        Any array-like object.

    max_norm : float
        Maximum possible norm value for the input array. Array will
        be clipped in case if its norm will be greater than
        specified limit.
    """
    with tf.name_scope('max-norm-clip'):
        array_norm = tf.norm(array)
        max_norm = asfloat(max_norm)

        return tf.where(
            tf.greater_equal(array_norm, max_norm),
            tf.multiply(max_norm, array) / array_norm,
            array,
        )


class MaxNormRegularization(WeightUpdateConfigurable):
    """
    Max-norm regularization algorithm will clip norm of the
    parameter in case if it will exceed maximum limit.

    .. code-block:: python

        if norm(weight) > max_norm:
            weight = max_norm * weight / norm(weight)

    .. raw:: html

        <br>

    Warns
    -----
    {WeightUpdateConfigurable.Warns}

    Parameters
    ----------
    max_norm : int, float
        Any parameter that has norm greater than this value
        will be clipped. Defaults to ``10``.

    Examples
    --------
    >>> from neupy import algorithms
    >>> bpnet = algorithms.GradientDescent(
    ...     (2, 4, 1),
    ...     step=0.1,
    ...     max_norm=4,
    ...     addons=[algorithms.MaxNormRegularization]
    ... )

    References
    ----------
    [1] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever,
      R. Salakhutdinov. Dropout: A Simple Way to Prevent
      Neural Networks from Overfitting.
      http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """
    max_norm = NumberProperty(default=10, minval=0)

    def init_train_updates(self):
        parameters = [param for _, _, param in iter_parameters(self.layers)]
        original_updates = super(
            MaxNormRegularization, self).init_train_updates()

        modified_updates = []

        for parameter, updated in original_updates:
            if parameter in parameters:
                updated = max_norm_clip(updated, self.max_norm)
            modified_updates.append((parameter, updated))

        return modified_updates
