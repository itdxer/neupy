import tensorflow as tf

from neupy.core.properties import ProperFractionProperty, NumberProperty
from .base import Identity


__all__ = ('Dropout', 'GaussianNoise')


class Dropout(Identity):
    """
    Dropout layer. It randomly switches of (multiplies by zero)
    input values, where probability to be switched per each value
    can be controlled with the ``proba`` parameter. For example,
    ``proba=0.2`` will mean that only 20% of the input values will
    be multiplied by 0 and 80% of the will be unchanged.

    It's important to note that output from the dropout is controled by
    the ``training`` parameter in the ``output`` method. Droput
    will be applied only in cases when ``training=True`` propagated
    through the network, otherwise it will act as an identity.

    Parameters
    ----------
    proba : float
        Fraction of the input units to drop. Value needs to be
        between ``0`` and ``1``.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = join(
    ...     Input(10),
    ...     Relu(5) >> Dropout(0.5),
    ...     Relu(5) >> Dropout(0.5),
    ...     Sigmoid(1),
    ... )
    >>> network
    (?, 10) -> [... 6 layers ...] -> (?, 1)
    """
    proba = ProperFractionProperty()

    def __init__(self, proba, name=None):
        super(Dropout, self).__init__(name=name)
        self.proba = proba

    def output(self, input_value, training=False):
        if not training:
            return input_value
        return tf.nn.dropout(input_value, keep_prob=(1.0 - self.proba))


class GaussianNoise(Identity):
    """
    Add gaussian noise to the input value. Mean and standard deviation
    of the noise can be controlled from the layers parameters.

    It's important to note that output from the layer is controled by
    the ``training`` parameter in the ``output`` method. Layer
    will be applied only in cases when ``training=True`` propagated
    through the network, otherwise it will act as an identity.

    Parameters
    ----------
    std : float
        Standard deviation of the gaussian noise. Values needs to
        be greater than zero. Defaults to ``1``.

    mean : float
        Mean of the gaussian noise. Defaults to ``0``.

    {Identity.name}

    Methods
    -------
    {Identity.Methods}

    Attributes
    ----------
    {Identity.Attributes}

    Examples
    --------
    >>> from neupy.layers import *
    >>> network = join(
    ...     Input(10),
    ...     Relu(5) >> GaussianNoise(std=0.1),
    ...     Relu(5) >> GaussianNoise(std=0.1),
    ...     Sigmoid(1),
    ... )
    >>> network
    (?, 10) -> [... 6 layers ...] -> (?, 1)
    """
    mean = NumberProperty()
    std = NumberProperty(minval=0)

    def __init__(self, mean=1, std=0, name=None):
        super(GaussianNoise, self).__init__(name=name)
        self.mean = mean
        self.std = std

    def output(self, input_value, training=False):
        if not training:
            return input_value

        noise = tf.random_normal(
            shape=tf.shape(input_value),
            mean=self.mean,
            stddev=self.std)

        return input_value + noise
