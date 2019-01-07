import tensorflow as tf

from neupy.core.properties import ProperFractionProperty, NumberProperty
from .base import Identity


__all__ = ('Dropout', 'GaussianNoise')


class Dropout(Identity):
    """
    Dropout layer

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
    """
    proba = ProperFractionProperty()

    def __init__(self, proba, name=None):
        super(Dropout, self).__init__(name=name)
        self.proba = proba

    def output(self, input_value, training=False):
        if not training:
            return input_value
        return tf.nn.dropout(input_value, keep_prob=(1.0 - self.proba))

    def __repr__(self):
        classname = self.__class__.__name__
        return "{}(proba={})".format(classname, self.proba)


class GaussianNoise(Identity):
    """
    Add gaussian noise to the input value. Mean and standard
    deviation are layer's parameters.

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
    """
    std = NumberProperty(minval=0)
    mean = NumberProperty()

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

    def __repr__(self):
        classname = self.__class__.__name__
        return "{}(mean={}, std={})".format(classname, self.mean, self.std)
