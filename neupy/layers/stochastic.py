from neupy.core.properties import ProperFractionProperty, NumberProperty
from neupy.utils import theano_random_stream
from .base import BaseLayer


__all__ = ('Dropout', 'GaussianNoise')


class Dropout(BaseLayer):
    """
    Dropout layer

    Parameters
    ----------
    proba : float
        Fraction of the input units to drop. Value needs to be
        between ``0`` and ``1``.

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    proba = ProperFractionProperty(required=True)

    def __init__(self, proba, **options):
        super(Dropout, self).__init__(proba=proba, **options)

    def output(self, input_value):
        if not self.training_state:
            return input_value

        theano_random = theano_random_stream()
        proba = (1.0 - self.proba)
        mask = theano_random.binomial(n=1, p=proba,
                                      size=input_value.shape,
                                      dtype=input_value.dtype)
        return (mask * input_value) / proba

    def __repr__(self):
        classname = self.__class__.__name__
        return "{}(proba={})".format(classname, self.proba)


class GaussianNoise(BaseLayer):
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

    {BaseLayer.Parameters}

    Methods
    -------
    {BaseLayer.Methods}

    Attributes
    ----------
    {BaseLayer.Attributes}
    """
    std = NumberProperty(default=1, minval=0)
    mean = NumberProperty(default=0)

    def __init__(self, mean=1, std=0, **options):
        super(GaussianNoise, self).__init__(mean=mean, std=std, **options)

    def output(self, input_value):
        if not self.training_state:
            return input_value

        theano_random = theano_random_stream()
        noise = theano_random.normal(size=input_value.shape,
                                     avg=self.mean, std=self.std)
        return input_value + noise

    def __repr__(self):
        classname = self.__class__.__name__
        return "{}(mean={}, std={})".format(classname, self.mean, self.std)
