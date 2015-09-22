from abc import abstractmethod

from neupy.core.config import ConfigurableWithABC


__all__ = ('Regression', 'Classification', 'Clustering')


class Regression(object):
    """ Mixin for regression.
    """


class Classification(ConfigurableWithABC):
    """ Mixin for classification.
    """
    @abstractmethod
    def predict_prob(self):
        """ Predict probabilities.
        """


class Clustering(object):
    """ Mixin for clustering.
    """
