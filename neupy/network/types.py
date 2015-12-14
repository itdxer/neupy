from abc import abstractmethod

from neupy.core.config import ConfigurableABC


__all__ = ('Regression', 'Classification', 'Clustering')


class Regression(object):
    """ Mixin for regression.
    """


class Classification(ConfigurableABC):
    """ Mixin for classification.
    """
    @abstractmethod
    def predict_proba(self):
        """ Predict probabilities.
        """


class Clustering(object):
    """ Mixin for clustering.
    """
