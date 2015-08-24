from abc import abstractmethod

from neuralpy.helpers.logs import Verbose
from .config import ConfigurableWithABC


__all__ = ('BaseSkeleton',)


class BaseSkeleton(ConfigurableWithABC, Verbose):
    """ Base class for all algorithms and networks.
    """
    @abstractmethod
    def train(self, input_data, target_data):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    def fit(self, X, y, *args, **kwargs):
        self.train(X, y, *args, **kwargs)
        return self
