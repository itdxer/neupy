from abc import abstractmethod

from neupy.core.base import BaseSkeleton
from neupy.core.config import ConfigurableABC


__all__ = ('BaseEnsemble',)


class BaseEnsemble(BaseSkeleton, ConfigurableABC):
    """ Base class for ensemble algorithms.
    """
    def __init__(self, networks):
        self.networks = networks

        if len(self.networks) < 2:
            raise ValueError("Ensemble algorithm should has at least "
                             "2 networks")

    @abstractmethod
    def train(self, input_data, target_data, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    def __repr__(self):
        return "{classname}(networks=[\n    {networks}\n])".format(
            classname=self.__class__.__name__,
            networks=',\n    '.join(map(repr, self.networks))
        )
