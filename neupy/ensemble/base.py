from abc import abstractmethod

from neupy.core.base import BaseSkeleton
from neupy.core.properties import Property
from neupy.core.config import ConfigurableWithABC


__all__ = ('BaseEnsemble',)


class BaseEnsemble(BaseSkeleton, ConfigurableWithABC):
    """ Base class for ensemlbe algorithms.
    """
    shuffle_data = Property(default=False, expected_type=bool)

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
