from neupy.core.base import BaseSkeleton
from neupy.core.config import Configurable


__all__ = ('BaseEnsemble',)


class BaseEnsemble(BaseSkeleton, Configurable):
    """
    Base class for ensemble algorithms.

    Parameters
    ----------
    networks : list or tuple
        List of networks.

    Raises
    ------
    ValueError
    """
    def __init__(self, networks):
        self.networks = networks
        n_networks = len(self.networks)

        if n_networks < 2:
            raise ValueError("Ensemble algorithm should have at least "
                             "2 networks, got {}.".format(n_networks))

    def __repr__(self):
        return "{classname}(networks=[\n    {networks}\n])".format(
            classname=self.__class__.__name__,
            networks=',\n    '.join(map(repr, self.networks))
        )
