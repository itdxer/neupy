from neupy.core.config import Configurable
from neupy.algorithms.gd import LEARING_RATE_UPDATE


__all__ = ('LearningRateConfigurable',)


class LearningRateConfigurable(Configurable):
    """ Configuration class for learning rate control algorithms.

    Warns
    -----
    It works with any algorithm based on backpropagation. Class can't
    work without it.
    """
    addon_type = LEARING_RATE_UPDATE
