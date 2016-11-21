from neupy.core.config import Configurable
from neupy.algorithms.gd import WEIGHT_PENALTY


__all__ = ('WeightUpdateConfigurable',)


class WeightUpdateConfigurable(Configurable):
    """
    Configuration class for algorithms that update weights.

    Warns
    -----
    It works with any algorithm based on the backpropagation.
    """
    addon_type = WEIGHT_PENALTY
