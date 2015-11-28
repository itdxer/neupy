from neupy.core.config import Configurable
from neupy.algorithms.gd import WEIGHT_PENALTY


__all__ = ('WeightUpdateConfigurable',)


class WeightUpdateConfigurable(Configurable):
    optimization_type = WEIGHT_PENALTY
