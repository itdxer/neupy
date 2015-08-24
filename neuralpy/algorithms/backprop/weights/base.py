from neuralpy.core.config import Configurable
from neuralpy.algorithms.backprop import WEIGHT_UPDATE


__all__ = ('WeightUpdateConfigurable',)


class WeightUpdateConfigurable(Configurable):
    optimization_type = WEIGHT_UPDATE
