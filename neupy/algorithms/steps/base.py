from neupy.core.config import Configurable
from neupy.algorithms.gd import LEARING_RATE_UPDATE


__all__ = ('LearningRateConfigurable',)


class LearningRateConfigurable(Configurable):
    optimization_type = LEARING_RATE_UPDATE
