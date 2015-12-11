from neupy.core.properties import ProperFractionProperty, BoundedProperty
from .base import LearningRateConfigurable


__all__ = ('DeltaBarDelta',)


class DeltaBarDelta(LearningRateConfigurable):
    beta = ProperFractionProperty(default=0.5)
    increase_factor = BoundedProperty(default=0.1, minsize=0)
    decrease_factor = ProperFractionProperty(default=0.9)
