from neupy.core.properties import ProperFractionProperty, BoundedProperty
from .base import SingleStepConfigurable


__all__ = ('DeltaBarDelta',)


class DeltaBarDelta(SingleStepConfigurable):
    beta = ProperFractionProperty(default=0.5)
    increase_factor = BoundedProperty(default=0.1, minval=0)
    decrease_factor = ProperFractionProperty(default=0.9)
