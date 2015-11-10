from neupy.core.properties import (BetweenZeroAndOneProperty,
                                   NonNegativeNumberProperty)
from .base import LearningRateConfigurable


__all__ = ('DeltaBarDelta',)


class DeltaBarDelta(LearningRateConfigurable):
    beta = BetweenZeroAndOneProperty(default=0.5)
    increase_factor = NonNegativeNumberProperty(default=0.1)
    decrease_factor = BetweenZeroAndOneProperty(default=0.9)
