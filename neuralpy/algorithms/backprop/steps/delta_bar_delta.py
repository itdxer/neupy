from neuralpy.core.properties import (SimpleNumberProperty,
                                      NonNegativeNumberProperty)
from .base import MultiSteps


__all__ = ('DeltaBarDelta',)


class DeltaBarDelta(MultiSteps):
    beta = SimpleNumberProperty(default=0.5)
    increase_factor = NonNegativeNumberProperty(default=0.1)
    decrease_factor = SimpleNumberProperty(default=0.9)
