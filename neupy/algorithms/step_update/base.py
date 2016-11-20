from neupy.core.config import Configurable
from neupy.algorithms.gd import SINGLE_STEP_UPDATE, MULTIPLE_STEP_UPDATE


__all__ = ('SingleStepConfigurable', 'MultipleStepConfigurable')


class SingleStepConfigurable(Configurable):
    """
    Configuration class for learning rate control algorithms.
    Works for algorithms that modify single learnig rate.

    Warns
    -----
    It works only with algorithms based on backpropagation.
    """
    addon_type = SINGLE_STEP_UPDATE


class MultipleStepConfigurable(Configurable):
    """
    Configuration class for learning rate control algorithms.
    Works for algorithms that modify multiple learnig rates for one
    neural network.

    Warns
    -----
    It works only with algorithms based on backpropagation.
    """
    addon_type = MULTIPLE_STEP_UPDATE
