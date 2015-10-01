from numpy import ones

from neupy.core.config import Configurable
from neupy.algorithms.backprop import LEARING_RATE_UPDATE


__all__ = ('SingleStep', 'MultiSteps')


class LearningRateConfigurable(Configurable):
    optimization_type = LEARING_RATE_UPDATE


class SingleStep(LearningRateConfigurable):
    """ Base class for backpropagation step algorithms which allow use single
    step for all layers.

    Attributes
    ----------
    {first_step}
    """
    def setup_defaults(self):
        self.first_step = self.step


class MultiSteps(LearningRateConfigurable):
    """ Base class for step algorithms which allow use unique step for
    every layer.

    Attributes
    ----------
    {steps}
    """
    def init_layers(self):
        super(MultiSteps, self).init_layers()
        steps = self.steps = []

        for layer in self.train_layers:
            steps.append(ones(layer.size) * self.step)

    def layer_step(self, layer_number):
        return self.steps[layer_number]
