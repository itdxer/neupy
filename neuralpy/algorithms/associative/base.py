from numpy import reshape

from neuralpy.core.properties import NonNegativeIntProperty
from neuralpy.layers import StepLayer
from neuralpy.network.base import BaseNetwork
from neuralpy.network.learning import UnsupervisedLearning


__all__ = ('BaseStepAssociative',)


class BaseAssociative(UnsupervisedLearning, BaseNetwork):
    def __init__(self, connection, **options):
        if len(connection) != 2:
            raise ValueError("Connection must contains only input and "
                             "output layers")

        super(BaseAssociative, self).__init__(connection, **options)

    def setup_defaults(self):
        del self.use_bias
        self.use_bias = False
        super(BaseAssociative, self).setup_defaults()

    def train(self, input_train, epochs=None, epsilon=None):
        if epsilon is not None:
            raise AttributeError("You can't converge this algorithm. Use "
                                 "`epochs` parameter.")
        return super(BaseAssociative, self).train(input_train, epochs, epsilon)

    def predict(self, input_data):
        result = input_data
        for layer in self.layers:
            result = layer.output(result)
        return result


class BaseStepAssociative(BaseAssociative):
    """ Base class for associative algorithms which have 2 layers and first
    one is has step function as activation.
    """
    n_unconditioned = NonNegativeIntProperty(default=1, min_size=1)

    def __init__(self, connection, **options):
        super(BaseStepAssociative, self).__init__(connection, **options)

        input_layer = self.input_layer
        n_unconditioned = self.n_unconditioned

        if not isinstance(input_layer, StepLayer):
            raise ValueError("Input layer must be `StepLayer`")

        if input_layer.input_size <= n_unconditioned:
            raise ValueError(
                "Number of uncondition features must be less than total "
                "number of features in network. #feature = {} and "
                "#unconditioned = {}".format(
                    input_layer.input_size,
                    n_unconditioned
                )
            )

    def train_epoch(self, input_train, target_train):
        weight = self.input_layer.weight
        unconditioned = self.n_unconditioned
        predict = self.predict
        weight_delta = self.weight_delta

        for input_row in input_train:
            input_row = reshape(input_row, (1, input_row.size))
            layer_output = predict(input_row)
            weight[unconditioned:, :] += weight_delta(input_row, layer_output)
