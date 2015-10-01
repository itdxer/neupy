from numpy import reshape

from neupy.utils import format_data, is_row1d
from neupy.core.properties import NonNegativeIntProperty
from neupy.layers import StepLayer
from neupy.network.base import BaseNetwork
from neupy.network.learning import UnsupervisedLearning


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
        row1d = is_row1d(self.input_layer)
        result = format_data(input_data, row1d=row1d)

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
        input_train = format_data(input_train)

        weight = self.input_layer.weight
        unconditioned = self.n_unconditioned
        predict = self.predict
        weight_delta = self.weight_delta

        for input_row in input_train:
            input_row = reshape(input_row, (1, input_row.size))
            layer_output = predict(input_row)
            weight[unconditioned:, :] += weight_delta(input_row, layer_output)
