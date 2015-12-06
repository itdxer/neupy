from numpy import reshape

from neupy.utils import format_data
from neupy.core.properties import NonNegativeIntProperty
from neupy.layers import Step
from neupy.network import ConstructableNetwork, UnsupervisedLearning


__all__ = ('BaseStepAssociative',)


class BaseAssociative(UnsupervisedLearning, ConstructableNetwork):
    def __init__(self, connection, **options):
        if len(connection) != 2:
            raise ValueError("Network should have only 2 layers")
        super(BaseAssociative, self).__init__(connection, **options)

    def train(self, input_train, epochs=100):
        return super(BaseAssociative, self).train(input_train, epochs,
                                                  epsilon=None)


class BaseStepAssociative(BaseAssociative):
    """ Base class for associative algorithms which have 2 layers and first
    one is has step function as activation.
    """
    n_unconditioned = NonNegativeIntProperty(default=1, min_size=1)

    def __init__(self, connection, **options):
        super(BaseStepAssociative, self).__init__(connection, **options)

        input_layer = self.input_layer
        n_unconditioned = self.n_unconditioned

        if not isinstance(input_layer, Step):
            raise ValueError("Input layer should be `Step` class instance.")

        if input_layer.input_size <= n_unconditioned:
            raise ValueError(
                "Number of uncondition should must be less than total "
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
