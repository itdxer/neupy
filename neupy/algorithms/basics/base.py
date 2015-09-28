from neupy.utils import format_data
from neupy.algorithms.feedforward import FeedForwardNetwork
from neupy.network.learning import SupervisedLearning
from neupy.functions import linear_error
from neupy.layers import StepLayer, OutputLayer


__all__ = ('SimpleTwoLayerNetwork',)


class SimpleTwoLayerNetwork(SupervisedLearning, FeedForwardNetwork):
    """ Base class for feedforward neural network without hidden layers.
    Input layer is always :layer:`StepLayer`.
    """
    def __init__(self, layer_sizes, **options):
        if len(layer_sizes) != 2:
            raise ValueError("This network must contains two layers.")

        input_layer_size, output_layer_size = layer_sizes

        super(SimpleTwoLayerNetwork, self).__init__(
            StepLayer(input_layer_size) > OutputLayer(output_layer_size),
            **options
        )

        self.input_layer = self.layers[0]
        self.error = linear_error

    def setup_defaults(self):
        del self.use_raw_predict_at_error

    def raw_predict(self, input_data):
        input_data = format_data(input_data)

        input_layer = self.input_layer
        input_data = input_layer.preformat_input(input_data)

        self.input_data = input_data
        self.summated = input_layer.summator(input_data)

        return input_layer.activation_function(self.summated)

    def update_weights(self, weight_deltas):
        self.input_layer.weight += self.step * weight_deltas
