from abc import abstractmethod

from neupy.layers.utils import generate_layers
from neupy.network.base import BaseNetwork


__all__ = ('FeedForwardNetwork',)


class FeedForwardNetwork(BaseNetwork):
    """ Base class for the feed forward neural network.
    """

    def __init__(self, connection, **options):
        islist_of_integers = (
            isinstance(connection, (list, tuple)) and
            isinstance(connection[0], int)
        )
        if islist_of_integers:
            connection = generate_layers(list(connection))

        super(FeedForwardNetwork, self).__init__(connection, **options)

    # ----------------- Active Neural Network State ---------------- #

    @abstractmethod
    def get_weight_delta(self, output_train, target_train):
        pass

    @abstractmethod
    def update_weights(self, weight_delta):
        pass

    def after_weight_update(self, input_train, target_train):
        pass

    def train_epoch(self, input_train, target_train):
        output_train = self.raw_predict(input_train)
        self.output_train = output_train

        self.weight_delta = self.get_weight_delta(output_train, target_train)

        self.update_weights(self.weight_delta)
        self.after_weight_update(input_train, target_train)

        return self.error(output_train, target_train)

    def raw_predict(self, input_data):
        layer_outputs = self.layer_outputs = []
        summated_data = self.summated_data = []

        for layer in self.train_layers:
            input_data = layer.preformat_input(input_data)
            layer_outputs.append(input_data)

            summated = layer.summator(input_data)
            summated_data.append(summated)

            input_data = layer.activation_function(summated)

        return input_data

    def predict(self, input_data):
        raw_prediction = self.raw_predict(input_data)
        return self.output_layer.output(raw_prediction)
