import numpy as np

from neupy.utils import format_data
from neupy.core.properties import IntProperty, ArrayProperty
from neupy.network import BaseNetwork, UnsupervisedLearning


__all__ = ('BaseStepAssociative',)


class BaseAssociative(UnsupervisedLearning, BaseNetwork):
    """ Base class for associative learning.

    Parameters
    ----------
    n_inputs : int
        Number of input units.
    n_outputs : int
        Number of output units.
    weight : array-like
        Neural network weights. ``None`` mean that network weight would
        be generated randomly. Value defined manualy should have
        shape ``(n_inputs, n_outputs)``. Defaults to ``None``.
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    train(input_train, epochs=100):
        Train neural network.
    {BaseSkeleton.fit}
    """

    n_inputs = IntProperty(minval=1, required=True)
    n_outputs = IntProperty(minval=1, required=True)
    weight = ArrayProperty()

    def __init__(self, **options):
        super(BaseAssociative, self).__init__(**options)
        self.init_layers()

    def init_layers(self):
        valid_weight_shape = (self.n_inputs, self.n_outputs)

        if self.weight is None:
            self.weight = np.random.randn(*valid_weight_shape)

        if self.weight.shape != valid_weight_shape:
            raise ValueError("Weight matrix has invalid shape. Got {}, "
                             "expected {}".format(self.weight.shape,
                                                  valid_weight_shape))

        self.weight = self.weight.astype(float)

    def train(self, input_train, epochs=100):
        return super(BaseAssociative, self).train(input_train, epochs,
                                                  epsilon=None)


class BaseStepAssociative(BaseAssociative):
    """ Base class for associative algorithms which have 2 layers and first
    one is has step function as activation.

    Parameters
    ----------
    {BaseAssociative.n_inputs}
    {BaseAssociative.n_outputs}
    n_unconditioned : int
        Number of unconditioned units in neraul networks. All these
        units wouldn't update during the training procedure.
        Unconditioned should be the first features in dataset.
    {BaseAssociative.weight}
    bias : array-like
        Neural network bias units. The same behaviour as for parameter
        ``weight``. Defaults to ``None``.
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {BaseAssociative.train}
    {BaseSkeleton.fit}
    """

    n_inputs = IntProperty(minval=2, required=True)
    n_unconditioned = IntProperty(minval=1, required=True)
    bias = ArrayProperty()

    def init_layers(self):
        if self.n_inputs <= self.n_unconditioned:
            raise ValueError(
                "Number of uncondition features should be less than total "
                "number of features. `n_inputs`={} and "
                "`n_unconditioned`={}".format(
                    self.n_inputs,
                    self.n_unconditioned
                )
            )

        valid_weight_shape = (self.n_inputs, self.n_outputs)
        valid_bias_shape = (self.n_outputs,)

        if self.weight is None:
            self.weight = np.zeros(valid_weight_shape)
            self.weight[:self.n_unconditioned, :] = 1

        if self.bias is None:
            self.bias = -0.5 * np.ones(valid_bias_shape)

        super(BaseStepAssociative, self).init_layers()

        if self.bias.shape != valid_bias_shape:
            raise ValueError("Bias vector has invalid shape. Got {}, "
                             "expected {}".format(self.bias.shape,
                                                  valid_bias_shape))

        self.bias = self.bias.astype(float)

    def predict(self, input_data):
        input_data = format_data(input_data, is_feature1d=False)
        raw_output = input_data.dot(self.weight) + self.bias
        return np.where(raw_output > 0, 1, 0)

    def train(self, input_train, *args, **kwargs):
        input_train = format_data(input_train, is_feature1d=False)
        return super(BaseStepAssociative, self).train(input_train, *args,
                                                      **kwargs)

    def train_epoch(self, input_train, target_train):
        weight = self.weight
        n_unconditioned = self.n_unconditioned
        predict = self.predict
        weight_delta = self.weight_delta

        for input_row in input_train:
            input_row = np.reshape(input_row, (1, input_row.size))
            layer_output = predict(input_row)
            weight[n_unconditioned:, :] += weight_delta(input_row,
                                                        layer_output)
