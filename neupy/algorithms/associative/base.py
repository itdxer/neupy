import numpy as np

from neupy import init
from neupy.utils import format_data
from neupy.core.properties import IntProperty, ParameterProperty, ArrayProperty
from neupy.algorithms.base import BaseNetwork


__all__ = ('BaseStepAssociative',)


class BaseAssociative(BaseNetwork):
    """
    Base class for associative learning.

    Parameters
    ----------
    n_inputs : int
        Number of input units.

    n_outputs : int
        Number of output units.

    weight : array-like, Initializer
        Neural network weights.
        Value defined manualy should have shape ``(n_inputs, n_outputs)``.
        Defaults to :class:`Normal() <neupy.init.Normal>`.

    {BaseNetwork.step}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}

    train(input_train, epochs=100)
        Train neural network.

    {BaseSkeleton.fit}
    """
    n_inputs = IntProperty(minval=1, required=True)
    n_outputs = IntProperty(minval=1, required=True)
    weight = ParameterProperty(default=init.Normal())

    def __init__(self, **options):
        super(BaseAssociative, self).__init__(**options)
        self.init_layers()

    def init_layers(self):
        valid_weight_shape = (self.n_inputs, self.n_outputs)

        if isinstance(self.weight, init.Initializer):
            self.weight = self.weight.sample(valid_weight_shape)

        if self.weight.shape != valid_weight_shape:
            raise ValueError("Weight matrix has invalid shape. Got {}, "
                             "expected {}".format(self.weight.shape,
                                                  valid_weight_shape))

        self.weight = self.weight.astype(float)

    def train(self, input_train, epochs=100):
        input_train = format_data(input_train, is_feature1d=True)
        return super(BaseAssociative, self).train(
            input_train=input_train, target_train=None,
            input_test=None, target_test=None,
            epochs=epochs, epsilon=None,
            summary='table'
        )


class BaseStepAssociative(BaseAssociative):
    """
    Base class for associative algorithms which have 2 layers and first
    one is has step function as activation.

    Parameters
    ----------
    {BaseAssociative.n_inputs}

    {BaseAssociative.n_outputs}

    n_unconditioned : int
        Number of unconditioned units in neraul networks. All these
        units wouldn't update during the training procedure.
        Unconditioned should be the first feature in the dataset.

    weight : array-like
        Neural network weights.
        Value defined manualy should have shape ``(n_inputs, n_outputs)``.
        Defaults to ``None`` which means that all unconditional
        weights will be equal to ``1``. Other weights equal to ``0``.

    bias : array-like, Initializer
        Neural network bias units.
        Defaults to :class:`Constant(-0.5) <neupy.init.Constant>`.

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

    weight = ArrayProperty()
    bias = ParameterProperty(default=init.Constant(-0.5))

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

        if isinstance(self.bias, init.Initializer):
            self.bias = self.bias.sample(valid_bias_shape)

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
