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
        Number of features (columns) in the input data.

    n_outputs : int
        Number of outputs in the  network.

    weight : array-like, Initializer
        Neural network weights.
        Value defined manually should have shape ``(n_inputs, n_outputs)``.
        Defaults to :class:`Normal() <neupy.init.Normal>`.

    {BaseNetwork.Parameters}

    Methods
    -------
    {BaseSkeleton.predict}

    train(X_train, epochs=100)
        Train neural network.

    {BaseSkeleton.fit}
    """
    n_inputs = IntProperty(minval=1, required=True)
    n_outputs = IntProperty(minval=1, required=True)
    weight = ParameterProperty(default=init.Normal())

    def __init__(self, **options):
        super(BaseAssociative, self).__init__(**options)
        self.init_weights()

    def init_weights(self):
        valid_weight_shape = (self.n_inputs, self.n_outputs)

        if isinstance(self.weight, init.Initializer):
            self.weight = self.weight.sample(
                valid_weight_shape, return_array=True)

        if self.weight.shape != valid_weight_shape:
            raise ValueError(
                "Weight matrix has invalid shape. Got {}, expected {}"
                "".format(self.weight.shape, valid_weight_shape))

        self.weight = self.weight.astype(float)

    def format_input_data(self, X):
        X = format_data(X, is_feature1d=(self.n_inputs == 1))

        if X.ndim != 2:
            raise ValueError(
                "Cannot make prediction, because input "
                "data has more than 2 dimensions")

        if X.shape[1] != self.n_inputs:
            raise ValueError(
                "Input data expected to have {} features, "
                "but got {}".format(self.n_inputs, X.shape[1]))

        return X

    def train(self, X_train, epochs=100):
        X_train = self.format_input_data(X_train)
        return super(BaseAssociative, self).train(
            X_train=X_train, epochs=epochs)


class BaseStepAssociative(BaseAssociative):
    """
    Base class for associative algorithms which have 2 layers and first
    one is has step function as activation.

    Parameters
    ----------
    {BaseAssociative.n_inputs}

    {BaseAssociative.n_outputs}

    n_unconditioned : int
        Number of unconditioned units in neural networks. All these
        units wouldn't update during the training procedure.
        Unconditioned should be the first feature in the dataset.

    weight : array-like
        Neural network weights.
        Value defined manually should have shape ``(n_inputs, n_outputs)``.
        Defaults to ``None`` which means that all unconditional
        weights will be equal to ``1``. Other weights equal to ``0``.

    bias : array-like, Initializer
        Neural network bias units.
        Defaults to :class:`Constant(-0.5) <neupy.init.Constant>`.

    {BaseNetwork.Parameters}

    Methods
    -------
    {BaseAssociative.Methods}
    """
    n_inputs = IntProperty(minval=2, required=True)
    n_unconditioned = IntProperty(minval=1, required=True)

    weight = ArrayProperty()
    bias = ParameterProperty(default=init.Constant(-0.5))

    def init_weights(self):
        if self.n_inputs <= self.n_unconditioned:
            raise ValueError(
                "Number of unconditioned features should be less than total "
                "number of features. `n_inputs`={} and `n_unconditioned`={}"
                "".format(self.n_inputs, self.n_unconditioned))

        valid_weight_shape = (self.n_inputs, self.n_outputs)
        valid_bias_shape = (self.n_outputs,)

        if self.weight is None:
            self.weight = np.zeros(valid_weight_shape)
            self.weight[:self.n_unconditioned, :] = 1

        if isinstance(self.bias, init.Initializer):
            self.bias = self.bias.sample(valid_bias_shape, return_array=True)

        super(BaseStepAssociative, self).init_weights()

        if self.bias.shape != valid_bias_shape:
            raise ValueError(
                "Bias vector has invalid shape. Got {}, expected {}"
                "".format(self.bias.shape, valid_bias_shape))

        self.bias = self.bias.astype(float)

    def predict(self, X):
        X = format_data(X, is_feature1d=False)
        raw_output = X.dot(self.weight) + self.bias
        return np.where(raw_output > 0, 1, 0)

    def train(self, X_train, *args, **kwargs):
        X_train = format_data(X_train, is_feature1d=False)
        return super(BaseStepAssociative, self).train(X_train, *args, **kwargs)

    def one_training_update(self, X_train, y_train):
        weight = self.weight
        n_unconditioned = self.n_unconditioned
        predict = self.predict
        weight_delta = self.weight_delta

        error = 0

        for x_row in X_train:
            x_row = np.expand_dims(x_row, axis=0)
            layer_output = predict(x_row)

            delta = weight_delta(x_row, layer_output)
            weight[n_unconditioned:, :] += delta

            # This error can tell us whether network has converged
            # to some value of weights. Low errors will mean that weights
            # hasn't been updated much during the training epoch.
            error += np.linalg.norm(delta)

        return error
