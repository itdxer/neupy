import sys

import theano
import theano.tensor as T
import numpy as np

from neupy.layers import Softmax, Output
from .base import BaseEnsemble


__all__ = ('MixtureOfExperts',)


class MixtureOfExperts(BaseEnsemble):
    """ Mixture of Experts ensemble algorithm for GradientDescent
    based Neural Networks.

    Parameters
    ----------
    networks : list
        List of networks based on :network:`GradientDescent` algorithm.
    gating_network : object
        2-Layer Neural Network based on :network:`GradientDescent` which
        has :layer:`Softmax` as first one and the last one must be
        the :layer:`Output`. Output layer size must be equal to number
        of networks in model. Also important to say that in every network
        input size must be equal.

    Methods
    -------
    train(self, input_data, target_data, epochs=100):
        Train neural network.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import algorithms, layers
    >>> from neupy.estimators import rmsle
    >>>
    >>> np.random.seed(100)
    >>>
    >>> dataset = datasets.load_diabetes()
    >>> data, target = dataset.data, dataset.target
    >>> input_scaler = preprocessing.MinMaxScaler((-1 ,1))
    >>> output_scaler = preprocessing.MinMaxScaler()
    >>>
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     input_scaler.fit_transform(data),
    ...     output_scaler.fit_transform(target),
    ...     train_size=0.8
    ... )
    ...
    >>>
    >>> insize, outsize = (10, 1)
    >>> networks = [
    ...     algorithms.GradientDescent((insize, 20, outsize), step=0.1),
    ...     algorithms.GradientDescent((insize, 20, outsize), step=0.1),
    ... ]
    >>> n_networks = len(networks)
    >>>
    >>> moe = algorithms.MixtureOfExperts(
    ...     networks=networks,
    ...     gating_network=algorithms.GradientDescent(
    ...         layers.Softmax(insize) > layers.Output(n_networks),
    ...         step=0.1,
    ...         verbose=False,
    ...     )
    ... )
    ...
    >>> moe.train(x_train, y_train, epochs=300)
    >>> y_predicted = moe.predict(x_test)
    >>>
    >>> rmsle(output_scaler.inverse_transform(y_test),
    ...       output_scaler.inverse_transform(y_predicted).round())
    0.44680253132714459
    """
    def __init__(self, networks, gating_network=None):
        super(MixtureOfExperts, self).__init__(networks)
        algorithms = sys.modules['neupy.algorithms']

        if not isinstance(gating_network, algorithms.GradientDescent):
            raise ValueError("Gating network must use GradientDescent "
                             "learning algorihtm")

        for network in self.networks:
            if not isinstance(network, algorithms.GradientDescent):
                raise ValueError(
                    "Network must use GradientDescent learning algorithm, "
                    "got {0}".format(network.__class__.__name__)
                )

            if network.output_layer.size != 1:
                raise ValueError("Network must contains one output unit, got "
                                 "{0}".format(network.output_layer.size))

            if network.error.__name__ != 'mse':
                raise ValueError(
                    "Use only Mean Square Error (MSE) function in network, "
                    "got {0}".format(network.error.__name__)
                )

            network.verbose = False

        if gating_network.input_layer.__class__ != Softmax:
            raise ValueError(
                "Input layer must be `Softmax`, got `{0}`".format(
                    gating_network.input_layer.__class__.__name__
                )
            )

        if gating_network.output_layer.__class__ != Output:
            raise ValueError(
                "First layer must be `Output`, got `{0}`".format(
                    gating_network.output_layer.__class__.__name__
                )
            )

        gating_network.verbose = False
        gating_network_output_size = gating_network.output_layer.size
        n_networks = len(self.networks)

        if gating_network_output_size != n_networks:
            raise ValueError("Invalid Gating network output size. Expected "
                             "{0}, got {1}".format(n_networks,
                                                   gating_network_output_size))

        if gating_network.error.__name__ != 'mse':
            raise ValueError(
                "Only Mean Square Error (MSE) function is available, "
                "got {0}".format(gating_network.error.__name__)
            )

        self.gating_network = gating_network

        x = T.matrix('x')
        y = T.matrix('y')

        gating_network.variables.network_input = x
        gating_network.init_variables()
        gating_network.init_methods()

        probs = gating_network.variables.prediction_func
        train_outputs, outputs = [], []
        for i, network in enumerate(self.networks):
            network.variables.network_input = x
            network.variables.network_output = y

            network.init_variables()
            network.init_methods()

            output = network.variables.prediction_func
            outputs.append(output * probs[:, i:i + 1])

            train_output = network.variables.train_prediction_func
            train_outputs.append(train_output * probs[:, i:i + 1])

        outputs_concat = T.concatenate(outputs, axis=1)
        self.prediction_func = sum(outputs)
        self.train_prediction_func = sum(train_outputs)

        for net in self.networks:
            net.methods.error_func = net.error(y, self.train_prediction_func)

        gating_network.methods.error_func = gating_network.error(
            gating_network.variables.network_output,
            outputs_concat
        )

        self._predict = theano.function([x], self.prediction_func)

    def train(self, input_data, target_data, epochs=100):
        if target_data.ndim == 1:
            target_data = target_data.reshape((target_data.size, 1))

        output_size = target_data.shape[1]
        if output_size != 1:
            raise ValueError("Target data must contains only 1 column, got "
                             "{0}".format(output_size))

        input_size = input_data.shape[1]

        gating_network = self.gating_network
        gating_network_input_size = gating_network.input_layer.size

        if gating_network_input_size != input_size:
            raise ValueError(
                "Gating Network expected get {0} input features, got "
                "{1}".format(gating_network_input_size, input_size)
            )

        networks = self.networks

        for epoch in range(epochs):
            predictions = []
            for i, network in enumerate(networks):
                predictions.append(network.predict(input_data))
                network.train_epoch(input_data, target_data)

            predictions = np.concatenate(predictions, axis=1)
            gating_network.train_epoch(input_data, predictions)

    def predict(self, input_data):
        return self._predict(input_data)

    def __repr__(self):
        indent = ' ' * 4
        return (
            "{classname}(networks=[\n"
            "{double_indent}{networks}\n"
            "{indent}],\n"
            "{indent}gating_network={gating_network}\n"
            ")"
        ).format(
            classname=self.__class__.__name__,
            networks=',\n        '.join(map(repr, self.networks)),
            gating_network=repr(self.gating_network),
            indent=indent,
            double_indent=(2 * indent)
        )
