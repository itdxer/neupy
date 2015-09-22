from numpy import multiply, concatenate

from neupy import algorithms
from neupy.layers import SoftmaxLayer, OutputLayer
from neupy.functions import errors
from .base import BaseEnsemble


__all__ = ('MixtureOfExperts',)


class MixtureOfExperts(BaseEnsemble):
    """ Mixture of Experts ensemble algorithm for Backpropagation
    based Neural Networks.

    Parameters
    ----------
    networks : list
        List of networks based on :network:`Backpropagation` algorithm.
    gating_network : object
        2 Layer Neural Network based on :network:`Backpropagation` which
        has :layer:`SoftmaxLayer` as first one and the last one must be
        the :layer:`OutputLayer`. Output layer size must be equal to number
        of networks in model. Also important to say that in every network
        input size must be equal.

    Methods
    -------
    {supervised_train_epochs}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import ensemble, algorithms, layers
    >>> from neupy.functions import rmsle
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
    ...     algorithms.Backpropagation((insize, 20, outsize), step=0.1,
    ...                                 verbose=False),
    ...     algorithms.Backpropagation((insize, 20, outsize), step=0.1,
    ...                                 verbose=False),
    ... ]
    >>> n_networks = len(networks)
    >>>
    >>> moe = ensemble.MixtureOfExperts(
    ...     networks=networks,
    ...     gating_network=algorithms.Backpropagation(
    ...         layers.SoftmaxLayer(insize) > layers.OutputLayer(n_networks),
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

        if not isinstance(gating_network, algorithms.Backpropagation):
            raise ValueError("Gating network must use Backpropagation "
                             "learning algorihtm")

        for network in self.networks:
            if not isinstance(network, algorithms.Backpropagation):
                raise ValueError(
                    "Network must use Backpropagation learning algorithm, "
                    "got {0}".format(network.__class__.__name__)
                )

            if network.output_layer.input_size != 1:
                raise ValueError("Network must contains one output unit, got "
                                 "{0}".format(network.output_layer.input_size))

            if network.error != errors.mse:
                raise ValueError(
                    "Use only Mean Square Error (MSE) function in network, "
                    "got {0}".format(network.error.__name__)
                )

            network.verbose = False

        if gating_network.input_layer.__class__ != SoftmaxLayer:
            raise ValueError(
                "Input layer must be `SoftmaxLayer`, got `{0}`".format(
                    gating_network.input_layer.__class__.__name__
                )
            )

        if gating_network.output_layer.__class__ != OutputLayer:
            raise ValueError(
                "First layer must be `OutputLayer`, got `{0}`".format(
                    gating_network.output_layer.__class__.__name__
                )
            )

        gating_network.verbose = False
        gating_network_output_size = gating_network.output_layer.input_size
        n_networks = len(self.networks)

        if gating_network_output_size != n_networks:
            raise ValueError("Invalid Gating network output size. Expected "
                             "{0}, got {1}".format(n_networks,
                                                   gating_network_output_size))

        if gating_network.error != errors.mse:
            raise ValueError(
                "Use only Mean Square Error (MSE) function in network, "
                "got {0}".format(gating_network.error.__name__)
            )

        self.gating_network = gating_network

    def train(self, input_data, target_data, epochs=100):
        if target_data.ndim == 1:
            target_data = target_data.reshape((target_data.size, 1))

        output_size = target_data.shape[1]
        if output_size != 1:
            raise ValueError("Target data must contains only 1 column, got "
                             "{0}".format(output_size))

        input_size = input_data.shape[1]

        gating_network = self.gating_network
        gating_network_input_size = gating_network.input_layer.input_size

        if gating_network_input_size != input_size:
            raise ValueError(
                "Gating Network expected get {0} input features, got "
                "{1}".format(gating_network_input_size, input_size)
            )

        for epoch in range(epochs):
            probs = self.gating_network.predict(input_data)
            total_output = 0
            outputs = []

            for i, network in enumerate(self.networks):
                output = network.predict(input_data)
                outputs.append(output)
                total_output += multiply(output, probs[:, i:i + 1])

            outputs = concatenate(outputs, axis=1)

            for i, network in enumerate(self.networks):
                # This is simple solution for error derivative update
                # g * (expected - actual) = g * expected - g * actual
                # It's simple hack and probably could broke other
                # algorithms, but now it works fine, but I need
                # change it later
                weight_delta = network.get_weight_delta(
                    total_output * probs[:, i:i + 1],
                    target_data * probs[:, i:i + 1]
                )
                network.update_weights(weight_delta)
                network.after_weight_update(input_data, target_data)

            # The same as at comment above
            weight_delta = gating_network.get_weight_delta(
                total_output * probs,
                target_data * probs
            )
            gating_network.update_weights(weight_delta)
            gating_network.after_weight_update(input_data, target_data)

    def predict(self, input_data):
        probs = self.gating_network.predict(input_data)
        total_output = 0

        for i, network in enumerate(self.networks):
            output = network.predict(input_data)
            total_output += multiply(output, probs[:, i:i + 1])

        return total_output

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
