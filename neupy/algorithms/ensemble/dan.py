import numpy as np

from .base import BaseEnsemble


__all__ = ('DynamicallyAveragedNetwork',)


class DynamicallyAveragedNetwork(BaseEnsemble):
    """ Dynamically Averaged Network (DAN) weighted ensamble for binary
    classification problems.

    Notes
    -----
    * Every network must has 1 output and result must be between 0 and 1.

    Parameters
    ----------
    networks : list
        List of Neural Networks.

    Methods
    -------
    train(self, input_data, target_data, *args, **kwargs)
        Use input data to train all neural network one by one.

    Attributes
    ----------
    weights : ndarray, shape = [n_predictors, n_networks]
        After you get prediction you can also check weight which you
        will get to combine the result.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets, metrics
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import algorithms
    >>>
    >>> np.random.seed(50)
    >>>
    >>> data, target = datasets.make_classification(300, n_features=4,
    >>>                                             n_classes=2)
    >>> x_train, x_test, y_train, y_test = train_test_split(
    >>>     data, target, train_size=0.7
    >>> )
    >>>
    >>> dan = algorithms.DynamicallyAveragedNetwork([
    >>>     algorithms.RPROP((4, 10, 1), maxstep=1),
    >>>     algorithms.GradientDescent((4, 5, 1)),
    >>> ])
    >>> dan.train(x_train, y_train, epochs=500)
    >>> y_predicted = dan.predict(x_test)
    >>>
    >>> metrics.accuracy_score(y_test, y_predicted)
    0.97777777777777775
    """
    def __init__(self, networks):
        super(DynamicallyAveragedNetwork, self).__init__(networks)
        self.weights = None

        for network in networks:
            output_layer_size = network.output_layer.size
            network.verbose = False

            if output_layer_size != 1:
                raise ValueError(
                    "Final layer at network `{}` must has 1 output, got "
                    "{}".format(self.__class__.__name__, output_layer_size)
                )

    def train(self, input_data, target_data, *args, **kwargs):
        for network in self.networks:
            network.train(input_data, target_data, *args, **kwargs)

    def predict_raw(self, input_data):
        number_of_inputs = input_data.shape[0]
        network_certainties = np.zeros((number_of_inputs, len(self.networks)))
        network_outputs = network_certainties.copy()

        for i, network in enumerate(self.networks):
            output = network.predict(input_data)
            minval, maxval = output.min(), output.max()

            if not (0 <= minval <= 1 and 0 <= maxval <= 1):
                raise ValueError(
                    "Netwrok output must be in range [0, 1]. Network output "
                    "was in range [{}, {}]".format(minval, maxval)
                )

            certainty = np.where(output > 0.5, output, 1 - output)

            network_certainties[:, i:i + 1] = certainty
            network_outputs[:, i:i + 1] = output

        total_output_sum = np.reshape(network_certainties.sum(axis=1),
                                      (number_of_inputs, 1))
        self.weights = network_certainties / total_output_sum
        return (self.weights * network_outputs).sum(axis=1)

    def predict(self, input_data):
        raw_output = self.predict_raw(input_data)
        return np.round(raw_output)
