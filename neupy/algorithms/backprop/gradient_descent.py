import numpy as np

from neupy.core.properties import NumberProperty
from .backpropagation import Backpropagation


__all__ = ('MinibatchGradientDescent',)


class MinibatchGradientDescent(Backpropagation):
    """ Mini-batch Gradient Descent algorithm.

    Parameters
    ----------
    batch_size : int
        Setup batch size for learning process, defaults to ``10``.
    {optimizations}
    {raw_predict_param}
    {full_params}

    Methods
    -------
    {supervised_train}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> mgdnet = algorithms.MinibatchGradientDescent(
    ...     (2, 3, 1),
    ...     verbose=False,
    ...     batch_size=1
    ... )
    >>> mgdnet.train(x_train, y_train)

    See Also
    --------
    :network:`Backpropagation` : Backpropagation algorithm.
    """
    batch_size = NumberProperty(default=10)

    def iter_batches(self, input_train, target_train):
        count_of_data = input_train.shape[0]

        for i in range(0, count_of_data, self.batch_size):
            batch = slice(i, i + self.batch_size)
            yield input_train[batch], target_train[batch]

    def train_batch(self, input_data, target_data):
        deltas = [np.zeros(l.weight.shape) for l in self.train_layers]

        predict_for_error = self.predict_for_error
        get_gradient = self.get_gradient

        for input_row, target_row in zip(input_data, target_data):
            output_train = predict_for_error(input_data)

            self.output_train = output_train

            weight_delta = get_gradient(output_train, target_data)
            deltas = map(sum, zip(deltas, weight_delta))

        self.weight_delta = [-delta / len(input_data) for delta in deltas]
        return self.weight_delta

    def train_epoch(self, input_train, target_train):
        batches = self.iter_batches(input_train, target_train)

        train_batch = self.train_batch
        update_weights = self.update_weights
        after_weight_update = self.after_weight_update

        for input_data, target_data in batches:
            weight_delta = train_batch(input_data, target_data)
            update_weights(weight_delta)
            after_weight_update(input_train, target_train)

        return self.error(self.predict_for_error(input_train), target_train)
