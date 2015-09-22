from numpy import dot

from .base import BaseStepAssociative


__all__ = ('Instar',)


class Instar(BaseStepAssociative):
    """ Instar is a simple unsupervised Neural Network algortihm which
    detect associations in features. Algorithm very similar to
    :network:`HebbRule` except the learning rule.

    Parameters
    ----------
    n_unconditioned : int
        This value control number of features which are unconditioned
        stimulus for network. Defaults to ``1``. Can be any integer value
        bigger than ``1``, but less than feature space.
    {step}
    {show_epoch}
    {shuffle_data}
    {error}
    {verbose}
    {full_signals}

    Methods
    -------
    {unsupervised_train_epochs}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms, layers
    >>>
    >>> train_data = np.array([
    ...     [0, 1, 0, 0],
    ...     [1, 1, 0, 0],
    ... ])
    >>> test_cases = np.array([
    ...     [0, 1, 0, 0],
    ...     [0, 0, 0, 0],
    ...     [0, 0, 1, 1],
    ... ])
    >>>
    >>>
    >>> scaler = {{'lower_value': 0, 'upper_value': 1}}
    >>> input_layer = layers.StepLayer(4, function_coef=scaler)
    >>> output_layer = layers.OutputLayer(1)
    >>>
    >>> instnet = algorithms.Instar(
    ...     input_layer > output_layer,
    ...     n_unconditioned=1,
    ...     step=1,
    ...     verbose=False,
    ... )
    >>>
    >>> instnet.train(train_data, epochs=2)
    >>> instnet.predict(test_cases)
    array([[1],
           [0],
           [0]])
    """
    def weight_delta(self, input_row, layer_output):
        unconditioned = self.n_unconditioned
        update_from_column = unconditioned - self.use_bias
        weight = self.input_layer.weight[unconditioned:, :]
        return self.step * dot(
            (input_row[:, update_from_column:].T - weight), layer_output
        )
