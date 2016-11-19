from numpy import dot

from .base import BaseStepAssociative


__all__ = ('Instar',)


class Instar(BaseStepAssociative):
    """
    Instar is a simple unsupervised Neural Network algortihm
    which detects associations.

    Parameters
    ----------
    {BaseAssociative.n_inputs}

    {BaseAssociative.n_outputs}

    {BaseStepAssociative.n_unconditioned}

    {BaseAssociative.weight}

    {BaseStepAssociative.bias}

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

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
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
    >>> instnet = algorithms.Instar(
    ...     n_inputs=4,
    ...     n_outputs=1,
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
        n_unconditioned = self.n_unconditioned
        weight = self.weight[n_unconditioned:, :]
        return self.step * dot(
            (input_row[:, n_unconditioned:].T - weight),
            layer_output.T
        )
