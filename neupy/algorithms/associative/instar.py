import numpy as np

from .base import BaseStepAssociative


__all__ = ('Instar',)


class Instar(BaseStepAssociative):
    """
    Instar is a simple unsupervised Neural Network algorithm
    which detects associations.

    Parameters
    ----------
    {BaseAssociative.Parameters}

    Methods
    -------
    {BaseAssociative.Methods}

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

        return self.step * np.dot(
            input_row[:, n_unconditioned:].T - weight,
            layer_output.T)
