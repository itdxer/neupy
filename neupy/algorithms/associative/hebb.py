from neupy.core.properties import BoundedProperty
from .base import BaseStepAssociative


__all__ = ('HebbRule',)


class HebbRule(BaseStepAssociative):
    """
    Neural Network with Hebbian Learning. It's an unsupervised algorithm.
    Network can learn associations from the data.

    Notes
    -----
    - Network always generates weights that contains ``0``
      weight for the conditioned stimulus and ``1`` for the other.
      Such initialization helps to control your default state
      for the feature learning.

    Parameters
    ----------
    decay_rate : float
        Decay rate controls network's weights. It helps network to
        'forget' information and control weight's size. Without this
        parameter network's weights will increase fast.
        Defaults to ``0.2``.

    {BaseStepAssociative.Parameters}

    Methods
    -------
    {BaseStepAssociative.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> pavlov_dog_data = np.array([
    ...     [1, 0],  # food, no bell
    ...     [1, 1],  # food, bell
    ... ])
    >>> dog_test_cases = np.array([
    ...     [0, 0],  # no food, no bell
    ...     [0, 1],  # no food, bell
    ...     [1, 0],  # food, no bell
    ...     [1, 1],  # food, bell
    ... ])
    >>>
    >>> hebbnet = algorithms.HebbRule(
    ...     n_inputs=2,
    ...     n_outputs=1,
    ...     n_unconditioned=1,
    ...     step=0.1,
    ...     decay_rate=0.8,
    ...     verbose=False
    ... )
    >>> hebbnet.train(pavlov_dog_data, epochs=2)
    >>> hebbnet.predict(dog_test_cases)
    array([[0],
           [1],
           [1],
           [1]])
    """
    decay_rate = BoundedProperty(default=0.2, minval=0)

    def weight_delta(self, input_row, layer_output):
        n_unconditioned = self.n_unconditioned
        weight = self.weight[n_unconditioned:, :]
        delta = input_row[:, n_unconditioned:].T.dot(layer_output)
        return -self.decay_rate * weight + self.step * delta
