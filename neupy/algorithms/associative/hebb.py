from neupy.core.properties import BoundedProperty
from .base import BaseStepAssociative


__all__ = ('HebbRule',)


class HebbRule(BaseStepAssociative):
    """ Hebbian Learning Unsupervised Neural Network.
    Network can learn associations from data and emulate similar behaviour
    as dog in Pavlov experiment.

    Notes
    -----
    * Network always generate weights which contains ``0`` weight for \
    conditioned stimulus and ``1`` otherwise. This setup helps you controll \
    your default state for learning features. Other type of weight you can \
    setup as optional parameter ``weight`` in input layer.
    * No bias.

    Parameters
    ----------
    decay_rate : float
        Decay rate is control your network weights. It helps network
        'forgote' information and control weight sizes. Without this
        parameter network weight will grow. Defaults to ``0.2``.
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
    >>> from neupy import algorithms, layers
    >>>
    ... pavlov_dog_data = np.array([
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
    ...     layers.Step(2) > layers.Output(1),
    ...     n_unconditioned=1,
    ...     step=0.1,
    ...     decay_rate=0.8,
    ...     verbose=False
    ... )
    >>> hebbnet.train(pavlov_dog_data, epochs=2)
    >>> hebbnet.predict(dog_test_cases)
    array([[-1],
           [ 1],
           [ 1],
           [ 1]])
    """

    decay_rate = BoundedProperty(default=0.2, minval=0)

    def weight_delta(self, input_row, layer_output):
        n_unconditioned = self.n_unconditioned
        weight = self.weight[n_unconditioned:, :]
        delta = input_row[:, n_unconditioned:].T.dot(layer_output)
        return -self.decay_rate * weight + self.step * delta
