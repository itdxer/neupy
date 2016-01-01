from numpy import eye, newaxis, sign, isinf, clip, inner, outer
from numpy.linalg import norm

from neupy.core.properties import (ChoiceProperty, BoundedProperty,
                                   ProperFractionProperty)
from neupy.algorithms.utils import (matrix_list_in_one_vector,
                                    vector_to_list_of_matrix)
from neupy.network import StopNetworkTraining
from .base import GradientDescent


__all__ = ('BFGS',)


class BFGS(GradientDescent):
    """ BFGS  algorithm optimization.

    Parameters
    ----------
    {GradientDescent.optimizations}
    {ConstructableNetwork.connection}
    {SupervisedConstructableNetwork.error}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}
    {BaseNetwork.plot_errors}
    {BaseNetwork.last_error}
    {BaseNetwork.last_validation_error}
    {BaseNetwork.previous_error}

    Examples
    --------
    Simple example

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> qnnet = algorithms.QuasiNewton(
    ...     (2, 3, 1),
    ...     update_function='bfgs',
    ...     verbose=False
    ... )
    >>> qnnet.train(x_train, y_train)

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    """

    def train_epoch(self, input_train, target_train):
        train_epoch = self.methods.train_epoch
        prediction_error = self.methods.prediction_error

        import scipy
        scipy.optimize.fmin_bfgs(

        )
