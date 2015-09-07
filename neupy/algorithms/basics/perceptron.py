import numpy as np

from neupy.algorithms.basics.base import SimpleTwoLayerNetwork


__all__ = ('Perceptron',)


class Perceptron(SimpleTwoLayerNetwork):
    """ Perceptron Neural Network. Simples linear model in Neural
    Networks.

    Parameters
    ----------
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
    >>> input_data = np.array([[1, 0], [2, 2], [3, 3], [0, 0]])
    >>> target_data = np.array([[1], [-1], [-1], [1]])
    >>>
    >>> mrnet = algorithms.Perceptron((2, 1), step=0.4, verbose=False)
    >>> mrnet.train(input_data, target_data, epochs=30)
    >>> mrnet.predict(np.array([[4, 4], [-1, -1]]))
    array([[-1],
           [ 1]])
    """
    def get_weight_delta(self, output_train, target_train):
        error_result = self.error(output_train, target_train)
        return np.dot(self.input_data.T, error_result)
