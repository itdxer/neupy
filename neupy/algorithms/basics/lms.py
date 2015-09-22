import numpy as np

from neupy.algorithms.basics.base import SimpleTwoLayerNetwork


__all__ = ('LMS',)


class LMS(SimpleTwoLayerNetwork):
    """ LMS Neural Network. Algorithm has several names, including the
    Widrow-Hoff or Delta rule. Algorithm similar to :network:`Perceptron`
    Neural Network, but has different idea behind learning process.

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
    >>> lmsnet = algorithms.LMS((2, 1), step=0.5, verbose=False)
    >>> lmsnet.train(input_data, target_data, epochs=200)
    >>> lmsnet.predict(np.array([[4, 4], [-1, -1]]))
    array([[-1],
           [ 1]])

    See Also
    --------
    :network:`Perceptron` : Perceptron Neural Network.
    """
    def get_weight_delta(self, output_train, target_train):
        input_data = self.input_data

        minimized_input = input_data / np.linalg.norm(input_data) ** 2
        return self.step * np.dot(minimized_input.T,
                                  target_train - self.summated)
