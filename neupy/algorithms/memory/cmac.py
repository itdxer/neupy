import numpy as np

from neupy.utils import format_data
from neupy.core.properties import IntProperty
from neupy.algorithms.base import BaseNetwork


__all__ = ('CMAC',)


class CMAC(BaseNetwork):
    """
    Cerebellar Model Articulation Controller (CMAC) Network based on memory.

    Notes
    -----
    - Network always use Mean Absolute Error (MAE).
    - Network works for multi dimensional target values.

    Parameters
    ----------
    quantization : int
        Network transforms every input to discrete value.
        Quantization value controls number of total number of
        categories after quantization, defaults to ``10``.

    associative_unit_size : int
        Number of associative blocks in memory, defaults to ``2``.

    {BaseNetwork.Parameters}

    Attributes
    ----------
    weight : dict
        Network's weight that contains memorized patterns.

    Methods
    -------
    {BaseSkeleton.predict}

    train(X_train, y_train, X_test=None, y_test=None, epochs=100)
        Trains the network to the data X. Network trains until maximum
        number of ``epochs`` was reached.

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms import CMAC
    >>>
    >>> train_space = np.linspace(0, 2 * np.pi, 100)
    >>> test_space = np.linspace(np.pi, 2 * np.pi, 50)
    >>>
    >>> X_train = np.reshape(train_space, (100, 1))
    >>> X_test = np.reshape(test_space, (50, 1))
    >>>
    >>> y_train = np.sin(X_train)
    >>> y_test = np.sin(X_test)
    >>>
    >>> cmac = CMAC(
    ...     quantization=100,
    ...     associative_unit_size=32,
    ...     step=0.2,
    ... )
    ...
    >>> cmac.train(X_train, y_train, epochs=100)
    >>>
    >>> predicted_test = cmac.predict(X_test)
    >>> cmac.score(y_test, predicted_test)
    0.0023639417543036569
    """
    quantization = IntProperty(default=10, minval=1)
    associative_unit_size = IntProperty(default=2, minval=2)

    def __init__(self, **options):
        self.weight = {}
        super(CMAC, self).__init__(**options)

    def predict(self, X):
        X = format_data(X)

        get_memory_coords = self.get_memory_coords
        get_result_by_coords = self.get_result_by_coords
        predicted = []

        for input_sample in self.quantize(X):
            coords = get_memory_coords(input_sample)
            predicted.append(get_result_by_coords(coords))

        return np.array(predicted)

    def get_result_by_coords(self, coords):
        return sum(
            self.weight.setdefault(coord, 0) for coord in coords
        ) / self.associative_unit_size

    def get_memory_coords(self, quantized_value):
        assoc_unit_size = self.associative_unit_size

        for i in range(assoc_unit_size):
            point = ((quantized_value + i) / assoc_unit_size).astype(int)
            yield tuple(np.concatenate([point, [i]]))

    def quantize(self, X):
        return (X * self.quantization).astype(int)

    def one_training_update(self, X_train, y_train):
        get_memory_coords = self.get_memory_coords
        get_result_by_coords = self.get_result_by_coords
        weight = self.weight
        step = self.step

        n_samples = X_train.shape[0]
        quantized_input = self.quantize(X_train)
        errors = 0

        for input_sample, target_sample in zip(quantized_input, y_train):
            coords = list(get_memory_coords(input_sample))
            predicted = get_result_by_coords(coords)

            error = target_sample - predicted
            for coord in coords:
                weight[coord] += step * error

            errors += sum(abs(error))

        return errors / n_samples

    def score(self, X, y):
        predicted = self.predict(X)
        return np.mean(np.abs(predicted - y))

    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=100):
        is_test_data_partialy_missed = (
            (X_test is None and y_test is not None) or
            (X_test is not None and y_test is None)
        )

        if is_test_data_partialy_missed:
            raise ValueError(
                "Input and target test samples are missed. "
                "They must be defined together or none of them.")

        X_train = format_data(X_train)
        y_train = format_data(y_train)

        if X_test is not None:
            X_test = format_data(X_test)
            y_test = format_data(y_test)

        return super(CMAC, self).train(
            X_train, y_train, X_test, y_test, epochs=epochs)
