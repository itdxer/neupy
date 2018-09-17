from abc import abstractmethod

from neupy.utils import preformat_value
from neupy.core.logs import Verbose
from .config import ConfigurableABC


__all__ = ('BaseSkeleton',)


class BaseSkeleton(ConfigurableABC, Verbose):
    """
    Base class for neural network algorithms.

    Methods
    -------
    fit(\*args, \*\*kwargs)
        Alias to the ``train`` method.

    predict(input_data)
        Predicts output for the specified input.
    """
    @abstractmethod
    def train(self, input_data, target_data):
        raise NotImplementedError

    def predict(self, input_data):
        raise NotImplementedError

    def transform(self, input_data):
        return self.predict(input_data)

    def fit(self, X, y, *args, **kwargs):
        self.train(X, y, *args, **kwargs)
        return self

    def class_name(self):
        return self.__class__.__name__

    def repr_options(self):
        options = []
        for option_name in self.options:
            option_value = getattr(self, option_name)
            option_value = preformat_value(option_value)

            option_repr = "{}={}".format(option_name, option_value)
            options.append(option_repr)

        return ', '.join(options)

    def __repr__(self):
        class_name = self.class_name()
        available_options = self.repr_options()
        return "{}({})".format(class_name, available_options)
