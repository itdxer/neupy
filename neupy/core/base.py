from abc import abstractmethod

from neupy.helpers import preformat_value
from neupy.helpers.logs import Verbose
from .config import ConfigurableWithABC


__all__ = ('BaseSkeleton',)


class BaseSkeleton(ConfigurableWithABC, Verbose):
    """ Base class for all algorithms and networks.
    """
    def get_params(self, deep=False):
        options = {}
        for name, option in self.options.items():
            value = getattr(self, name)
            # Default value is not always valid type. For this reason we
            # need to ignore all the values that have the same value as
            # in default attibute.
            if value != option.value.default:
                options[name] = value
        return options

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

    @abstractmethod
    def train(self, input_data, target_data):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    def fit(self, X, y, *args, **kwargs):
        self.train(X, y, *args, **kwargs)
        return self

    def _repr_options(self):
        options = []
        for option_name in self.options:
            option_value = getattr(self, option_name)
            option_value = preformat_value(option_value)

            option_repr = "{}={}".format(option_name, option_value)
            options.append(option_repr)

        return ', '.join(options)

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self._repr_options()
        )
