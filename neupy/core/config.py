from abc import ABCMeta
from collections import namedtuple

import numpy as np
import tensorflow as tf
from six import with_metaclass

from neupy.utils import tensorflow_eval
from .properties import BaseProperty, WithdrawProperty
from .docs import SharedDocsMeta


__all__ = ('ConfigMeta', 'ConfigABCMeta', 'Configurable', 'ConfigurableABC',
           'ExtractParameters', 'DumpableObject')


Option = namedtuple('Option', 'class_name value')


class ExtractParameters(object):
    def get_params(self, deep=False):
        options = {}

        for property_name, option in self.options.items():
            value = getattr(self, property_name)

            if isinstance(value, tf.Variable):
                value = tensorflow_eval(value)

            property_ = option.value
            is_numpy_array = isinstance(value, np.ndarray)

            if hasattr(option.value, 'choices'):
                choices = property_.choices

                if not is_numpy_array and value in choices.values():
                    choices = {v: k for k, v in choices.items()}
                    value = choices[value]

            options[property_name] = value

        return options

    def set_params(self, **params):
        self.__dict__.update(params)
        return self


def initialize_with_kwargs(class_, kwargs):
    return class_(**kwargs)


class DumpableObject(ExtractParameters):
    def __reduce__(self):
        return initialize_with_kwargs, (self.__class__, self.get_params())


class ConfigMeta(SharedDocsMeta):
    """
    Meta-class that configure initialized properties. Also it helps
    inheit properties from parent classes and use them.
    """
    def __new__(cls, clsname, bases, attrs):
        new_class = super(ConfigMeta, cls).__new__(cls, clsname, bases, attrs)
        parents = [kls for kls in bases if isinstance(kls, ConfigMeta)]

        if not hasattr(new_class, 'options'):
            new_class.options = {}

        for base_class in parents:
            new_class.options = dict(base_class.options, **new_class.options)

        options = new_class.options

        # Set properties names and save options for different classes
        for key, value in attrs.items():
            if isinstance(value, BaseProperty):
                value.name = key
                options[key] = Option(class_name=clsname, value=value)

            if isinstance(value, WithdrawProperty) and key in options:
                del options[key]

        return new_class


class BaseConfigurable(ExtractParameters):
    """
    Base configuration class. It help set up and validate
    initialized property values.

    Parameters
    ----------
    **options
        Available properties.
    """
    def __init__(self, **options):
        available_options = set(self.options.keys())
        invalid_options = set(options) - available_options

        if invalid_options:
            clsname = self.__class__.__name__
            raise ValueError(
                "The `{}` object contains invalid properties: {}"
                "".format(clsname, ', '.join(invalid_options)))

        for key, value in options.items():
            setattr(self, key, value)

        for option_name, option in self.options.items():
            if option.value.required and option_name not in options:
                raise ValueError(
                    "Option `{}` is required.".format(option_name))


class Configurable(with_metaclass(ConfigMeta, BaseConfigurable)):
    """
    Class that combine ``BaseConfigurable`` class functionality and
    ``ConfigMeta`` meta-class.
    """


class ConfigABCMeta(ABCMeta, ConfigMeta):
    """
    Meta-class that combines ``ConfigMeta`` and ``abc.ABCMeta``
    meta-classes.
    """


class ConfigurableABC(with_metaclass(ConfigABCMeta, BaseConfigurable)):
    """
    Class that combine ``BaseConfigurable`` class functionality,
    ``ConfigMeta`` and ``abc.ABCMeta`` meta-classes.
    """
