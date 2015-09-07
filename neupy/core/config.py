from abc import ABCMeta
from collections import namedtuple
from functools import reduce

from six import with_metaclass

from .properties import Property
from .docs import docs


__all__ = ('ConfigMeta', 'ConfigWithABCMeta', 'Configurable',
           'ConfigurableWithABC')


Option = namedtuple('Option', 'class_name value')


def merge_dicts(left_dict, right_dict):
    return dict(left_dict, **right_dict)


class ConfigMeta(type):
    def __new__(cls, clsname, bases, attrs):
        parents = [kls for kls in bases if isinstance(kls, ConfigMeta)]
        new_class = super(ConfigMeta, cls).__new__(cls, clsname, bases, attrs)

        if new_class.__doc__ is not None:
            maindocs = docs.copy()
            mro_classes = new_class.__mro__

            # Collect parameter `shared_docs` for all MRO classes and
            # combine them in one big dictionary
            shared_docs = [getattr(b, 'shared_docs', {}) for b in mro_classes]
            all_params = reduce(merge_dicts, shared_docs, maindocs)

            new_class.__doc__ = new_class.__doc__.format(**all_params)

        if not hasattr(new_class, 'options'):
            new_class.options = {}

        # Populate parent classes options
        for base_class in parents:
            if hasattr(base_class, 'options'):
                new_class.options = dict(base_class.options,
                                         **new_class.options)

        # Set properties names and save options for different classes
        for key, value in attrs.items():
            if isinstance(value, Property):
                value.name = key
                new_class.options[key] = Option(
                    class_name=clsname,
                    value=value,
                )

        return new_class


class ConfigWithABCMeta(ABCMeta, ConfigMeta):
    pass


class BaseConfigurable(object):
    def __init__(self, **options):
        available_options = set(self.options.keys())
        invalid_options = set(options) - available_options

        if invalid_options:
            raise ValueError("Network `{}` contains invalid properties: "
                             "{}".format(self.__class__.__name__,
                                         ', '.join(invalid_options)))

        for key, value in options.items():
            setattr(self, key, value)


class Configurable(with_metaclass(ConfigMeta, BaseConfigurable)):
    pass


class ConfigurableWithABC(with_metaclass(ConfigWithABCMeta,
                                         BaseConfigurable)):
    pass
