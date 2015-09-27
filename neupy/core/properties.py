import types

import numpy as np

from neupy.helpers.base import preformat_value


__all__ = ('Property', 'CheckSizeProperty', 'NumberProperty', 'BoolProperty',
           'BetweenZeroAndOneProperty', 'FuncProperty', 'NumberBoundProperty',
           'ArrayProperty', 'ListOfTypesProperty', 'ListProperty',
           'DictProperty', 'IntProperty', 'StringProperty',
           'NonNegativeIntProperty', 'NonNegativeNumberProperty',
           'ChoiceProperty')


class Property(object):
    expected_type = object
    disable = True

    def __init__(self, name=None, default=None, **options):
        self.name = name
        self.default = default

        for key, value in options.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            availabe_types = self.expected_type
            if isinstance(availabe_types, (list, tuple)):
                availabe_types = ', '.join(t.__name__ for t in availabe_types)
            else:
                availabe_types = availabe_types.__name__

            raise TypeError(
                "Wrong data type `{0}` for `{1}` property. Expected "
                "types: {2}".format(value.__class__.__name__, self.name,
                                    availabe_types)
            )

        self.validate(value)
        instance.__dict__[self.name] = value

    def __get__(self, instance, value):
        if instance is None:
            return

        if self.default is not None and self.name not in instance.__dict__:
            self.__set__(instance, self.default)

        return instance.__dict__.get(self.name, None)

    def __delete__(self, instance):
        name = self.name

        if name in instance.__dict__:
            del instance.__dict__[name]

        if name in instance.options:
            del instance.options[name]

    def __str__(self):
        return str(preformat_value(self.default))

    def __repr__(self):
        return self.__str__()

    def validate(self, value):
        pass


# -----------------------------------------------------#
#                   Typed properties                   #
# -----------------------------------------------------#


class StringProperty(Property):
    expected_type = str


class IntProperty(Property):
    expected_type = int


class NumberProperty(Property):
    expected_type = (float, int)


class BoolProperty(Property):
    expected_type = bool


class DictProperty(Property):
    expected_type = dict


class FuncProperty(Property):
    expected_type = types.FunctionType


class ArrayProperty(Property):
    expected_type = np.ndarray


class ListProperty(Property):
    expected_type = list


# -----------------------------------------------------#
#                 Special properties                   #
# -----------------------------------------------------#


class ListOfTypesProperty(Property):
    expected_type = (list, tuple, set)
    inner_list_type = int
    count = None

    def validate(self, value):
        super(ListOfTypesProperty, self).validate(value)

        if self.count is not None and len(value) != self.count:
            raise ValueError("Expected list with {} variables".format(
                self.count
            ))

        if not all(isinstance(v, self.inner_list_type) for v in value):
            raise TypeError("Expected list with {}".format(
                self.inner_list_type.__name__
            ))


class NumberBoundProperty(ListOfTypesProperty):
    count_of_values = 2


class VectorProperty(Property):
    expected_type = np.ndarray

    def validate(self, value):
        if value.ndim != 1:
            raise ValueError(
                "Value `{}` must be 1-D shape vector".format(self.name)
            )


class Matrix2DProperty(Property):
    expected_type = (np.ndarray, np.matrix)

    def validate(self, value):
        if value.ndim != 2:
            raise ValueError(
                "Value `{}` must be 2-D shape array/matrix".format(self.name)
            )


# -----------------------------------------------------#
#                  Choices properties                  #
# -----------------------------------------------------#


class ChoiceProperty(Property):
    choices = {}

    def __init__(self, *args, **kwargs):
        super(ChoiceProperty, self).__init__(*args, **kwargs)
        choices = self.choices

        if isinstance(choices, (list, tuple)):
            self.choices = dict(zip(choices, choices))

        if not isinstance(self.choices, dict):
            raise ValueError("Choice properties can be only a `dict`, got "
                             "`{0}`".format(self.choices.__class__.__name__))

        if not self.choices:
            raise ValueError("Must be at least one choice in property "
                             "`{0}`".format(self.name))

    def __set__(self, instance, value):
        if value not in self.choices:
            raise ValueError(
                "Wrong value `{0}` for property `{1}`. Available values: "
                "{2}".format(value, self.name, ", ".join(self.choices.keys()))
            )
        return super(ChoiceProperty, self).__set__(instance, value)

    def __get__(self, instance, value):
        if instance is not None:
            choice_key = super(ChoiceProperty, self).__get__(instance, value)
            return self.choices[choice_key]


# -----------------------------------------------------#
#                   Sized properties                   #
# -----------------------------------------------------#


class CheckSizeProperty(Property):
    min_size = -np.inf
    max_size = np.inf

    def __init__(self, min_size=None, max_size=None, *args, **kwargs):
        if min_size is not None:
            self.min_size = min_size

        if max_size is not None:
            self.max_size = max_size

        super(CheckSizeProperty, self).__init__(*args, **kwargs)

    def validate(self, value):
        if not self.min_size <= value <= self.max_size:
            raise ValueError("Value `{}` must be between {} and {}".format(
                self.name, self.min_size, self.max_size
            ))


class BetweenZeroAndOneProperty(NumberProperty, CheckSizeProperty):
    min_size = 0
    max_size = 1


class NonNegativeIntProperty(IntProperty, CheckSizeProperty):
    min_size = 0


class NonNegativeNumberProperty(NumberProperty, CheckSizeProperty):
    min_size = 0
